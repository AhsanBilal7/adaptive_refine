"""
Unit tests for src/arr (CPU only, no model download).

    python -m unittest tests/test_arr.py -v
"""

import os
import sys
import tempfile
import unittest
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arr.config import ARRConfig, config_from_env, load_shell_env, validate_config  # noqa: E402
from src.arr.engine import Request, prompt_lookup  # noqa: E402
from src.arr.ledger import FlopModel, Ledger  # noqa: E402
from src.arr.methods import Solve  # noqa: E402
from src.arr.runner import auroc, depth_accounting, drive, grade, summarize  # noqa: E402
from src.arr.tasks import get_task, sanity_check_tasks  # noqa: E402
from src.arr.sanity import engine_checks  # noqa: E402
from src.arr.stats import paired_compare  # noqa: E402
from src.arr.voting import Grader  # noqa: E402

EOS = 0


class FakeEngine:
    """Scripted engine: ``script(req) -> (out_ids, text, finish)``; every call costs ``cost``."""

    def __init__(self, script, cost=10.0):
        self.script, self.cost = script, cost
        self.queue, self.rows, self.n_rid = deque(), 4, 0
        self.stats = dict(passes=0, padded_positions=0, real_positions=1)
        self.log = []

    def format(self, system, user):
        return [7] * 5

    def worst_case(self, plen, max_new, measure):
        return self.cost

    def submit(self, r):
        r.rid, self.n_rid = self.n_rid, self.n_rid + 1
        r.wc = self.cost
        self.queue.append(r)

    @property
    def busy(self):
        return bool(self.queue)

    def n_running(self):
        return 0

    def step(self):
        done = []
        while self.queue:
            r = self.queue.popleft()
            self.log.append((r.role, r.tag[1].id if r.tag and hasattr(r.tag[1], "id") else None))
            if not r.ledger.affordable(self.cost, 0):
                r.finish = "budget"
            else:
                r.ledger.charge(self.cost, 0, r.role, "decoded", 1)
                r.spent = self.cost
                r.out, r.text, r.finish = self.script(r)
            r.stats["passes"] = 1
            done.append(r)
        return done


def cohort(answers, finish=None):
    finish = finish or ["eos"] * len(answers)
    return [dict(ids=[10 + k, a, EOS], text=f"\\boxed{{{a}}}", finish=f, tflops=10.0, passes=1,
                 stats=dict(prefill=5, decoded=3)) for k, (a, f) in enumerate(zip(answers, finish))]


def run(method, script, answers, budget, n=2, finish=None, **kw):
    cfg = ARRConfig(n_active=n, max_new_tokens=16, fixed_depth=kw.get("D", 3))
    eng = FakeEngine(script)
    s = Solve(method, cfg, eng, "q:0000", "What?", 0, budget, cohort(answers, finish))
    drive(eng, [s], lambda x: None, cap=4, label="t", every=1e9)
    return s, eng


def fixed(r):                     # the revision reproduces the draft exactly → fixed point
    if r.role == "sample":
        return [99, 5, EOS], "\\boxed{5}", "eos"
    return list(r.draft), "same", "eos"


class TestDecoder(unittest.TestCase):
    def test_replay_equals_greedy_and_fixed_points(self):
        self.assertEqual(engine_checks(), [])

    def test_prompt_lookup(self):
        import numpy as np
        seq = np.array([5, 6, 7, 8, 9, 1, 2, 6, 7])
        self.assertEqual(prompt_lookup(seq, 3, 3), [8, 9, 1])
        self.assertEqual(prompt_lookup(np.array([1, 2, 3]), 3, 3), [])

    def test_flops_and_ledger(self):
        fm = FlopModel(params=1e9, n_layers=2, attn_dim=8)
        self.assertAlmostEqual(fm.positions(0, 3), 2e9 * 3 + 4 * 2 * 8 * (1 + 2 + 3))
        self.assertAlmostEqual(fm.positions(10, 2), fm.positions(10, 1) + fm.positions(11, 1))
        led = Ledger(5.0, "passes")
        self.assertTrue(led.affordable(1e9, 1))
        led.charge(1.0, 1, "x", "decoded", 3)
        led.charge(1.0, 1, "x", "decoded", 3)      # same pass → charged once in passes
        self.assertEqual(led.spent, 1.0)
        self.assertEqual(led.counts["decoded"], 6)


class TestAllocation(unittest.TestCase):
    def test_fixed_point_starts_new_roots_until_budget(self):
        s, _ = run("arr", fixed, ["3", "4"], budget=60.0)
        rec = s.record()
        # cohort 20, then per root: revise (fixed) → sample new → revise ... until a complete attempt no longer fits
        self.assertGreater(len(rec["roots"]), 2)
        self.assertTrue(all(r["status"] in ("fixed_point", "active", "budget") for r in rec["roots"]))
        self.assertLessEqual(rec["cost"]["spent"], 60.0 + 1e-9)
        self.assertEqual(rec["n_voting_roots"], sum(r["valid"] for r in rec["roots"]))

    def test_no_replace_ablation_keeps_width(self):
        s, _ = run("arr_no_replace", fixed, ["3", "4"], budget=200.0)
        self.assertEqual(len(s.roots), 2)
        self.assertTrue(all(r.status == "fixed_point" for r in s.roots))

    def test_changed_revision_is_revisited_and_votes_once(self):
        def script(r):
            if r.role == "sample":
                return [99, 5, EOS], "\\boxed{5}", "eos"
            if len(r.draft) < 5:                          # change twice, then reproduce
                return list(r.draft[:-1]) + [8, EOS], "\\boxed{8}", "eos"
            return list(r.draft), "same", "eos"
        s, _ = run("arr", script, ["3"], budget=45.0, n=1)
        r0 = s.roots[0]
        self.assertEqual(r0.answers[0], "3")
        self.assertEqual(r0.answers[1], "8")
        self.assertEqual(r0.status, "fixed_point")
        self.assertEqual(r0.revisions, 3)
        (norm, final, counts), voters = s.vote()
        self.assertEqual(sum(counts.values()), len(voters))   # one vote per root

    def test_incomplete_revision_keeps_previous_and_is_not_retried(self):
        def script(r):
            if r.role == "revise":
                return list(r.draft) + [4, 4, 4], "", "length"
            return [99, 5, EOS], "\\boxed{5}", "eos"
        s, eng = run("arr", script, ["3", "4"], budget=500.0)
        self.assertTrue(all(r.status == "incomplete" for r in s.roots))
        self.assertEqual([r.answer for r in s.roots], ["3", "4"])
        self.assertEqual(sum(1 for role, _ in eng.log if role == "revise"), 2)
        self.assertEqual(len(s.roots), 2)                          # no new root after an incomplete revision

    def test_invalid_initial_sample_releases_slot(self):
        s, _ = run("arr", fixed, ["3", "4"], budget=40.0, finish=["length", "eos"])
        self.assertEqual(s.roots[0].status, "invalid")
        self.assertFalse(s.roots[0].valid)
        self.assertGreaterEqual(len(s.roots), 3)                   # slot 0 got a new root

    def test_direct_fixed_runs_exactly_D(self):
        s, eng = run("direct_fixed", fixed, ["3", "4"], budget=1e9, D=3)
        self.assertTrue(all(r.revisions == 3 and r.status == "depth_done" for r in s.roots))
        self.assertEqual(sum(1 for role, _ in eng.log if role == "revise"), 6)

    def test_majority_spends_the_budget_on_samples(self):
        s, eng = run("maj", fixed, ["3", "4"], budget=100.0)
        self.assertTrue(all(role == "sample" for role, _ in eng.log))
        self.assertEqual(len(s.roots), 2 + len(eng.log))
        self.assertLessEqual(s.ledger.spent, 100.0 + 1e-9)

    def test_critic_correct_pipeline(self):
        def script(r):
            if r.role == "critique":
                return [1, EOS], "The solution appears correct.", "eos"
            if r.role in ("sample", "continue"):
                return [99, 6, EOS], "\\boxed{6}", "eos"
            return (list(r.draft) if r.draft else [5, EOS]), "\\boxed{6}", "eos"
        s, eng = run("critic_correct", script, ["3"], budget=35.0, n=1)
        self.assertEqual([role for role, _ in eng.log][:2], ["critique", "correct"])


class TestReporting(unittest.TestCase):
    def test_grade_summary_and_telescoping(self):
        recs = []
        for q, (ans, gold) in enumerate([(["3", "4"], "4"), (["4", "4"], "4"), (["1", "2"], "4")]):
            s, _ = run("arr", fixed, ans, budget=60.0)
            rec = s.record()
            rec["qid"] = f"q:{q}"
            recs.append(grade(rec, gold, get_task("math500"), 2))
        summ = summarize(recs, 32)
        self.assertEqual(summ["n_questions"], 3)
        for k in ("vote_share", "vote_margin", "fixed_point_support", "output_tokens", "n_model_calls"):
            self.assertIn(k, recs[0])
        self.assertGreater(summ["mean_output_tokens_per_question"], 0)
        self.assertIn("auroc_vote_share", summ)
        self.assertIn("0", summ["correct_k_of_n"])
        acc = depth_accounting([dict(cohort_vote_correct_by_depth=t) for t in
                                ([False, True, True], [True, False], [True], [False, False, True, False])])
        self.assertTrue(acc["telescopes"])
        self.assertAlmostEqual(acc["A"][-1] - acc["A"][0], sum(acc["rho_rec"]) - sum(acc["rho_reg"]))

    def test_auroc(self):
        self.assertAlmostEqual(auroc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]), 0.75)
        self.assertAlmostEqual(auroc([1, 1, 1], [1, 0, 1]), 0.5)
        self.assertIsNone(auroc([0.5, 0.6], [1, 1]))

    def test_benchmark_tasks_roundtrip(self):
        # official scorers: every golden NATURAL PLAN plan is graded correct after extraction,
        # LiveBench answers are graded by the 2024 release scorers
        self.assertEqual(sanity_check_tasks(), [])

    def test_non_math_task_in_solve(self):
        t = get_task("np_calendar")
        meta = dict(task="np_calendar")
        gold = "Here is the proposed time: Monday, 14:30 - 15:30"
        a, n = t.extract("Maybe Monday, 9:00 - 9:30? ... Here is the proposed time: Monday, 14:30 - 15:30", meta)
        self.assertEqual(a, "Monday, 14:30 - 15:30")
        self.assertTrue(t.grade(a, gold, meta))
        self.assertFalse(t.grade("Monday, 9:00 - 10:00", gold, meta))
        self.assertNotIn("\\boxed", t.format_text("End with exactly one line: Final answer: \\boxed{<answer>}"))

    def test_paired_compare(self):
        a = [dict(qid=f"q{i}", correct=1.0) for i in range(10)]
        b = [dict(qid=f"q{i}", correct=float(i % 2)) for i in range(10)]
        out = paired_compare(a, b, "correct")
        self.assertEqual(out["n_questions"], 10)
        self.assertAlmostEqual(out["mean_diff"], 0.5)


class TestConfig(unittest.TestCase):
    def test_env_and_conf_precedence(self):
        with tempfile.TemporaryDirectory() as d:
            env = os.path.join(d, ".env")
            conf = os.path.join(d, "m.conf")
            with open(env, "w") as f:
                f.write('export ARR_N_ACTIVE="${ARR_N_ACTIVE:-4}"\nexport ARR_BUDGETS="${ARR_BUDGETS:-1,2}"\n')
            with open(conf, "w") as f:
                f.write('ARR_BUDGETS="100,200"\nARR_BLOCK_SIZE=16\n')
            load_shell_env([env, conf])
            cfg = config_from_env()
            validate_config(cfg)
            self.assertEqual(cfg.budget_list(), [100.0, 200.0])
            self.assertEqual(cfg.block_size, 16)
            for k in ("ARR_N_ACTIVE", "ARR_BUDGETS", "ARR_BLOCK_SIZE"):
                os.environ.pop(k, None)


if __name__ == "__main__":
    unittest.main()
