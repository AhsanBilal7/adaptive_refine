"""
Unit tests for src/rsi (no GPU / no model download needed).

    python -m unittest tests/test_rsi.py -v
"""

import json
import os
import random
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rsi.backends import FakeBackend, GenRequest  # noqa: E402
from src.rsi.config import RSIConfig, config_from_env, load_shell_env, validate_config  # noqa: E402
from src.rsi.counterfactual import compare_actions, evidence_yield, solve_and_log  # noqa: E402
from src.rsi.editor import LeakGuard, Patch, apply_patch, compile_patches  # noqa: E402
from src.rsi.memory import EvidenceMemory  # noqa: E402
from src.rsi.procedure import Decision, Plan, Procedure, Rule, h0_procedure, heuristic_procedure  # noqa: E402
from src.rsi.runner import run_solves, summarize  # noqa: E402
from src.rsi.scheduler import run_generators  # noqa: E402
from src.rsi.solver import SolveContext, solve_gen  # noqa: E402
from src.rsi.stats import acceptance_test, required_block_size, sign_flip_pvalue  # noqa: E402
from src.rsi.voting import Grader, extract_answer, normalize_answer, plurality_vote, sanity_check_grader  # noqa: E402


def tiny_cfg(**kw) -> RSIConfig:
    cfg = RSIConfig(n_slots=3, h0_depth=2, budget_cycles=2, cost_measure="calls",
                    init_max_new_tokens=50, rewrite_max_new_tokens=50, critique_max_new_tokens=30,
                    correct_max_new_tokens=50, cf_seeds=2, cf_states_per_problem=1, rounds=1,
                    search_batch_size=3, confirm_block_size=3, min_support=1, rule_delta_min=0.0,
                    pair_repro_frac=0.0, editor_mode="compiler", editor_num_proposals=4,
                    permutation_samples=200)
    for k, v in kw.items():
        setattr(cfg, k, v)
    validate_config(cfg)
    return cfg


def responder_factory(correct_prob=0.5, empty_roles=()):
    """Answers depend on the request seed so different seeds disagree."""
    def respond(req: GenRequest, n: int) -> str:
        if req.role in empty_roles:
            return ""
        if req.role == "critique":
            return "Step 2 has an arithmetic slip.\nERROR CATEGORY: arithmetic" if (req.seed or 0) % 2 \
                else "The solution appears correct.\nERROR CATEGORY: no_specific_error"
        if req.role == "editor":
            return json.dumps({"type": "edit_prompt", "prompt_key": "critique_extra",
                               "prompt_text": "Check every sign carefully.",
                               "supporting_evidence": ["E0"], "contradicting_evidence": []})
        r = random.Random(req.seed or n)
        ans = "4" if r.random() < correct_prob else str(r.choice([3, 5, 7]))
        return f"Step 1: compute.\nFinal answer: \\boxed{{{ans}}}"
    return respond


class TestVoting(unittest.TestCase):
    def test_extract_and_normalize(self):
        self.assertEqual(extract_answer("x \\boxed{\\frac{1}{2}} y"), "\\frac{1}{2}")
        self.assertEqual(extract_answer("no box", fallback_last_number=False), "")
        self.assertEqual(normalize_answer("3.500"), "3.5")
        self.assertEqual(normalize_answer("{7}"), "7")
        self.assertEqual(normalize_answer("\\dfrac{3}{4}"), normalize_answer("\\frac{3}{4}"))

    def test_plurality_tiebreak_lowest_slot(self):
        w, _, _ = plurality_vote(["5", "4", "4", "5"])
        self.assertEqual(w, "5")                   # tie 2–2: answer of slot 0 wins
        w, disp, _ = plurality_vote(["", "", ""])
        self.assertEqual(w, "")
        self.assertEqual(disp, "[invalid]")
        w, _, _ = plurality_vote(["", "4", "7"])  # abstain + tie → lowest supporting slot
        self.assertEqual(w, "4")

    def test_grader_sanity(self):
        self.assertEqual(sanity_check_grader(Grader("math500")), [])


class TestInnerLoop(unittest.TestCase):
    def run_one(self, cfg, proc, responder=None, forced=None):
        be = FakeBackend(responder or responder_factory())
        ctx = SolveContext(cfg, be)
        st = ctx.new_state("q0", "What is 2+2?", seed_base=11)
        res = run_generators(be, [solve_gen(ctx, st, proc, forced=forced)])[0]
        return res, be

    def test_h0_reproduces_paper_schedule(self):
        cfg = tiny_cfg()
        res, be = self.run_one(cfg, h0_procedure(2))
        st = res.state
        # N initial + N * D * 3 calls  (paper: N(1+3D))
        self.assertEqual(st.meter.calls, 3 * (1 + 3 * 2))
        self.assertEqual([s.depth for s in st.slots], [2, 2, 2])
        self.assertEqual(st.stop_reason, "H0:done")
        self.assertLessEqual(st.meter.spent, st.meter.budget + 1e-9)
        # round-robin order: slot 0,1,2,0,1,2
        self.assertEqual([d["slot"] for d in st.decisions if d["action"] == "FULL_REFINE"], [0, 1, 2, 0, 1, 2])

    def test_budget_cap_stops_early(self):
        cfg = tiny_cfg(budget_cycles=1)
        res, _ = self.run_one(cfg, h0_procedure(3))
        st = res.state
        self.assertEqual(st.meter.calls, 3 + 3 * 3)
        self.assertEqual(st.stop_reason, "H0:budget")
        self.assertLessEqual(st.meter.spent, st.meter.budget + 1e-9)

    def test_budget_tokens_reservation_never_exceeded(self):
        cfg = tiny_cfg(cost_measure="tokens", budget_cycles=1.5)
        res, _ = self.run_one(cfg, h0_procedure(4))
        st = res.state
        self.assertLessEqual(st.meter.spent, st.meter.budget + 1e-9)
        self.assertGreater(st.meter.spent, 0)

    def test_incomplete_op_leaves_slot_but_charges(self):
        cfg = tiny_cfg()
        res, _ = self.run_one(cfg, h0_procedure(1), responder=responder_factory(empty_roles=("correct",)))
        st = res.state
        for s in st.slots:
            self.assertEqual(s.depth, 0)             # never committed
            self.assertEqual(len(s.history), 1)
        # every failed attempt is charged; H0 keeps retrying the unrefined slots
        # until the cap (3 init + 6 FULL_REFINE attempts × 3 calls) is exhausted
        self.assertEqual(st.meter.calls, 3 + 6 * 3)
        self.assertEqual(st.stop_reason, "H0:budget")
        self.assertLessEqual(st.meter.spent, st.meter.budget + 1e-9)

    def test_keep_freezes_and_terminates(self):
        cfg = tiny_cfg()
        proc = Procedure(items=[Rule("k", [], "KEEP")], h0_depth=2)
        res, _ = self.run_one(cfg, proc)
        st = res.state
        self.assertTrue(all(s.frozen for s in st.slots))
        self.assertEqual(st.meter.calls, 3)
        self.assertEqual(st.stop_reason, "H0:done")

    def test_restart_loop_hits_budget_or_cap(self):
        cfg = tiny_cfg()
        proc = Procedure(items=[Rule("rs", [], "RESTART")], h0_depth=2)
        res, _ = self.run_one(cfg, proc)
        st = res.state
        self.assertIn(st.stop_reason, ("H0:budget", "decision_cap", "H0:done"))
        self.assertLessEqual(st.meter.spent, st.meter.budget + 1e-9)
        self.assertEqual(len(st.slots), 3)           # restarts replace, never add votes

    def test_pair_plan_and_guard(self):
        cfg = tiny_cfg()
        plan = Plan("p", first_guard=[["depth", "==", 0]], second_guard=[["depth", "==", 0]],
                    first_op="DIRECT_REPAIR", second_op="RESTART",
                    continuation_guard=[["valid", "==", 1]])
        proc = Procedure(items=[plan], h0_depth=1)
        res, _ = self.run_one(cfg, proc)
        pairs = [d for d in res.state.decisions if d["action"] == "PAIR"]
        self.assertTrue(pairs)
        d = pairs[0]
        self.assertNotEqual(d["slot"], d["slot2"])
        self.assertTrue(d["guard_passed"])
        self.assertTrue(d["second"]["complete"])

    def test_pair_guard_blocks_second_op(self):
        cfg = tiny_cfg()
        plan = Plan("p", first_guard=[], second_guard=[], first_op="RESTART", second_op="RESTART",
                    continuation_guard=[["depth", ">=", 99]])
        proc = Procedure(items=[plan], h0_depth=1)
        res, _ = self.run_one(cfg, proc)
        d = [d for d in res.state.decisions if d["action"] == "PAIR"][0]
        self.assertFalse(d["guard_passed"])
        self.assertTrue(d["second"].get("skipped"))

    def test_forced_infeasible(self):
        cfg = tiny_cfg(budget_cycles=0.1)
        res, _ = self.run_one(cfg, h0_procedure(2), forced=[Decision("FULL_REFINE", slot=0)])
        self.assertTrue(res.infeasible_forced)

    def test_heuristic_and_maj_matched_run(self):
        cfg = tiny_cfg()
        be = FakeBackend(responder_factory())
        ctx = SolveContext(cfg, be)
        probs = [dict(qid=f"q{i}", question=f"What is 2+2? ({i})", gold="4") for i in range(4)]
        g = Grader("math500")
        for kind, proc in (("procedure", heuristic_procedure(2)), ("maj_matched", None)):
            recs = run_solves(ctx, be, probs, kind, proc, [0, 1], g, 0.2)
            s = summarize(recs)
            self.assertEqual(s["n_solves"], 8)
            self.assertTrue(all(r["cost"]["spent"] <= r["cost"]["budget"] + 1e-9 for r in recs))
        # compute-matched majority: B affords N + N*cycles*3 calls → 3 + 18 samples
        self.assertEqual(recs[0]["n_votes"], 21)

    def test_scheduler_batches_across_solves(self):
        cfg = tiny_cfg()
        be = FakeBackend(responder_factory())
        ctx = SolveContext(cfg, be)
        gens = [solve_gen(ctx, ctx.new_state(f"q{i}", "What is 2+2?", i), h0_procedure(1)) for i in range(5)]
        run_generators(be, gens)
        self.assertEqual(max(be.batch_sizes), 15)   # 5 solves × 3 initial samples in one batch


class TestCounterfactual(unittest.TestCase):
    def test_compare_actions_and_interaction(self):
        cfg = tiny_cfg(cf_explore_frac=1.0)
        be = FakeBackend(responder_factory(0.4))
        ctx = SolveContext(cfg, be)
        probs = [dict(qid=f"q{i}", question=f"What is 2+2? ({i})", gold="4") for i in range(3)]
        g = Grader("math500")
        rng = random.Random(0)
        logged, snaps = solve_and_log(ctx, be, probs, h0_procedure(2), g, rng)
        self.assertEqual(len(snaps), 3)
        ev = compare_actions(ctx, be, snaps, h0_procedure(2), {p["qid"]: "4" for p in probs}, g, rng, 0)
        kinds = {e["kind"] for e in ev}
        self.assertTrue({"incumbent", "single", "stop", "pair"} <= kinds)
        for e in ev:
            if e["kind"] == "pair" and e["status"] == "ok":
                v0, va, vb, vab = e["value_0"]["Q"], e["value_a"]["Q"], e["value_b"]["Q"], e["value_ab"]["Q"]
                self.assertAlmostEqual(e["I"], vab - va - vb + v0)
            if e["kind"] == "single" and e.get("delta") is not None:
                self.assertIn("parent_correct", e)
        y = evidence_yield(ev)
        self.assertEqual(y["n_states"], 3)
        # memory + compiler round-trip
        with tempfile.TemporaryDirectory() as d:
            mem = EvidenceMemory(os.path.join(d, "m.jsonl"))
            mem.add(ev)
            self.assertTrue(mem.summary_text())
            mem2 = EvidenceMemory(os.path.join(d, "m.jsonl"))
            self.assertEqual(len(mem2.records), len(ev))
            patches = compile_patches(mem, h0_procedure(2), cfg, rng, limit=10)
            for pt in patches:
                newp, errs = apply_patch(h0_procedure(2), pt, cfg, None)
                self.assertIsNotNone(newp, errs)


class TestEditor(unittest.TestCase):
    def test_validation_rejects_bad_rules(self):
        cfg = tiny_cfg()
        bad = Patch("add_rule", rule=dict(when=[["problem_id", "==", 3]], action="RESTART"),
                    supporting_evidence=["E1"])
        p, errs = apply_patch(h0_procedure(2), bad, cfg, None)
        self.assertIsNone(p)
        self.assertTrue(any("unknown feature" in e for e in errs))
        nosupport = Patch("add_rule", rule=dict(when=[["depth", ">=", 1]], action="KEEP"))
        self.assertIsNone(apply_patch(h0_procedure(2), nosupport, cfg, None)[0])
        glob = Patch("add_rule", rule=dict(when=[], action="STOP", scope="global"), supporting_evidence=["E1"])
        self.assertIsNone(apply_patch(h0_procedure(2), glob, cfg, None)[0])

    def test_leak_guard(self):
        cfg = tiny_cfg()
        guard = LeakGuard(["Find the number of ordered pairs of integers whose sum is ten and product six"],
                          ["\\frac{3}{5}", "123"])
        for text in ("The answer to problem 12 is 123.", "Remember answers 11, 22, 33, 44, 55.",
                     "find the number of ordered pairs of integers whose sum is ten",
                     "If unsure use 123"):
            pt = Patch("edit_prompt", prompt_key="critique_extra", prompt_text=text, supporting_evidence=["E1"])
            self.assertIsNone(apply_patch(h0_procedure(2), pt, cfg, guard)[0], text)
        ok = Patch("edit_prompt", prompt_key="critique_extra", prompt_text="Re-check each sign change.",
                   supporting_evidence=["E1"])
        self.assertIsNotNone(apply_patch(h0_procedure(2), ok, cfg, guard)[0])
        nonedit = Patch("edit_prompt", prompt_key="initial_system", prompt_text="x", supporting_evidence=["E1"])
        self.assertIsNone(apply_patch(h0_procedure(2), nonedit, cfg, guard)[0])

    def test_procedure_roundtrip_and_hash(self):
        p = heuristic_procedure(2)
        q = Procedure.from_dict(json.loads(json.dumps(p.to_dict())))
        self.assertEqual(p.hash(), q.hash())


class TestStats(unittest.TestCase):
    def test_sign_flip(self):
        import numpy as np
        self.assertLess(sign_flip_pvalue(np.ones(10) * 0.5), 0.01)
        self.assertGreater(sign_flip_pvalue(np.zeros(10)), 0.5)

    def test_acceptance_and_block_size(self):
        cand = [dict(qid=f"q{i}", utility=1.0, correct=True) for i in range(30)]
        inc = [dict(qid=f"q{i}", utility=0.0 if i < 20 else 1.0, correct=i >= 20) for i in range(30)]
        res = acceptance_test(cand, inc, 0.01, 0.02, n_samples=2000)
        self.assertTrue(res["accept"])
        self.assertGreater(required_block_size(0.2, 0.02, 0.01), 1000)


class TestConfig(unittest.TestCase):
    def test_env_and_conf_precedence(self):
        with tempfile.TemporaryDirectory() as d:
            env = os.path.join(d, ".env")
            conf = os.path.join(d, "x.conf")
            with open(env, "w") as f:
                f.write('export RSI_N_SLOTS="${RSI_N_SLOTS:-6}"\nexport RSI_CF_SEEDS="${RSI_CF_SEEDS:-3}"\n'
                        'if [[ -z "${FOO_X:-}" ]]; then export FOO_X=1; fi\n')
            with open(conf, "w") as f:
                f.write('RSI_CF_SEEDS=5\nMODEL_NAME="org/model-b"\n')
            saved = dict(os.environ)
            try:
                for k in ("RSI_N_SLOTS", "RSI_CF_SEEDS", "MODEL_NAME", "RSI_MODEL_NAME"):
                    os.environ.pop(k, None)
                load_shell_env([env, conf])
                cfg = config_from_env()
                self.assertEqual(cfg.n_slots, 6)
                self.assertEqual(cfg.cf_seeds, 5)            # conf overrides .env
                self.assertEqual(cfg.model_name, "org/model-b")
            finally:
                os.environ.clear()
                os.environ.update(saved)


class TestOuterLoop(unittest.TestCase):
    def test_one_round_end_to_end(self):
        from src.rsi.outer_loop import OuterLoop
        from src.rsi.splits import make_splits, save_splits
        cfg = tiny_cfg(editor_mode="both", editor_num_proposals=2, confirm_seeds=1)
        be = FakeBackend(responder_factory(0.45))
        dev = [dict(qid=f"d{i}", question=f"Compute {i} plus {i} then some words {i}", gold="4",
                    dataset="math_train", template=f"t{i}") for i in range(10)]
        test = [dict(qid="t0", question="Totally different sealed question", gold="4", dataset="math500",
                     template="tt")]
        with tempfile.TemporaryDirectory() as d:
            splits = make_splits(dev, test, rounds=1, search_size=3, confirm_size=3, seed=0)
            probs = {p["qid"]: p for p in dev + test}
            save_splits(splits, probs, d)
            loop = OuterLoop(cfg, be, d, probs, splits)
            H = loop.run()
            self.assertTrue(os.path.exists(os.path.join(d, "final_procedure.json")))
            self.assertTrue(os.path.exists(os.path.join(d, "rounds", "round_00.json")))
            with open(os.path.join(d, "rounds", "round_00.json")) as f:
                rr = json.load(f)
            self.assertIn(rr["decision"], ("accept", "retain"))
            with open(os.path.join(d, "memory.jsonl")) as f:
                mem = [json.loads(l) for l in f]
            self.assertTrue(all(m["qid"] != "t0" for m in mem))   # sealed test never enters memory
            self.assertGreater(loop.ledger.calls, 0)



class Crash(Exception):
    pass


def crashing_responder(limit, inner):
    def respond(req, n):
        if n >= limit:
            raise Crash(f"simulated kill at call {n}")
        return inner(req, n)
    return respond


class TestResume(unittest.TestCase):
    def test_jobs_resume_mid_run(self):
        from src.rsi.runner import run_solves_multi
        cfg = tiny_cfg(checkpoint_every_s=0.0)
        probs = [dict(qid=f"q{i}", question=f"What is 2+2? ({i})", gold="4") for i in range(4)]
        g = Grader("math500")
        with tempfile.TemporaryDirectory() as d:
            ck = os.path.join(d, "ck")
            be1 = FakeBackend(crashing_responder(40, responder_factory()))
            with self.assertRaises(Crash):
                run_solves_multi(SolveContext(cfg, be1), be1, probs, [("procedure", h0_procedure(2))], [0], g,
                                 0.2, ckpt=ck)
            self.assertTrue(os.path.exists(ck + ".inflight.pkl"))
            be2 = FakeBackend(responder_factory())
            recs = run_solves_multi(SolveContext(cfg, be2), be2, probs, [("procedure", h0_procedure(2))], [0],
                                    g, 0.2, ckpt=ck)[0]
            self.assertEqual(len(recs), 4)
            for r in recs:   # every solve finished its full H0 schedule after resuming
                self.assertEqual(r["cost"]["calls"], 3 * (1 + 3 * 2))
                self.assertEqual(r["stop_reason"], "H0:done")
            self.assertLess(be2.n_calls, 4 * 21)    # resumed work was not redone
            be3 = FakeBackend(responder_factory())
            run_solves_multi(SolveContext(cfg, be3), be3, probs, [("procedure", h0_procedure(2))], [0], g, 0.2,
                             ckpt=ck)
            self.assertEqual(be3.n_calls, 0)        # everything already done

    def test_outer_loop_resume_after_crash(self):
        from src.rsi.outer_loop import OuterLoop
        from src.rsi.splits import make_splits, save_splits
        cfg = tiny_cfg(editor_mode="compiler", confirm_seeds=1, rounds=2, checkpoint_every_s=0.0)
        dev = [dict(qid=f"d{i}", question=f"Compute {i} plus {i} then some words {i}", gold="4",
                    dataset="math_train", template=f"t{i}") for i in range(16)]
        test = [dict(qid="t0", question="sealed", gold="4", dataset="math500", template="tt")]
        probs = {p["qid"]: p for p in dev + test}
        with tempfile.TemporaryDirectory() as d:
            splits = make_splits(dev, test, rounds=2, search_size=3, confirm_size=3, seed=0)
            save_splits(splits, probs, d)
            full = FakeBackend(responder_factory(0.45))
            with tempfile.TemporaryDirectory() as d2:
                save_splits(splits, probs, d2)
                OuterLoop(cfg, full, d2, probs, splits).run()
            for limit in (150, 400):             # crash twice at different phases, then finish
                be = FakeBackend(crashing_responder(limit, responder_factory(0.45)))
                with self.assertRaises(Crash):
                    OuterLoop(cfg, be, d, probs, splits).run()
            be = FakeBackend(responder_factory(0.45))
            OuterLoop(cfg, be, d, probs, splits).run()
            for r in range(2):
                self.assertTrue(os.path.exists(os.path.join(d, "rounds", f"round_{r:02d}.json")))
            with open(os.path.join(d, "lineage.json")) as f:
                self.assertEqual([e["round"] for e in json.load(f)], [0, 1])
            with open(os.path.join(d, "memory.jsonl")) as f:
                eids = [json.loads(l)["eid"] for l in f]
            self.assertEqual(len(eids), len(set(eids)))     # no double-counted evidence
            self.assertLess(be.n_calls, full.n_calls)       # the final run only did the remaining work

if __name__ == "__main__":
    unittest.main()
