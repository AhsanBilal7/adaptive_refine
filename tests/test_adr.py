"""
Unit tests for src/adr (no GPU / no model download).

    python -m unittest tests/test_adr.py -v
"""

import json
import os
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adr.backends import FakeBackend  # noqa: E402
from src.adr.checkpoint import Job, run_jobs  # noqa: E402
from src.adr.config import ADRConfig, config_from_env, load_shell_env, validate_config  # noqa: E402
from src.adr.methods import Ctx, cimr_gen, direct_gen, maj_gen, prefix_gen  # noqa: E402
from src.adr.prompts import parse_feedback  # noqa: E402
from src.adr.runner import grade, summarize  # noqa: E402
from src.adr.scheduler import run_generators  # noqa: E402
from src.adr.state import QState  # noqa: E402
from src.adr.stats import paired_compare  # noqa: E402
from src.adr.structure import (apply_patch, apply_suffix, descendants, parse_full, parse_steps,  # noqa: E402
                               repair_set_adr, repair_set_suffix, with_all_deps)
from src.adr.voting import Grader  # noqa: E402

Q = "A rectangle is 12 m by 5 m. Painting costs $3 per m^2 and fencing $2 per m. Combined cost?"
WRONG = ("1. uses: [] Area: 12*5 = 50\n2. uses: [] Perimeter: 2(12+5) = 34\n3. uses: [1] Paint: 3*50 = 150\n"
         "4. uses: [2] Fence: 2*34 = 68\n5. uses: [3, 4] Total: 150+68 = \\boxed{218}")
RIGHT = WRONG.replace("= 50", "= 60").replace("3*50 = 150", "3*60 = 180").replace("150+68 = \\boxed{218}",
                                                                                   "180+68 = \\boxed{248}")
PATCH = {1: "1. uses: [] Area: 12*5 = 60", 3: "3. uses: [1] Paint: 3*60 = 180",
         5: "5. uses: [3, 4] Total: 180+68 = \\boxed{248}", 2: "2. uses: [] Perimeter: 2(12+5) = 34",
         4: "4. uses: [2] Fence: 2*34 = 68"}


def responder(bad_patch=False, malformed_init=False):
    def respond(req, n):
        u = req.user
        if req.role == "generate":
            return "I will just say the answer is \\boxed{7}" if malformed_init else WRONG
        if req.role == "critique":
            if "CURRENT" in u or "NUMBERED SOLUTION" in u:
                sol = u.split("NUMBERED SOLUTION", 1)[1]
                return "Looks right.\nDECISION: KEEP" if "248" in sol else \
                    "Step 1 multiplies wrongly.\nDECISION: FIX 1 | 12*5 is 60, not 50"
            return "The solution appears correct."                # CIMR critic
        if req.role == "patch":
            if bad_patch:
                return "1. uses: [] Area: 60"                    # missing ids 3, 5 → invalid
            ids = [int(x) for x in re.search(r"Rewrite ONLY steps ([\d, ]+)\.", u).group(1).split(",")]
            return "\n".join(PATCH[i] for i in ids)
        if req.role == "suffix":
            j = int(re.search(r"from step (\d+) onward", u).group(1))
            return "\n".join(PATCH[i] for i in range(j, 6))
        if req.role == "rewrite" and "refinement depth" not in u:
            return RIGHT
        return "Step 1: compute. Final answer: \\boxed{248}"     # CIMR rewrite / correct
    return respond


def cfg_(**kw):
    c = ADRConfig(n_rollouts=3, budgets="1000000", cost_measure="calls", gen_max_new_tokens=200,
                  critique_max_new_tokens=50, patch_max_new_tokens=100, cimr_depth=2, batch_size=8)
    for k, v in kw.items():
        setattr(c, k, v)
    validate_config(c)
    return c


def run_one(cfg, method, resp=None, budget=None):
    be = FakeBackend(resp or responder())
    ctx = Ctx(cfg, be)
    st = QState("q0", Q, cfg.n_rollouts, ctx.new_meter(float("inf")), seed_base=1)
    if method == "cimr":
        return run_generators(be, [cimr_gen(ctx, st)])[0], be
    if method == "maj_matched":
        st.meter.budget = budget
        return run_generators(be, [maj_gen(ctx, st)])[0], be
    st = run_generators(be, [prefix_gen(ctx, st)])[0]
    st.meter.budget = budget if budget is not None else cfg.budget_list()[0]
    return run_generators(be, [direct_gen(ctx, st, method)])[0], be


class TestStructure(unittest.TestCase):
    def test_parse_and_repair_sets(self):
        steps, errs = parse_full(WRONG)
        self.assertEqual(errs, [])
        self.assertEqual([s.uses for s in steps], [[], [], [1], [2], [3, 4]])
        self.assertEqual(descendants(steps, 1), {3, 5})
        self.assertEqual(repair_set_adr(steps, 1), [1, 3, 5])           # §4 worked example
        self.assertEqual(repair_set_adr(steps, 2), [2, 4, 5])
        self.assertEqual(repair_set_suffix(steps, 1), [1, 2, 3, 4, 5])
        self.assertEqual(repair_set_adr(with_all_deps(steps), 2), [2, 3, 4, 5])   # ablation 1 = suffix
        for j in range(1, 6):   # A(j) ⊆ suffix always
            self.assertTrue(set(repair_set_adr(steps, j)) <= set(repair_set_suffix(steps, j)))

    def test_tolerant_formats(self):
        txt = "Here is my plan.\n**1.** uses: [] a = 2\nStep 2: uses: [1] b = a+1\n3) (uses: [1, 2]) \\boxed{5}"
        steps, errs = parse_full(txt)
        self.assertEqual(errs, [])
        self.assertEqual([s.uses for s in steps], [[], [1], [1, 2]])

    def test_invalid_structures(self):
        self.assertIsNone(parse_full("1. uses: [2] x\n2. uses: [] y")[0])            # forward reference
        self.assertIsNone(parse_full("1. uses: [] x\n3. uses: [1] y")[0])            # gap in ids
        self.assertIsNone(parse_full("no steps at all \\boxed{3}")[0])

    def test_patch_keeps_other_steps_verbatim(self):
        steps, _ = parse_full(WRONG)
        new, errs = apply_patch(steps, "\n".join(PATCH[i] for i in (1, 3, 5)), [1, 3, 5])
        self.assertEqual(errs, [])
        self.assertEqual(new[1].raw, steps[1].raw)
        self.assertEqual(new[3].raw, steps[3].raw)
        self.assertIn("248", new[4].raw)
        self.assertIsNone(apply_patch(steps, PATCH[1], [1, 3, 5])[0])                 # missing ids
        self.assertIsNone(apply_patch(steps, PATCH[1] + "\n" + PATCH[2], [1])[0])     # extra id
        bad = "1. uses: [] a\n3. uses: [4] b\n5. uses: [3] \\boxed{1}"
        self.assertIsNone(apply_patch(steps, bad, [1, 3, 5])[0])                      # forward ref

    def test_suffix(self):
        steps, _ = parse_full(WRONG)
        new, errs = apply_suffix(steps, "3. uses: [1] p\n4. uses: [3] \\boxed{9}", 3)
        self.assertEqual(errs, [])
        self.assertEqual(len(new), 4)
        self.assertEqual(new[0].raw, steps[0].raw)

    def test_feedback_parser(self):
        self.assertEqual(parse_feedback("ok\nDECISION: KEEP").kind, "KEEP")
        f = parse_feedback("x\nDECISION: FIX 3 | wrong sum")
        self.assertEqual((f.kind, f.step, f.text), ("FIX", 3, "wrong sum"))
        f = parse_feedback("DECISION: FIX(step 2, sign error)")
        self.assertEqual((f.kind, f.step), ("FIX", 2))
        self.assertEqual(parse_feedback("DECISION: REBUILD | wrong approach").kind, "REBUILD")
        self.assertEqual(parse_feedback("I think it is fine").kind, "INVALID")


class TestMethods(unittest.TestCase):
    def test_adr_patches_only_dependency_closure(self):
        st, be = run_one(cfg_(), "adr")
        fixes = [e for e in st.events if e["feedback"] == "FIX"]
        self.assertEqual(len(fixes), 3)
        for e in fixes:
            self.assertEqual(e["repair_ids"], [1, 3, 5])
            self.assertEqual(e["suffix_size"], 5)
            self.assertEqual(e["update"], "patch")
            self.assertEqual(e["critique"], "shared_first")        # first critique reused
        self.assertTrue(all(r.answer == "248" and r.freeze_reason == "keep" for r in st.rollouts))
        self.assertEqual(st.stop_reason, "all_frozen")
        # calls: 3 generate + 3 first critiques + 3 patches + 3 second critiques (KEEP)
        self.assertEqual(st.meter.calls, 12)
        self.assertLessEqual(st.meter.spent, st.meter.budget)

    def test_variants_scope(self):
        for m, size, upd in (("adr_all_deps", 5, "patch"), ("suffix", 5, "suffix")):
            st, _ = run_one(cfg_(), m)
            e = [x for x in st.events if x["feedback"] == "FIX"][0]
            self.assertEqual((e["repair_size"], e["update"]), (size, upd), m)
            self.assertEqual(st.rollouts[0].answer, "248")
        st, _ = run_one(cfg_(), "full_rewrite")
        e = [x for x in st.events if x["feedback"] == "FIX"][0]
        self.assertEqual(e["update"], "rewrite")
        self.assertEqual(e["regenerated_frac"], 1.0)

    def test_invalid_patch_falls_back_to_rewrite(self):
        st, _ = run_one(cfg_(), "adr", responder(bad_patch=True))
        e = [x for x in st.events if x["feedback"] == "FIX"][0]
        self.assertTrue(e["fallback"])
        self.assertEqual(e["update"], "fallback_rewrite")
        self.assertEqual(st.rollouts[0].answer, "248")

    def test_malformed_initial_uses_rewrite_until_structured(self):
        st, _ = run_one(cfg_(), "adr", responder(malformed_init=True))
        first = [e for e in st.events if e["rollout"] == 0][0]
        self.assertEqual(first["critique"], "skipped_malformed")
        self.assertEqual(first["update"], "rewrite")
        self.assertIsNotNone(st.rollouts[0].steps)
        self.assertEqual(st.rollouts[0].answer, "248")

    def test_budget_stops_and_is_respected(self):
        st, _ = run_one(cfg_(), "adr", budget=8)   # 6 calls for the prefix; room for ~one more turn
        self.assertEqual(st.stop_reason, "budget")
        self.assertLessEqual(st.meter.spent, 8)

    def test_cycle_freezes(self):
        def resp(req, n):
            if req.role == "critique" and "NUMBERED SOLUTION" in req.user:
                return "DECISION: REBUILD | start over"
            return responder()(req, n) if req.role != "rewrite" else WRONG
        st, _ = run_one(cfg_(), "adr", resp)
        self.assertTrue(all(r.freeze_reason == "cycle" for r in st.rollouts))

    def test_cimr_call_count(self):
        st, _ = run_one(cfg_(), "cimr")
        self.assertEqual(st.meter.calls, 3 * (1 + 3 * 2))           # N(1+3D)
        self.assertTrue(all(r.answer == "248" for r in st.rollouts))

    def test_maj_matched_budget(self):
        st, _ = run_one(cfg_(), "maj_matched", budget=10)
        self.assertEqual(len(st.extra["answers"]), 10)
        self.assertLessEqual(st.meter.spent, 10)

    def test_grade_and_summary(self):
        st, _ = run_one(cfg_(), "adr")
        rec = grade(st, "248", Grader("math500"), "adr")
        self.assertTrue(rec["correct"])
        s = summarize([dict(rec, seed=0)])
        self.assertEqual(s["mean_repair_over_suffix"], 0.6)
        self.assertEqual(s["frac_repair_smaller_than_suffix"], 1.0)
        self.assertEqual(s["wrong_to_correct"], 3)
        self.assertEqual(s["correct_to_wrong"], 0)


class TestResume(unittest.TestCase):
    def test_prefix_resume_does_not_double_charge(self):
        class Crash(Exception):
            pass
        cfg = cfg_(checkpoint_every_s=0.0)

        def crashing(req, n):
            if req.role == "critique":
                raise Crash()
            return responder()(req, n)
        with tempfile.TemporaryDirectory() as d:
            be = FakeBackend(crashing)
            ctx = Ctx(cfg, be)
            mk = lambda: [Job("q0|0", QState("q0", Q, 3, ctx.new_meter(float("inf")), seed_base=1),
                              lambda s: prefix_gen(ctx, s))]
            with self.assertRaises(Crash):
                run_jobs(be, mk(), os.path.join(d, "ck"), save_every_s=0.0)
            be2 = FakeBackend(responder())
            ctx2 = Ctx(cfg, be2)
            jobs = [Job("q0|0", QState("q0", Q, 3, ctx2.new_meter(float("inf")), seed_base=1),
                        lambda s: prefix_gen(ctx2, s))]
            st = run_jobs(be2, jobs, os.path.join(d, "ck"), save_every_s=0.0)[0]
            self.assertEqual(be2.n_calls, 3)                       # only the 3 critiques after resume
            self.assertEqual(st.meter.calls, 6)                    # 3 generations charged once + 3 critiques


class TestStatsConfig(unittest.TestCase):
    def test_paired_compare(self):
        a = [dict(qid=f"q{i}", correct=1.0) for i in range(20)]
        b = [dict(qid=f"q{i}", correct=float(i >= 10)) for i in range(20)]
        r = paired_compare(a, b)
        self.assertAlmostEqual(r["mean_diff"], 0.5)
        self.assertLess(r["p_value"], 0.01)

    def test_env_conf_precedence(self):
        with tempfile.TemporaryDirectory() as d:
            env, conf = os.path.join(d, ".env"), os.path.join(d, "x.conf")
            with open(env, "w") as f:
                f.write('export ADR_N_ROLLOUTS="${ADR_N_ROLLOUTS:-4}"\nexport ADR_BUDGETS="${ADR_BUDGETS:-1,2}"\n')
            with open(conf, "w") as f:
                f.write('ADR_BUDGETS="30000,60000"\nMODEL_NAME="org/m"\n')
            saved = dict(os.environ)
            try:
                for k in ("ADR_N_ROLLOUTS", "ADR_BUDGETS", "MODEL_NAME", "ADR_MODEL_NAME"):
                    os.environ.pop(k, None)
                load_shell_env([env, conf])
                c = config_from_env()
                self.assertEqual((c.n_rollouts, c.budget_list(), c.model_name), (4, [30000.0, 60000.0], "org/m"))
            finally:
                os.environ.clear()
                os.environ.update(saved)


if __name__ == "__main__":
    unittest.main()
