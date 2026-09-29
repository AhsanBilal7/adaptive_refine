"""
runner.py
=========
Full-solve execution + sealed grading + run-level metrics.

``run_solves`` executes a procedure on a list of problems × seeds with all
solves batched together, then grades each terminal vote against gold (the
grader is never visible to the controller).

Baselines (§8) sharing identical extraction / vote / budget conventions:
  h0          — original fixed refinement (paper round-robin, H0)
  heuristic   — fixed hand-written adaptive rules
  maj_matched — compute-matched majority sampling: draw problem-only samples
                while the SAME per-problem cap B affords another one; every
                sample votes (it is a sampling baseline, not a slot method)
  <file.json> — any saved procedure (e.g. the frozen H after development)
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .backends import GenerationBackend
from .procedure import Procedure
from .scheduler import run_generators
from .solver import SolveContext, SolveResult, _charge, solve_gen
from .state import SolveState
from .voting import Grader, INVALID_ANSWER, normalize_answer, plurality_vote

logger = logging.getLogger(__name__)


def seed_base(global_seed: int, qid: str, k: int) -> int:
    """Process-independent seed (Python's str hash is salted per process)."""
    h = hashlib.sha256(f"{int(global_seed)}|{qid}|{int(k)}".encode()).hexdigest()
    return int(h[:8], 16) & 0x7FFFFFFF


# =============================================================================
# Compute-matched majority baseline
# =============================================================================

def maj_matched_gen(ctx: SolveContext, st: SolveState, max_samples: int = 256):
    from src.cimr_step import truncate_at_first_boxed  # type: ignore

    answers: List[str] = st.extra.setdefault("answers", [])   # in the state → survives checkpoints
    norms: List[str] = st.extra.setdefault("norms", [])
    st.initialized = True
    while len(answers) < max_samples and st.meter.can_afford(ctx.wc_init(st.meter, st.question)):
        # Batch as many samples as the remaining cap can reserve.
        k = 0
        rem = st.meter.remaining
        per = ctx.wc_init(st.meter, st.question)
        while (k + 1) * per <= rem + 1e-12 and len(answers) + k < max_samples:
            k += 1
        reqs = [ctx.req_init(st) for _ in range(max(k, 1))]
        results = yield reqs
        for req, res in zip(reqs, results):
            _charge(st, req, res)
            text = truncate_at_first_boxed(res.text) if res.ok else ""
            a, n = ctx.extract(text)
            answers.append(a)
            norms.append(n)
    winner, disp, counts = plurality_vote(norms, answers)
    st.stopped, st.stop_reason = True, "budget"
    return SolveResult(st.qid, winner, disp, st, meta=dict(n_samples=len(answers), answers=answers,
                                                           norms=norms))


# =============================================================================
# Grading
# =============================================================================

def grade_result(res: SolveResult, gold: Optional[str], grader: Grader, lam: float) -> Dict[str, Any]:
    st = res.state
    m = st.meter
    correct = bool(grader.is_correct(res.final_answer, gold)) if gold is not None else None
    frac_cost = m.spent / m.budget if m.budget > 0 else 0.0
    if "answers" in res.meta:   # maj_matched
        slot_answers = res.meta["answers"]
    else:
        slot_answers = [s.answer for s in st.slots]
    slot_correct = [bool(grader.is_correct(a, gold)) if a else False for a in slot_answers] if gold is not None else []
    rec: Dict[str, Any] = dict(
        qid=res.qid, gold=gold, final_answer=res.final_answer, final_norm=res.final_norm,
        correct=correct, utility=(float(correct) - lam * frac_cost) if correct is not None else None,
        cost=m.snapshot(), cost_frac=frac_cost, by_role=m.by_role,
        n_decisions=st.n_decisions, stop_reason=st.stop_reason,
        slot_answers=slot_answers, slot_correct=slot_correct,
        extraction_failures=sum(1 for a in slot_answers if not normalize_answer(a)),
        n_votes=len(slot_answers), infeasible_forced=res.infeasible_forced,
    )
    # per-decision graded dynamics (after the fact, never visible to control)
    dec_out = []
    for d in st.decisions:
        vb = grader.is_correct(d.get("vote_before_answer", ""), gold) if gold is not None else None
        va = grader.is_correct(d.get("vote_after_answer", ""), gold) if gold is not None else None
        flips = []
        if d["action"] == "PAIR":
            acted = [(d["slot"], d.get("first", {})), (d["slot2"], d.get("second", {}))]
        elif d["action"] in ("FULL_REFINE", "DIRECT_REPAIR", "RESTART"):
            acted = [(d["slot"], d.get("outcome", {}))]
        else:
            acted = []
        for s, out in acted:
            if not out.get("complete"):
                continue
            pb, pa = d["answers_before"][s], d["answers_after"][s]
            flips.append(dict(slot=s, before=bool(grader.is_correct(pb, gold)) if gold is not None else None,
                              after=bool(grader.is_correct(pa, gold)) if gold is not None else None,
                              changed=d["norms_before"][s] != d["norms_after"][s]))
        dec_out.append(dict(idx=d["idx"], key=d["key"], action=d["action"], item_id=d["item_id"],
                            forced=d["forced"], vote_before_correct=vb, vote_after_correct=va,
                            cost=d["spent_after"] - d["spent_before"], flips=flips,
                            plan=d.get("plan"), guard_passed=d.get("guard_passed")))
    rec["decisions"] = dec_out
    return rec


# =============================================================================
# Running many solves
# =============================================================================

def make_job(ctx: SolveContext, method: str, proc: Optional[Procedure], job_id: str, qid: str,
             question: str, sb: int):
    from .checkpoint import Job
    st = ctx.new_state(qid, question, sb)
    if method == "maj_matched":
        return Job(job_id, st, lambda s, forced: maj_matched_gen(ctx, s))
    return Job(job_id, st, lambda s, forced: solve_gen(ctx, s, proc, forced=forced))


def run_solves_multi(ctx: SolveContext, backend: GenerationBackend, problems: Sequence[Dict[str, Any]],
                     arms: Sequence[tuple], seeds: Sequence[int], grader: Grader, lam: float,
                     max_inflight: int = 0, desc: str = "", ckpt: Optional[str] = None
                     ) -> List[List[Dict[str, Any]]]:
    """Run several (method, procedure) arms on the same problems × seeds in ONE
    scheduler pass so independent solves share batches.  Arms use identical
    seed bases (paired comparisons).  Returns one record list per arm.

    ``ckpt`` (path prefix) enables continuous saving + resume (checkpoint.py)."""
    from .checkpoint import run_jobs
    jobs, meta = [], []
    for a, (method, proc) in enumerate(arms):
        tag = f"{method}:{proc.hash() if proc is not None else '-'}"
        for p in problems:
            for k in seeds:
                jobs.append(make_job(ctx, method, proc, f"{a}|{tag}|{p['qid']}|{k}", p["qid"], p["question"],
                                     seed_base(ctx.cfg.seed, p["qid"], k)))
                meta.append((a, p, k))
    outs = run_jobs(backend, jobs, ckpt, desc=desc, max_inflight=max_inflight or ctx.cfg.resolved_max_inflight(),
                    save_every_s=ctx.cfg.checkpoint_every_s)
    recs: List[List[Dict[str, Any]]] = [[] for _ in arms]
    for (a, p, k), res in zip(meta, outs):
        r = grade_result(res, p.get("gold"), grader, lam)
        r["seed"] = k
        r["dataset"] = p.get("dataset", "")
        recs[a].append(r)
    return recs


def run_solves(ctx: SolveContext, backend: GenerationBackend, problems: Sequence[Dict[str, Any]],
               method: str, proc: Optional[Procedure], seeds: Sequence[int], grader: Grader,
               lam: float, max_inflight: int = 0, desc: str = "", ckpt: Optional[str] = None
               ) -> List[Dict[str, Any]]:
    return run_solves_multi(ctx, backend, problems, [(method, proc)], seeds, grader, lam,
                            max_inflight, desc, ckpt)[0]


# =============================================================================
# Metrics (§8 reporting list)
# =============================================================================

def summarize(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not recs:
        return {}
    graded = [r for r in recs if r["correct"] is not None]
    by_q: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in graded:
        by_q[r["qid"]].append(r)
    q_acc = [np.mean([float(x["correct"]) for x in v]) for v in by_q.values()]
    out: Dict[str, Any] = dict(
        n_solves=len(recs), n_questions=len(by_q),
        accuracy=float(np.mean(q_acc)) if q_acc else None,
        solved_count=float(np.sum(q_acc)) if q_acc else None,
        seed_accuracy_std=float(np.std([np.mean([float(x["correct"]) for x in graded if x["seed"] == s])
                                         for s in sorted({x["seed"] for x in graded})])) if graded else None,
        utility=float(np.mean([r["utility"] for r in graded])) if graded else None,
        mean_cost_frac=float(np.mean([r["cost_frac"] for r in recs])),
        mean_calls=float(np.mean([r["cost"]["calls"] for r in recs])),
        mean_tokens_in=float(np.mean([r["cost"]["tokens_in"] for r in recs])),
        mean_tokens_out=float(np.mean([r["cost"]["tokens_out"] for r in recs])),
        mean_tflops=float(np.mean([r["cost"]["tflops"] for r in recs])),
        total_tflops=float(np.sum([r["cost"]["tflops"] for r in recs])),
        extraction_failure_rate=float(np.sum([r["extraction_failures"] for r in recs]) /
                                      max(1, np.sum([r["n_votes"] for r in recs]))),
        candidate_accuracy=float(np.mean([np.mean(r["slot_correct"]) for r in graded if r["slot_correct"]]))
        if graded else None,
    )
    acts = Counter(d["action"] for r in recs for d in r["decisions"])
    out["action_frequencies"] = dict(acts)
    out["mean_decisions"] = float(np.mean([r["n_decisions"] for r in recs]))
    # op-level flips (conditional rates)
    w2c = w_tot = c2w = c_tot = 0
    rec_n = reg_n = n_dec = 0
    for r in graded:
        for d in r["decisions"]:
            for f in d["flips"]:
                if f["before"] is False:
                    w_tot += 1
                    w2c += int(bool(f["after"]))
                elif f["before"] is True:
                    c_tot += 1
                    c2w += int(not f["after"])
            if d["action"] != "STOP" and d["vote_before_correct"] is not None:
                n_dec += 1
                rec_n += int((not d["vote_before_correct"]) and d["vote_after_correct"])
                reg_n += int(d["vote_before_correct"] and (not d["vote_after_correct"]))
    out.update(
        cond_wrong_to_correct=(w2c / w_tot) if w_tot else None, n_wrong_parents=w_tot,
        cond_correct_to_wrong=(c2w / c_tot) if c_tot else None, n_correct_parents=c_tot,
        vote_recoveries=rec_n, vote_regressions=reg_n, n_graded_decisions=n_dec,
    )
    return out


__all__ = ["run_solves", "run_solves_multi", "grade_result", "summarize", "maj_matched_gen", "seed_base", "make_job"]
