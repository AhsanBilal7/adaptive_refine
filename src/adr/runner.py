"""
runner.py
=========
Run the §7 comparison on one dataset and write everything under
``<root>/<model>/<dataset>/``:

    prefix/                      shared structured solutions + first critiques (checkpointed)
    B<budget>/                   one cell per predeclared budget
        results_<method>.jsonl   per question × seed: answers, cost, latency, every turn
        summary.json             metrics per method + paired comparisons vs adr
        ckpt/                    resumable checkpoints
    cimr_D<depth>/               the original fixed-depth method (budget-independent)
    summary_all.json             every cell × method in one table

Reported (§7): accuracy vs total input/output tokens and estimated FLOPs, wall-clock
latency (same batching), fraction of reasoning regenerated per repair, wrong→correct
and correct→wrong answer changes, dependency/format failures and full-rewrite
fallback rate — plus paired question-level uncertainty.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .backends import GenerationBackend
from .checkpoint import Job, atomic_json, run_jobs
from .config import DIRECT_CRITIQUE_METHODS, ADRConfig, dump_config
from .methods import Ctx, cimr_gen, direct_gen, maj_gen, prefix_gen
from .progress import set_context
from .state import QState
from .stats import paired_compare
from .voting import Grader, normalize_answer, plurality_vote

logger = logging.getLogger(__name__)
BIG = float("inf")


def load_questions(dataset: str, max_samples: int = 0, cache_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    from src.dataset_loader import load_dataset_split  # type: ignore
    exs = load_dataset_split(data_name=dataset, max_samples=max_samples or None, cache_dir=cache_dir or None)
    return [dict(qid=f"{dataset}:{i:04d}", question=e["question"], gold=str(e.get("answer", "")))
            for i, e in enumerate(exs) if e.get("question")]


def _seed_base(cfg: ADRConfig, qid: str, k: int) -> int:
    import hashlib
    return int(hashlib.sha256(f"{cfg.seed}|{qid}|{k}".encode()).hexdigest()[:8], 16)


# =============================================================================
# Grading + per-record metrics
# =============================================================================

def grade(st: QState, gold: str, grader: Grader, method: str) -> Dict[str, Any]:
    if method == "maj_matched":
        answers, norms = st.extra.get("answers", []), st.extra.get("norms", [])
        _, final, _ = plurality_vote(norms, answers)
    else:
        answers = [r.answer for r in st.rollouts]
        _, final, _ = st.vote()
    ok = lambda a: bool(grader.is_correct(a, gold)) if a else False
    events = []
    for ev in st.events:
        e = {k: v for k, v in ev.items() if k not in ("error",)}
        if "answer_before" in ev:
            e["before_correct"], e["after_correct"] = ok(ev["answer_before"]), ok(ev["answer_after"])
        events.append(e)
    m = st.meter
    return dict(
        qid=st.qid, gold=gold, method=method, final_answer=final, correct=ok(final),
        rollout_answers=answers, rollout_correct=[ok(a) for a in answers],
        extraction_failures=sum(1 for a in answers if not normalize_answer(a)), n_votes=len(answers),
        cost=m.snapshot(), by_role=m.by_role, elapsed_sec=st.elapsed, stop_reason=st.stop_reason,
        freeze_reasons=[r.freeze_reason or ("active" if r.active else "") for r in st.rollouts],
        format_ok_initial=st.extra.get("prefix", {}).get("format_ok"), events=events,
    )


def summarize(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not recs:
        return {}
    by_q: Dict[str, List[float]] = {}
    for r in recs:
        by_q.setdefault(r["qid"], []).append(float(r["correct"]))
    q_acc = [np.mean(v) for v in by_q.values()]
    ev = [e for r in recs for e in r["events"]]
    repairs = [e for e in ev if e.get("feedback") in ("FIX", "REBUILD")]
    fix = [e for e in ev if e.get("feedback") == "FIX" and e.get("repair_size")]
    applied = [e for e in ev if e.get("update_result") == "applied"]
    changed = [e for e in applied if e.get("answer_before") != e.get("answer_after")]
    w_parents = [e for e in applied if e.get("before_correct") is False]
    c_parents = [e for e in applied if e.get("before_correct") is True]
    fmt = [x for r in recs for x in (r.get("format_ok_initial") or [])]
    out = dict(
        n_solves=len(recs), n_questions=len(by_q),
        accuracy=float(np.mean(q_acc)), solved_count=float(np.sum(q_acc)),
        mean_tokens_in=float(np.mean([r["cost"]["tokens_in"] for r in recs])),
        mean_tokens_out=float(np.mean([r["cost"]["tokens_out"] for r in recs])),
        mean_tokens_total=float(np.mean([r["cost"]["tokens_in"] + r["cost"]["tokens_out"] for r in recs])),
        mean_tflops=float(np.mean([r["cost"]["tflops"] for r in recs])),
        mean_calls=float(np.mean([r["cost"]["calls"] for r in recs])),
        mean_latency_sec=float(np.mean([r["elapsed_sec"] for r in recs])),
        extraction_failure_rate=float(sum(r["extraction_failures"] for r in recs) / max(1, sum(r["n_votes"] for r in recs))),
        candidate_accuracy=float(np.mean([np.mean(r["rollout_correct"]) for r in recs if r["rollout_correct"]])),
        # repair scope (§7: fraction of reasoning regenerated per repair)
        n_repairs=len(repairs), n_fix=len(fix),
        mean_repair_size=float(np.mean([e["repair_size"] for e in fix])) if fix else None,
        mean_suffix_size=float(np.mean([e["suffix_size"] for e in fix])) if fix else None,
        mean_repair_over_suffix=float(np.mean([e["repair_size"] / e["suffix_size"] for e in fix])) if fix else None,
        frac_repair_smaller_than_suffix=float(np.mean([e["repair_size"] < e["suffix_size"] for e in fix])) if fix else None,
        mean_regenerated_frac=float(np.mean([e["regenerated_frac"] for e in applied if "regenerated_frac" in e])) if applied else None,
        # answer changes per applied update
        n_updates_applied=len(applied), answer_change_rate=len(changed) / len(applied) if applied else None,
        wrong_to_correct=sum(1 for e in w_parents if e.get("after_correct")),
        correct_to_wrong=sum(1 for e in c_parents if not e.get("after_correct")),
        cond_wrong_to_correct=(sum(1 for e in w_parents if e.get("after_correct")) / len(w_parents)) if w_parents else None,
        cond_correct_to_wrong=(sum(1 for e in c_parents if not e.get("after_correct")) / len(c_parents)) if c_parents else None,
        # failures
        initial_format_failure_rate=float(1 - np.mean(fmt)) if fmt else None,
        fallback_rate=(sum(1 for e in repairs if e.get("fallback")) / len(repairs)) if repairs else None,
        invalid_update_rate=(sum(1 for e in repairs if e.get("update_result") == "invalid_kept_previous") / len(repairs)) if repairs else None,
        feedback_counts=dict(Counter(e.get("feedback") for e in ev if e.get("feedback"))),
        freeze_reasons=dict(Counter(x for r in recs for x in r["freeze_reasons"] if x)),
        stop_reasons=dict(Counter(r["stop_reason"] for r in recs)),
    )
    return out


# =============================================================================
# Running cells
# =============================================================================

def _run_and_save(backend, jobs, cell: str, method: str, cfg: ADRConfig, questions, grader, label: str
                  ) -> List[Dict[str, Any]]:
    path = os.path.join(cell, f"results_{method}.jsonl")
    if os.path.exists(path):
        with open(path) as f:
            recs = [json.loads(l) for l in f if l.strip()]
        if len(recs) == len(jobs):
            logger.info(f"[adr] {label}: already complete, loaded {path}")
            return recs
    set_context(label)
    outs = run_jobs(backend, jobs, os.path.join(cell, "ckpt", method), desc=f"{len(jobs)} solves",
                    save_every_s=cfg.checkpoint_every_s, max_inflight=cfg.resolved_max_inflight())
    gold = {q["qid"]: q["gold"] for q in questions}
    recs = []
    for j, st in zip(jobs, outs):
        r = grade(st, gold[st.qid], grader, method)
        r["seed"] = int(j.id.rsplit("|", 1)[1])
        recs.append(r)
    with open(path + ".tmp", "w") as f:
        for r in recs:
            f.write(json.dumps(r, default=str) + "\n")
    os.replace(path + ".tmp", path)
    s = summarize(recs)
    logger.info(f"[adr] {label}: acc={s['accuracy']:.3f} solved={s['solved_count']:.1f}/{s['n_questions']} "
                f"tokens/q={s['mean_tokens_total']:.0f} calls/q={s['mean_calls']:.1f}")
    return recs


def run_dataset(cfg: ADRConfig, backend: GenerationBackend, dataset: str, methods: Sequence[str],
                questions: Optional[List[Dict[str, Any]]] = None, tag: str = "") -> Dict[str, Any]:
    ctx = Ctx(cfg, backend)
    questions = questions or load_questions(dataset, cfg.max_samples, cfg.hf_cache_dir)
    grader = Grader(dataset)
    seeds = list(range(max(1, cfg.seeds)))
    ds_dir = os.path.join(cfg.model_dir(), dataset + (f"_{tag}" if tag else ""))
    os.makedirs(ds_dir, exist_ok=True)
    dump_config(cfg, os.path.join(ds_dir, "config.json"))
    all_summary: Dict[str, Any] = {}
    budgets = cfg.budget_list()
    n_cells = (1 if "cimr" in methods else 0) + len(budgets)
    cell_i = 0

    # 1. original CIMR (fixed depth; budget-independent)
    if "cimr" in methods:
        cell_i += 1
        cell = os.path.join(ds_dir, f"cimr_D{cfg.cimr_depth}")
        os.makedirs(cell, exist_ok=True)
        jobs = [Job(f"{q['qid']}|{k}", QState(q["qid"], q["question"], cfg.n_rollouts, ctx.new_meter(BIG),
                                              seed_base=_seed_base(cfg, q["qid"], k)),
                    lambda s: cimr_gen(ctx, s)) for q in questions for k in seeds]
        recs = _run_and_save(backend, jobs, cell, "cimr", cfg, questions, grader,
                             f"{dataset} · cell {cell_i}/{n_cells} cimr_D{cfg.cimr_depth} · cimr")
        s = summarize(recs)
        atomic_json({"cimr": s}, os.path.join(cell, "summary.json"))
        all_summary[f"cimr_D{cfg.cimr_depth}/cimr"] = s

    # 2. shared prefix for the direct-critique variants (budget-independent)
    direct = [m for m in methods if m in DIRECT_CRITIQUE_METHODS]
    prefix_states: Dict[str, QState] = {}
    if direct:
        pdir = os.path.join(ds_dir, "prefix")
        os.makedirs(pdir, exist_ok=True)
        jobs = [Job(f"{q['qid']}|{k}", QState(q["qid"], q["question"], cfg.n_rollouts, ctx.new_meter(BIG),
                                              seed_base=_seed_base(cfg, q["qid"], k)),
                    lambda s: prefix_gen(ctx, s)) for q in questions for k in seeds]
        set_context(f"{dataset} · shared prefix (structured solutions + first critiques)")
        outs = run_jobs(backend, jobs, os.path.join(pdir, "ckpt"), desc=f"{len(jobs)} solves",
                        save_every_s=cfg.checkpoint_every_s, max_inflight=cfg.resolved_max_inflight())
        prefix_states = {j.id: st for j, st in zip(jobs, outs)}
        fmt = [ok for st in outs for ok in st.extra["prefix"]["format_ok"]]
        atomic_json(dict(n=len(outs), initial_format_ok_rate=float(np.mean(fmt)) if fmt else None),
                    os.path.join(pdir, "prefix_summary.json"))

    # 3. budgeted methods, one cell per predeclared budget
    for B in budgets:
        cell_i += 1
        cell = os.path.join(ds_dir, f"B{int(B) if float(B).is_integer() else B}")
        os.makedirs(cell, exist_ok=True)
        cell_recs: Dict[str, List[Dict[str, Any]]] = {}
        run_methods = [m for m in methods if m != "cimr"]
        for mi, method in enumerate(run_methods):
            label = f"{dataset} · cell {cell_i}/{n_cells} B={B:g} · method {mi + 1}/{len(run_methods)} {method}"
            if method == "maj_matched":
                jobs = [Job(f"{q['qid']}|{k}", QState(q["qid"], q["question"], cfg.n_rollouts, ctx.new_meter(B),
                                                      seed_base=_seed_base(cfg, q["qid"], k) + 7),
                            lambda s: maj_gen(ctx, s)) for q in questions for k in seeds]
            else:
                jobs = []
                for q in questions:
                    for k in seeds:
                        st = prefix_states[f"{q['qid']}|{k}"].copy()   # same solutions + first critiques
                        st.meter.budget = B
                        jobs.append(Job(f"{q['qid']}|{k}", st, (lambda m: (lambda s: direct_gen(ctx, s, m)))(method)))
            cell_recs[method] = _run_and_save(backend, jobs, cell, method, cfg, questions, grader, label)
        summ = {m: summarize(r) for m, r in cell_recs.items()}
        if "adr" in cell_recs:
            summ["paired_vs_adr"] = {
                m: dict(accuracy=paired_compare(cell_recs["adr"], r, "correct"),
                        tokens=paired_compare([dict(x, t=x["cost"]["tokens_in"] + x["cost"]["tokens_out"])
                                               for x in cell_recs["adr"]],
                                              [dict(x, t=x["cost"]["tokens_in"] + x["cost"]["tokens_out"]) for x in r], "t"))
                for m, r in cell_recs.items() if m != "adr"}
        atomic_json(summ, os.path.join(cell, "summary.json"))
        for m, s in summ.items():
            if m != "paired_vs_adr":
                all_summary[f"{os.path.basename(cell)}/{m}"] = s
    atomic_json(all_summary, os.path.join(ds_dir, "summary_all.json"))
    return all_summary


def pilot_report(summary: Dict[str, Any]) -> Dict[str, Any]:
    """§7 feasibility pilot: are dependencies well formed, and are repair sets
    substantially smaller than suffixes?  Fits nothing; failures stay visible."""
    s = next((v for k, v in summary.items() if k.endswith("/adr")), {})
    out = dict(
        initial_format_failure_rate=s.get("initial_format_failure_rate"),
        n_fix=s.get("n_fix"), mean_repair_over_suffix=s.get("mean_repair_over_suffix"),
        frac_repair_smaller_than_suffix=s.get("frac_repair_smaller_than_suffix"),
        fallback_rate=s.get("fallback_rate"), invalid_update_rate=s.get("invalid_update_rate"),
        accuracy=s.get("accuracy"),
        # literal reading of §7: "mostly linear" = repair set equals the suffix in most FIX repairs;
        # "fallbacks dominate" = more than half of repairs fall back to a full rewrite
        mostly_linear=(s.get("frac_repair_smaller_than_suffix") is not None
                       and s["frac_repair_smaller_than_suffix"] < 0.5),
        fallbacks_dominate=(s.get("fallback_rate") is not None and s["fallback_rate"] > 0.5),
    )
    rep = out
    rep["verdict"] = ("do NOT begin a large sweep (§7)" if rep["mostly_linear"] or rep["fallbacks_dominate"]
                      else "feasible: proceed to the main study")
    return rep



__all__ = ["load_questions", "grade", "summarize", "run_dataset", "pilot_report"]
