#!/usr/bin/env python3
"""
export_arr_figures.py — figure-ready data from every ARR result on disk.

Reads $ARR_OUTPUT_ROOT/<model>/<dataset>/B<budget>/results_<method>.jsonl and writes
$ARR_OUTPUT_ROOT/figures_data/ (or --out):

  compute_accuracy.csv  one row per model × dataset × budget × method: accuracy (+ per seed),
                        TFLOPs, budget spent, serial passes (critical path p50/p95), wall-clock
                        latency p50/p95, output tokens per question / per call, prompt / checked /
                        accepted / discarded / decoded positions, roots, voting roots,
                        revisions per root, fixed-point rate, accepted-prefix fraction,
                        wrong→correct / correct→wrong, AUROCs                 (compute figures)
  pareto.csv            the same cells with non-dominated flags for accuracy vs TFLOPs, vs
                        critical-path passes, vs output tokens, per model × dataset (Pareto fronts)
  per_question.csv      one row per question × seed × method × budget: correctness, cost,
                        latency, roots, vote share / margin, fixed-point support, output tokens
  trajectories.csv      one row per root version: answer and correctness after every completed
                        revision, final status (trajectory figures)
  revisions.csv         one row per revision / correction call: draft length, accepted prefix,
                        fixed point, output tokens, passes, checked / discarded positions,
                        TFLOPs, correctness before / after
  auroc.csv             AUROC of each label-free confidence signal for predicting correctness
  robustness.csv        per-seed accuracy, paired question-level differences vs arr (mean, 95 %
                        bootstrap CI, sign-flip p), and the fixed-cohort depth accounting
                        A(d), ρ_rec(d), ρ_reg(d)

    python export_arr_figures.py [--root $ARR_OUTPUT_ROOT] [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.arr.runner import auroc, depth_accounting, summarize  # noqa: E402
from src.arr.stats import paired_compare  # noqa: E402
from src.arr.tasks import get_task  # noqa: E402

CELL_KEYS = [
    "n_questions", "n_solves", "accuracy", "solved_count", "mean_tflops", "median_tflops", "mean_budget_spent",
    "critical_path_passes_p50", "critical_path_passes_p95", "mean_question_passes", "latency_sec_p50",
    "latency_sec_p95", "mean_output_tokens_per_question", "mean_output_tokens_per_call",
    "mean_final_solution_tokens", "mean_prompt_positions", "mean_draft_checked", "mean_draft_accepted",
    "mean_draft_discarded", "mean_draft_recompute", "mean_decoded", "mean_calls", "mean_root_births", "mean_voting_roots",
    "mean_revisions_per_root", "fixed_point_rate", "mean_fixed_point_retirements", "mean_accepted_prefix_frac",
    "mean_first_mismatch", "frac_changed_within_first_block", "frac_changed_at_first_token",
    "incomplete_revision_rate", "wrong_to_correct", "correct_to_wrong", "n_completed_updates", "invalid_samples",
    "mean_vote_share", "auroc_vote_share", "auroc_vote_margin", "auroc_fixed_point_support",
    "fixed_point_root_accuracy", "non_fixed_point_root_accuracy",
]


def enrich(rec: Dict[str, Any], task) -> Dict[str, Any]:
    """Fill label-free confidence fields for records written before they existed."""
    votes = rec.get("votes") or {}
    total = sum(votes.values())
    win = task.norm(rec["final_answer"]) if rec["final_answer"] != "[invalid]" else ""
    if "vote_share" not in rec:
        rec["vote_share"] = votes.get(win, 0) / total if total else 0.0
        rec["vote_margin"] = ((votes.get(win, 0) - max([c for k, c in votes.items() if k != win] + [0])) / total
                              if total else 0.0)
        fixed = [r for r in rec["roots"] if r["status"] == "fixed_point" and r["valid"]]
        rec["n_fixed_point_roots"] = len(fixed)
        rec["fixed_point_support"] = (sum(task.norm(r["answer"]) == win for r in fixed) / rec["n_voting_roots"]
                                      if rec["n_voting_roots"] else 0.0)
    if "output_tokens" not in rec:
        rec["output_tokens"] = int(sum(e.get("n_out", 0) for e in rec["events"]))
        rec["n_model_calls"] = int(sum(1 for e in rec["events"] if "n_out" in e))
    return rec


def load(root: str):
    cells: Dict[tuple, List[Dict[str, Any]]] = {}
    for path in sorted(glob.glob(os.path.join(root, "*", "*", "B*", "results_*.jsonl"))):
        parts = path.split(os.sep)
        model, dataset, cell = parts[-4], parts[-3], parts[-2]
        method = re.sub(r"\.s\d+of\d+$", "", os.path.basename(path)[len("results_"):-len(".jsonl")])
        task = get_task(dataset.replace("_pilot", ""))
        key = (model, dataset, float(cell[1:]), method)
        recs = cells.get(key, [])
        seen = {(r["qid"], r["seed"]) for r in recs}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if (r["qid"], r["seed"]) not in seen:
                        seen.add((r["qid"], r["seed"]))
                        recs.append(enrich(r, task))
        if recs:
            cells[key] = recs
    return cells


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


def pareto_flags(rows: List[Dict[str, Any]], cost: str) -> None:
    for r in rows:
        c, a = r.get(cost), r.get("accuracy")
        if c is None or a is None:
            r[f"pareto_{cost}"] = False
            continue
        r[f"pareto_{cost}"] = not any(
            (o.get(cost) is not None and o["accuracy"] is not None and o[cost] <= c and o["accuracy"] >= a
             and (o[cost] < c or o["accuracy"] > a)) for o in rows if o is not r)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("ARR_OUTPUT_ROOT", "experiments_results/arr"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--block_size", type=int, default=int(os.environ.get("ARR_BLOCK_SIZE", 32)))
    args = ap.parse_args()
    out = args.out or os.path.join(args.root, "figures_data")
    os.makedirs(out, exist_ok=True)
    cells = load(args.root)

    comp, perq, traj, revs, aur, rob = [], [], [], [], [], []
    for (model, ds, B, m), recs in sorted(cells.items()):
        key = dict(model=model, dataset=ds, budget=B, method=m)
        s = summarize(recs, args.block_size)
        row = dict(key, **{k: s.get(k) for k in CELL_KEYS})
        row["correct_k_of_n"] = s.get("correct_k_of_n")
        comp.append(row)
        for sig in ("vote_share", "vote_margin", "fixed_point_support"):
            aur.append(dict(key, signal=sig, auroc=auroc([r.get(sig, 0.0) for r in recs], [r["correct"] for r in recs]),
                            n_correct=sum(r["correct"] for r in recs), n_wrong=sum(not r["correct"] for r in recs)))
        for seed, kn in (s.get("correct_k_of_n") or {}).items():
            k, n = kn.split("/")
            rob.append(dict(key, kind="per_seed_accuracy", seed=seed, value=int(k) / int(n), k=int(k), n=int(n)))
        acc = s.get("depth_accounting") or {}
        for d, A in enumerate(acc.get("A", [])):
            rob.append(dict(key, kind="depth_accounting", depth=d, A=A,
                            rho_rec=acc["rho_rec"][d - 1] if d else None, rho_reg=acc["rho_reg"][d - 1] if d else None,
                            telescopes=acc.get("telescopes")))
        for r in recs:
            c = r["cost"]
            perq.append(dict(key, qid=r["qid"], seed=r["seed"], correct=r["correct"], final_answer=r["final_answer"],
                             gold=r.get("gold"), tflops=c.get("tflops"), spent=c.get("spent"), passes=c.get("passes"),
                             critical_path_passes=r.get("critical_path_passes"), latency_sec=r.get("latency_sec"),
                             prompt_positions=c.get("prefill"), draft_checked=c.get("draft_checked"),
                             draft_accepted=c.get("draft_accepted"), draft_discarded=c.get("draft_discarded"),
                             decoded=c.get("decoded"), calls=c.get("calls"), output_tokens=r.get("output_tokens"),
                             n_roots=len(r["roots"]), n_voting_roots=r["n_voting_roots"],
                             n_fixed_point_roots=r.get("n_fixed_point_roots"), vote_share=r.get("vote_share"),
                             vote_margin=r.get("vote_margin"), fixed_point_support=r.get("fixed_point_support"),
                             stop_reason=r.get("stop_reason")))
            for x in r["roots"]:
                for v, (a, ok) in enumerate(zip(x["answers"], x.get("correct") or [None] * len(x["answers"]))):
                    traj.append(dict(key, qid=r["qid"], seed=r["seed"], root=x["id"], lane=x["lane"],
                                     cohort_root=x["id"] < 4, version=v, answer=a, correct=ok,
                                     is_final=v == len(x["answers"]) - 1, final_status=x["status"],
                                     final_tokens=x.get("n_tokens")))
            for e in r["events"]:
                if e.get("op") in ("revise", "correct", "continue"):
                    revs.append(dict(key, qid=r["qid"], seed=r["seed"], root=e["root"], op=e["op"], mode=e.get("mode"),
                                     finish=e.get("finish"), draft_len=e.get("draft_len"), lcp=e.get("lcp"),
                                     fixed_point=e.get("identical"), n_out=e.get("n_out"), passes=e.get("passes"),
                                     checked=e.get("checked"), accepted=e.get("accepted"),
                                     discarded=e.get("discarded"), decoded=e.get("decoded"),
                                     prefill=e.get("prefill"), tflops=e.get("spent"),
                                     revisions_before=e.get("revisions_before"),
                                     before_correct=e.get("before_correct"), after_correct=e.get("after_correct")))
    # paired comparisons against arr in every cell that has it
    for (model, ds, B, m), recs in sorted(cells.items()):
        base = cells.get((model, ds, B, "arr"))
        if m == "arr" or base is None:
            continue
        for metric, fn in (("accuracy", lambda x: x), ("tflops", lambda x: dict(x, t=x["cost"]["tflops"])),
                           ("output_tokens", lambda x: dict(x, t=x.get("output_tokens", 0))),
                           ("critical_path_passes", lambda x: dict(x, t=x.get("critical_path_passes", 0)))):
            k = "correct" if metric == "accuracy" else "t"
            pc = paired_compare([fn(x) for x in base], [fn(x) for x in recs], k)
            rob.append(dict(model=model, dataset=ds, budget=B, method=m, kind="paired_arr_minus_method",
                            metric=metric, mean_diff=pc["mean_diff"], ci95_lo=pc["ci95"][0], ci95_hi=pc["ci95"][1],
                            p_value=pc["p_value"], n_questions=pc["n_questions"], arr_better=pc["a_better"],
                            method_better=pc["b_better"], ties=pc["ties"]))
    by_ds = defaultdict(list)
    for r in comp:
        by_ds[(r["model"], r["dataset"])].append(r)
    for rows in by_ds.values():
        for cost in ("mean_tflops", "critical_path_passes_p50", "mean_output_tokens_per_question"):
            pareto_flags(rows, cost)
    for name, rows in (("compute_accuracy", comp), ("pareto", comp), ("per_question", perq),
                       ("trajectories", traj), ("revisions", revs), ("auroc", aur), ("robustness", rob)):
        write_csv(os.path.join(out, f"{name}.csv"), rows)
    print(f"[export] {len(cells)} cells, {len(perq)} solves → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
