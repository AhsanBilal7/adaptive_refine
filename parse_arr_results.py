#!/usr/bin/env python3
"""
parse_arr_results.py — print (or save) the current ARR results.

Reads every $ARR_OUTPUT_ROOT/<model>/<dataset>/B<budget>/results_<method>[.s<i>of<n>].jsonl
(shards merged, duplicates dropped; partial runs included and flagged) and prints, per
model × dataset, one table of budget × method:

  acc        mean accuracy over seeds (every question in the denominator)
  k/n        correct / questions, per seed
  done       finished solves / (questions in the benchmark × seeds); partial cells are flagged "*"
  TFLOPs     mean estimated TFLOPs per question
  cp50       median serial forward passes on the critical path (latency proxy)
  tok/q      mean output tokens per question
  roots      mean roots born per question        votes   mean voting roots
  fixpt      fixed-point rate of complete revisions
  acc.pref   mean accepted-prefix fraction of complete revisions
  W→C C→W    answer changes after completed revisions (labels used for evaluation only)
  AUROC      vote share as a predictor of correctness
  Δacc vs arr  paired question-level accuracy difference (method − arr) with 95 % CI

    python parse_arr_results.py                         # all models / datasets
    python parse_arr_results.py --dataset aime24 math500
    python parse_arr_results.py --markdown results.md   # also write a Markdown report
    python parse_arr_results.py --csv results.csv       # also write one flat CSV
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from export_arr_figures import load  # noqa: E402
from src.arr.runner import summarize  # noqa: E402
from src.arr.stats import paired_compare  # noqa: E402

DATASET_SIZE = {"aime24": 30, "aime25": 30, "amc23": 40, "math500": 500, "olympiadbench": 674, "gsm8k": 1319,
                "np_calendar": 1000, "np_meeting": 1000, "np_trip": 1600, "livebench_reasoning": 150,
                "math500_pilot": 50}
METHOD_ORDER = ["arr", "replace_ordinary", "replace_pld", "arr_no_replace", "direct_fixed", "critic_correct",
                "continuation", "maj"]
COLS = [("acc", 6), ("k/n", 14), ("done", 10), ("TFLOPs", 8), ("cp50", 7), ("tok/q", 8), ("roots", 6),
        ("votes", 6), ("fixpt", 6), ("acc.pref", 8), ("W→C", 5), ("C→W", 5), ("AUROC", 6), ("Δacc vs arr [95% CI]", 24)]


def fmt(x, p=3):
    return "-" if x is None else f"{x:.{p}f}"


def rows_for(cells, model, ds):
    out = []
    budgets = sorted({b for (m_, d, b, _) in cells if m_ == model and d == ds})
    size = DATASET_SIZE.get(ds)
    n_seeds = max(len({r["seed"] for r in recs}) for (m_, d, _, _), recs in cells.items() if m_ == model and d == ds)
    for B in budgets:
        methods = [m for (m_, d, b, m) in cells if m_ == model and d == ds and b == B]
        methods.sort(key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)
        base = cells.get((model, ds, B, "arr"))
        for m in methods:
            recs = cells[(model, ds, B, m)]
            s = summarize(recs, 32)
            nq = s["n_solves"]
            expected = size * n_seeds if size else None
            delta = ""
            if base and m != "arr":
                pc = paired_compare(recs, base, "correct")          # method − arr on shared questions
                if pc["mean_diff"] is not None:
                    delta = f"{pc['mean_diff']:+.3f} [{pc['ci95'][0]:+.3f},{pc['ci95'][1]:+.3f}] n={pc['n_questions']}"
            partial = expected is not None and nq < expected
            out.append(dict(
                model=model, dataset=ds, budget=int(B) if float(B).is_integer() else B, method=m,
                acc=s["accuracy"], kn=",".join(s["correct_k_of_n"].values()),
                done=f"{nq}/{expected}" + ("*" if partial else "") if expected else str(nq), partial=partial,
                tflops=s["mean_tflops"], cp50=s["critical_path_passes_p50"],
                tokq=s.get("mean_output_tokens_per_question"), roots=s["mean_root_births"],
                votes=s["mean_voting_roots"], fixpt=s["fixed_point_rate"], accpref=s["mean_accepted_prefix_frac"],
                w2c=s["wrong_to_correct"], c2w=s["correct_to_wrong"], auroc=s.get("auroc_vote_share"), delta=delta))
    return out


def text_table(rows):
    lines = [f"{'budget':>7} {'method':<17}" + "".join(f"{c:>{w}}" for c, w in COLS)]
    last_b = None
    for r in rows:
        if last_b is not None and r["budget"] != last_b:
            lines.append("")
        last_b = r["budget"]
        vals = [fmt(r["acc"]), r["kn"], r["done"], fmt(r["tflops"], 0), fmt(r["cp50"], 0), fmt(r["tokq"], 0),
                fmt(r["roots"], 1), fmt(r["votes"], 1), fmt(r["fixpt"], 2), fmt(r["accpref"], 2), str(r["w2c"]),
                str(r["c2w"]), fmt(r["auroc"], 2), r["delta"]]
        lines.append(f"{'B' + str(r['budget']):>7} {r['method']:<17}" + "".join(f"{v:>{w}}" for v, (_, w) in zip(vals, COLS)))
    return "\n".join(lines)


def md_table(rows):
    head = ["budget", "method"] + [c for c, _ in COLS]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        vals = [f"B{r['budget']}", f"`{r['method']}`", fmt(r["acc"]), r["kn"], r["done"], fmt(r["tflops"], 0),
                fmt(r["cp50"], 0), fmt(r["tokq"], 0), fmt(r["roots"], 1), fmt(r["votes"], 1), fmt(r["fixpt"], 2),
                fmt(r["accpref"], 2), str(r["w2c"]), str(r["c2w"]), fmt(r["auroc"], 2), r["delta"] or "-"]
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("ARR_OUTPUT_ROOT", "experiments_results/arr"))
    ap.add_argument("--model", nargs="*", default=None)
    ap.add_argument("--dataset", nargs="*", default=None)
    ap.add_argument("--markdown", default=None, help="write a Markdown report here")
    ap.add_argument("--csv", default=None, help="write one flat CSV here")
    args = ap.parse_args()

    cells = load(args.root)
    groups = defaultdict(list)
    for (model, ds, _, _) in cells:
        if (args.model and model not in args.model) or (args.dataset and ds not in args.dataset):
            continue
        groups[(model, ds)] = []
    all_rows, md = [], ["# ARR results", "",
                        f"Source: `{args.root}`. `*` = partial cell (run still in progress); "
                        "Δacc = paired question-level difference (method − arr) with 95 % bootstrap CI.", ""]
    for (model, ds) in sorted(groups):
        rows = rows_for(cells, model, ds)
        all_rows += rows
        title = f"{model} · {ds}"
        print("\n" + "=" * 150 + f"\n{title}\n" + "=" * 150)
        print(text_table(rows))
        md += [f"## {title}", "", md_table(rows), ""]
    if args.markdown:
        with open(args.markdown, "w") as f:
            f.write("\n".join(md) + "\n")
        print(f"\n[parse] Markdown → {args.markdown}")
    if args.csv and all_rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
            w.writeheader()
            w.writerows(all_rows)
        print(f"[parse] CSV → {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
