#!/usr/bin/env python
"""
Evaluate accuracy of DORA output files.

Usage:
    # Single file
    python scripts/eval_accuracy.py --file result/MATH500/dora_cosine/samples_cosine_n_16_...jsonl

    # Directory  (all samples_*.jsonl inside)
    python scripts/eval_accuracy.py --dir result/MATH500/dora_cosine

    # Multiple directories at once
    python scripts/eval_accuracy.py \
        --dir result/MATH500/dora_cosine \
        --dir result/MATH500/dora_nli \
        --dir result/MATH500/dora_hybrid
"""

import argparse
import glob
import json
import os
import re
import sys
from concurrent.futures import TimeoutError
from pathlib import Path

import numpy as np
from pebble import ProcessPool
from tqdm import tqdm

# ── Resolve project root and evaluation/ directory ────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent          # scripts/
PROJECT_ROOT = SCRIPT_DIR.parent                      # adaptive_scientific_discovery/
EVAL_DIR = PROJECT_ROOT / "evaluation"

if not EVAL_DIR.exists():
    sys.exit(f"[ERROR] evaluation/ directory not found at {EVAL_DIR}")

sys.path.insert(0, str(EVAL_DIR))

from grader import math_equal_process          # noqa: E402
from parser import parse_ground_truth, extract_answer_map  # noqa: E402


# ── Filename parser ────────────────────────────────────────────────────────────

def parse_filename(path: str) -> dict:
    """
    Extract metadata from a DORA output filename.

    Supported patterns (all start with samples_):
        samples_cosine_n_{n}_alpha_{a}_seed_{s}_temp_{t}_*.jsonl
        samples_nli_{model}_lambda{l}_n_{n}_alpha_{a}_seed_{s}_temp_{t}_*.jsonl
        samples_hybrid_{model}_topk{k}_lambda{l}_n_{n}_alpha_{a}_seed_{s}_temp_{t}_*.jsonl
        samples_ablation_{mode}_n_{n}_alpha_{a}_seed_{s}_temp_{t}_*.jsonl
    """
    name = Path(path).stem  # drop .jsonl

    meta = {"filename": Path(path).name, "approach": "unknown",
            "n": None, "seed": None, "alpha": None, "temp": None}

    # similarity / approach
    if name.startswith("samples_cosine"):
        meta["approach"] = "cosine"
    elif name.startswith("samples_hybrid"):
        meta["approach"] = "hybrid"
        m = re.search(r"topk(\d+)", name)
        if m:
            meta["nli_top_k"] = int(m.group(1))
        m = re.search(r"lambda([\d.]+)", name)
        if m:
            meta["lambda"] = float(m.group(1))
    elif name.startswith("samples_nli"):
        meta["approach"] = "nli"
        m = re.search(r"lambda([\d.]+)", name)
        if m:
            meta["lambda"] = float(m.group(1))
    elif name.startswith("samples_ablation"):
        m = re.match(r"samples_ablation_(\w+)_n_", name)
        meta["approach"] = f"ablation_{m.group(1)}" if m else "ablation"

    # numeric fields
    for key, pattern in [("n", r"_n_(\d+)"), ("seed", r"_seed_(\d+)"),
                         ("alpha", r"_alpha_([\d.]+)"), ("temp", r"_temp_([\d.]+)")]:
        m = re.search(pattern, name)
        if m:
            meta[key] = int(m.group(1)) if key in ("n", "seed") else float(m.group(1))

    return meta


# ── Per-sample grading ─────────────────────────────────────────────────────────

def grade_file(filepath: str, benchmark: str = "math",
               pred_col: str = "pred", max_samples: int = None) -> dict:
    """
    Load a DORA .jsonl, extract predictions and ground-truth, grade each
    sample with math_equal, return accuracy stats.
    """
    samples = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if max_samples:
        samples = samples[:max_samples]

    if not samples:
        return {"error": "empty file", "acc": None, "n_samples": 0}

    # Parse ground truth + extract predicted answer for every sample
    params = []
    for idx, sample in enumerate(samples):
        try:
            _, gt = parse_ground_truth(sample, benchmark)
        except Exception as e:
            gt = sample.get("answer", "")

        try:
            pred_text = sample.get(pred_col, "")
            # extract_answer_map always returns {"pred": extracted_answer}
            pred = extract_answer_map({pred_col: pred_text},
                                      data_name=benchmark, col=pred_col)["pred"]
        except Exception:
            pred = sample.get(pred_col, "")

        params.append((idx, pred, gt))

    # Grade with timeout protection (same as evaluate_dir.py)
    scores = []
    timeout_cnt = 0
    # math_equal_process returns (bool_result, idx) — we only need [0]
    with ProcessPool(max_workers=min(8, os.cpu_count() or 1)) as pool:
        future = pool.map(math_equal_process, params, timeout=3)
        iterator = future.result()
        with tqdm(total=len(params), desc="  grading", leave=False) as pbar:
            while True:
                try:
                    result = next(iterator)
                    scores.append(result[0] if isinstance(result, tuple) else result)
                except StopIteration:
                    break
                except TimeoutError:
                    scores.append(False)
                    timeout_cnt += 1
                except Exception:
                    scores.append(False)
                finally:
                    pbar.update(1)

    acc = float(np.mean(scores)) * 100 if scores else 0.0
    return {
        "acc": acc,
        "n_samples": len(scores),
        "n_correct": int(sum(scores)),
        "timeout_cnt": timeout_cnt,
    }


# ── Pretty table printer ───────────────────────────────────────────────────────

def print_table(rows: list):
    """Print results as a fixed-width table sorted by approach then n."""
    rows = sorted(rows, key=lambda r: (r["approach"], r.get("n") or 0, r.get("seed") or 0))

    cols = ["approach", "n", "seed", "acc (%)", "correct", "total", "timeouts", "file"]
    widths = {c: len(c) for c in cols}
    for r in rows:
        widths["approach"] = max(widths["approach"], len(str(r["approach"])))
        widths["n"]        = max(widths["n"],        len(str(r.get("n", ""))))
        widths["seed"]     = max(widths["seed"],     len(str(r.get("seed", ""))))
        widths["acc (%)"]  = max(widths["acc (%)"],  len(f"{r['acc']:.2f}" if r["acc"] is not None else "error"))
        widths["correct"]  = max(widths["correct"],  len(str(r.get("n_correct", ""))))
        widths["total"]    = max(widths["total"],    len(str(r.get("n_samples", ""))))
        widths["timeouts"] = max(widths["timeouts"], len(str(r.get("timeout_cnt", ""))))
        widths["file"]     = max(widths["file"],     len(r["filename"]))

    sep = "+-" + "-+-".join("-" * widths[c] for c in cols) + "-+"
    hdr = "| " + " | ".join(c.ljust(widths[c]) for c in cols) + " |"

    print()
    print(sep)
    print(hdr)
    print(sep)
    for r in rows:
        acc_str = f"{r['acc']:.2f}" if r["acc"] is not None else "error"
        print("| " + " | ".join([
            str(r["approach"]).ljust(widths["approach"]),
            str(r.get("n", "")).ljust(widths["n"]),
            str(r.get("seed", "")).ljust(widths["seed"]),
            acc_str.ljust(widths["acc (%)"]),
            str(r.get("n_correct", "")).ljust(widths["correct"]),
            str(r.get("n_samples", "")).ljust(widths["total"]),
            str(r.get("timeout_cnt", "")).ljust(widths["timeouts"]),
            r["filename"].ljust(widths["file"]),
        ]) + " |")
    print(sep)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def collect_files(args) -> list:
    files = []
    if args.file:
        for f in args.file:
            if not os.path.isfile(f):
                print(f"[WARN] file not found: {f}")
            else:
                files.append(f)
    if args.dir:
        for d in args.dir:
            if not os.path.isdir(d):
                print(f"[WARN] directory not found: {d}")
                continue
            found = sorted(glob.glob(os.path.join(d, "samples_*.jsonl")))
            if not found:
                print(f"[WARN] no samples_*.jsonl files in {d}")
            files.extend(found)
    return files


def main():
    parser = argparse.ArgumentParser(description="Evaluate DORA output accuracy")
    parser.add_argument("--file", metavar="PATH", action="append",
                        help="Path to a specific .jsonl file (repeatable)")
    parser.add_argument("--dir", metavar="PATH", action="append",
                        help="Directory containing samples_*.jsonl files (repeatable)")
    parser.add_argument("--benchmark", default="math",
                        help="Benchmark name passed to grader (default: math)")
    parser.add_argument("--pred_col", default="pred",
                        help="Column name for predictions (default: pred)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Evaluate only the first N samples (for quick checks)")
    args = parser.parse_args()

    if not args.file and not args.dir:
        parser.error("Provide at least one --file or --dir argument.")

    files = collect_files(args)
    if not files:
        sys.exit("[ERROR] No files to evaluate.")

    print(f"\nFound {len(files)} file(s) to evaluate.\n")

    rows = []
    for filepath in files:
        meta = parse_filename(filepath)
        print(f"[{meta['approach']}] n={meta['n']} seed={meta['seed']}  {Path(filepath).name}")
        result = grade_file(filepath, benchmark=args.benchmark,
                            pred_col=args.pred_col, max_samples=args.max_samples)
        rows.append({**meta, **result})
        if result["acc"] is not None:
            print(f"  → acc = {result['acc']:.2f}%  "
                  f"({result['n_correct']}/{result['n_samples']} correct, "
                  f"{result['timeout_cnt']} timeouts)\n")
        else:
            print(f"  → ERROR: {result.get('error')}\n")

    if len(rows) > 1:
        print_table(rows)

    # Also dump to JSON next to this script for later inspection
    out_json = os.path.join(PROJECT_ROOT, "eval_results.json")
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Results saved to {out_json}")


if __name__ == "__main__":
    main()
