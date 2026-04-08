#!/usr/bin/env python3
"""
run_ablations.py – Reproduce all ablation studies from the AdaptSci paper.

Usage:
    python run_ablations.py \
        --model-name Qwen3-Coder-480B \
        --model-url http://localhost:8000/v1 \
        --sandbox-urls http://localhost:8080/run_code \
        --parquet-file-path data/inference/lsr_synth.parquet \
        --output-dir results/ablations \
        --mape-threshold 0.001 \
        --num-turns 40 \
        --top-k 3

Each ablation row from Section 6 of the AdaptSci paper is run sequentially.
Results are saved as separate JSON files and a summary CSV is written.
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Ablation configurations (Section 6, "Key ablations")
# ---------------------------------------------------------------------------

ABLATION_CONFIGS = [
    # Name, extra CLI args
    (
        "full_adaptsci",
        "Full AdaptSci (all mechanisms on)",
        [],
    ),
    (
        "no_early_stopping",
        "Ablation (i): remove early stopping",
        ["--ablation-disable-early-stopping"],
    ),
    (
        "fix_n_restarts",
        "Ablation (ii): fix n_restarts=1",
        ["--ablation-fix-n-restarts"],
    ),
    (
        "alpha_0",
        "Ablation (iii): α=0 (diversity only)",
        ["--ablation-alpha-override", "0"],
    ),
    (
        "alpha_0.25",
        "Ablation (iii): α=0.25",
        ["--ablation-alpha-override", "0.25"],
    ),
    (
        "alpha_0.5",
        "Ablation (iii): α=0.5 (default)",
        ["--ablation-alpha-override", "0.5"],
    ),
    (
        "alpha_0.75",
        "Ablation (iii): α=0.75",
        ["--ablation-alpha-override", "0.75"],
    ),
    (
        "alpha_1",
        "Ablation (iii): α=1 (improvement only)",
        ["--ablation-alpha-override", "1"],
    ),
    (
        "no_c_diversity",
        "Ablation (iv): C-diversity off",
        ["--ablation-disable-c-diversity"],
    ),
    (
        "random_frontier",
        "Ablation (iv): random frontier sampling",
        ["--ablation-random-frontier"],
    ),
    (
        "hessian_at_first_run",
        "Ablation (v): Hessian at first BFGS run",
        ["--ablation-hessian-at-first-run"],
    ),
    (
        "sr_scientist_baseline",
        "Baseline: SR-Scientist (no adaptive compute)",
        ["--no-adaptive-compute"],
    ),
]


def build_command(base_args: list, extra_args: list, output_path: str) -> list:
    return (
        [sys.executable, "main.py"]
        + base_args
        + ["--output-json-path", output_path]
        + extra_args
    )


def extract_overall_acc(json_path: str) -> dict:
    """Extract overall Acc0.01 and Acc0.001 from a results JSON file."""
    try:
        with open(json_path) as f:
            results = json.load(f)
    except Exception:
        return {}

    all_subs = [
        r["submitted_results"]
        for r in results
        if r.get("submitted_results") and isinstance(r["submitted_results"].get("mse"), (int, float))
    ]
    if not all_subs:
        return {}

    import numpy as np
    return {
        "n": len(all_subs),
        "acc_0.01":  float(np.mean([s.get("acc_0.01", 0) for s in all_subs])),
        "acc_0.001": float(np.mean([s.get("acc_0.001", 0) for s in all_subs])),
        "success":   float(np.mean([s.get("success", 0) for s in all_subs])),
    }


def main():
    parser = argparse.ArgumentParser(description="Run all AdaptSci ablation studies.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-url", required=True)
    parser.add_argument("--sandbox-urls", nargs="+", required=True)
    parser.add_argument("--parquet-file-path", required=True)
    parser.add_argument("--output-dir", default="results/ablations")
    parser.add_argument("--mape-threshold", type=float, default=0.001)
    parser.add_argument("--num-turns", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-assistant-turns", type=int, default=25)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--source", nargs="+", default=None)
    parser.add_argument(
        "--run-only",
        nargs="*",
        default=None,
        help="Run only these ablation names (space-separated). Default: all.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Base CLI args (shared across all ablations)
    base_args = [
        "--model-name", args.model_name,
        "--model-url", args.model_url,
        "--sandbox-urls", *args.sandbox_urls,
        "--parquet-file-path", args.parquet_file_path,
        "--mape-threshold", str(args.mape_threshold),
        "--num-turns", str(args.num_turns),
        "--top-k", str(args.top_k),
        "--max-assistant-turns", str(args.max_assistant_turns),
        "--api_key", args.api_key,
    ]
    if args.source:
        base_args += ["--source"] + args.source

    summary_rows = []

    for name, description, extra_args in ABLATION_CONFIGS:
        if args.run_only and name not in args.run_only:
            continue

        output_path = os.path.join(args.output_dir, f"{name}.json")
        print(f"\n{'='*60}")
        print(f"Running ablation: {name}")
        print(f"Description: {description}")
        print(f"Output: {output_path}")
        print("=" * 60)

        cmd = build_command(base_args, extra_args, output_path)
        print("Command:", " ".join(cmd))

        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"WARNING: Ablation '{name}' exited with code {ret.returncode}.")

        # Extract results
        metrics = extract_overall_acc(output_path)
        metrics["name"] = name
        metrics["description"] = description
        summary_rows.append(metrics)
        print(f"  → Acc0.01={metrics.get('acc_0.01', 'N/A'):.2%}  "
              f"Acc0.001={metrics.get('acc_0.001', 'N/A'):.2%}")

    # Write summary CSV
    try:
        import csv
        summary_path = os.path.join(args.output_dir, "ablation_summary.csv")
        fields = ["name", "description", "n", "acc_0.01", "acc_0.001", "success"]
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nAblation summary saved to: {summary_path}")

        # Pretty print table
        print(f"\n{'Ablation':<35} {'Acc0.01':>10} {'Acc0.001':>10} {'Success':>10}")
        print("-" * 70)
        for row in summary_rows:
            print(
                f"{row.get('name', ''):<35} "
                f"{row.get('acc_0.01', 0):>9.2%} "
                f"{row.get('acc_0.001', 0):>9.2%} "
                f"{row.get('success', 0):>9.2%}"
            )
    except Exception as e:
        print(f"Warning: could not write summary CSV: {e}")


if __name__ == "__main__":
    main()
