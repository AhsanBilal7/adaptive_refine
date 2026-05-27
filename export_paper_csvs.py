#!/usr/bin/env python3
"""
export_paper_csvs.py
====================
Read the pre-computed metrics_summary.json files and export three tidy CSV
files that feed directly into the MATLAB paper plots.

Target models  : Qwen2.5-Math-7B, Qwen2.5-1.5B, Ministral-8B, LLaMA-3.1-8B
Target config  : N=8, D=4  (no-critique ablations excluded)

Output CSVs (saved to --output_dir, default: experiments_results/paper_csvs/)
────────────────────────────────────────────────────────────────────────────
depth_accuracy_metrics.csv
    model, dataset, depth,
    improvement_rate, recovery_rate, regression_rate

correction_effectiveness.csv
    model, dataset, depth,
    answer_change_rate,
    flip_wrong_to_correct, flip_correct_to_wrong, flip_wrong_to_wrong,
    net_correction_benefit

diversity.csv
    model, dataset,
    mean_n_unique_final_answers, frac_questions_diverse,
    mean_final_prm_std, mean_final_prm_range,
    depth_1_agreement, depth_2_agreement, depth_3_agreement, depth_4_agreement

Usage
-----
    python export_paper_csvs.py                          # use defaults
    python export_paper_csvs.py --metrics_dir experiments_results/cimr_metrics \
                                --output_dir  experiments_results/paper_csvs
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List

# ─── Configuration ────────────────────────────────────────────────────────────

# Map folder prefix → short display name for plots / CSV
MODEL_MAP: Dict[str, str] = {
    "Qwen2.5-Math-7B-Instruct": "Qwen2.5-Math-7B",
    "Qwen2.5-1.5B-Instruct":    "Qwen2.5-1.5B",
    "Ministral-8B-Instruct":    "Ministral-8B",
    "Llama-3.1-8B-Instruct":    "LLaMA-3.1-8B",
}

TARGET_N = 8
TARGET_D = 4
FILE_SUFFIX = f"_N{TARGET_N}_d{TARGET_D}.json"    # e.g. _N8_d4.json
EXCLUDE_TAG = "no_critique"

DEPTHS = list(range(1, TARGET_D + 1))

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _match_model(folder_name: str) -> str | None:
    """Return the short display name if folder_name starts with a known prefix."""
    for prefix, short in MODEL_MAP.items():
        if folder_name.startswith(prefix):
            return short
    return None


def _dataset_from_key(key: str) -> str:
    """Extract dataset name from a JSON key like 'cimr_full_results_amc23_N8_d4.json'."""
    # strip prefix / suffix
    s = key.replace("cimr_full_results_", "").replace(FILE_SUFFIX, "")
    return s


def _write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote  {os.path.relpath(path)}  ({len(rows)} rows)")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Export paper CSVs from CIMR metrics.")
    parser.add_argument(
        "--metrics_dir",
        default="experiments_results/cimr_metrics",
        help="Folder containing one metrics_summary.json per model sub-directory",
    )
    parser.add_argument(
        "--output_dir",
        default="experiments_results/paper_csvs",
        help="Destination folder for the three CSV files",
    )
    args = parser.parse_args()

    depth_accuracy_rows: List[Dict] = []
    correction_rows: List[Dict] = []
    diversity_rows: List[Dict] = []

    for folder in sorted(os.listdir(args.metrics_dir)):
        short_name = _match_model(folder)
        if short_name is None:
            continue                                    # skip non-target models

        summary_path = os.path.join(args.metrics_dir, folder, "metrics_summary.json")
        if not os.path.isfile(summary_path):
            print(f"  [WARN] missing {summary_path}")
            continue

        with open(summary_path, "r", encoding="utf-8") as f:
            summary: Dict[str, Any] = json.load(f)

        for key, entry in summary.items():
            # filter: must be N8_d4 and not an ablation
            if not key.endswith(FILE_SUFFIX):
                continue
            if EXCLUDE_TAG in key:
                continue

            dataset = _dataset_from_key(key)
            acc     = entry.get("accuracy", {})
            corr    = entry.get("correction_effectiveness", {})
            div     = entry.get("diversity", {})

            impr  = acc.get("depth_improvement_rate", {})
            recov = acc.get("depth_recovery_rate", {})
            regr  = acc.get("depth_regression_rate", {})

            # ── depth_accuracy_metrics.csv ────────────────────────────────
            for d in DEPTHS:
                depth_accuracy_rows.append({
                    "model":            short_name,
                    "dataset":          dataset,
                    "depth":            d,
                    "improvement_rate": impr.get(str(d), ""),
                    "recovery_rate":    recov.get(str(d), ""),
                    "regression_rate":  regr.get(str(d), ""),
                })

            # ── correction_effectiveness.csv ──────────────────────────────
            for d in DEPTHS:
                c = corr.get(str(d), {})
                correction_rows.append({
                    "model":                short_name,
                    "dataset":              dataset,
                    "depth":                d,
                    "answer_change_rate":   c.get("answer_change_rate", ""),
                    "flip_wrong_to_correct": c.get("flip_wrong_to_correct", ""),
                    "flip_correct_to_wrong": c.get("flip_correct_to_wrong", ""),
                    "flip_wrong_to_wrong":  c.get("flip_wrong_to_wrong", ""),
                    "net_correction_benefit": c.get("net_correction_benefit", ""),
                })

            # ── diversity.csv ─────────────────────────────────────────────
            dw_agree = div.get("depth_wise_answer_agreement", {})
            diversity_rows.append({
                "model":                        short_name,
                "dataset":                      dataset,
                "mean_n_unique_final_answers":   div.get("mean_n_unique_final_answers", ""),
                "frac_questions_diverse":        div.get("frac_questions_diverse", ""),
                "mean_final_prm_std":            div.get("mean_final_prm_std", ""),
                "mean_final_prm_range":          div.get("mean_final_prm_range", ""),
                "depth_1_agreement":             dw_agree.get("1", ""),
                "depth_2_agreement":             dw_agree.get("2", ""),
                "depth_3_agreement":             dw_agree.get("3", ""),
                "depth_4_agreement":             dw_agree.get("4", ""),
            })

    # ── write CSVs ────────────────────────────────────────────────────────────
    _write_csv(
        os.path.join(args.output_dir, "depth_accuracy_metrics.csv"),
        depth_accuracy_rows,
        ["model", "dataset", "depth",
         "improvement_rate", "recovery_rate", "regression_rate"],
    )
    _write_csv(
        os.path.join(args.output_dir, "correction_effectiveness.csv"),
        correction_rows,
        ["model", "dataset", "depth",
         "answer_change_rate",
         "flip_wrong_to_correct", "flip_correct_to_wrong", "flip_wrong_to_wrong",
         "net_correction_benefit"],
    )
    _write_csv(
        os.path.join(args.output_dir, "diversity.csv"),
        diversity_rows,
        ["model", "dataset",
         "mean_n_unique_final_answers", "frac_questions_diverse",
         "mean_final_prm_std", "mean_final_prm_range",
         "depth_1_agreement", "depth_2_agreement",
         "depth_3_agreement", "depth_4_agreement"],
    )
    print("Done.")


if __name__ == "__main__":
    main()
