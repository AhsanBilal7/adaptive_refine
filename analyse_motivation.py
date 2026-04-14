"""
analyse_motivation.py
---------------------
Produces all data tables and CSV files needed for the five motivation
experiments of DN-BoN.  Does NOT produce plots — it saves every number
needed to build any figure or table downstream.

Verifier consistency
--------------------
The JSONL files must be generated with the SAME verifier used in your main
experiments (PRM by default).  The `raw_scores` field in each row contains
PRM scores — continuous quality estimates per trace.  The `is_correct` and
`all_correct` fields are always outcome-based (gold answer comparison) and
are used ONLY for measuring accuracy, never for candidate selection.

This separation is intentional:
  raw_scores    — PRM signal (selection signal, same as main method)
  is_correct    — ground truth accuracy (evaluation only)
  all_correct   — per-trace ground truth (Exp 3, 5 oracle analysis only)

With PRM scores, Exp 4 (score gap analysis) is particularly meaningful
because the gaps are real-valued — a small gap means the top-scored trace
barely beats the best trace from another direction, which is exactly where
normalization matters most.

Token-length measurement
------------------------
Real token counts via HuggingFace tokenizer (--model arg).
Falls back to NLTK word-count, then chars/4, in that order.
Never uses raw character length as a proxy for compute.

Output files (all written to --out_dir)
----------------------------------------
exp1_redundancy_vs_budget.csv
    Per (dataset, N) row:
    n_questions, mean_n_clusters, mean_cluster_entropy,
    mean_largest_cluster_frac, mean_unique_over_N,
    bon_acc, dnbon_acc, majority_acc

exp2_accuracy_by_redundancy_bin.csv
    Per (dataset, method, redundancy_bin) row:
    n_questions, accuracy, mean_largest_cluster_frac,
    mean_cluster_entropy, redundancy_bin_label

exp3_majority_cluster_analysis.csv
    Per (dataset) row:
    frac_largest_correct, frac_largest_wrong,
    frac_largest_correct_but_overrepresented,
    bon_acc_when_largest_wrong, dnbon_acc_when_largest_wrong,
    majority_acc_when_largest_wrong, n_total, n_largest_wrong

exp4_score_gap_vs_redundancy.csv
    Per (dataset, score_gap_bin, redundancy_bin) row:
    n_questions, bon_acc, dnbon_acc, mean_score_gap,
    mean_largest_cluster_frac

exp5_oracle_cluster_analysis.csv
    Per (dataset) row:
    oracle_cluster_acc (select best cluster, take max score within),
    oracle_direction_acc (if we knew correct cluster, accuracy),
    dnbon_acc, bon_acc, majority_acc,
    frac_gain_from_direction (oracle_dir - bon) / (oracle_dir - bon + eps),
    n_questions

summary_stats.csv
    One row per (dataset, method, N):
    accuracy, mean_token_length, std_token_length,
    mean_n_clusters, mean_cluster_entropy, mean_largest_frac

Usage
-----
  python analyse_motivation.py \\
      --results_dir results/Qwen2.5-Math-7B-Instruct_mnt2048_t0.7_p0.95/ \\
      --model Qwen/Qwen2.5-Math-7B-Instruct \\
      --out_dir motivation_analysis/

  # Without tokenizer (uses NLTK word count fallback)
  python analyse_motivation.py \\
      --results_dir results/Qwen2.5-Math-7B-Instruct_mnt2048_t0.7_p0.95/ \\
      --out_dir motivation_analysis/
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Token counting
# ──────────────────────────────────────────────────────────────────────────────

_TOKENIZER = None


def load_tokenizer(model_name: str, cache_dir: Optional[str] = None):
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER
    try:
        from transformers import AutoTokenizer
        print(f"Loading tokenizer: {model_name}")
        _TOKENIZER = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, cache_dir=cache_dir
        )
        print(f"Tokenizer loaded (vocab={_TOKENIZER.vocab_size})")
    except Exception as e:
        print(f"[warn] Tokenizer load failed ({e})")
        _TOKENIZER = None
    return _TOKENIZER


def token_count(text: str, tok=None) -> int:
    """Count tokens: real tokenizer > NLTK words > chars/4."""
    if not isinstance(text, str) or not text.strip():
        return 0
    if tok is not None:
        try:
            return len(tok.encode(text, add_special_tokens=False))
        except Exception:
            pass
    try:
        import nltk
        words = nltk.word_tokenize(text)
        return max(1, len(words))
    except Exception:
        pass
    return max(1, len(text) // 4)


# ──────────────────────────────────────────────────────────────────────────────
# JSONL loading and parsing
# ──────────────────────────────────────────────────────────────────────────────

KNOWN_DATASETS = ["math500", "aime24", "aime25", "amc23", "gsm8k"]
KNOWN_METHODS  = ["dn_bon", "bon", "bon_majority", "greedy", "greedy_pass_n"]


def _infer_dataset(fname: str) -> str:
    fname = fname.lower()
    for ds in KNOWN_DATASETS:
        if ds in fname:
            return ds
    return "unknown"


def _infer_method(fname: str) -> str:
    fname = fname.lower()
    # Order matters: check longer names first
    for m in ["greedy_pass_n", "bon_majority", "dn_bon", "bon", "greedy"]:
        if m in fname:
            return m
    return "unknown"


def _infer_N(fname: str) -> int:
    import re
    m = re.search(r"_N(\d+)", fname)
    return int(m.group(1)) if m else -1


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def discover_files(results_dir: str) -> Dict[Tuple[str, str, int], str]:
    """
    Scan results_dir for JSONL files and return
    {(dataset, method, N): filepath}
    """
    d = Path(results_dir)
    mapping = {}
    for fp in sorted(d.glob("*.jsonl")):
        if "_summary" in fp.name:
            continue
        dataset = _infer_dataset(fp.name)
        method  = _infer_method(fp.name)
        N       = _infer_N(fp.name)
        if dataset != "unknown" and method != "unknown" and N > 0:
            mapping[(dataset, method, N)] = str(fp)
    return mapping


# ──────────────────────────────────────────────────────────────────────────────
# Per-row feature extraction
# ──────────────────────────────────────────────────────────────────────────────

def cluster_entropy(cluster_sizes: List[int]) -> float:
    """Shannon entropy of cluster size distribution (base 2)."""
    total = sum(cluster_sizes)
    if total == 0:
        return 0.0
    probs = [s / total for s in cluster_sizes]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def cluster_stats(row: Dict) -> Dict:
    """
    Extract cluster-level statistics from one JSONL row.
    Works for dn_bon rows (have cluster_labels + cluster_sizes).
    Returns {} if not a DN-BoN row.
    """
    cs = row.get("cluster_sizes", [])
    cl = row.get("cluster_labels", [])
    if not cs or not cl:
        return {}

    # Unique clusters and their sizes
    label_to_size: Dict[int, int] = {}
    for label, size in zip(cl, cs):
        label_to_size[label] = size

    unique_sizes = list(label_to_size.values())
    n_clusters   = len(unique_sizes)
    total        = sum(unique_sizes)
    largest_size = max(unique_sizes)

    # Entropy of direction distribution
    entropy = cluster_entropy(unique_sizes)

    # Fraction of samples in the largest cluster
    largest_frac = largest_size / total if total > 0 else 0.0

    # Effective sample size = total / (1 + sum((p_i^2 - 1/n)) correction)
    # Simple proxy: unique directions / N
    unique_over_N = n_clusters / len(cl) if cl else 0.0

    return {
        "n_clusters":    n_clusters,
        "largest_size":  largest_size,
        "largest_frac":  largest_frac,
        "entropy":       entropy,
        "unique_over_N": unique_over_N,
    }


def raw_score_gap(row: Dict) -> float:
    """
    Gap between the top raw verifier score and the best score from
    a DIFFERENT cluster.

    With PRM (continuous scores):
      A small gap means the top-scored trace barely beats the best trace
      from another cluster — normalization matters here because the raw
      signal alone cannot decisively pick a direction.
      This is the primary use case for this metric.

    With outcome verifier (binary 0/1):
      Gap is either 0 (both clusters have a correct trace — tie) or 1
      (only one cluster is correct).  Less informative than PRM.
    """
    raw    = row.get("raw_scores", [])
    labels = row.get("cluster_labels", [])
    if not raw or not labels or len(raw) != len(labels):
        if raw:
            sorted_r = sorted(raw, reverse=True)
            return sorted_r[0] - sorted_r[1] if len(sorted_r) > 1 else 0.0
        return 0.0

    best_label = labels[int(np.argmax(raw))]
    best_score = max(raw)
    # Best score from a DIFFERENT cluster
    other_scores = [s for s, l in zip(raw, labels) if l != best_label]
    if not other_scores:
        return 0.0
    return best_score - max(other_scores)


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 1 — Redundancy vs Budget
# ──────────────────────────────────────────────────────────────────────────────

def exp1_redundancy_vs_budget(
    file_map: Dict,
    out_dir: str,
    tok=None,
):
    """
    For each (dataset, N): cluster stats + accuracy for dn_bon, bon, majority.
    """
    print("\n[Exp 1] Redundancy vs Budget")

    # Group dn_bon files by (dataset, N)
    dn_bon_files  = {(ds, N): p for (ds, m, N), p in file_map.items() if m == "dn_bon"}
    bon_files     = {(ds, N): p for (ds, m, N), p in file_map.items() if m == "bon"}
    maj_files     = {(ds, N): p for (ds, m, N), p in file_map.items() if m == "bon_majority"}

    rows_out = []
    for (dataset, N), path in sorted(dn_bon_files.items()):
        rows = load_jsonl(path)
        if not rows:
            continue

        stats_list = []
        tok_lengths = []
        for row in rows:
            s = cluster_stats(row)
            if s:
                stats_list.append(s)
            trace = row.get("raw_llm_output", "")
            if trace:
                tok_lengths.append(token_count(trace, tok))

        if not stats_list:
            continue

        dnbon_acc = np.mean([int(bool(r.get("is_correct", False))) for r in rows])
        bon_acc   = None
        maj_acc   = None

        if (dataset, N) in bon_files:
            bon_rows = load_jsonl(bon_files[(dataset, N)])
            if bon_rows:
                bon_acc = np.mean([int(bool(r.get("is_correct", False))) for r in bon_rows])
        if (dataset, N) in maj_files:
            maj_rows = load_jsonl(maj_files[(dataset, N)])
            if maj_rows:
                maj_acc = np.mean([int(bool(r.get("is_correct", False))) for r in maj_rows])

        mean = lambda key: float(np.mean([s[key] for s in stats_list]))

        rows_out.append({
            "dataset":               dataset,
            "N":                     N,
            "n_questions":           len(stats_list),
            "mean_n_clusters":       round(mean("n_clusters"), 3),
            "mean_cluster_entropy":  round(mean("entropy"), 4),
            "mean_largest_frac":     round(mean("largest_frac"), 4),
            "mean_unique_over_N":    round(mean("unique_over_N"), 4),
            "mean_token_length":     round(float(np.mean(tok_lengths)), 1) if tok_lengths else None,
            "std_token_length":      round(float(np.std(tok_lengths)), 1)  if tok_lengths else None,
            "dnbon_acc":             round(float(dnbon_acc), 4),
            "bon_acc":               round(float(bon_acc), 4) if bon_acc is not None else None,
            "majority_acc":          round(float(maj_acc), 4) if maj_acc is not None else None,
        })
        print(f"  {dataset} N={N}: n_clusters={rows_out[-1]['mean_n_clusters']:.2f}  "
              f"unique/N={rows_out[-1]['mean_unique_over_N']:.3f}  "
              f"largest_frac={rows_out[-1]['mean_largest_frac']:.3f}  "
              f"dn_bon_acc={rows_out[-1]['dnbon_acc']:.3f}")

    _write_csv(rows_out, os.path.join(out_dir, "exp1_redundancy_vs_budget.csv"))


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 2 — Accuracy Conditioned on Redundancy Bin
# ──────────────────────────────────────────────────────────────────────────────

def exp2_accuracy_by_redundancy(
    file_map: Dict,
    out_dir: str,
    n_bins: int = 3,
):
    """
    Bin problems by redundancy (largest_frac) into low/medium/high.
    Compare BoN, majority, DN-BoN accuracy within each bin.
    """
    print("\n[Exp 2] Accuracy Conditioned on Redundancy")

    dn_bon_files = {(ds, N): p for (ds, m, N), p in file_map.items() if m == "dn_bon"}
    bon_files    = {(ds, N): p for (ds, m, N), p in file_map.items() if m == "bon"}
    maj_files    = {(ds, N): p for (ds, m, N), p in file_map.items() if m == "bon_majority"}

    rows_out = []
    bin_labels = ["low_redundancy", "medium_redundancy", "high_redundancy"][:n_bins]

    for (dataset, N), dn_path in sorted(dn_bon_files.items()):
        dn_rows = load_jsonl(dn_path)
        if not dn_rows:
            continue

        # Load matching BoN and majority rows — index by question prefix
        def idx_by_q(rows):
            return {r.get("question", "")[:100]: r for r in rows}

        bon_idx = idx_by_q(load_jsonl(bon_files[(dataset, N)])) \
                  if (dataset, N) in bon_files else {}
        maj_idx = idx_by_q(load_jsonl(maj_files[(dataset, N)])) \
                  if (dataset, N) in maj_files else {}

        # Compute redundancy scores (largest_frac) for all dn_bon rows
        fracs, valid_rows = [], []
        for row in dn_rows:
            s = cluster_stats(row)
            if s:
                fracs.append(s["largest_frac"])
                valid_rows.append(row)

        if not fracs:
            continue

        # Bin edges from quantiles for balanced bins
        q_edges = np.quantile(fracs, np.linspace(0, 1, n_bins + 1))
        q_edges[0]  -= 1e-9  # include the minimum
        q_edges[-1] += 1e-9  # include the maximum

        for b in range(n_bins):
            lo, hi = q_edges[b], q_edges[b + 1]
            mask = [(lo < f <= hi) for f in fracs]
            bin_dn_rows = [r for r, m in zip(valid_rows, mask) if m]
            if not bin_dn_rows:
                continue

            dn_acc  = np.mean([int(bool(r.get("is_correct", False))) for r in bin_dn_rows])
            qs = [r.get("question", "")[:100] for r in bin_dn_rows]
            bon_acc = np.mean([int(bool(bon_idx[q].get("is_correct", False)))
                               for q in qs if q in bon_idx]) \
                      if bon_idx else None
            maj_acc = np.mean([int(bool(maj_idx[q].get("is_correct", False)))
                               for q in qs if q in maj_idx]) \
                      if maj_idx else None

            bin_fracs    = [f for f, m in zip(fracs, mask) if m]
            mean_entropy = np.mean([cluster_stats(r)["entropy"]
                                    for r in bin_dn_rows if cluster_stats(r)])

            rows_out.append({
                "dataset":              dataset,
                "N":                    N,
                "redundancy_bin":       b,
                "redundancy_bin_label": bin_labels[b] if b < len(bin_labels) else f"bin{b}",
                "largest_frac_lo":      round(float(lo), 4),
                "largest_frac_hi":      round(float(hi), 4),
                "mean_largest_frac":    round(float(np.mean(bin_fracs)), 4),
                "mean_entropy":         round(float(mean_entropy), 4),
                "n_questions":          len(bin_dn_rows),
                "dnbon_acc":            round(float(dn_acc), 4),
                "bon_acc":              round(float(bon_acc), 4) if bon_acc is not None else None,
                "majority_acc":         round(float(maj_acc), 4) if maj_acc is not None else None,
                "dnbon_minus_bon":      round(float(dn_acc - bon_acc), 4)
                                        if bon_acc is not None else None,
            })
            print(f"  {dataset} N={N} [{bin_labels[b] if b < len(bin_labels) else b}]  "
                  f"n={len(bin_dn_rows)}  dn_bon={dn_acc:.3f}  bon={bon_acc:.3f if bon_acc else 'N/A'}")

    _write_csv(rows_out, os.path.join(out_dir, "exp2_accuracy_by_redundancy_bin.csv"))


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 3 — "Majority is Correct" Analysis
# ──────────────────────────────────────────────────────────────────────────────

def exp3_majority_cluster_analysis(
    file_map: Dict,
    out_dir: str,
):
    """
    For each problem: is the largest cluster the correct one?
    Report accuracy of each method in each subgroup.
    """
    print("\n[Exp 3] Majority Cluster Correctness Analysis")

    dn_bon_files = {(ds, N): p for (ds, m, N), p in file_map.items() if m == "dn_bon"}
    bon_files    = {(ds, N): p for (ds, m, N), p in file_map.items() if m == "bon"}
    maj_files    = {(ds, N): p for (ds, m, N), p in file_map.items() if m == "bon_majority"}

    rows_out = []

    for (dataset, N), dn_path in sorted(dn_bon_files.items()):
        dn_rows = load_jsonl(dn_path)
        if not dn_rows:
            continue

        def idx_by_q(rows):
            return {r.get("question", "")[:100]: r for r in rows}

        bon_idx = idx_by_q(load_jsonl(bon_files[(dataset, N)])) \
                  if (dataset, N) in bon_files else {}
        maj_idx = idx_by_q(load_jsonl(maj_files[(dataset, N)])) \
                  if (dataset, N) in maj_files else {}

        # For each problem, determine whether the largest cluster is correct
        largest_correct, largest_wrong = [], []

        for row in dn_rows:
            cl     = row.get("cluster_labels", [])
            cs     = row.get("cluster_sizes",  [])
            ac     = row.get("all_correct",    [])   # per-trace correctness
            if not cl or not cs:
                continue

            # Find the label of the largest cluster
            label_to_size: Dict[int, int] = {}
            for label, size in zip(cl, cs):
                label_to_size[label] = size
            largest_label = max(label_to_size, key=label_to_size.__getitem__)
            largest_sz    = label_to_size[largest_label]

            # Is any trace in the largest cluster correct?
            if ac:
                largest_members_correct = any(
                    ac[i] for i, l in enumerate(cl) if l == largest_label
                )
            else:
                # Fall back: is the selected trace in the largest cluster?
                sel_idx = row.get("selected_index", 0)
                largest_members_correct = (
                    cl[sel_idx] == largest_label and bool(row.get("is_correct", False))
                )

            q_key = row.get("question", "")[:100]
            entry = {
                "question_key":      q_key,
                "dn_bon_correct":    bool(row.get("is_correct", False)),
                "largest_correct":   largest_members_correct,
                "largest_frac":      largest_sz / len(cl),
                "bon_correct":       bool(bon_idx[q_key].get("is_correct", False))
                                     if q_key in bon_idx else None,
                "maj_correct":       bool(maj_idx[q_key].get("is_correct", False))
                                     if q_key in maj_idx else None,
            }
            if largest_members_correct:
                largest_correct.append(entry)
            else:
                largest_wrong.append(entry)

        n_total  = len(largest_correct) + len(largest_wrong)
        if n_total == 0:
            continue

        n_lc = len(largest_correct)
        n_lw = len(largest_wrong)

        # Overrepresented-correct: largest is correct AND frac > 0.5
        n_overrep = sum(1 for e in largest_correct if e["largest_frac"] > 0.5)

        def acc(entries, key):
            vals = [int(e[key]) for e in entries if e[key] is not None]
            return round(float(np.mean(vals)), 4) if vals else None

        rows_out.append({
            "dataset":                        dataset,
            "N":                              N,
            "n_total":                        n_total,
            "n_largest_correct":              n_lc,
            "n_largest_wrong":                n_lw,
            "frac_largest_correct":           round(n_lc / n_total, 4),
            "frac_largest_wrong":             round(n_lw / n_total, 4),
            "frac_largest_correct_overrep":   round(n_overrep / n_total, 4),
            # Accuracy of each method WHEN LARGEST CLUSTER IS WRONG
            "dnbon_acc_when_largest_wrong":   acc(largest_wrong, "dn_bon_correct"),
            "bon_acc_when_largest_wrong":     acc(largest_wrong, "bon_correct"),
            "majority_acc_when_largest_wrong": acc(largest_wrong, "maj_correct"),
            # Overall
            "dnbon_acc_overall":              acc(largest_correct + largest_wrong, "dn_bon_correct"),
            "bon_acc_overall":                acc(largest_correct + largest_wrong, "bon_correct"),
            "majority_acc_overall":           acc(largest_correct + largest_wrong, "maj_correct"),
        })
        print(f"  {dataset} N={N}: largest_wrong_frac={rows_out[-1]['frac_largest_wrong']:.3f}  "
              f"dn_bon_when_wrong={rows_out[-1]['dnbon_acc_when_largest_wrong']}")

    _write_csv(rows_out, os.path.join(out_dir, "exp3_majority_cluster_analysis.csv"))


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 4 — Score Gap vs Redundancy
# ──────────────────────────────────────────────────────────────────────────────

def exp4_score_gap_vs_redundancy(
    file_map: Dict,
    out_dir: str,
    n_bins: int = 3,
):
    """
    When the top raw score gap is small AND redundancy is high,
    normalization helps most.
    """
    print("\n[Exp 4] Score Gap vs Redundancy")

    dn_bon_files = {(ds, N): p for (ds, m, N), p in file_map.items() if m == "dn_bon"}
    bon_files    = {(ds, N): p for (ds, m, N), p in file_map.items() if m == "bon"}

    rows_out = []
    bin_labels = ["small_gap", "medium_gap", "large_gap"][:n_bins]
    red_labels = ["low_redundancy", "medium_redundancy", "high_redundancy"][:n_bins]

    for (dataset, N), dn_path in sorted(dn_bon_files.items()):
        dn_rows = load_jsonl(dn_path)
        if not dn_rows:
            continue

        def idx_by_q(rows):
            return {r.get("question", "")[:100]: r for r in rows}
        bon_idx = idx_by_q(load_jsonl(bon_files[(dataset, N)])) \
                  if (dataset, N) in bon_files else {}

        # Extract (gap, redundancy, dn_correct, bon_correct) per row
        gaps, fracs, dn_corrects, bon_corrects = [], [], [], []
        for row in dn_rows:
            s = cluster_stats(row)
            if not s:
                continue
            gap = raw_score_gap(row)
            q   = row.get("question", "")[:100]
            gaps.append(gap)
            fracs.append(s["largest_frac"])
            dn_corrects.append(int(bool(row.get("is_correct", False))))
            bon_corrects.append(int(bool(bon_idx[q].get("is_correct", False)))
                                if q in bon_idx else None)

        if not gaps:
            continue

        # Bin by gap quantiles
        gap_edges  = np.quantile(gaps,  np.linspace(0, 1, n_bins + 1))
        frac_edges = np.quantile(fracs, np.linspace(0, 1, n_bins + 1))
        gap_edges[0]  -= 1e-9; gap_edges[-1]  += 1e-9
        frac_edges[0] -= 1e-9; frac_edges[-1] += 1e-9

        for gb in range(n_bins):
            for rb in range(n_bins):
                glo, ghi = gap_edges[gb],  gap_edges[gb + 1]
                rlo, rhi = frac_edges[rb], frac_edges[rb + 1]
                mask = [
                    (glo < g <= ghi) and (rlo < f <= rhi)
                    for g, f in zip(gaps, fracs)
                ]
                idxs = [i for i, m in enumerate(mask) if m]
                if len(idxs) < 3:
                    continue

                dn_acc  = float(np.mean([dn_corrects[i] for i in idxs]))
                bon_v   = [bon_corrects[i] for i in idxs if bon_corrects[i] is not None]
                bon_acc = float(np.mean(bon_v)) if bon_v else None

                rows_out.append({
                    "dataset":           dataset,
                    "N":                 N,
                    "gap_bin":           gb,
                    "gap_bin_label":     bin_labels[gb] if gb < len(bin_labels) else f"gap{gb}",
                    "redundancy_bin":    rb,
                    "redundancy_bin_label": red_labels[rb] if rb < len(red_labels) else f"red{rb}",
                    "gap_lo":            round(float(glo), 4),
                    "gap_hi":            round(float(ghi), 4),
                    "redundancy_lo":     round(float(rlo), 4),
                    "redundancy_hi":     round(float(rhi), 4),
                    "mean_score_gap":    round(float(np.mean([gaps[i] for i in idxs])), 4),
                    "mean_largest_frac": round(float(np.mean([fracs[i] for i in idxs])), 4),
                    "n_questions":       len(idxs),
                    "dnbon_acc":         round(dn_acc, 4),
                    "bon_acc":           round(float(bon_acc), 4) if bon_acc is not None else None,
                    "dnbon_minus_bon":   round(dn_acc - bon_acc, 4)
                                        if bon_acc is not None else None,
                })

    _write_csv(rows_out, os.path.join(out_dir, "exp4_score_gap_vs_redundancy.csv"))
    print(f"  Written {len(rows_out)} (gap_bin, redundancy_bin) cells")


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 5 — Oracle Cluster Analysis
# ──────────────────────────────────────────────────────────────────────────────

def exp5_oracle_cluster_analysis(
    file_map: Dict,
    out_dir: str,
):
    """
    Oracle 1: select the cluster whose representative has the highest raw score
              (cluster-level BoN — eliminates within-cluster duplicates).
    Oracle 2: select the correct cluster (if any trace is correct in it).
    These bound how much gain is theoretically available from better direction
    selection vs better within-direction selection.
    """
    print("\n[Exp 5] Oracle Cluster Analysis")

    dn_bon_files = {(ds, N): p for (ds, m, N), p in file_map.items() if m == "dn_bon"}
    bon_files    = {(ds, N): p for (ds, m, N), p in file_map.items() if m == "bon"}
    maj_files    = {(ds, N): p for (ds, m, N), p in file_map.items() if m == "bon_majority"}

    rows_out = []

    for (dataset, N), dn_path in sorted(dn_bon_files.items()):
        dn_rows = load_jsonl(dn_path)
        if not dn_rows:
            continue

        def idx_by_q(rows):
            return {r.get("question", "")[:100]: r for r in rows}
        bon_idx = idx_by_q(load_jsonl(bon_files[(dataset, N)])) \
                  if (dataset, N) in bon_files else {}
        maj_idx = idx_by_q(load_jsonl(maj_files[(dataset, N)])) \
                  if (dataset, N) in maj_files else {}

        oracle_cluster_correct = []   # Oracle 1: max score per cluster
        oracle_direction_correct = [] # Oracle 2: any correct trace in any cluster
        dnbon_correct = []
        bon_correct   = []
        maj_correct   = []

        for row in dn_rows:
            cl  = row.get("cluster_labels", [])
            raw = row.get("raw_scores",     [])
            ac  = row.get("all_correct",    [])
            if not cl or not raw:
                continue

            q = row.get("question", "")[:100]

            # Oracle 1: for each unique cluster, take the max raw score trace.
            # Then pick the cluster whose representative has the highest score.
            label_to_best_correct: Dict[int, bool] = {}
            label_to_best_score:   Dict[int, float] = {}
            for i, (label, score) in enumerate(zip(cl, raw)):
                if label not in label_to_best_score or score > label_to_best_score[label]:
                    label_to_best_score[label] = score
                    label_to_best_correct[label] = bool(ac[i]) if ac else False
            best_cluster = max(label_to_best_score, key=label_to_best_score.__getitem__)
            oracle_cluster_correct.append(int(label_to_best_correct[best_cluster]))

            # Oracle 2: is there ANY cluster that contains a correct trace?
            if ac:
                oracle_direction_correct.append(int(any(ac)))
            else:
                oracle_direction_correct.append(int(bool(row.get("is_correct", False))))

            dnbon_correct.append(int(bool(row.get("is_correct", False))))
            if q in bon_idx:
                bon_correct.append(int(bool(bon_idx[q].get("is_correct", False))))
            if q in maj_idx:
                maj_correct.append(int(bool(maj_idx[q].get("is_correct", False))))

        if not oracle_cluster_correct:
            continue

        n = len(oracle_cluster_correct)
        oracle_cl_acc  = float(np.mean(oracle_cluster_correct))
        oracle_dir_acc = float(np.mean(oracle_direction_correct))
        dn_acc         = float(np.mean(dnbon_correct))
        bon_acc_v      = float(np.mean(bon_correct)) if bon_correct else None
        maj_acc_v      = float(np.mean(maj_correct)) if maj_correct else None

        # Fraction of the oracle direction gap that DN-BoN captures
        oracle_gap = oracle_dir_acc - (bon_acc_v if bon_acc_v else dn_acc)
        dnbon_gain = dn_acc         - (bon_acc_v if bon_acc_v else dn_acc)
        frac_captured = (dnbon_gain / oracle_gap) if abs(oracle_gap) > 1e-6 else None

        rows_out.append({
            "dataset":                  dataset,
            "N":                        N,
            "n_questions":              n,
            "oracle_cluster_acc":       round(oracle_cl_acc,  4),
            "oracle_direction_acc":     round(oracle_dir_acc, 4),
            "dnbon_acc":                round(dn_acc, 4),
            "bon_acc":                  round(float(bon_acc_v), 4) if bon_acc_v else None,
            "majority_acc":             round(float(maj_acc_v), 4) if maj_acc_v else None,
            # Gap analysis
            "oracle_direction_gap":     round(oracle_dir_acc - (bon_acc_v or dn_acc), 4),
            "dnbon_vs_bon_gain":        round(dn_acc - (bon_acc_v or dn_acc), 4),
            "frac_oracle_gap_captured": round(float(frac_captured), 4)
                                        if frac_captured is not None else None,
        })
        print(f"  {dataset} N={N}: oracle_dir={oracle_dir_acc:.3f}  "
              f"oracle_cluster={oracle_cl_acc:.3f}  "
              f"dn_bon={dn_acc:.3f}  bon={bon_acc_v:.3f if bon_acc_v else 'N/A'}")

    _write_csv(rows_out, os.path.join(out_dir, "exp5_oracle_cluster_analysis.csv"))


# ──────────────────────────────────────────────────────────────────────────────
# Summary statistics table
# ──────────────────────────────────────────────────────────────────────────────

def summary_stats(
    file_map: Dict,
    out_dir: str,
    tok=None,
):
    """One row per (dataset, method, N) with accuracy + token length stats."""
    print("\n[Summary] Per-file statistics")
    rows_out = []
    for (dataset, method, N), path in sorted(file_map.items()):
        rows = load_jsonl(path)
        if not rows:
            continue

        acc = float(np.mean([int(bool(r.get("is_correct", False))) for r in rows]))
        tok_lens = [token_count(r.get("raw_llm_output", ""), tok)
                    for r in rows if r.get("raw_llm_output")]

        cs_stats = [cluster_stats(r) for r in rows if method == "dn_bon"]
        cs_stats = [s for s in cs_stats if s]

        rows_out.append({
            "dataset":          dataset,
            "method":           method,
            "N":                N,
            "verifier":         rows[0].get("method", "unknown"),   # from row metadata
            "n_questions":      len(rows),
            "accuracy":         round(acc, 4),
            "mean_token_length": round(float(np.mean(tok_lens)), 1) if tok_lens else None,
            "std_token_length":  round(float(np.std(tok_lens)), 1)  if tok_lens else None,
            "p25_token_length":  round(float(np.percentile(tok_lens, 25)), 1) if tok_lens else None,
            "p75_token_length":  round(float(np.percentile(tok_lens, 75)), 1) if tok_lens else None,
            "mean_n_clusters":   round(float(np.mean([s["n_clusters"] for s in cs_stats])), 3)
                                 if cs_stats else None,
            "mean_entropy":      round(float(np.mean([s["entropy"] for s in cs_stats])), 4)
                                 if cs_stats else None,
            "mean_largest_frac": round(float(np.mean([s["largest_frac"] for s in cs_stats])), 4)
                                 if cs_stats else None,
        })
        print(f"  {dataset:10s}  {method:15s}  N={N}  acc={acc:.3f}  n={len(rows)}")

    _write_csv(rows_out, os.path.join(out_dir, "summary_stats.csv"))


# ──────────────────────────────────────────────────────────────────────────────
# CSV writer
# ──────────────────────────────────────────────────────────────────────────────

def _write_csv(rows: List[Dict], path: str):
    if not rows:
        print(f"  [skip] {path} — no data")
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  → {path}  ({len(rows)} rows)")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate all DN-BoN motivation experiment data from JSONL results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results_dir", required=True,
                   help="Directory containing JSONL result files.")
    p.add_argument("--out_dir",     default="motivation_analysis",
                   help="Output directory for CSV tables.")
    p.add_argument("--model",       default=None,
                   help="HF model id for real token counting (e.g. Qwen/Qwen2.5-Math-7B-Instruct).")
    p.add_argument("--hf_cache",    default=None)
    p.add_argument("--n_bins",      type=int, default=3,
                   help="Number of bins for Exp 2 and Exp 4.")
    p.add_argument("--exps",        nargs="+",
                   default=["1", "2", "3", "4", "5", "summary"],
                   help="Which experiments to run: 1 2 3 4 5 summary")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load tokenizer
    tok = None
    if args.model:
        tok = load_tokenizer(args.model, args.hf_cache)

    # Discover all result files
    print(f"\nScanning: {args.results_dir}")
    file_map = discover_files(args.results_dir)
    print(f"Found {len(file_map)} result file(s):")
    for (ds, m, N), path in sorted(file_map.items()):
        print(f"  {ds:10s}  {m:15s}  N={N:2d}  {Path(path).name}")

    if not file_map:
        print("ERROR: No JSONL result files found.")
        sys.exit(1)

    exps = set(args.exps)

    if "summary" in exps:
        summary_stats(file_map, args.out_dir, tok)
    if "1" in exps:
        exp1_redundancy_vs_budget(file_map, args.out_dir, tok)
    if "2" in exps:
        exp2_accuracy_by_redundancy(file_map, args.out_dir, args.n_bins)
    if "3" in exps:
        exp3_majority_cluster_analysis(file_map, args.out_dir)
    if "4" in exps:
        exp4_score_gap_vs_redundancy(file_map, args.out_dir, args.n_bins)
    if "5" in exps:
        exp5_oracle_cluster_analysis(file_map, args.out_dir)

    print(f"\n{'='*60}")
    print(f"  All analysis complete.  Output: {args.out_dir}/")
    print(f"{'='*60}")
    print("  Files written:")
    for f in sorted(Path(args.out_dir).glob("*.csv")):
        size = f.stat().st_size
        print(f"    {f.name:<45} {size:>8} bytes")


if __name__ == "__main__":
    main()