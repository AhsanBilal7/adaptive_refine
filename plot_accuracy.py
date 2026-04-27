#!/usr/bin/env python3
"""
plot_accuracy.py
================
Recursively scan the results folder structure produced by run_experiments.sh,
parse every summary JSON + JSONL file, aggregate into a master CSV, produce
publication-quality bar plots, and save supplementary CSVs for:

  • accuracy           — one row per (folder, method, dataset, N)
  • latency            — avg_elapsed_sec per question, sourced from summary JSONs
  • theoretical FLOPs  — computed from actual token counts read from JSONL files
                         using a real tokenizer (tiktoken cl100k_base, or the
                         transformers AutoTokenizer for the run model when available)
  • cluster entropy    — per-question Shannon entropy over cluster-size distribution,
                         averaged per (method, dataset, N), sourced from JSONL files

FLOPs formula (model-size × token count, theoretical)
------------------------------------------------------
For a transformer with P parameters the dominant cost is the
matrix-multiply at each token: approximately 2P FLOPs per token
(forward pass; multiply by 2 for training, but here we do inference).
  FLOPs ≈ 2 × P × T_total
where T_total is the actual total number of tokens (prompt + generated)
measured with a real tokenizer from the stored traces.

Per-method component breakdown
-------------------------------
  greedy        : 2 × P_policy × T_total
  bon           : 2 × P_policy × T_total  +  2 × P_prm × T_total
  dn_bon        : 2 × P_policy × T_total  +  2 × P_prm × T_total
                  + 2 × P_embed × T_total   (sentence encoder)
  dn_bon_updated: same as dn_bon
  greedy_pass_n : 2 × P_policy × T_total   (no PRM, no embedding)
  bon_majority  : 2 × P_policy × T_total   (no PRM, no embedding)

P_policy, P_prm, P_embed are read from --policy_params, --prm_params,
--embed_params CLI flags (in billions; default sensible values given).

Token counting
--------------
1. Try to load the exact tokenizer from the HuggingFace model name stored in
   the folder name (e.g. Qwen2.5-Math-7B-Instruct).  This requires the model
   to be cached locally or network access.
2. Fall back to tiktoken cl100k_base (GPT-4 / Llama tokeniser family) which
   is a very close proxy for most modern LLMs.
3. Fall back to whitespace split as a last resort.

Token counts are measured on the actual "raw_llm_output" field of every
JSONL row.  If the JSONL is absent, token count is estimated as the mean
trace_length (characters) divided by a 4-char-per-token proxy.

Cluster entropy
---------------
For each question in a JSONL file, if cluster_sizes are stored (DN-BoN rows):
  p_m = k_m / N   for each cluster m
  H   = -Σ_m p_m log2(p_m)   (Shannon entropy, bits)
Averaged over all questions → avg_cluster_entropy per (method, dataset, N).

Folder structure expected
--------------------------
results/
  {ModelName}_mnt{MAX_TOKENS}_t{TEMP}_p{TOP_P}[_4bit|_8bit]/
    {method}_{dataset}_N{N}_summary.json
    {method}_{dataset}_N{N}.jsonl
    ...

Output
------
  results_master.csv                    — all models, all methods, all metrics in one file
  results_master_{model}_flops.csv      — FLOPs for every method for that model
  results_master_{model}_latency.csv    — latency for every method for that model
  results_master_{model}_entropy.csv    — entropy columns for every method (NaN for
                                          non-clustering methods like greedy/bon/majority)
  results_master_all_models_flops.csv   — combined across all models (when >1 model)
  results_master_all_models_latency.csv
  results_master_all_models_entropy.csv
  plots/{model}/accuracy_*.pdf/.png     — per-model accuracy bar charts
  plots/{model}/avg_elapsed_sec_*.pdf   — per-model latency (all methods)
  plots/{model}/flops_total_*.pdf       — per-model total FLOPs (all methods)
  plots/{model}/flops_stacked_*.pdf     — per-model stacked FLOPs by component
  plots/{model}/avg_entropy_*.pdf       — per-model entropy (dn_bon only has data)
  plots/{model}/avg_entropy_norm_*.pdf
  plots/{model}/avg_frac_largest_*.pdf
  plots/{model}/avg_unique_over_N_*.pdf
  plots/all_models/…                    — combined cross-model plots (when >1 model)
"""

import argparse
import json
import math
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Aesthetics (NeurIPS / publication style) ──────────────────────────────────
plt.rcParams.update({
    "text.usetex":         False,
    "font.family":         "sans-serif",
    "font.size":           9,
    "axes.titlesize":      9,
    "axes.labelsize":      9,
    "xtick.labelsize":     8,
    "ytick.labelsize":     8,
    "legend.fontsize":     8,
    "legend.framealpha":   1.0,
    "legend.edgecolor":    "#cccccc",
    "axes.grid":           True,
    "grid.alpha":          0.25,
    "grid.linestyle":      ":",
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "figure.dpi":          150,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "savefig.pad_inches":  0.05,
})

# ── Wong colour-blind-safe palette ────────────────────────────────────────────
METHOD_COLOURS = {
    "greedy":         "#999999",
    "bon":            "#0072B2",
    "dn_bon":         "#009E73",
    "dn_bon_updated": "#CC79A7",
    "greedy_pass_n":  "#56B4E9",
    "bon_majority":   "#D55E00",
}

METHOD_LABELS = {
    "greedy":         "Greedy",
    "bon":            "BoN",
    "dn_bon":         "DN-BoN (ours)",
    "dn_bon_updated": "DN-BoN Updated",
    "greedy_pass_n":  "Greedy Pass@N",
    "bon_majority":   "Majority",
}

METHOD_ORDER = [
    "greedy",
    "bon_majority",
    "greedy_pass_n",
    "bon",
    "dn_bon",
    "dn_bon_updated",
]

# Datasets that use a PRM (verifier) during candidate selection
_PRM_METHODS = {"bon", "dn_bon", "dn_bon_updated"}
# Datasets that use a sentence encoder (clustering)
_EMBED_METHODS = {"dn_bon", "dn_bon_updated"}

DATASET_ORDER = [
    "gsm8k", "math500", "college_math", "minerva_math",
    "amc23", "aime24", "aime25", "olympiadbench",
]

DATASET_LABELS = {
    "gsm8k":         "GSM-8K",
    "math500":       "MATH-500",
    "college_math":  "College Math",
    "minerva_math":  "Minerva Math",
    "amc23":         "AMC-23",
    "aime24":        "AIME-24",
    "aime25":        "AIME-25",
    "olympiadbench": "OlympiadBench",
}


# =============================================================================
# Tokenizer — real token counting from actual traces
# =============================================================================

class _TokenCounter:
    """
    Wraps a tokenizer so token counting always succeeds.

    Priority order when model_name is given:
      1. transformers AutoTokenizer.from_pretrained(model_name)
      2. tiktoken cl100k_base (close proxy for Llama / Qwen / Mistral)
      3. Whitespace split (last-resort fallback)
    """

    def __init__(self, model_name: Optional[str] = None):
        self._enc     = None   # tiktoken encoder
        self._hf_tok  = None   # transformers tokenizer
        self._mode    = "whitespace"

        # ── Try transformers first if model name known ─────────────────────
        if model_name:
            try:
                from transformers import AutoTokenizer  # type: ignore
                self._hf_tok = AutoTokenizer.from_pretrained(
                    model_name, use_fast=True, trust_remote_code=True
                )
                self._mode = "transformers"
                print(f"[tokenizer] Loaded HuggingFace tokenizer: {model_name}")
                return
            except Exception as e:
                warnings.warn(
                    f"[tokenizer] Could not load HF tokenizer for '{model_name}': {e}. "
                    "Falling back to tiktoken cl100k_base."
                )

        # ── tiktoken fallback ─────────────────────────────────────────────
        try:
            import tiktoken  # type: ignore
            self._enc  = tiktoken.get_encoding("cl100k_base")
            self._mode = "tiktoken"
            print("[tokenizer] Using tiktoken cl100k_base (GPT-4 / Llama proxy)")
            return
        except Exception as e:
            warnings.warn(f"[tokenizer] tiktoken unavailable: {e}. Using whitespace split.")

        self._mode = "whitespace"
        print("[tokenizer] Using whitespace split (least accurate fallback)")

    def count(self, text: str) -> int:
        """Return the number of tokens in text."""
        if not text:
            return 0
        if self._mode == "transformers":
            return len(self._hf_tok.encode(text, add_special_tokens=False))
        if self._mode == "tiktoken":
            return len(self._enc.encode(text))
        # whitespace fallback
        return len(text.split())

    @property
    def mode(self) -> str:
        return self._mode


# =============================================================================
# Theoretical FLOP computation
# =============================================================================

def _flops_for_row(
    method: str,
    n_tokens: int,          # total tokens (prompt + generated) for ONE question × N samples
    policy_params: float,   # policy model params in billions
    prm_params: float,      # PRM params in billions
    embed_params: float,    # sentence encoder params in billions
) -> float:
    """
    Theoretical FLOPs for one (question, method, N) evaluation.

    Formula: FLOPs ≈ 2 × P × T
    where P is parameter count and T is total tokens processed.

    Component rules per method:
      greedy        → policy only
      bon           → policy + PRM
      dn_bon        → policy + PRM + embedding encoder
      dn_bon_updated→ policy + PRM + embedding encoder
      greedy_pass_n → policy only  (no verifier, no embedding)
      bon_majority  → policy only  (no verifier, no embedding)
    """
    P_policy = policy_params * 1e9
    P_prm    = prm_params    * 1e9
    P_embed  = embed_params  * 1e9

    flops = 2.0 * P_policy * n_tokens
    if method in _PRM_METHODS:
        flops += 2.0 * P_prm * n_tokens
    if method in _EMBED_METHODS:
        flops += 2.0 * P_embed * n_tokens
    return flops


def _flops_breakdown(
    method: str,
    n_tokens: int,
    policy_params: float,
    prm_params: float,
    embed_params: float,
) -> Dict[str, float]:
    """Return per-component FLOPs as a dict."""
    P_policy = policy_params * 1e9
    P_prm    = prm_params    * 1e9
    P_embed  = embed_params  * 1e9
    d: Dict[str, float] = {}
    d["flops_policy"] = 2.0 * P_policy * n_tokens
    d["flops_prm"]    = 2.0 * P_prm    * n_tokens if method in _PRM_METHODS   else 0.0
    d["flops_embed"]  = 2.0 * P_embed  * n_tokens if method in _EMBED_METHODS else 0.0
    d["flops_total"]  = d["flops_policy"] + d["flops_prm"] + d["flops_embed"]
    return d


# =============================================================================
# Parameter inference from model name
# =============================================================================

def _infer_policy_params(model_name: str) -> Optional[float]:
    """
    Infer the policy model parameter count (billions) from the model name
    string encoded in the results folder name.

    The folder name format is:
        {ModelShort}_mnt{MAX_TOKENS}_t{TEMP}_p{TOP_P}[_4bit|_8bit]
    e.g.  Qwen2.5-Math-7B-Instruct_mnt2048_t0.7_p0.95
          Llama-3.2-1B-Instruct_mnt2048_t0.7_p0.95

    Parsing rules (applied in priority order)
    ------------------------------------------
    1. Strip everything from ``_mnt`` onwards (folder suffix).
    2. Find all occurrences of the pattern  <number>[B|b]  in the remaining
       string, e.g. ``7B``, ``1.5B``, ``72b``, ``0.5B``.
       Take the LAST match so that the ``2.5`` in ``Qwen2.5-Math-7B``
       is not mistaken for the model size.
    3. If no numeric B-tag is found, fall back to keyword heuristics:
         mini   → 3.8B   (Phi-3-mini family)
         medium → 14.0B  (Phi-3-medium family)
         small  → 7.0B
         large  → 70.0B
    4. Returns None if nothing matches; the caller falls back to the
       ``--policy_params`` CLI default.

    Examples
    --------
    ``Qwen2.5-Math-7B-Instruct``       → 7.0
    ``Qwen2.5-Math-1.5B-Instruct``     → 1.5
    ``Llama-3.1-70B-Instruct``         → 70.0
    ``deepseek-math-7b-instruct``      → 7.0
    ``gemma-2-27b-it``                 → 27.0
    ``phi-3-mini-4k-instruct``         → 3.8
    ``unknown-model-v2``               → None  (caller uses CLI default)
    """
    # Strip the folder-name suffix (_mnt…) if present
    name = re.split(r'_mnt', model_name)[0]

    # Find all occurrences of  <digits>[optional dot + digits><B|b>
    # followed by a non-alpha character or end-of-string
    matches = re.findall(r'(\d+(?:\.\d+)?)\s*[bB](?:[^a-zA-Z]|$)', name)
    if matches:
        return float(matches[-1])  # last match = the model-size tag

    # Keyword fallbacks for names that spell out the size
    name_lower = name.lower()
    if 'mini'   in name_lower: return 3.8
    if 'medium' in name_lower: return 14.0
    if 'small'  in name_lower: return 7.0
    if 'large'  in name_lower: return 70.0

    return None  # cannot infer; caller uses CLI default




def _cluster_entropy_from_labels(
    cluster_labels: List[int],
) -> Optional[float]:
    """
    Compute the Shannon entropy of the cluster-size distribution (bits).

    Definition (matches Eq. in the paper appendix):
        Let {C_1, …, C_M} be the M clusters discovered for a question.
        Define p_m = k_m / N,  where k_m = |C_m| and N = total traces.
        Then:
            H = -∑_{m=1}^{M}  p_m · log2(p_m)           [bits]

    H = 0 when all N traces fall into a single cluster (maximum redundancy).
    H = log2(N) when every trace is its own cluster (maximum diversity).

    Parameters
    ----------
    cluster_labels : list of int, length N
        Integer cluster assignment for each of the N generated traces.
        Sourced from the ``cluster_labels`` field in the JSONL rows.

    Returns
    -------
    H : float (bits), or None if input is empty.
    """
    if not cluster_labels:
        return None
    from collections import Counter
    N = len(cluster_labels)
    if N == 0:
        return None
    counts = Counter(cluster_labels)           # {cluster_id: k_m}
    H = 0.0
    for k_m in counts.values():
        p_m = k_m / N                          # fraction of traces in cluster m
        if p_m > 0.0:
            H -= p_m * math.log2(p_m)
    return float(H)


def _cluster_entropy_from_sizes(
    cluster_sizes: List[int],
) -> Optional[float]:
    """
    Compute Shannon entropy of the cluster-size distribution when only
    per-trace cluster-size counts are available (no explicit label integers).

    The ``cluster_sizes`` field stores, for each trace i, the integer k_{d_i}
    (size of the cluster to which trace i belongs).  From this array we
    reconstruct the unique cluster probabilities as follows:

        - Group traces that share the same cluster size.
        - The number of distinct clusters of size k is  count(k) / k,
          because each cluster of size k contributes k identical entries.
        - The probability mass assigned to each such cluster is  k / N.

    Therefore:
        H = -∑_{k}  (count(k)/k) · (k/N) · log2(k/N)
          = -∑_{k}  (count(k)/N) · log2(k/N)

    Note: when multiple clusters have the same size k, each is treated as a
    separate term; the formula above handles this correctly because count(k)/k
    gives the number of such clusters and each contributes k/N.

    This is the fallback when ``cluster_labels`` is absent from the JSONL.
    Prefer ``_cluster_entropy_from_labels`` when labels are available, as it
    is exact without the equal-size-cluster ambiguity.

    Parameters
    ----------
    cluster_sizes : list of int, length N
        For each trace i, the integer size k_{d_i} of its cluster.
        Sourced from the ``cluster_sizes`` field in the JSONL rows.

    Returns
    -------
    H : float (bits), or None if input is empty.
    """
    if not cluster_sizes:
        return None
    N = len(cluster_sizes)
    if N == 0:
        return None
    from collections import Counter
    size_counts = Counter(cluster_sizes)   # {k: how many traces have cluster size k}
    H = 0.0
    for k, cnt in size_counts.items():
        # cnt / k  = number of distinct clusters of size k
        # k / N    = probability mass of one such cluster
        p_m = k / N
        # There are cnt/k such clusters; each contributes p_m * log2(p_m)
        n_clusters_k = cnt / k
        if p_m > 0.0:
            H -= n_clusters_k * (p_m * math.log2(p_m))
    return float(H)


def _cluster_entropy_stats(
    cluster_labels: List[int],
    cluster_sizes: List[int],
) -> Dict:
    """
    Compute a full set of cluster-diversity statistics for one question.

    Returns a dict with:
      entropy          : H = -∑ p_m log2(p_m)        [bits]
      entropy_norm     : H / log2(N)                  normalised to [0, 1]
                         (= 0 for single cluster; = 1 for all-singleton)
      n_clusters       : M, number of distinct clusters
      frac_largest     : k_{max} / N  (fraction in dominant cluster)
      unique_over_N    : M / N        (fraction of samples that are unique)
      n_total          : N, total traces
    """
    if not cluster_labels and not cluster_sizes:
        return {}

    N = len(cluster_labels) if cluster_labels else len(cluster_sizes)
    if N == 0:
        return {}

    # Prefer labels (exact) over sizes (approximate)
    if cluster_labels:
        H = _cluster_entropy_from_labels(cluster_labels)
        from collections import Counter
        counts = Counter(cluster_labels)
        M       = len(counts)
        k_max   = max(counts.values())
    else:
        H = _cluster_entropy_from_sizes(cluster_sizes)
        from collections import Counter
        size_counts = Counter(cluster_sizes)
        # Reconstruct M = sum(cnt/k for k, cnt in size_counts.items())
        M     = int(round(sum(cnt / k for k, cnt in size_counts.items())))
        k_max = max(cluster_sizes)

    H_ideal = math.log2(N) if N > 1 else 1.0   # entropy if all singleton clusters
    H_norm  = (H / H_ideal) if H_ideal > 0 else 0.0

    return {
        "entropy":       H,
        "entropy_norm":  H_norm,
        "n_clusters":    M,
        "frac_largest":  k_max / N,
        "unique_over_N": M / N,
        "n_total":       N,
    }


# =============================================================================
# Parsing helpers
# =============================================================================

# Folder name: {ModelShort}_mnt{tokens}_t{temp}_p{top_p}[_4bit|_8bit]
_FOLDER_RE = re.compile(
    r"^(?P<model>.+?)_mnt(?P<max_tokens>\d+)_t(?P<temp>[\d.]+)_p(?P<top_p>[\d.]+)"
    r"(?P<quant>_4bit|_8bit)?$"
)

_KNOWN_METHODS = list(METHOD_COLOURS.keys())
_KNOWN_METHODS_SORTED = sorted(_KNOWN_METHODS, key=len, reverse=True)

_SUMMARY_RE = re.compile(
    r"^(?P<method>.+?)_(?P<dataset>[^_]+(?:_[^_]+)?)_N(?P<N>\d+)_summary\.json$"
)


def _parse_folder_name(name: str):
    """Return (model, max_tokens, temp, top_p, quant) or None."""
    m = _FOLDER_RE.match(name)
    if not m:
        return None
    return (
        m.group("model"),
        int(m.group("max_tokens")),
        float(m.group("temp")),
        float(m.group("top_p")),
        (m.group("quant") or "").lstrip("_"),
    )


def _parse_summary_filename(fname: str):
    """Return (method, dataset, N) from a summary filename, or None."""
    if not fname.endswith("_summary.json"):
        return None
    stem = fname[: -len("_summary.json")]

    for method in _KNOWN_METHODS_SORTED:
        prefix = method + "_"
        if stem.startswith(prefix):
            remainder = stem[len(prefix):]
            dm = re.match(r"^(.+)_N(\d+)$", remainder)
            if dm:
                return method, dm.group(1), int(dm.group(2))

    m = _SUMMARY_RE.match(fname)
    if m:
        return m.group("method"), m.group("dataset"), int(m.group("N"))
    return None


def _load_summary(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  [warn] Could not read {path}: {e}", file=sys.stderr)
        return None


def _load_jsonl(path: str) -> List[dict]:
    """Load a JSONL file, silently skip malformed lines."""
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    except Exception:
        pass
    return rows


# =============================================================================
# Per-JSONL statistics: token counts + entropy
# =============================================================================

def _jsonl_stats(
    jsonl_path: str,
    token_counter: _TokenCounter,
    n_samples: int,          # N (sample budget) for this run
) -> Dict:
    """
    Read one JSONL file and compute:
      - avg_tokens_per_question : mean number of tokens in selected trace
      - avg_total_tokens        : avg tokens × N  (all N samples approximated
                                  from trace_lengths when full traces not stored)
      - avg_cluster_entropy     : mean Shannon entropy over cluster distribution
      - n_questions             : number of questions processed
    """
    rows = _load_jsonl(jsonl_path)
    if not rows:
        return {}

    all_tokens_per_q  = []   # total token count for all N traces of one question
    all_per_q_stats   = []   # list of per-question entropy stat dicts

    for row in rows:
        # ── Token counting ─────────────────────────────────────────────────
        selected_trace = row.get("raw_llm_output", "")
        trace_lengths  = row.get("trace_lengths", [])  # list of char lengths

        if selected_trace:
            # Count tokens in the actual selected trace
            tokens_selected = token_counter.count(str(selected_trace))
            # Estimate total tokens for all N traces:
            #   If we have trace_lengths (char counts for every trace), use them
            #   with the same ratio tokens_selected / len(selected_trace)
            if trace_lengths and len(selected_trace) > 0:
                char_per_token = len(str(selected_trace)) / max(tokens_selected, 1)
                tokens_all_traces = sum(
                    int(cl / char_per_token) for cl in trace_lengths
                )
            else:
                # Only selected trace available → scale by N
                tokens_all_traces = tokens_selected * n_samples
        elif trace_lengths:
            # No stored trace text — use character-to-token proxy (4 chars/token)
            tokens_selected   = int(trace_lengths[0] / 4.0) if trace_lengths else 0
            tokens_all_traces = int(sum(trace_lengths) / 4.0)
        else:
            tokens_selected   = 0
            tokens_all_traces = 0

        all_tokens_per_q.append(tokens_all_traces)

        # ── Cluster entropy and diversity statistics ───────────────────────
        # Uses _cluster_entropy_stats which returns: entropy, entropy_norm,
        # n_clusters, frac_largest, unique_over_N, n_total.
        # Only DN-BoN / DN-BoN-Updated rows have cluster data; all other
        # methods will have empty lists and stats will be NaN in the CSV.
        cluster_labels = row.get("cluster_labels", [])
        cluster_sizes  = row.get("cluster_sizes",  [])

        stats_q = _cluster_entropy_stats(cluster_labels, cluster_sizes)
        if stats_q:
            all_per_q_stats.append(stats_q)

    result: Dict = {"n_questions": len(rows)}

    if all_tokens_per_q:
        result["avg_tokens_per_question"] = float(np.mean(all_tokens_per_q))
    else:
        result["avg_tokens_per_question"] = 0.0

    # Aggregate per-question entropy stats → means and stds
    if all_per_q_stats:
        for key in ("entropy", "entropy_norm", "n_clusters",
                    "frac_largest", "unique_over_N"):
            vals = [s[key] for s in all_per_q_stats if key in s]
            if vals:
                result[f"avg_{key}"]    = float(np.mean(vals))
                result[f"std_{key}"]    = float(np.std(vals))
            else:
                result[f"avg_{key}"]    = float("nan")
                result[f"std_{key}"]    = float("nan")
    else:
        for key in ("entropy", "entropy_norm", "n_clusters",
                    "frac_largest", "unique_over_N"):
            result[f"avg_{key}"] = float("nan")
            result[f"std_{key}"] = float("nan")

    return result


# =============================================================================
# Main scan
# =============================================================================

def scan_results(
    results_root: str,
    token_counter: _TokenCounter,
    policy_params_override: Optional[float],   # None = auto-infer from folder name
    prm_params: float,
    embed_params: float,
) -> pd.DataFrame:
    """
    Walk results_root recursively.  For each summary JSON found, also look
    for the matching JSONL to compute token counts, FLOPs, and entropy.

    Policy model parameter count
    ----------------------------
    When ``policy_params_override`` is None (the default), the parameter
    count is inferred automatically from the model name encoded in each
    folder name via ``_infer_policy_params()``.  For example:
        Qwen2.5-Math-7B-Instruct_mnt2048_t0.7_p0.95  →  7.0 B
        Llama-3.1-70B-Instruct_mnt2048_t0.7_p0.95    →  70.0 B
    When the user passes ``--policy_params <value>``, that value is used
    for every folder regardless of the name.

    Returns a DataFrame with columns:
      folder, model, max_tokens, temp, top_p, quant,
      method, dataset, N,
      accuracy, n_correct, n_total, avg_elapsed_sec, total_wall_sec,
      policy_params_B,               ← actual value used per folder
      avg_tokens_per_question,
      flops_policy, flops_prm, flops_embed, flops_total,
      avg_entropy, std_entropy,
      avg_entropy_norm, std_entropy_norm,
      avg_n_clusters, std_n_clusters,
      avg_frac_largest, std_frac_largest,
      avg_unique_over_N, std_unique_over_N,
      tokenizer_mode
    """
    root = Path(results_root)
    if not root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    records = []

    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue

        parsed = _parse_folder_name(folder.name)
        if parsed is None:
            model      = folder.name
            max_tokens = None
            temp       = None
            top_p      = None
            quant      = ""
        else:
            model, max_tokens, temp, top_p, quant = parsed

        # ── Resolve policy parameter count for this folder ─────────────────
        # Priority: user CLI override → auto-inferred from model name → 7.0B default
        if policy_params_override is not None:
            # User explicitly set --policy_params → honour it for all folders
            policy_params = policy_params_override
            policy_params_source = "cli_override"
        else:
            inferred = _infer_policy_params(model)
            if inferred is not None:
                policy_params = inferred
                policy_params_source = "inferred"
            else:
                policy_params = 7.0   # safe fallback when name has no size tag
                policy_params_source = "default_fallback"
                print(
                    f"  [warn] Could not infer policy params from model name "
                    f"'{model}'. Using fallback {policy_params}B.",
                    file=sys.stderr,
                )

        summary_files = sorted(folder.glob("*_summary.json"))
        if not summary_files:
            summary_files = sorted(folder.rglob("*_summary.json"))

        for sf in summary_files:
            parsed_fn = _parse_summary_filename(sf.name)
            if parsed_fn is None:
                continue
            method, dataset, N = parsed_fn

            data = _load_summary(str(sf))
            if data is None:
                continue

            # ── Core metrics from summary JSON ─────────────────────────────
            accuracy        = float(data.get("accuracy",        0.0))
            n_correct       = int(  data.get("n_correct",       0  ))
            n_total         = int(  data.get("n_total",         0  ))
            avg_elapsed_sec = float(data.get("avg_elapsed_sec", 0.0))
            total_wall_sec  = float(data.get("total_wall_sec",  0.0))

            # ── JSONL-derived metrics ──────────────────────────────────────
            # The JSONL has the same base name as the summary JSON minus _summary.json
            jsonl_name = sf.name.replace("_summary.json", ".jsonl")
            jsonl_path = sf.parent / jsonl_name

            jstats = _jsonl_stats(str(jsonl_path), token_counter, N)

            avg_tokens = jstats.get("avg_tokens_per_question", 0.0)

            # ── Theoretical FLOPs ─────────────────────────────────────────
            # avg_tokens is the mean TOTAL tokens for all N traces of one question.
            # FLOPs are per question (all N traces combined).
            fb = _flops_breakdown(method, avg_tokens, policy_params, prm_params, embed_params)

            rec = {
                "folder":          folder.name,
                "model":           model,
                "max_tokens":      max_tokens,
                "temp":            temp,
                "top_p":           top_p,
                "quant":           quant,
                "method":          method,
                "dataset":         dataset,
                "N":               N,
                # accuracy
                "accuracy":        accuracy,
                "n_correct":       n_correct,
                "n_total":         n_total,
                # latency (from summary JSON avg_elapsed_sec per question)
                "avg_elapsed_sec": avg_elapsed_sec,
                "total_wall_sec":  total_wall_sec,
                # model size used for FLOPs (inferred or overridden)
                "policy_params_B": policy_params,
                # tokens
                "avg_tokens_per_question": avg_tokens,
                # FLOPs
                "flops_policy":    fb["flops_policy"],
                "flops_prm":       fb["flops_prm"],
                "flops_embed":     fb["flops_embed"],
                "flops_total":     fb["flops_total"],
                # cluster entropy  (H = -∑ p_m log2(p_m), bits)
                "avg_entropy":          jstats.get("avg_entropy",      float("nan")),
                "std_entropy":          jstats.get("std_entropy",      float("nan")),
                # normalised entropy  (H / log2(N), in [0,1])
                "avg_entropy_norm":     jstats.get("avg_entropy_norm", float("nan")),
                "std_entropy_norm":     jstats.get("std_entropy_norm", float("nan")),
                # mean number of clusters M per question
                "avg_n_clusters":       jstats.get("avg_n_clusters",   float("nan")),
                "std_n_clusters":       jstats.get("std_n_clusters",   float("nan")),
                # fraction of traces in dominant cluster  (f_max = k_max / N)
                "avg_frac_largest":     jstats.get("avg_frac_largest", float("nan")),
                "std_frac_largest":     jstats.get("std_frac_largest", float("nan")),
                # unique reasoning directions per sample  (M / N)
                "avg_unique_over_N":    jstats.get("avg_unique_over_N",float("nan")),
                "std_unique_over_N":    jstats.get("std_unique_over_N",float("nan")),
                # tokenizer mode used for token counting
                "tokenizer_mode":       token_counter.mode,
            }
            records.append(rec)

    if not records:
        raise ValueError(
            f"No summary JSON files found under {results_root}.\n"
            f"Expected filenames like: bon_aime24_N8_summary.json"
        )

    df = pd.DataFrame(records)
    df["N"] = df["N"].astype(int)
    return df


# =============================================================================
# CSV export
# =============================================================================

def save_csv(df: pd.DataFrame, path: str):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"CSV saved → {out}  ({len(df)} rows)")


def save_supplementary_csvs(df: pd.DataFrame, base_path: str):
    """
    Save focused supplementary CSVs — one file per model per metric.

    Files produced
    --------------
    For each unique model found in df:

      {stem}_{model_tag}_flops.csv
          All methods × datasets × N.
          Columns: folder, model, max_tokens, method, dataset, N,
                   policy_params_B, avg_tokens_per_question,
                   flops_policy, flops_prm, flops_embed, flops_total,
                   tokenizer_mode.
          Every method present; greedy/bon_majority/greedy_pass_n have
          flops_prm=0 and flops_embed=0 (correctly zero, not NaN).

      {stem}_{model_tag}_latency.csv
          All methods × datasets × N.
          Columns: folder, model, max_tokens, method, dataset, N,
                   avg_elapsed_sec, total_wall_sec.

      {stem}_{model_tag}_entropy.csv
          All methods × datasets × N.
          Columns: folder, model, max_tokens, method, dataset, N,
                   avg_entropy, std_entropy, avg_entropy_norm, std_entropy_norm,
                   avg_n_clusters, std_n_clusters,
                   avg_frac_largest, std_frac_largest,
                   avg_unique_over_N, std_unique_over_N.
          Non-clustering methods (greedy, bon, bon_majority, greedy_pass_n)
          have NaN in all entropy columns — kept deliberately so the file
          contains a complete row for every (model, method, dataset, N)
          combination and can be merged with other CSVs without missing rows.

    A combined all-models file is also saved for each metric when more
    than one model is present.
    """
    base    = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    stem    = base.stem
    suf     = base.suffix
    out_dir = base.parent

    # ── Column definitions ─────────────────────────────────────────────────────
    id_cols   = ["folder", "model", "max_tokens", "method", "dataset", "N"]

    flop_cols = id_cols + [
        "policy_params_B",
        "avg_tokens_per_question",
        "flops_policy", "flops_prm", "flops_embed", "flops_total",
        "tokenizer_mode",
    ]

    lat_cols  = id_cols + [
        "avg_elapsed_sec", "total_wall_sec",
    ]

    ent_cols  = id_cols + [
        "avg_entropy",       "std_entropy",
        "avg_entropy_norm",  "std_entropy_norm",
        "avg_n_clusters",    "std_n_clusters",
        "avg_frac_largest",  "std_frac_largest",
        "avg_unique_over_N", "std_unique_over_N",
    ]

    # Helper: keep only existing columns and sort rows for readability
    def _extract(frame: pd.DataFrame, cols: list) -> pd.DataFrame:
        present = [c for c in cols if c in frame.columns]
        return (
            frame[present]
            .copy()
            .sort_values(
                [c for c in ["model", "dataset", "method", "N"] if c in present]
            )
            .reset_index(drop=True)
        )

    # ── Per-model files ────────────────────────────────────────────────────────
    models = sorted(df["model"].dropna().unique())

    for mdl in models:
        mdf = df[df["model"] == mdl].copy()

        # Safe short name for filenames
        mdl_tag = re.sub(r"[^\w\-.]", "_", mdl).strip("_")[:40]

        # FLOPs  — all methods; prm/embed columns are 0 (not NaN) for methods
        #          that don't use them, so no filtering needed
        flop_df = _extract(mdf, flop_cols)
        flop_path = out_dir / f"{stem}_{mdl_tag}_flops{suf}"
        flop_df.to_csv(flop_path, index=False)
        print(f"FLOPs CSV saved   → {flop_path}  "
              f"({len(flop_df)} rows, {flop_df['method'].nunique()} methods)")

        # Latency — all methods
        lat_df = _extract(mdf, lat_cols)
        lat_path = out_dir / f"{stem}_{mdl_tag}_latency{suf}"
        lat_df.to_csv(lat_path, index=False)
        print(f"Latency CSV saved → {lat_path}  "
              f"({len(lat_df)} rows, {lat_df['method'].nunique()} methods)")

        # Entropy — ALL methods retained; NaN values kept for non-clustering
        #           methods so the file has a complete row per (method, dataset, N)
        ent_df = _extract(mdf, ent_cols)
        ent_path = out_dir / f"{stem}_{mdl_tag}_entropy{suf}"
        ent_df.to_csv(ent_path, index=False)
        n_with_data = ent_df["avg_entropy"].notna().sum()
        print(f"Entropy CSV saved → {ent_path}  "
              f"({len(ent_df)} rows total; "
              f"{n_with_data} rows with cluster data, "
              f"{len(ent_df)-n_with_data} rows NaN for non-clustering methods)")

    # ── Combined all-models files (when multiple models present) ───────────────
    if len(models) > 1:
        print(f"\nSaving combined all-models CSVs...")

        combined_flop = _extract(df, flop_cols)
        p = out_dir / f"{stem}_all_models_flops{suf}"
        combined_flop.to_csv(p, index=False)
        print(f"  All-models FLOPs  → {p}")

        combined_lat = _extract(df, lat_cols)
        p = out_dir / f"{stem}_all_models_latency{suf}"
        combined_lat.to_csv(p, index=False)
        print(f"  All-models Latency → {p}")

        combined_ent = _extract(df, ent_cols)
        p = out_dir / f"{stem}_all_models_entropy{suf}"
        combined_ent.to_csv(p, index=False)
        print(f"  All-models Entropy → {p}")


# =============================================================================
# Plot helpers
# =============================================================================

def _dataset_order_key(ds: str) -> int:
    try:
        return DATASET_ORDER.index(ds)
    except ValueError:
        return len(DATASET_ORDER)


def _method_order_key(m: str) -> int:
    try:
        return METHOD_ORDER.index(m)
    except ValueError:
        return len(METHOD_ORDER)


def _bar_figure(
    sub: pd.DataFrame,
    metric: str,
    ylabel: str,
    title_prefix: str,
    tag: str,
    model_tag: str,
    out_path: Path,
    yformatter=None,
    ylim_top: float = None,
):
    """
    Generic grouped-bar figure: one subplot per dataset, x-axis = N,
    bars grouped by method.  Used for accuracy, latency, FLOPs, entropy.
    """
    datasets = sorted(sub["dataset"].unique(), key=_dataset_order_key)
    N_vals   = sorted(sub["N"].unique())
    methods  = sorted(sub["method"].unique(), key=_method_order_key)
    methods  = [m for m in methods if m in sub["method"].values]

    n_ds = len(datasets)
    n_m  = len(methods)
    n_N  = len(N_vals)
    if n_ds == 0 or n_m == 0 or n_N == 0:
        return

    ncols = min(4, n_ds)
    nrows = int(np.ceil(n_ds / ncols))
    fig_w = 3.8 * ncols
    fig_h = 3.2 * nrows + 0.8

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    title_model = f"  |  {model_tag}" if model_tag else ""
    fig.suptitle(
        f"{title_prefix}  |  max_tokens={tag.replace('mnt','')}{title_model}",
        fontsize=10, fontweight="bold", y=1.005,
    )

    bw     = 0.75 / n_m
    x_base = np.arange(n_N, dtype=float)

    for idx, ds in enumerate(datasets):
        row_i = idx // ncols
        col_i = idx % ncols
        ax    = axes[row_i][col_i]
        ds_label = DATASET_LABELS.get(ds, ds)

        for m_idx, method in enumerate(methods):
            vals = []
            for N in N_vals:
                cell = sub[(sub["dataset"] == ds) & (sub["N"] == N) & (sub["method"] == method)]
                if cell.empty or pd.isna(cell[metric].iloc[0]):
                    vals.append(np.nan)
                else:
                    vals.append(float(cell[metric].iloc[0]))

            x_pos = x_base - (n_m - 1) * bw / 2 + m_idx * bw
            col   = METHOD_COLOURS.get(method, "#888888")
            ax.bar(
                x_pos,
                [v if not np.isnan(v) else 0 for v in vals],
                bw * 0.88,
                color=col, edgecolor="white", linewidth=0.5,
                label=METHOD_LABELS.get(method, method),
                zorder=3, alpha=0.92,
            )

        ax.set_title(ds_label, fontsize=9, fontweight="bold", pad=3)
        ax.set_xticks(x_base)
        ax.set_xticklabels([f"N={N}" for N in N_vals], fontsize=7.5)
        ax.set_ylabel(ylabel, fontsize=8)
        if yformatter:
            ax.yaxis.set_major_formatter(yformatter)
        cur_top = ax.get_ylim()[1]
        ax.set_ylim(0, min(ylim_top, cur_top * 1.12) if ylim_top else cur_top * 1.12)
        ax.tick_params(axis="both", labelsize=7.5)

    for idx in range(n_ds, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    handles, labels_list = [], []
    seen: dict = {}
    for ax_row in axes:
        for ax in ax_row:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in seen:
                    seen[l] = True
                    handles.append(h); labels_list.append(l)

    fig.legend(handles, labels_list, loc="lower center",
               bbox_to_anchor=(0.5, -0.03), ncol=min(n_m, 5),
               frameon=True, fontsize=8.5)
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    pdf_path = out_path / f"{metric}_{tag}.pdf"
    png_path = out_path / f"{metric}_{tag}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    print(f"Plot saved → {pdf_path}")


# =============================================================================
# Accuracy bar plots (grid: datasets × N)
# =============================================================================

def plot_accuracy_bars(
    df: pd.DataFrame,
    out_dir: str,
    max_tokens_filter: int = None,
    model_tag: str = "",
):
    """Original accuracy bar charts (one row per dataset, one column per N)."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    token_groups = (
        [max_tokens_filter]
        if max_tokens_filter is not None
        else sorted(df["max_tokens"].dropna().unique().astype(int)) or [None]
    )

    for mnt in token_groups:
        sub = df[df["max_tokens"] == mnt].copy() if mnt is not None else df.copy()
        tag = f"mnt{mnt}" if mnt is not None else "all"
        if sub.empty:
            continue

        datasets = sorted(sub["dataset"].unique(), key=_dataset_order_key)
        N_vals   = sorted(sub["N"].unique())
        methods  = sorted(sub["method"].unique(), key=_method_order_key)
        methods  = [m for m in methods if m in sub["method"].values]

        n_ds, n_N, n_m = len(datasets), len(N_vals), len(methods)
        if not (n_ds and n_N and n_m):
            continue

        fig_w = max(4.5, 2.2 * n_N)
        fig_h = max(3.5, 2.5 * n_ds)
        fig, axes = plt.subplots(n_ds, n_N, figsize=(fig_w, fig_h),
                                  squeeze=False, sharey="row")
        title_model = f"  |  {model_tag}" if model_tag else ""
        fig.suptitle(
            f"Accuracy by dataset and budget  |  max_tokens={mnt}{title_model}",
            fontsize=10, fontweight="bold", y=1.01,
        )

        bw     = 0.70 / n_m
        x_base = np.arange(1)

        for r_idx, ds in enumerate(datasets):
            ds_label = DATASET_LABELS.get(ds, ds)
            for c_idx, N in enumerate(N_vals):
                ax   = axes[r_idx][c_idx]
                cell = sub[(sub["dataset"] == ds) & (sub["N"] == N)]
                any_bar = False
                for m_idx, method in enumerate(methods):
                    mrow = cell[cell["method"] == method]
                    if mrow.empty:
                        continue
                    acc   = float(mrow["accuracy"].iloc[0])
                    col   = METHOD_COLOURS.get(method, "#888888")
                    x_pos = x_base - (n_m - 1) * bw / 2 + m_idx * bw
                    ax.bar(x_pos, acc, bw * 0.90, color=col, edgecolor="white",
                           linewidth=0.4, label=METHOD_LABELS.get(method, method), zorder=3)
                    any_bar = True
                ax.set_xlim(-0.5, 0.5)
                ax.set_xticks([])
                ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
                ax.set_ylim(0, min(1.05, ax.get_ylim()[1] + 0.08))
                if r_idx == 0:
                    ax.set_title(f"$N={N}$", fontsize=8.5, pad=4)
                if c_idx == 0:
                    ax.set_ylabel(ds_label, fontsize=8.5, labelpad=4)
                if not any_bar:
                    ax.text(0, 0.5, "—", ha="center", va="center",
                            transform=ax.transAxes, color="#aaaaaa")

        handles, lbls = axes[0][0].get_legend_handles_labels()
        seen: dict = {}
        h_d, l_d = [], []
        for h, l in zip(handles, lbls):
            if l not in seen:
                seen[l] = True
                h_d.append(h); l_d.append(l)
        fig.legend(h_d, l_d, loc="lower center", bbox_to_anchor=(0.5, -0.04),
                   ncol=min(n_m, 5), frameon=True, fontsize=8.5)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(out_path / f"accuracy_bars_{tag}.{ext}")
        plt.close(fig)
        print(f"Plot saved → {out_path / f'accuracy_bars_{tag}.pdf'}")

    # Also produce grouped-by-N layout
    _plot_metric_vs_N(df, "accuracy", "Accuracy", out_dir, max_tokens_filter,
                      model_tag, yformatter=mticker.PercentFormatter(xmax=1, decimals=0),
                      ylim_top=1.08)


# =============================================================================
# Generic metric vs N grouped plots
# =============================================================================

def _plot_metric_vs_N(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    out_dir: str,
    max_tokens_filter: int = None,
    model_tag: str = "",
    yformatter=None,
    ylim_top: float = None,
):
    """
    For each max_tokens group, produce one figure per metric showing
    each dataset as a subplot, x-axis = N, grouped bars by method.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    token_groups = (
        [max_tokens_filter]
        if max_tokens_filter is not None
        else sorted(df["max_tokens"].dropna().unique().astype(int)) or [None]
    )

    for mnt in token_groups:
        sub = df[df["max_tokens"] == mnt].copy() if mnt is not None else df.copy()
        tag = f"mnt{mnt}" if mnt is not None else "all"
        if sub.empty:
            continue
        # Skip metric if all NaN
        if sub[metric].isna().all() or (sub[metric] == 0).all():
            continue
        _bar_figure(sub, metric, ylabel,
                    ylabel + " vs Sample Budget $N$",
                    tag, model_tag, out_path,
                    yformatter=yformatter, ylim_top=ylim_top)


# =============================================================================
# Latency plot
# =============================================================================

def plot_latency(df: pd.DataFrame, out_dir: str,
                 max_tokens_filter: int = None, model_tag: str = ""):
    """Bar plots of avg_elapsed_sec (latency per question) vs N."""
    _plot_metric_vs_N(df, "avg_elapsed_sec", "Avg Latency (s/question)",
                      out_dir, max_tokens_filter, model_tag)


# =============================================================================
# FLOPs plots
# =============================================================================

class _SciFormatter(mticker.ScalarFormatter):
    """Format y-axis as X.Xe+YY for FLOPs."""
    def __init__(self):
        super().__init__(useMathText=True)
        self.set_scientific(True)
        self.set_powerlimits((-2, 2))


def plot_flops(df: pd.DataFrame, out_dir: str,
               max_tokens_filter: int = None, model_tag: str = ""):
    """Bar plots of total theoretical FLOPs vs N."""
    _plot_metric_vs_N(df, "flops_total", "Total FLOPs (theoretical)",
                      out_dir, max_tokens_filter, model_tag,
                      yformatter=mticker.FuncFormatter(
                          lambda x, _: f"{x:.1e}" if x > 0 else "0"))

    # Also plot per-component FLOPs stacked
    _plot_flops_stacked(df, out_dir, max_tokens_filter, model_tag)


def _plot_flops_stacked(
    df: pd.DataFrame,
    out_dir: str,
    max_tokens_filter: int = None,
    model_tag: str = "",
):
    """
    Stacked bar chart of FLOPs: policy / PRM / embedding, per method × N.
    One figure per dataset per max_tokens.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    token_groups = (
        [max_tokens_filter]
        if max_tokens_filter is not None
        else sorted(df["max_tokens"].dropna().unique().astype(int)) or [None]
    )

    component_cols    = ["flops_policy", "flops_prm", "flops_embed"]
    component_labels  = ["Policy", "PRM", "Embedding"]
    component_colours = ["#0072B2", "#D55E00", "#009E73"]

    for mnt in token_groups:
        sub = df[df["max_tokens"] == mnt].copy() if mnt is not None else df.copy()
        tag = f"mnt{mnt}" if mnt is not None else "all"
        if sub.empty:
            continue

        datasets = sorted(sub["dataset"].unique(), key=_dataset_order_key)
        N_vals   = sorted(sub["N"].unique())
        methods  = sorted(sub["method"].unique(), key=_method_order_key)
        methods  = [m for m in methods if m in sub["method"].values]

        n_ds, n_N, n_m = len(datasets), len(N_vals), len(methods)
        if not (n_ds and n_N and n_m):
            continue

        ncols = min(4, n_ds)
        nrows = int(np.ceil(n_ds / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(3.8 * ncols, 3.2 * nrows + 0.8),
                                  squeeze=False)
        title_model = f"  |  {model_tag}" if model_tag else ""
        fig.suptitle(
            f"FLOPs Breakdown vs N  |  max_tokens={tag.replace('mnt','')}{title_model}",
            fontsize=10, fontweight="bold", y=1.005,
        )

        bw     = 0.75 / n_m
        x_base = np.arange(n_N, dtype=float)

        for idx, ds in enumerate(datasets):
            row_i = idx // ncols
            col_i = idx % ncols
            ax    = axes[row_i][col_i]
            ds_label = DATASET_LABELS.get(ds, ds)

            for m_idx, method in enumerate(methods):
                x_pos = x_base - (n_m - 1) * bw / 2 + m_idx * bw
                bottoms = np.zeros(n_N)
                for comp_col, comp_label, comp_col_hex in zip(
                        component_cols, component_labels, component_colours):
                    vals = []
                    for N in N_vals:
                        cell = sub[(sub["dataset"] == ds) & (sub["N"] == N) &
                                   (sub["method"] == method)]
                        vals.append(float(cell[comp_col].iloc[0]) if not cell.empty else 0.0)
                    vals = np.array(vals)
                    ax.bar(x_pos, vals, bw * 0.88, bottom=bottoms,
                           color=comp_col_hex, edgecolor="white", linewidth=0.3,
                           label=comp_label if m_idx == 0 else "_nolegend_",
                           zorder=3, alpha=0.88)
                    bottoms += vals

                # Method label annotation on top bar at first N
                max_h = bottoms[0]
                ax.text(x_pos[0], max_h * 1.02,
                        METHOD_LABELS.get(method, method)[:4],
                        ha="center", va="bottom", fontsize=5.5, rotation=0)

            ax.set_title(ds_label, fontsize=9, fontweight="bold", pad=3)
            ax.set_xticks(x_base)
            ax.set_xticklabels([f"N={N}" for N in N_vals], fontsize=7.5)
            ax.set_ylabel("FLOPs", fontsize=8)
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"{x:.1e}" if x > 0 else "0"))
            ax.tick_params(axis="both", labelsize=7.5)

        for idx in range(n_ds, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        # Component legend (not method legend — use separate note)
        from matplotlib.patches import Patch
        comp_handles = [Patch(facecolor=c, label=l)
                        for c, l in zip(component_colours, component_labels)]
        fig.legend(comp_handles, component_labels, loc="lower center",
                   bbox_to_anchor=(0.5, -0.03), ncol=3,
                   frameon=True, fontsize=8.5,
                   title="FLOPs component (stacked across methods of same N)")
        fig.tight_layout(rect=[0, 0.04, 1, 1])

        for ext in ("pdf", "png"):
            fig.savefig(out_path / f"flops_stacked_{tag}.{ext}")
        plt.close(fig)
        print(f"Plot saved → {out_path / f'flops_stacked_{tag}.pdf'}")


def save_per_folder_accuracy_csvs(df: pd.DataFrame, results_root: str):
    """
    Save one accuracy CSV inside each scanned model/result folder.
    Example output:
      results_updated/Qwen2.5-Math-7B-Instruct_mnt32768_t0.7_p0.95/accuracy_summary.csv
    """
    root = Path(results_root)

    for folder_name in sorted(df["folder"].dropna().unique()):
        sub = df[df["folder"] == folder_name].copy()
        if sub.empty:
            continue

        out_cols = [
            "model", "folder", "max_tokens", "temp", "top_p", "quant",
            "method", "dataset", "N",
            "accuracy", "n_correct", "n_total",
            "avg_elapsed_sec", "total_wall_sec"
        ]
        out_cols = [c for c in out_cols if c in sub.columns]

        out_path = root / folder_name / "accuracy_summary.csv"
        sub[out_cols].sort_values(["dataset", "method", "N"]).to_csv(out_path, index=False)
        print(f"Per-folder accuracy CSV saved → {out_path}")

# =============================================================================
# Entropy plot
# =============================================================================

def plot_entropy(df: pd.DataFrame, out_dir: str,
                 max_tokens_filter: int = None, model_tag: str = ""):
    """
    Bar plots of cluster diversity metrics vs N (DN-BoN / DN-BoN Updated only).

    Produces four figures:
      avg_entropy         — raw Shannon entropy H (bits)
      avg_entropy_norm    — H / log2(N) ∈ [0,1];  0=all in one cluster, 1=all unique
      avg_frac_largest    — dominant cluster fraction f_max = k_max / N
      avg_unique_over_N   — unique directions per sample M / N
    """
    ent_df = df[df["avg_entropy"].notna()].copy()
    if ent_df.empty:
        print("  [info] No entropy data to plot (only DN-BoN rows carry cluster info).")
        return

    _plot_metric_vs_N(ent_df, "avg_entropy",
                      "Avg Cluster Entropy H (bits)",
                      out_dir, max_tokens_filter, model_tag)

    _plot_metric_vs_N(ent_df, "avg_entropy_norm",
                      "Normalised Entropy H/log₂(N)  [0=redundant, 1=diverse]",
                      out_dir, max_tokens_filter, model_tag,
                      ylim_top=1.05)

    _plot_metric_vs_N(ent_df, "avg_frac_largest",
                      "Dominant Cluster Fraction f_max = k_max / N",
                      out_dir, max_tokens_filter, model_tag,
                      ylim_top=1.05)

    _plot_metric_vs_N(ent_df, "avg_unique_over_N",
                      "Unique Directions per Sample M / N",
                      out_dir, max_tokens_filter, model_tag,
                      ylim_top=1.05)


# =============================================================================
# Model filter helper
# =============================================================================

def _filter_by_model(df: pd.DataFrame, model_query: str) -> pd.DataFrame:
    available = sorted(df["model"].dropna().unique())
    if not model_query:
        return df

    mask     = df["model"].str.contains(model_query, case=False, na=False)
    filtered = df[mask].copy()

    if filtered.empty:
        print(
            f"\nERROR: No results found for model matching '{model_query}'.\n"
            f"Available models:\n" + "\n".join(f"  • {m}" for m in available),
            file=sys.stderr,
        )
        sys.exit(1)

    matched = sorted(filtered["model"].unique())
    if len(matched) > 1:
        print(f"[info] Model filter '{model_query}' matched {len(matched)} model(s):")
        for m in matched:
            print(f"       • {m}")
    else:
        print(f"[info] Model filter: {matched[0]}")

    return filtered


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Parse DN-BoN results folders → master CSV + accuracy/latency/FLOPs/"
            "entropy bar plots.\n\n"
            "FLOPs are theoretical (2 × P × T) using actual token counts from JSONL\n"
            "files measured with a real tokenizer (HuggingFace or tiktoken).\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results_dir", default="./results_updated",
                        help="Root results folder")
    parser.add_argument("--out_csv", default="results_master.csv",
                        help="Output master CSV path")
    parser.add_argument("--plot_dir", default="./plots",
                        help="Output directory for plots")
    parser.add_argument("--model", default=None, metavar="MODEL",
                        help="Filter by model name (case-insensitive substring)")
    parser.add_argument("--list_models", action="store_true",
                        help="Print all model names found in results_dir and exit")
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Only plot/compute for this max_tokens value")
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip plotting (CSV only)")

    # ── Model-size parameters for FLOPs ──────────────────────────────────────
    g = parser.add_argument_group(
        "Model sizes for theoretical FLOPs",
        "All values in billions (e.g. 7 for a 7B model)."
    )
    g.add_argument(
        "--policy_params", type=float, default=None,
        metavar="BILLIONS",
        help=(
            "Policy (generator) model parameter count in billions.\n"
            "DEFAULT: auto-inferred from the model name in each folder name.\n"
            "  Qwen2.5-Math-7B-Instruct  → 7.0\n"
            "  Llama-3.1-70B-Instruct    → 70.0\n"
            "  Qwen2.5-Math-1.5B         → 1.5\n"
            "Pass an explicit value to override auto-inference for all folders,\n"
            "e.g.  --policy_params 13.0  for a 13B model."
        ),
    )
    g.add_argument("--prm_params", type=float, default=7.0,
                   help="PRM (verifier) model parameter count in billions  [default: 7.0]")
    g.add_argument(
        "--embed_params", type=float, default=0.022,
        metavar="BILLIONS",
        help=(
            "Sentence encoder parameter count in billions.\n"
            "  0.022 → all-MiniLM-L6-v2  (default, used in DN-BoN)\n"
            "  0.568 → BGE-M3\n"
            "The embedding model name is not stored in the folder name,\n"
            "so this must be set manually if you used a model other than MiniLM."
        ),
    )

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--tokenizer", default=None, metavar="MODEL_NAME_OR_PATH",
        help=(
            "HuggingFace tokenizer to use for token counting.  Pass the model id "
            "(e.g. 'Qwen/Qwen2.5-Math-7B-Instruct') or a local path.  "
            "Falls back to tiktoken cl100k_base if omitted or unavailable."
        ),
    )

    args = parser.parse_args()

    # ── Build token counter ───────────────────────────────────────────────────
    print(f"\nInitialising tokenizer...")
    tc = _TokenCounter(args.tokenizer)

    # ── Scan ─────────────────────────────────────────────────────────────────
    print(f"\nScanning: {args.results_dir}")
    print(f"  policy_params : {'auto-infer from model name' if args.policy_params is None else f'{args.policy_params}B (override)'}")
    print(f"  prm_params    : {args.prm_params}B")
    print(f"  embed_params  : {args.embed_params}B")
    df_all = scan_results(
        args.results_dir, tc,
        policy_params_override=args.policy_params,
        prm_params=args.prm_params,
        embed_params=args.embed_params,
    )

    # ── --list_models ─────────────────────────────────────────────────────────
    if args.list_models:
        models = sorted(df_all["model"].dropna().unique())
        print(f"\nFound {len(models)} model(s) in {args.results_dir}:")
        for m in models:
            n_rows = len(df_all[df_all["model"] == m])
            mnts   = sorted(df_all[df_all["model"] == m]["max_tokens"]
                            .dropna().unique().astype(int))
            print(f"  • {m}  ({n_rows} results,  max_tokens={mnts})")
        sys.exit(0)

    # ── Model filter ──────────────────────────────────────────────────────────
    df = _filter_by_model(df_all, args.model or "")

    # ── Model tag for filenames ───────────────────────────────────────────────
    if args.model:
        model_tag = re.sub(r"[^\w\-.]", "_", args.model).strip("_")
    elif len(df["model"].unique()) == 1:
        raw = str(df["model"].iloc[0])
        model_tag = re.sub(r"[^\w\-.]", "_", raw).strip("_")[:40]
    else:
        model_tag = "all_models"

    # ── Summary printout ──────────────────────────────────────────────────────
    print(f"\nResults after model filter ({model_tag}):")
    print(f"  Rows            : {len(df)}")
    print(f"  Models          : {sorted(df['model'].unique())}")
    print(f"  Datasets        : {sorted(df['dataset'].unique())}")
    print(f"  Methods         : {sorted(df['method'].unique())}")
    print(f"  N values        : {sorted(df['N'].unique())}")
    if df["max_tokens"].notna().any():
        mnts = sorted(df["max_tokens"].dropna().unique().astype(int))
        print(f"  Max tokens      : {mnts}")
    print(f"  Tokenizer mode  : {tc.mode}")
    if "policy_params_B" in df.columns:
        unique_params = df[["model","policy_params_B"]].drop_duplicates().sort_values("model")
        print(f"  Policy params   :")
        for _, row in unique_params.iterrows():
            src = "(override)" if args.policy_params is not None else "(auto-inferred)"
            print(f"    {row['model']:<45s} → {row['policy_params_B']}B  {src}")
    print(f"  PRM params      : {args.prm_params}B")
    print(f"  Embed params    : {args.embed_params}B")

    # Print FLOPs summary
    has_flops = (df["flops_total"] > 0).any()
    if has_flops:
        print(f"\n  FLOPs summary (total, per question × N samples):")
        for method in sorted(df["method"].unique()):
            mdf = df[df["method"] == method]
            mean_flops = mdf["flops_total"].mean()
            print(f"    {method:<20s}: {mean_flops:.2e} FLOPs avg")

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = args.out_csv
    if args.model and args.out_csv == "results_master.csv":
        stem = Path(args.out_csv).stem
        suf  = Path(args.out_csv).suffix
        csv_path = str(Path(args.out_csv).parent / f"{stem}_{model_tag}{suf}")

    save_csv(df, csv_path)
    save_supplementary_csvs(df, csv_path)
    save_per_folder_accuracy_csvs(df, args.results_dir)
    # ── Plots ─────────────────────────────────────────────────────────────────
    if not args.no_plot:
        # Determine the set of models to plot
        # If a model filter was applied: plot only that model (already filtered)
        # If no filter: plot every model separately AND one combined view
        models_to_plot = sorted(df["model"].dropna().unique())

        for mdl in models_to_plot:
            mdf = df[df["model"] == mdl].copy()

            # Per-model output subdirectory: plots/<mdl_tag>/
            mdl_tag = re.sub(r"[^\w\-.]", "_", mdl).strip("_")[:40]
            model_plot_dir = Path(args.plot_dir) / mdl_tag
            print(f"\nGenerating plots for model: {mdl}  →  {model_plot_dir}")

            # Accuracy
            plot_accuracy_bars(mdf, str(model_plot_dir), args.max_tokens, mdl_tag)

            # Latency — all methods
            print(f"  [{mdl_tag}] Plotting latency ({mdf['method'].nunique()} methods)...")
            plot_latency(mdf, str(model_plot_dir), args.max_tokens, mdl_tag)

            # FLOPs — all methods (prm/embed=0 for non-verifier methods shown in plot)
            print(f"  [{mdl_tag}] Plotting FLOPs ({mdf['method'].nunique()} methods)...")
            plot_flops(mdf, str(model_plot_dir), args.max_tokens, mdl_tag)

            # Entropy — all methods passed in; plot_entropy internally restricts
            #           to rows with actual cluster data (dn_bon / dn_bon_updated)
            #           so non-clustering methods are correctly absent from the plot
            #           (they have NaN for entropy, not meaningful to plot as a bar)
            print(f"  [{mdl_tag}] Plotting entropy (dn_bon / dn_bon_updated only)...")
            plot_entropy(mdf, str(model_plot_dir), args.max_tokens, mdl_tag)

        # If multiple models exist, also produce combined cross-model plots
        if len(models_to_plot) > 1:
            combined_dir = Path(args.plot_dir) / "all_models"
            print(f"\nGenerating combined plots for all {len(models_to_plot)} models "
                  f"→  {combined_dir}")
            plot_accuracy_bars(df, str(combined_dir), args.max_tokens, "all_models")
            plot_latency(df, str(combined_dir), args.max_tokens, "all_models")
            plot_flops(df, str(combined_dir), args.max_tokens, "all_models")
            plot_entropy(df, str(combined_dir), args.max_tokens, "all_models")

    print("\nDone.")


if __name__ == "__main__":
    main()