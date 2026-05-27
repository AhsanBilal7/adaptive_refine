#!/usr/bin/env python3
"""
extract_baseline_flop.py
========================
Compute theoretical FLOPs for **baseline** inference methods (BoN, DN-BoN,
Majority Voting, Greedy) from JSONL result files using tiktoken for exact
token counting of the selected response and a per-question char-to-token
ratio to estimate tokens for the remaining N-1 candidates.

Supported methods
-----------------
  bon             – Best-of-N with PRM selection            (gen × N + PRM × N)
  dn_bon          – Diversity-normalised BoN with PRM        (gen × N + PRM × N)
  dn_bon_updated  – DN-BoN updated alpha                     (gen × N + PRM × N)
  bon_majority    – Best-of-N with majority voting           (gen × N, no PRM)
  greedy_pass_n   – N samples, log-prob selection            (gen × N, no PRM)
  greedy          – Single greedy sample                     (gen × 1, no PRM)

Token counting
--------------
Output tokens for the SELECTED response are counted exactly via tiktoken.
The remaining N-1 responses are not stored as text; only their character
lengths are available in ``trace_lengths``.  A per-question char-per-token
ratio is derived from the selected response and applied to estimate token
counts for all N candidates.

FLOP formula (same as extract_compute_flop.py)
----------------------------------------------
  generation :  FLOPs = 2 × P × (T_in + T_out)
  PRM scoring:  FLOPs = 2 × P_prm × T_in          (prefill-only)

  T_in  = SYSTEM_OVERHEAD_TOKENS + question_tokens   (same for all N)
  T_out = estimated output tokens per candidate

Usage
-----
    python extract_baseline_flop.py \\
        --results_dir  experiments_results/results_baselines \\
        --output_dir   experiments_results/compute_flops_baseline \\
        --models "Ministral-8B" "Qwen2.5-1.5B" "Qwen2.5-Math-7B" "LLaMA-3.1-8B"

    # Also process dn_bon_updated files
    python extract_baseline_flop.py --include_dn_bon_updated

Output
------
Per-model JSON summary:   <output_dir>/<model_folder>/baseline_flop_summary.json
Master CSV:               <output_dir>/baseline_flop_summary.csv

CSV columns
-----------
model, dataset, method, N, max_tokens,
total_tflop_per_question,
generator_tflop_per_question,
verifier_tflop_per_question,
effective_bon_n,
mean_out_tokens_per_candidate,
mean_in_tokens,
total_tflop_all_questions,
accuracy,
n_questions,
gen_params_b,
tiktoken_encoding
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── tiktoken import (required) ────────────────────────────────────────────────
try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False

# =============================================================================
# Model parameter registry  (identical to extract_compute_flop.py)
# =============================================================================

MODEL_REGISTRY: Dict[str, Tuple[float, str, str]] = {
    # folder-prefix → (params_B, display_name, tiktoken_encoding)
    "Qwen2.5-Math-7B-Instruct":    (7.617, "Qwen2.5-Math-7B", "cl100k_base"),
    "Qwen2.5-Math-7B":             (7.617, "Qwen2.5-Math-7B", "cl100k_base"),
    "Qwen2.5-1.5B-Instruct":       (1.543, "Qwen2.5-1.5B",    "cl100k_base"),
    "Qwen2.5-1.5B":                (1.543, "Qwen2.5-1.5B",    "cl100k_base"),
    "Ministral-8B-Instruct-2410":  (8.030, "Ministral-8B",    "cl100k_base"),
    "Ministral-8B":                (8.030, "Ministral-8B",    "cl100k_base"),
    "Llama-3.1-8B-Instruct":       (8.030, "LLaMA-3.1-8B",    "cl100k_base"),
    "Llama-3.1-8B":                (8.030, "LLaMA-3.1-8B",    "cl100k_base"),
    "Llama-3.2-1B-Instruct":       (1.236, "LLaMA-3.2-1B",    "cl100k_base"),
    "Llama-3.2-3B-Instruct":       (3.212, "LLaMA-3.2-3B",    "cl100k_base"),
    "DeepSeek-R1-Distill-Qwen-7B": (7.617, "DeepSeek-R1-7B",  "cl100k_base"),
}

VERIFIER_PARAMS_B: float = 7.617   # Qwen2.5-Math-PRM-7B
VERIFIER_ENCODING: str   = "cl100k_base"

# Methods that run a PRM verifier to score all N candidates
PRM_METHODS = {"bon", "dn_bon", "dn_bon_updated"}

# Approximate system-prompt + chat-template overhead in tokens (same constant
# as extract_compute_flop.py so FLOPs are directly comparable)
SYSTEM_OVERHEAD_TOKENS: int = 120

# Fallback chars-per-token ratio when the selected response is too short to
# give a reliable estimate (< 50 chars).  ~3.5 chars/tok is typical for
# English mathematical text with LaTeX.
FALLBACK_CHARS_PER_TOKEN: float = 3.5

# Ordered list of known method prefixes (longest first to avoid prefix clash)
KNOWN_METHODS = [
    "dn_bon_updated",
    "dn_bon",
    "bon_majority",
    "bon",
    "greedy_pass_n",
    "greedy",
]


# =============================================================================
# Tiktoken helpers
# =============================================================================

@lru_cache(maxsize=None)
def _get_encoder(encoding_name: str):
    if not _TIKTOKEN_AVAILABLE:
        return None
    return tiktoken.get_encoding(encoding_name)


def _count_tokens(text: str, encoding_name: str) -> int:
    if not text:
        return 0
    enc = _get_encoder(encoding_name)
    if enc is None:
        raise RuntimeError(
            "tiktoken is required but not installed.\n"
            "Install with:  uv pip install 'tiktoken==0.7.0'"
        )
    return len(enc.encode(text))


# =============================================================================
# FLOP formulas  (identical to extract_compute_flop.py)
# =============================================================================

def flops_generation(params_b: float, t_in: int, t_out: int) -> float:
    """FLOPs for one autoregressive generation call: 2 × P × (T_in + T_out)."""
    return 2.0 * (params_b * 1e9) * (t_in + t_out)


def flops_forward_pass(params_b: float, t_in: int) -> float:
    """FLOPs for prefill-only forward pass (PRM scoring): 2 × P × T_in."""
    return 2.0 * (params_b * 1e9) * t_in


def to_tflops(flops: float) -> float:
    return flops / 1e12


def _safe_mean(vals: list) -> float:
    return sum(vals) / len(vals) if vals else 0.0


# =============================================================================
# File-name / folder parsing helpers
# =============================================================================

def _parse_filename(fname: str) -> Optional[Tuple[str, str, int]]:
    """
    Parse a baseline JSONL filename into (method, dataset, N).

    Accepts: bon_aime24_N8.jsonl, dn_bon_updated_math500_N8.jsonl, etc.
    Rejects:  files with spaces, _summary.json, non-JSONL, etc.
    Returns None if unparseable.
    """
    if " " in fname or not fname.endswith(".jsonl") or "_summary" in fname:
        return None
    stem = fname[:-6]  # strip .jsonl
    for method in KNOWN_METHODS:
        prefix = method + "_"
        if stem.startswith(prefix):
            rest = stem[len(prefix):]          # e.g. "aime24_N8"
            m = re.match(r"(.+?)_N(\d+)$", rest)
            if m:
                return method, m.group(1), int(m.group(2))
    return None


def _parse_folder(folder: str, encoding_override: Optional[str] = None
                  ) -> Optional[Tuple[float, str, str, Optional[int]]]:
    """
    Match a subdirectory name to MODEL_REGISTRY.

    Returns (gen_params_b, display_name, encoding, max_tokens) or None.
    """
    folder_base = folder.split("_mnt")[0].split("_t0.")[0]
    for prefix, (params_b, display, enc) in MODEL_REGISTRY.items():
        if folder_base.startswith(prefix):
            mnt_match = re.search(r"mnt(\d+)", folder)
            max_tok = int(mnt_match.group(1)) if mnt_match else None
            return params_b, display, (encoding_override or enc), max_tok
    return None


# Token-source labels used in diagnostics / warnings
_SRC_TRACE    = "trace_lengths"        # all N char-lengths present → ratio-estimated
_SRC_SEL_X_N  = "selected_x_N"        # only selected text stored → replicated × N
_SRC_NO_DATA  = "no_data"             # neither trace_lengths nor raw text → skip


# =============================================================================
# Per-record FLOP computation
# =============================================================================

def compute_record_flops(
    record: Dict[str, Any],
    method: str,
    file_N: int,
    gen_params_b: float,
    prm_params_b: float,
    gen_encoding: str,
    prm_encoding: str,
) -> Dict[str, Any]:
    """
    Compute FLOPs for one question record.

    Three token-counting strategies, in priority order:

    A) ``trace_lengths`` present with ≥1 entry
       → char-length-to-token ratio derived from the selected response (exact
         tiktoken) is applied to every candidate's stored char length.
         The selected response itself uses its exact token count.

    B) ``trace_lengths`` absent/empty BUT ``raw_llm_output`` (selected response)
       is present
       → selected response tokens counted exactly; replicated × N so that all
         N generation passes are counted fairly.  ``token_source = "selected_x_N"``.

    C) Both absent (old-format records that stored nothing)
       → ``token_source = "no_data"``; caller must skip this record.

    ``file_N`` is the N parsed from the filename and is used as a fallback
    when the record's own ``N`` field is 0 / missing.
    """
    question_text  = record.get("question") or ""
    selected_text  = record.get("raw_llm_output") or ""
    trace_lengths  = record.get("trace_lengths") or []   # char lengths, all N
    selected_index = int(record.get("selected_index") or 0)

    # Prefer record-level N (reflects actual candidates); fall back to file_N
    N_candidates = int(record.get("N") or 0) or file_N or 1

    # ── Input tokens (identical for every candidate) ─────────────────────────
    q_tokens = _count_tokens(question_text, gen_encoding)
    t_in     = SYSTEM_OVERHEAD_TOKENS + q_tokens

    # ── Exact token count for the stored (selected) response ─────────────────
    sel_chars  = len(selected_text)
    sel_tokens = _count_tokens(selected_text, gen_encoding) if selected_text else 0

    # chars-per-token ratio — derived from the selected response when reliable
    if sel_chars >= 50 and sel_tokens > 0:
        chars_per_tok = sel_chars / sel_tokens
    else:
        chars_per_tok = FALLBACK_CHARS_PER_TOKEN

    # ── Choose token-counting strategy ───────────────────────────────────────
    candidate_out_tokens: List[int]
    token_source: str

    if trace_lengths:
        # ── Strategy A: use stored char lengths for all N candidates ─────────
        # The selected candidate gets its exact tiktoken count; all others are
        # estimated via the per-question chars-per-token ratio.
        token_source = _SRC_TRACE
        candidate_out_tokens = []
        for i, clen in enumerate(trace_lengths):
            if i == selected_index and sel_tokens > 0:
                candidate_out_tokens.append(sel_tokens)         # exact
            else:
                candidate_out_tokens.append(max(1, round(clen / chars_per_tok)))

    elif selected_text:
        # ── Strategy B: only selected response stored; replicate × N ─────────
        # This is the fair-comparison case: N responses were generated but only
        # the selected one was persisted.  We use the selected response length
        # as a representative proxy for all N candidates and multiply by N so
        # that total generation FLOPs equal N × flops_per_single_pass — the
        # same budget as BoN / greedy-pass-N at inference time.
        token_source = _SRC_SEL_X_N
        candidate_out_tokens = [sel_tokens] * N_candidates

    else:
        # ── Strategy C: no usable data — record must be skipped ──────────────
        return dict(token_source=_SRC_NO_DATA, n_candidates=N_candidates)

    # ── Generation FLOPs: one autoregressive call per candidate ──────────────
    generation_flop = sum(
        flops_generation(gen_params_b, t_in, t_out)
        for t_out in candidate_out_tokens
    )

    # ── Verifier FLOPs: PRM forward pass per candidate (when applicable) ─────
    verifier_flop = 0.0
    if method in PRM_METHODS:
        for t_out in candidate_out_tokens:
            t_in_prm = q_tokens + t_out   # question + response (prefill only)
            verifier_flop += flops_forward_pass(prm_params_b, t_in_prm)

    total_flop = generation_flop + verifier_flop

    return dict(
        token_source=token_source,
        generation_flop=generation_flop,
        verifier_flop=verifier_flop,
        total_flop=total_flop,
        t_in=t_in,
        q_tokens=q_tokens,
        candidate_out_tokens=candidate_out_tokens,
        mean_out_tokens=_safe_mean(candidate_out_tokens),
        chars_per_tok=chars_per_tok,
        n_candidates=len(candidate_out_tokens),
    )


# =============================================================================
# Per-file FLOP aggregation
# =============================================================================

def compute_file_flops(
    records: List[Dict[str, Any]],
    source_file: str,
    method: str,
    dataset: str,
    N: int,
    max_tokens: Optional[int],
    gen_params_b: float,
    prm_params_b: float,
    gen_encoding: str,
    prm_encoding: str,
) -> Dict[str, Any]:
    """
    Aggregate FLOPs over all question records in one JSONL file.

    Records that have neither trace_lengths nor raw_llm_output (old-format
    ``bestofn`` / ``greedypassn`` entries) are counted but excluded from the
    FLOP averages and accuracy calculation — they are reported separately as
    ``n_skipped_no_data``.

    Records that fall back to the ``selected_x_N`` strategy (only the selected
    response was stored; all N candidates are represented by replicating it)
    are included in all metrics but reported as ``n_fallback_selected_x_N`` so
    the caller can audit the fraction of questions using the proxy approach.
    """
    if not records:
        return {"source_file": source_file, "n_questions": 0}

    q_total_flops:      List[float] = []
    q_gen_flops:        List[float] = []
    q_vrfy_flops:       List[float] = []
    all_out_toks:       List[float] = []
    all_in_toks:        List[int]   = []
    n_correct            = 0
    n_skipped_no_data    = 0
    n_fallback_sel_x_n   = 0

    for rec in records:
        rf = compute_record_flops(
            record=rec,
            method=method,
            file_N=N,
            gen_params_b=gen_params_b,
            prm_params_b=prm_params_b,
            gen_encoding=gen_encoding,
            prm_encoding=prm_encoding,
        )

        if rf["token_source"] == _SRC_NO_DATA:
            # Old-format record with nothing to estimate from — exclude entirely
            n_skipped_no_data += 1
            continue

        if rf["token_source"] == _SRC_SEL_X_N:
            # Selected-response proxy replicated × N: included but flagged
            n_fallback_sel_x_n += 1

        q_total_flops.append(rf["total_flop"])
        q_gen_flops.append(rf["generation_flop"])
        q_vrfy_flops.append(rf["verifier_flop"])
        all_out_toks.append(rf["mean_out_tokens"])
        all_in_toks.append(rf["t_in"])
        if rec.get("is_correct"):
            n_correct += 1

    n_total = len(records)          # all records in file (incl. skipped)
    n_valid = len(q_total_flops)    # records that contributed to FLOPs

    if n_valid == 0:
        logger.warning(
            f"No computable records in {os.path.basename(source_file)} "
            f"(all {n_total} records had no usable token data)."
        )
        return {
            "source_file": source_file,
            "n_questions": n_total,
            "n_valid": 0,
            "n_skipped_no_data": n_skipped_no_data,
        }

    if n_skipped_no_data:
        logger.warning(
            f"{os.path.basename(source_file)}: "
            f"skipped {n_skipped_no_data}/{n_total} records (no token data); "
            f"FLOPs computed from {n_valid} records."
        )
    if n_fallback_sel_x_n:
        logger.debug(
            f"{os.path.basename(source_file)}: "
            f"{n_fallback_sel_x_n}/{n_valid} records used "
            f"'selected_response × N' token proxy."
        )

    # ── BoN effective-N: total FLOPs vs. a single generation pass ────────────
    mean_t_in  = _safe_mean(all_in_toks)
    mean_t_out = _safe_mean(all_out_toks)
    flop_single = flops_generation(gen_params_b, int(mean_t_in), int(mean_t_out))
    mean_total  = _safe_mean(q_total_flops)
    effective_bon_n = (mean_total / flop_single) if flop_single > 0 else float("nan")

    return dict(
        source_file=source_file,
        method=method,
        dataset=dataset,
        N=N,
        max_tokens=max_tokens,
        n_questions=n_valid,          # questions with computable FLOPs
        n_skipped_no_data=n_skipped_no_data,
        n_fallback_selected_x_N=n_fallback_sel_x_n,
        gen_params_b=gen_params_b,
        prm_params_b=prm_params_b,
        tiktoken_encoding=gen_encoding,
        uses_prm=(method in PRM_METHODS),
        # ── FLOPs per question (TFLOPs) ──────────────────────────────────────
        total_tflop_per_question     =round(to_tflops(mean_total), 6),
        generator_tflop_per_question =round(to_tflops(_safe_mean(q_gen_flops)), 6),
        verifier_tflop_per_question  =round(to_tflops(_safe_mean(q_vrfy_flops)), 6),
        # ── BoN comparison ───────────────────────────────────────────────────
        effective_bon_n              =round(effective_bon_n, 2),
        # ── Token stats ──────────────────────────────────────────────────────
        mean_out_tokens_per_candidate=round(mean_t_out, 1),
        mean_in_tokens               =round(mean_t_in, 1),
        # ── Totals ───────────────────────────────────────────────────────────
        total_tflop_all_questions    =round(to_tflops(sum(q_total_flops)), 4),
        # ── Accuracy (based on valid records only) ───────────────────────────
        accuracy=round(n_correct / max(n_valid, 1), 4),
    )


# =============================================================================
# CSV
# =============================================================================

CSV_FIELDS = [
    "model", "dataset", "method", "N", "max_tokens",
    "total_tflop_per_question",
    "generator_tflop_per_question",
    "verifier_tflop_per_question",
    "effective_bon_n",
    "mean_out_tokens_per_candidate",
    "mean_in_tokens",
    "total_tflop_all_questions",
    "accuracy",
    "n_questions",
    "n_skipped_no_data",
    "n_fallback_selected_x_N",
    "gen_params_b",
    "tiktoken_encoding",
    "uses_prm",
]


def _metrics_to_csv_row(m: Dict[str, Any], display_name: str) -> Dict[str, Any]:
    row = {f: m.get(f, "") for f in CSV_FIELDS}
    row["model"] = display_name
    return row


# =============================================================================
# JSON output helper
# =============================================================================

def _write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed line in {path}: {e}")
    return records


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Compute theoretical FLOPs for CIMR baseline methods.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results_dir",
        default="experiments_results/results_baselines",
        help="Root folder containing baseline JSONL files (default: %(default)s)",
    )
    parser.add_argument(
        "--output_dir",
        default="experiments_results/compute_flops_baseline",
        help="Destination folder for outputs (default: %(default)s)",
    )
    parser.add_argument(
        "--encoding",
        default=None,
        help="Override tiktoken encoding for all models (e.g. cl100k_base). "
             "Default: per-model value from MODEL_REGISTRY.",
    )
    parser.add_argument(
        "--verifier_params_b",
        type=float,
        default=VERIFIER_PARAMS_B,
        help=f"Verifier model parameters in billions (default: {VERIFIER_PARAMS_B}B).",
    )
    parser.add_argument(
        "--verifier_encoding",
        default=VERIFIER_ENCODING,
        help=f"tiktoken encoding for verifier (default: {VERIFIER_ENCODING}).",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Restrict to specific display names, e.g. 'Qwen2.5-Math-7B'. "
             "Processes all recognised models if omitted.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Restrict to specific methods, e.g. 'bon' 'dn_bon'. "
             "Processes all methods if omitted.",
    )
    parser.add_argument(
        "--include_dn_bon_updated",
        action="store_true",
        help="Include dn_bon_updated files (excluded by default to avoid "
             "double-counting with dn_bon).",
    )
    args = parser.parse_args()

    if not _TIKTOKEN_AVAILABLE:
        raise SystemExit(
            "ERROR: tiktoken is not installed.\n"
            "Install with:  uv pip install 'tiktoken==0.7.0'"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    prm_b   = args.verifier_params_b
    prm_enc = args.verifier_encoding

    # Warm up encoders
    encs_to_warm = {args.encoding} if args.encoding else {
        enc for (_, _, enc) in MODEL_REGISTRY.values()
    }
    encs_to_warm.add(prm_enc)
    for enc_name in encs_to_warm:
        _get_encoder(enc_name)
        logger.info(f"  loaded tiktoken encoder: {enc_name}")

    # ── Walk results tree ─────────────────────────────────────────────────────
    per_dir:      Dict[str, Dict[str, Any]] = {}
    all_csv_rows: List[Dict[str, Any]]      = []
    n_ok = n_err = n_skip = 0

    for root, dirs, files in os.walk(args.results_dir):
        dirs.sort()
        folder = os.path.basename(root)
        model_info = _parse_folder(folder, args.encoding)
        if model_info is None:
            if root != args.results_dir:
                n_skip += 1
            continue

        gen_b, display_name, gen_enc, max_tokens = model_info

        if args.models and display_name not in args.models:
            n_skip += 1
            continue

        for fname in sorted(files):
            parsed = _parse_filename(fname)
            if parsed is None:
                continue

            method, dataset, N = parsed

            # Filter: dn_bon_updated excluded unless flag set
            if method == "dn_bon_updated" and not args.include_dn_bon_updated:
                continue

            # Filter: user-specified methods
            if args.methods and method not in args.methods:
                continue

            full_path = os.path.join(root, fname)
            try:
                records = _load_jsonl(full_path)
                if not records:
                    logger.warning(f"Empty file: {full_path}")
                    continue

                metrics = compute_file_flops(
                    records=records,
                    source_file=full_path,
                    method=method,
                    dataset=dataset,
                    N=N,
                    max_tokens=max_tokens,
                    gen_params_b=gen_b,
                    prm_params_b=prm_b,
                    gen_encoding=gen_enc,
                    prm_encoding=prm_enc,
                )
                metrics["model_display"] = display_name

                dir_rel = os.path.relpath(root, args.results_dir)
                per_dir.setdefault(dir_rel, {})[fname] = metrics
                all_csv_rows.append(_metrics_to_csv_row(metrics, display_name))

                logger.info(
                    f"  {display_name:20s}  {dataset:18s}  {method:18s}"
                    f"  N={N:2d}  mnt={max_tokens or '?':>6}"
                    f"  {metrics['total_tflop_per_question']:8.3f} TFLOPs/q"
                    f"  eff_BoN={metrics['effective_bon_n']:5.1f}"
                    f"  acc={metrics['accuracy']:.3f}"
                    f"  prm={'Y' if metrics['uses_prm'] else 'N'}"
                )
                n_ok += 1

            except Exception as e:
                logger.error(f"Failed on {full_path}: {e}", exc_info=True)
                n_err += 1

    # ── Write per-model JSON summaries ────────────────────────────────────────
    for dir_rel, summary in per_dir.items():
        out_path = os.path.join(args.output_dir, dir_rel, "baseline_flop_summary.json")
        _write_json(out_path, summary)
        logger.info(f"  written JSON  {os.path.relpath(out_path)}")

    # ── Write master CSV ──────────────────────────────────────────────────────
    all_csv_rows.sort(key=lambda r: (
        str(r["model"]),
        str(r["dataset"]),
        str(r["method"]),
        int(r["N"]) if str(r["N"]).isdigit() else 0,
        int(r["max_tokens"]) if str(r.get("max_tokens", "")).isdigit() else 0,
    ))

    csv_path = os.path.join(args.output_dir, "baseline_flop_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_csv_rows)
    logger.info(
        f"  written CSV  {os.path.relpath(csv_path)}  ({len(all_csv_rows)} rows)"
    )
    logger.info(
        f"\nDone.  {n_ok} files processed, {n_err} errors, {n_skip} folders skipped."
    )

    if all_csv_rows:
        _print_summary_table(all_csv_rows)


def _print_summary_table(rows: List[Dict[str, Any]]) -> None:
    print(
        f"\n{'Model':<22} {'Dataset':<18} {'Method':<18} {'N':>3} {'mnt':>6}"
        f" {'TFLOPs/q':>10} {'eff_BoN-N':>10} {'acc':>7} {'PRM':>4}"
    )
    print("-" * 110)
    for r in rows:
        try:
            print(
                f"{str(r['model']):<22} {str(r['dataset']):<18}"
                f" {str(r['method']):<18}"
                f" {int(r['N']):>3}"
                f" {str(r.get('max_tokens','?')):>6}"
                f" {float(r['total_tflop_per_question']):>10.3f}"
                f" {float(r['effective_bon_n']):>10.1f}"
                f" {float(r['accuracy']):>7.3f}"
                f" {'Y' if str(r.get('uses_prm','')).lower() in ('true','1','yes') else 'N':>4}"
            )
        except (ValueError, TypeError):
            pass


if __name__ == "__main__":
    main()
