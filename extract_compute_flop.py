#!/usr/bin/env python3
"""
extract_compute_flop.py
=======================
Compute theoretical FLOPs for the CIMR pipeline from stored JSON results.

Token counts are measured with **tiktoken** (exact BPE tokenization) rather
than a character-ratio heuristic.  Output tokens are counted directly from
the stored response text; input tokens are reconstructed from the pipeline
structure and config-based truncation limits.

FLOPs are estimated using the standard LLM scaling-law approximation:

    generation :  FLOPs = 2 × P × (T_in + T_out)
    PRM scoring:  FLOPs = 2 × P_prm × T_in          (prefill-only, no decode)

Pipeline phases counted per (rollout × depth) step
---------------------------------------------------
  Phase A  – reasoning generation          (generator model)
  Phase B  – self-critique generation      (generator model)  [0 if no_critique]
  Phase C  – self-correction generation    (generator model)  [0 if no_correction]
  Phase E  – PRM scoring × 2               (verifier model,
                reasoning_trace + corrected_trace as separate forward passes)

BoN comparison baseline
-----------------------
Best-of-N(N) runs N independent reasoning passes only (no critique/correction):
    bon_flop_per_q = N × 2 × P_gen × (T_in_depth1 + mean_T_out_A)

    effective_bon_n = total_cimr_flop_per_q / flop_per_single_bon_sample

Tokeniser registry
------------------
Each model is mapped to the tiktoken encoding that best approximates its
tokenizer.  All four target models (Qwen2.5-Math-7B, Qwen2.5-1.5B,
Ministral-8B, LLaMA-3.1-8B) use BPE vocabularies derived from or closely
matching cl100k_base.

Usage
-----
    python extract_compute_flop.py \\
        --results_dir  experiments_results/results_paired \\
        --output_dir   experiments_results/compute_flops

    # Override tiktoken encoding globally
    python extract_compute_flop.py --encoding o200k_base

    # Process only specific display-name models
    python extract_compute_flop.py --models "Ministral-8B" "Qwen2.5-1.5B" "Qwen2.5-Math-7B" "LLaMA-3.1-8B"
    python extract_compute_flop.py --results_dir  experiments_results/results_baselines --output_dir   experiments_results/compute_flops_baseline --models "Ministral-8B" "Qwen2.5-1.5B" "Qwen2.5-Math-7B" "LLaMA-3.1-8B"
    # Also include no_critique ablation files
    python extract_compute_flop.py --include_no_critique

Output
------
Per model sub-directory (mirrors extract_cimr_metrics.py layout):
    <output_dir>/<model>/flop_summary.json

Master CSV across all models:
    <output_dir>/compute_flop_summary.csv

CSV columns
-----------
model, dataset, N, D, suffix,
total_tflop_per_question,          # generator + verifier (TFLOPs, mean over qs)
generator_tflop_per_question,      # phases A + B + C only
verifier_tflop_per_question,       # PRM verifier (Phase E) only
phase_a_tflop_per_question,        # reasoning generation
phase_b_tflop_per_question,        # self-critique
phase_c_tflop_per_question,        # self-correction
bon_baseline_tflop_per_question,   # BoN(N) — N reasoning-only passes
effective_bon_n,                   # BoN samples equivalent to CIMR total FLOPs
mean_reasoning_out_tokens,         # mean Phase A output tokens per step
mean_critique_out_tokens,          # mean Phase B output tokens per step
mean_correction_out_tokens,        # mean Phase C output tokens per step
mean_total_out_tokens_per_rollout,
total_tflop_all_questions,         # sum over all questions in file
majority_accuracy,
max_prm_accuracy,
n_questions,
gen_params_b,
tiktoken_encoding,
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── tiktoken import (required – no fallback) ─────────────────────────────────
try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False


# =============================================================================
# Model parameter registry
# =============================================================================

# folder-name prefix  →  (generator_params_B, display_name, tiktoken_encoding)
#
# Encoding notes:
#   Qwen 2.5  — officially uses tiktoken with a vocabulary built on cl100k_base
#               plus ~16 k extra tokens (Chinese, math, code).  cl100k_base is
#               the closest public encoding.
#   LLaMA 3.x — Meta's tiktoken-based tokenizer with 128 k vocab; cl100k_base
#               is used as an approximation (vocab overlap is ~80 %).
#   Ministral — SentencePiece BPE with 32 k vocab; cl100k_base slightly
#               over-counts tokens on average but is the best public proxy.
#   Verifier  — Qwen2.5-Math-PRM-7B uses the same tokenizer as Qwen 2.5.

MODEL_REGISTRY: Dict[str, Tuple[float, str, str]] = {
    # (params_B, display_name, tiktoken_encoding)
    "Qwen2.5-Math-7B-Instruct":    (7.617, "Qwen2.5-Math-7B", "cl100k_base"),
    "Qwen2.5-Math-7B":             (7.617, "Qwen2.5-Math-7B", "cl100k_base"),
    "Qwen2.5-1.5B-Instruct":       (1.543, "Qwen2.5-1.5B",    "cl100k_base"),
    "Qwen2.5-1.5B":                (1.543, "Qwen2.5-1.5B",    "cl100k_base"),
    "Ministral-8B-Instruct-2410":  (8.030, "Ministral-8B",    "cl100k_base"),
    "Ministral-8B":                (8.030, "Ministral-8B",    "cl100k_base"),
    "Llama-3.1-8B-Instruct":       (8.030, "LLaMA-3.1-8B",    "cl100k_base"),
    "Llama-3.1-8B":                (8.030, "LLaMA-3.1-8B",    "cl100k_base"),
    # Additional models present in experiments_results
    "Llama-3.2-1B-Instruct":       (1.236, "LLaMA-3.2-1B",    "cl100k_base"),
    "Llama-3.2-3B-Instruct":       (3.212, "LLaMA-3.2-3B",    "cl100k_base"),
    "DeepSeek-R1-Distill-Qwen-7B": (7.617, "DeepSeek-R1-7B",  "cl100k_base"),
}

# Qwen2.5-Math-PRM-7B verifier (always used as PRM scorer)
VERIFIER_PARAMS_B: float   = 7.617
VERIFIER_ENCODING: str     = "cl100k_base"

# =============================================================================
# Pipeline config defaults  (must match run_paired_ablation.py defaults)
# =============================================================================

# These are character-level truncation limits applied in cimr_step.py *before*
# the text reaches the model.  We truncate stored text to the same limits
# before tokenising, so Phase-B/C input token counts faithfully reflect what
# the model actually sees.
CRITIQUE_MAX_TRACE_CHARS:      int = 4000
CORRECTION_MAX_TRACE_CHARS:    int = 4000
CORRECTION_MAX_CRITIQUE_CHARS: int = 1500
CONTINUATION_MAX_PRIOR_CHARS:  int = 4000

# Approximate overhead for system-prompt + chat-template markup tokens that
# are NOT stored in the JSON.  Measured as ~120 tokens by counting the CIMR
# prompt templates (REASONING_SYSTEM_PROMPT, CRITIQUE_SYSTEM_PROMPT,
# CORRECTION_SYSTEM_PROMPT) plus the chat-template header/footer tokens.
SYSTEM_OVERHEAD_TOKENS: int = 120


# =============================================================================
# Tiktoken encoder cache
# =============================================================================

@lru_cache(maxsize=None)
def _get_encoder(encoding_name: str):
    """Load (and cache) a tiktoken encoding by name."""
    if not _TIKTOKEN_AVAILABLE:
        return None
    return tiktoken.get_encoding(encoding_name)


def _count_tokens(text: str, encoding_name: str) -> int:
    """
    Return the exact token count for *text* using tiktoken.

    Raises ``RuntimeError`` if tiktoken is not installed – install with:
        uv pip install 'tiktoken==0.7.0'

    Empty or whitespace-only text returns 0 without calling the encoder.
    """
    if not text:
        return 0
    enc = _get_encoder(encoding_name)
    if enc is None:
        raise RuntimeError(
            "tiktoken is required for token counting but is not installed.\n"
            "Install it with:  uv pip install 'tiktoken==0.7.0'"
        )
    return len(enc.encode(text))


def _count_tokens_truncated(text: str, max_chars: int, encoding_name: str) -> int:
    """
    Count tokens *after* applying the pipeline's character-level truncation.

    This mirrors cimr_step.py: the text is sliced to ``max_chars`` chars
    before being fed to the model, so we truncate first and then tokenise.
    """
    truncated = text[:max_chars] if len(text) > max_chars else text
    return _count_tokens(truncated, encoding_name)


# =============================================================================
# FLOP formulas
# =============================================================================

def flops_generation(params_b: float, t_in: int, t_out: int) -> float:
    """
    FLOPs for one autoregressive generation call.

    Standard LLM approximation (Chinchilla / PaLM scaling laws):
        FLOPs ≈ 2 × P × (T_in + T_out)

    *params_b* is in billions; result is in raw FLOPs.
    """
    return 2.0 * (params_b * 1e9) * (t_in + t_out)


def flops_forward_pass(params_b: float, t_in: int) -> float:
    """
    FLOPs for a prefill-only forward pass (PRM scoring – no decoding).
        FLOPs ≈ 2 × P × T_in
    """
    return 2.0 * (params_b * 1e9) * t_in


def to_tflops(flops: float) -> float:
    """Convert raw FLOPs to TFLOPs (10^12)."""
    return flops / 1e12


# =============================================================================
# Per-step FLOP computation
# =============================================================================

def compute_step_flops(
    step: Dict[str, Any],
    question: str,
    prior_corrected_trace: Optional[str],
    gen_params_b: float,
    prm_params_b: float,
    gen_encoding: str,
    prm_encoding: str,
) -> Dict[str, Any]:
    """
    Compute FLOPs for one (rollout × depth) step.

    Token counts come from tiktoken, with character-level truncation applied
    to match the pipeline's ``*_max_*_chars`` limits before tokenisation.

    Returns a dict with:
        phase_a_flop, phase_b_flop, phase_c_flop, phase_e_flop,
        generator_flop, verifier_flop, total_flop,
        reasoning_out_tokens, critique_out_tokens, correction_out_tokens,
        reasoning_in_tokens, critique_in_tokens, correction_in_tokens,
        prm_in_tokens_reasoning, prm_in_tokens_corrected.
    """
    step_idx  = int(step.get("step_idx", 1))
    is_depth1 = (step_idx == 1)

    # ── Stored output text ─────────────────────────────────────────────────
    reasoning_out_text  = step.get("reasoning_response")  or ""
    critique_out_text   = step.get("critique_response")   or ""
    correction_out_text = step.get("correction_response") or ""
    corrected_trace     = step.get("corrected_trace")     or ""

    has_critique   = bool(critique_out_text.strip())
    has_correction = bool(correction_out_text.strip())

    # ── Output token counts (exact, from tiktoken) ─────────────────────────
    t_out_A = _count_tokens(reasoning_out_text,  gen_encoding)
    t_out_B = _count_tokens(critique_out_text,   gen_encoding) if has_critique   else 0
    t_out_C = _count_tokens(correction_out_text, gen_encoding) if has_correction else 0

    # ── Input token counts ──────────────────────────────────────────────────
    q_tokens = _count_tokens(question, gen_encoding)

    # Phase A: question (+ prior corrected trace at depth > 1) + system overhead
    if is_depth1:
        t_in_A = SYSTEM_OVERHEAD_TOKENS + q_tokens
    else:
        prior_tok = _count_tokens_truncated(
            prior_corrected_trace or "",
            CONTINUATION_MAX_PRIOR_CHARS,
            gen_encoding,
        )
        t_in_A = SYSTEM_OVERHEAD_TOKENS + q_tokens + prior_tok

    # Phase B: system + question + reasoning_trace (char-truncated to CRITIQUE_MAX)
    if has_critique:
        t_in_B = (SYSTEM_OVERHEAD_TOKENS
                  + q_tokens
                  + _count_tokens_truncated(
                      reasoning_out_text, CRITIQUE_MAX_TRACE_CHARS, gen_encoding
                  ))
    else:
        t_in_B = 0

    # Phase C: system + question + reasoning_trace (trunc) + critique (trunc)
    if has_correction:
        t_in_C = (SYSTEM_OVERHEAD_TOKENS
                  + q_tokens
                  + _count_tokens_truncated(
                      reasoning_out_text, CORRECTION_MAX_TRACE_CHARS, gen_encoding
                  )
                  + _count_tokens_truncated(
                      critique_out_text, CORRECTION_MAX_CRITIQUE_CHARS, gen_encoding
                  ))
    else:
        t_in_C = 0

    # Phase E: PRM forward pass over (a) raw reasoning trace, (b) corrected trace
    t_in_E_reason    = q_tokens + _count_tokens(reasoning_out_text, prm_encoding)
    t_in_E_corrected = q_tokens + _count_tokens(corrected_trace,    prm_encoding)

    # ── FLOP computation ─────────────────────────────────────────────────────
    phase_a_flop = flops_generation(  gen_params_b, t_in_A, t_out_A)
    phase_b_flop = flops_generation(  gen_params_b, t_in_B, t_out_B) if has_critique   else 0.0
    phase_c_flop = flops_generation(  gen_params_b, t_in_C, t_out_C) if has_correction else 0.0
    phase_e_flop = (flops_forward_pass(prm_params_b, t_in_E_reason)
                  + flops_forward_pass(prm_params_b, t_in_E_corrected))

    generator_flop = phase_a_flop + phase_b_flop + phase_c_flop
    verifier_flop  = phase_e_flop
    total_flop     = generator_flop + verifier_flop

    return dict(
        phase_a_flop=phase_a_flop,
        phase_b_flop=phase_b_flop,
        phase_c_flop=phase_c_flop,
        phase_e_flop=phase_e_flop,
        generator_flop=generator_flop,
        verifier_flop=verifier_flop,
        total_flop=total_flop,
        # token counts (for diagnostics / CSV columns)
        reasoning_out_tokens=t_out_A,
        critique_out_tokens=t_out_B,
        correction_out_tokens=t_out_C,
        reasoning_in_tokens=t_in_A,
        critique_in_tokens=t_in_B,
        correction_in_tokens=t_in_C,
        prm_in_tokens_reasoning=t_in_E_reason,
        prm_in_tokens_corrected=t_in_E_corrected,
    )


# =============================================================================
# Per-file FLOP aggregation
# =============================================================================

def _safe_mean(vals: list) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def compute_file_flops(
    store: Dict[str, Any],
    source_file: str,
    gen_params_b: float,
    prm_params_b: float,
    gen_encoding: str,
    prm_encoding: str,
) -> Dict[str, Any]:
    """
    Aggregate FLOPs across all questions / rollouts / steps in a results store.
    Returns a flat metrics dict ready for JSON / CSV export.
    """
    questions = list(store.values())
    if not questions:
        return {"source_file": source_file, "n_questions": 0}

    sample_q = questions[0]
    N = int(sample_q.get("N", 0))
    D = int(sample_q.get("depth", 0))
    n_total = len(questions)

    # Per-question accumulators
    q_total_flops:    List[float] = []
    q_gen_flops:      List[float] = []
    q_vrfy_flops:     List[float] = []
    q_phase_a_flops:  List[float] = []
    q_phase_b_flops:  List[float] = []
    q_phase_c_flops:  List[float] = []

    # Token averages across ALL rollout × depth steps
    all_reasoning_out_tok:   List[int] = []
    all_critique_out_tok:    List[int] = []
    all_correction_out_tok:  List[int] = []
    all_rollout_total_tok:   List[int] = []

    # For BoN baseline: cost of one depth-1 reasoning-only pass
    bon_sample_flops: List[float] = []

    for q in questions:
        question_text = q.get("question") or ""
        rollouts = [r for r in (q.get("rollouts") or []) if r]

        q_tf = q_gf = q_vf = q_af = q_bf = q_cf = 0.0

        for rollout in rollouts:
            steps = rollout.get("steps") or []
            rollout_out_tok = 0
            prior_corrected: Optional[str] = None

            for step in steps:
                sf = compute_step_flops(
                    step=step,
                    question=question_text,
                    prior_corrected_trace=prior_corrected,
                    gen_params_b=gen_params_b,
                    prm_params_b=prm_params_b,
                    gen_encoding=gen_encoding,
                    prm_encoding=prm_encoding,
                )

                q_tf += sf["total_flop"]
                q_gf += sf["generator_flop"]
                q_vf += sf["verifier_flop"]
                q_af += sf["phase_a_flop"]
                q_bf += sf["phase_b_flop"]
                q_cf += sf["phase_c_flop"]

                all_reasoning_out_tok.append(sf["reasoning_out_tokens"])
                if sf["critique_out_tokens"] > 0:
                    all_critique_out_tok.append(sf["critique_out_tokens"])
                if sf["correction_out_tokens"] > 0:
                    all_correction_out_tok.append(sf["correction_out_tokens"])

                rollout_out_tok += (sf["reasoning_out_tokens"]
                                    + sf["critique_out_tokens"]
                                    + sf["correction_out_tokens"])

                # Advance prior-corrected trace for next depth
                prior_corrected = step.get("corrected_trace") or prior_corrected

                # BoN reference cost: depth-1 step, reasoning only (no B/C/E)
                if int(step.get("step_idx", -1)) == 1:
                    bon_flop = flops_generation(
                        gen_params_b,
                        sf["reasoning_in_tokens"],
                        sf["reasoning_out_tokens"],
                    )
                    bon_sample_flops.append(bon_flop)

            all_rollout_total_tok.append(rollout_out_tok)

        q_total_flops.append(q_tf)
        q_gen_flops.append(q_gf)
        q_vrfy_flops.append(q_vf)
        q_phase_a_flops.append(q_af)
        q_phase_b_flops.append(q_bf)
        q_phase_c_flops.append(q_cf)

    # ── BoN baseline ─────────────────────────────────────────────────────────
    flop_per_bon_sample   = _safe_mean(bon_sample_flops)
    bon_baseline_flop_per_q = N * flop_per_bon_sample
    mean_total_flop_per_q   = _safe_mean(q_total_flops)
    effective_bon_n = (mean_total_flop_per_q / flop_per_bon_sample
                       if flop_per_bon_sample > 0 else float("nan"))

    # ── Accuracy from aggregation field ──────────────────────────────────────
    n_maj = sum(1 for q in questions
                if (q.get("aggregation") or {}).get("majority_is_correct") is True)
    n_prm = sum(1 for q in questions
                if (q.get("aggregation") or {}).get("max_prm_is_correct") is True)
    denom = max(n_total, 1)

    # ── Depth-wise cumulative FLOPs (for staircase / depth-accuracy plots) ──
    depth_flops: Dict[str, Dict[str, float]] = {}
    for d in range(1, D + 1):
        d_tot: List[float] = []
        d_gen: List[float] = []
        d_vrf: List[float] = []
        for q in questions:
            question_text = q.get("question") or ""
            rollouts = [r for r in (q.get("rollouts") or []) if r]
            dt = dg = dv = 0.0
            for rollout in rollouts:
                steps = [s for s in (rollout.get("steps") or [])
                         if int(s.get("step_idx", -1)) <= d]
                prior_corrected = None
                for step in steps:
                    sf = compute_step_flops(
                        step=step,
                        question=question_text,
                        prior_corrected_trace=prior_corrected,
                        gen_params_b=gen_params_b,
                        prm_params_b=prm_params_b,
                        gen_encoding=gen_encoding,
                        prm_encoding=prm_encoding,
                    )
                    dt += sf["total_flop"]
                    dg += sf["generator_flop"]
                    dv += sf["verifier_flop"]
                    prior_corrected = step.get("corrected_trace") or prior_corrected
            d_tot.append(dt)
            d_gen.append(dg)
            d_vrf.append(dv)

        depth_flops[str(d)] = dict(
            mean_total_tflop    =round(to_tflops(_safe_mean(d_tot)), 6),
            mean_gen_tflop      =round(to_tflops(_safe_mean(d_gen)), 6),
            mean_verifier_tflop =round(to_tflops(_safe_mean(d_vrf)), 6),
        )

    return dict(
        source_file=source_file,
        model_display=_infer_model_display(source_file),
        dataset=_infer_dataset(source_file),
        N=N, D=D,
        suffix=_infer_suffix(source_file),
        n_questions=n_total,
        gen_params_b=gen_params_b,
        prm_params_b=prm_params_b,
        tiktoken_encoding=gen_encoding,
        # ── FLOPs per question (TFLOPs) ──────────────────────────────────────
        total_tflop_per_question      =round(to_tflops(mean_total_flop_per_q), 6),
        generator_tflop_per_question  =round(to_tflops(_safe_mean(q_gen_flops)), 6),
        verifier_tflop_per_question   =round(to_tflops(_safe_mean(q_vrfy_flops)), 6),
        phase_a_tflop_per_question    =round(to_tflops(_safe_mean(q_phase_a_flops)), 6),
        phase_b_tflop_per_question    =round(to_tflops(_safe_mean(q_phase_b_flops)), 6),
        phase_c_tflop_per_question    =round(to_tflops(_safe_mean(q_phase_c_flops)), 6),
        # ── BoN comparison ────────────────────────────────────────────────────
        bon_baseline_tflop_per_question=round(to_tflops(bon_baseline_flop_per_q), 6),
        flop_per_bon_sample_tflop      =round(to_tflops(flop_per_bon_sample), 6),
        effective_bon_n                =round(effective_bon_n, 2),
        # ── Totals ────────────────────────────────────────────────────────────
        total_tflop_all_questions=round(to_tflops(sum(q_total_flops)), 4),
        # ── Token counts (means) ──────────────────────────────────────────────
        mean_reasoning_out_tokens         =round(_safe_mean(all_reasoning_out_tok), 1),
        mean_critique_out_tokens          =round(_safe_mean(all_critique_out_tok), 1),
        mean_correction_out_tokens        =round(_safe_mean(all_correction_out_tok), 1),
        mean_total_out_tokens_per_rollout =round(_safe_mean(all_rollout_total_tok), 1),
        # ── Accuracy ──────────────────────────────────────────────────────────
        majority_accuracy=round(n_maj / denom, 4),
        max_prm_accuracy =round(n_prm / denom, 4),
        # ── Depth-wise breakdown (JSON only) ─────────────────────────────────
        depth_flops=depth_flops,
    )


# =============================================================================
# Path / name helpers
# =============================================================================

def _infer_dataset(path: str) -> str:
    m = re.search(r"cimr_full_results_(.+?)_N\d", os.path.basename(path))
    return m.group(1) if m else ""


def _infer_suffix(path: str) -> str:
    """Return tag suffix like '_no_critique' / '_no_correction', or ''."""
    base = os.path.basename(path)
    m = re.search(r"_N\d+_d\d+(.*?)\.json$", base)
    return m.group(1) if m else ""


def _infer_model_display(path: str) -> str:
    parts = path.replace("\\", "/").split("/")
    for part in reversed(parts):
        if part.startswith("cimr_full_results_"):
            continue
        if "_mnt" in part or "_t0." in part or "_p0." in part:
            folder_base = part.split("_mnt")[0]
            for prefix, (_, display, _enc) in MODEL_REGISTRY.items():
                if folder_base.startswith(prefix):
                    return display
            return folder_base
    return ""


def _match_model_from_folder(
    folder: str, encoding_override: Optional[str] = None
) -> Optional[Tuple[float, str, str]]:
    """Return (gen_params_b, display_name, encoding) or None."""
    folder_base = folder.split("_mnt")[0].split("_t0.")[0]
    for prefix, (params_b, display, enc) in MODEL_REGISTRY.items():
        if folder_base.startswith(prefix):
            return params_b, display, (encoding_override or enc)
    return None


def _load_store(path: str) -> Optional[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {str(r.get("id", i)): r for i, r in enumerate(data)}
    if isinstance(data, dict):
        return data
    logger.warning(f"Unexpected JSON structure in {path}; skipping.")
    return None


def _write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


# =============================================================================
# CSV export
# =============================================================================

CSV_FIELDS = [
    "model", "dataset", "N", "D", "suffix",
    "total_tflop_per_question",
    "generator_tflop_per_question",
    "verifier_tflop_per_question",
    "phase_a_tflop_per_question",
    "phase_b_tflop_per_question",
    "phase_c_tflop_per_question",
    "bon_baseline_tflop_per_question",
    "effective_bon_n",
    "mean_reasoning_out_tokens",
    "mean_critique_out_tokens",
    "mean_correction_out_tokens",
    "mean_total_out_tokens_per_rollout",
    "total_tflop_all_questions",
    "majority_accuracy",
    "max_prm_accuracy",
    "n_questions",
    "gen_params_b",
    "tiktoken_encoding",
]


def _metrics_to_csv_row(m: Dict[str, Any]) -> Dict[str, Any]:
    return {f: m.get(f if f != "model" else "model_display", "") for f in CSV_FIELDS}


# fix the "model" key (CSV calls it "model", dict stores it as "model_display")
def metrics_to_csv_row(m: Dict[str, Any]) -> Dict[str, Any]:
    row = {f: m.get(f, "") for f in CSV_FIELDS}
    row["model"] = m.get("model_display", "")
    return row


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Compute theoretical FLOPs for CIMR results using tiktoken.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results_dir",
        default="experiments_results/results_paired",
        help="Root folder containing cimr_full_results_*.json files "
             "(default: %(default)s)",
    )
    parser.add_argument(
        "--output_dir",
        default="experiments_results/compute_flops",
        help="Destination folder for outputs (default: %(default)s)",
    )
    parser.add_argument(
        "--encoding",
        default=None,
        help="Override tiktoken encoding for all models "
             "(e.g. cl100k_base, o200k_base). "
             "If omitted, the per-model default from MODEL_REGISTRY is used.",
    )
    parser.add_argument(
        "--verifier_params_b",
        type=float,
        default=VERIFIER_PARAMS_B,
        help=f"Verifier model parameter count in billions "
             f"(default: {VERIFIER_PARAMS_B}B = Qwen2.5-Math-PRM-7B).",
    )
    parser.add_argument(
        "--verifier_encoding",
        default=VERIFIER_ENCODING,
        help=f"tiktoken encoding for the verifier model "
             f"(default: {VERIFIER_ENCODING}).",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Restrict to specific display names, e.g. 'Qwen2.5-Math-7B'. "
             "If omitted, all recognised models are processed.",
    )
    parser.add_argument(
        "--include_no_critique",
        action="store_true",
        help="Also process _no_critique ablation files (excluded by default).",
    )
    args = parser.parse_args()

    if not _TIKTOKEN_AVAILABLE:
        raise SystemExit(
            "ERROR: tiktoken is not installed.\n"
            "Install it with:  uv pip install 'tiktoken==0.7.0'"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    prm_b    = args.verifier_params_b
    prm_enc  = args.verifier_encoding

    # Warm up the encoder(s) so the first file doesn't pay the load cost
    if _TIKTOKEN_AVAILABLE:
        encs_to_warm = {args.encoding} if args.encoding else {
            enc for (_, _, enc) in MODEL_REGISTRY.values()
        }
        encs_to_warm.add(prm_enc)
        for enc_name in encs_to_warm:
            _get_encoder(enc_name)
            logger.info(f"  loaded tiktoken encoder: {enc_name}")

    # ── Walk results tree ────────────────────────────────────────────────────
    per_dir: Dict[str, Dict[str, Any]] = {}
    all_csv_rows: List[Dict[str, Any]] = []
    n_ok = n_err = n_skip = 0

    for root, dirs, files in os.walk(args.results_dir):
        dirs.sort()
        folder = os.path.basename(root)
        model_info = _match_model_from_folder(folder, args.encoding)
        if model_info is None:
            if root != args.results_dir:
                n_skip += 1
            continue

        gen_b, display_name, gen_enc = model_info

        if args.models and display_name not in args.models:
            n_skip += 1
            continue

        for fname in sorted(files):
            if not (fname.startswith("cimr_full_results_") and fname.endswith(".json")):
                continue
            if not args.include_no_critique and "_no_critique" in fname:
                continue

            full_path = os.path.join(root, fname)
            try:
                store = _load_store(full_path)
                if store is None:
                    continue

                metrics = compute_file_flops(
                    store=store,
                    source_file=full_path,
                    gen_params_b=gen_b,
                    prm_params_b=prm_b,
                    gen_encoding=gen_enc,
                    prm_encoding=prm_enc,
                )
                metrics["model_display"] = display_name

                dir_rel = os.path.relpath(root, args.results_dir)
                per_dir.setdefault(dir_rel, {})[fname] = metrics
                all_csv_rows.append(metrics_to_csv_row(metrics))

                logger.info(
                    f"  {display_name:20s}  {metrics['dataset']:18s}"
                    f"  N={metrics['N']:2d} D={metrics['D']:2d}"
                    f"  {metrics['total_tflop_per_question']:8.3f} TFLOPs/q"
                    f"  eff_BoN={metrics['effective_bon_n']:5.1f}"
                    f"  maj_acc={metrics['majority_accuracy']:.3f}"
                    f"  enc={gen_enc}"
                )
                n_ok += 1

            except Exception as e:
                logger.error(f"Failed on {full_path}: {e}", exc_info=True)
                n_err += 1

    # ── Write per-model JSON summaries ──────────────────────────────────────
    for dir_rel, summary in per_dir.items():
        out_path = os.path.join(args.output_dir, dir_rel, "flop_summary.json")
        _write_json(out_path, summary)
        logger.info(f"  written JSON  {os.path.relpath(out_path)}")

    # ── Write master CSV ─────────────────────────────────────────────────────
    all_csv_rows.sort(key=lambda r: (
        str(r["model"]), str(r["dataset"]),
        int(r["N"]) if str(r["N"]).isdigit() else 0,
        int(r["D"]) if str(r["D"]).isdigit() else 0,
        str(r["suffix"]),
    ))

    csv_path = os.path.join(args.output_dir, "compute_flop_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_csv_rows)
    logger.info(
        f"  written CSV   {os.path.relpath(csv_path)}  ({len(all_csv_rows)} rows)"
    )
    logger.info(
        f"\nDone.  {n_ok} files processed, {n_err} errors, {n_skip} folders skipped."
    )

    if all_csv_rows:
        _print_summary_table(all_csv_rows)


def _print_summary_table(rows: List[Dict[str, Any]]) -> None:
    print(
        f"\n{'Model':<22} {'Dataset':<18} {'N':>3} {'D':>3}"
        f" {'TFLOPs/q':>10} {'eff_BoN-N':>10}"
        f" {'maj_acc':>8} {'prm_acc':>8} {'encoding'}"
    )
    print("-" * 100)
    for r in rows:
        suffix_tag = f"[{r['suffix']}]" if r["suffix"] else ""
        try:
            print(
                f"{str(r['model']) + suffix_tag:<22} {str(r['dataset']):<18}"
                f" {int(r['N']):>3} {int(r['D']):>3}"
                f" {float(r['total_tflop_per_question']):>10.3f}"
                f" {float(r['effective_bon_n']):>10.1f}"
                f" {float(r['majority_accuracy']):>8.3f}"
                f" {float(r['max_prm_accuracy']):>8.3f}"
                f"  {r['tiktoken_encoding']}"
            )
        except (ValueError, TypeError):
            pass


if __name__ == "__main__":
    main()
