"""
cimr_step.py
============
One step of the CIMR pipeline.

A "step" is a self-contained unit of work:

  1. GENERATE  — produce a reasoning trace.
                 Step 1 generates from the question alone.
                 Steps 2..d generate given the question PLUS the
                 corrected output of the previous step.
  2. CRITIQUE  — the self-critique agent reads (PROBLEM, TRACE) and
                 emits a short paragraph naming the error TYPE / LOCATION
                 (or confirming correctness).  Does NOT re-solve.
  3. CORRECT   — the self-correction agent reads
                 (PROBLEM, TRACE, CRITIQUE) and emits a CORRECTED TRACE
                 ending in ``Final answer: \\boxed{<answer>}``.
  4. SCORE     — the corrected trace is PRM-scored (Qwen2.5-Math-PRM-7B)
                 → V_step.

The corrected trace is what gets passed forward to the next step (or
returned as the rollout's final answer if this is the last step).

This module does **one step**.  It is stateless w.r.t. the rollout —
the rollout layer (``cimr_rollout.py``) threads corrected outputs from
step k into step k+1.

Public symbols
--------------
  CIMRStepConfig   — dataclass-ish bag of step-level knobs.
  run_one_step()   — the main function; returns a CIMRStepResult dict.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers (boxed-answer extraction, post-correction trim)
# =============================================================================

_BOXED_OPEN_RE = re.compile(r"\\boxed\{")


def extract_boxed(text: str) -> str:
    """Return the content of the LAST ``\\boxed{...}`` in `text`,
    handling nested braces.  Falls back to the last number/expression in
    the text when no ``\\boxed{}`` is found."""
    if not text:
        return ""
    matches = list(re.finditer(r"\\boxed\{", text))
    if not matches:
        nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
        return nums[-1] if nums else ""
    start = matches[-1].end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start: i - 1].strip() if depth == 0 else text[start:].strip()


def truncate_at_first_boxed(text: str) -> str:
    """If `text` contains ``\\boxed{...}``, return the substring up to
    and including the FIRST such expression.  Removes any post-answer
    rambling, which both saves tokens downstream and makes PRM scoring
    cleaner."""
    if not text:
        return text
    m = _BOXED_OPEN_RE.search(text)
    if not m:
        return text
    start = m.end()
    depth = 1
    i = start
    n = len(text)
    while i < n and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return text
    return text[:i]


def extract_answer(text: str, ev=None, data_name: str = "") -> str:
    """Multi-strategy answer extraction.  Priority: \\boxed{} →
    evaluator's own → last numeric token."""
    ans = extract_boxed(text)
    if ans:
        return ans
    if ev is not None:
        try:
            from inference_methods import _extract_pred  # type: ignore
            ans = _extract_pred(text, ev, data_name)
            if ans:
                return ans
        except Exception:
            pass
    nums = re.findall(r"-?\d+(?:\.\d+)?", (text or "").replace(",", ""))
    return nums[-1] if nums else ""


# =============================================================================
# Chat-template prompt builder (works with HFModelWrapper-like models)
# =============================================================================

def build_chat_prompt(
    model, system_content: str, user_content: str,
) -> str:
    """Format a (system, user) prompt using the model's chat template.
    Falls back to plain concatenation if the template is unavailable."""
    tok = getattr(model, "tokenizer", None)
    if tok is not None and getattr(tok, "chat_template", None):
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user",   "content": user_content},
        ]
        try:
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception as e:
            logger.warning(f"apply_chat_template failed ({e}); falling back.")
    return f"{system_content}\n\n{user_content}"


# =============================================================================
# Generic generation helper (single prompt, no batching)
# =============================================================================

def _generate_one(
    model,
    full_prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float = 0.7,
    top_p: float = 0.95,
    repetition_penalty: float = 1.10,
    no_repeat_ngram_size: int = 12,
) -> str:
    """Run the model on ONE prompt and return the raw response text.

    Most generator wrappers in this project expose ``_generate_batch_raw``
    as the lowest-level call.  We pass a single-element batch and unpack.
    """
    if not full_prompt:
        return ""
    kwargs: Dict[str, Any] = dict(max_new_tokens=max_new_tokens)
    if do_sample:
        kwargs.update(
            do_sample=True, temperature=temperature, top_p=top_p,
        )
    else:
        kwargs.update(do_sample=False)
    kwargs["repetition_penalty"]  = repetition_penalty
    kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

    try:
        out = model._generate_batch_raw([full_prompt], **kwargs)
    except TypeError:
        # Wrapper rejected one of our kwargs — strip the niche knobs
        # and retry.
        kwargs.pop("repetition_penalty", None)
        kwargs.pop("no_repeat_ngram_size", None)
        try:
            out = model._generate_batch_raw([full_prompt], **kwargs)
        except Exception as e:
            logger.error(f"[CIMR] generation failed: {e}")
            return ""
    except Exception as e:
        logger.error(f"[CIMR] generation failed: {e}")
        return ""
    if not out:
        return ""
    return out[0] or ""


# =============================================================================
# PRM scoring helper for a single trace
# =============================================================================

def score_one_trace(
    verifier,
    question: str,
    trace:    str,
    gold_answer: Optional[str] = None,
) -> float:
    """PRM-score a single trace.  Returns 0.0 on any failure or when the
    verifier is unavailable."""
    if verifier is None or not trace or not trace.strip():
        return 0.0
    try:
        from verifier import score_candidates  # type: ignore
        scored = score_candidates(
            [trace], verifier, gold_answer, question=question,
        )
        return float(np.asarray(scored, dtype=float).reshape(-1)[0])
    except Exception as e:
        logger.warning(f"[CIMR] PRM scoring failed: {e}")
        return 0.0


# =============================================================================
# Step config / result containers
# =============================================================================

@dataclass
class CIMRStepConfig:
    """Step-level knobs.  Created once per rollout and reused across
    all d steps."""
    # Generation
    reasoning_max_new_tokens:  int = 2048
    reasoning_temperature:     float = 0.7
    reasoning_top_p:           float = 0.95
    reasoning_do_sample:       bool = True   # step 1 samples; later steps usually greedy
    # Critique
    critique_max_tokens:       int = 384
    critique_max_trace_chars:  int = 4000
    # Correction
    correction_max_tokens:     int = 1024
    correction_max_trace_chars:    int = 4000
    correction_max_critique_chars: int = 1500
    # Generation continuation prompt
    continuation_max_prior_chars: int = 4000
    # Other-answer-aware critique/correction (sibling rollouts)
    other_answer_aware:        bool = False


# =============================================================================
# THE STEP
# =============================================================================

def run_one_step(
    *,
    model,
    verifier,
    ev,
    data_name:        str,
    question:         str,
    gold_answer:      Optional[str],
    step_idx:         int,                 # 1-indexed
    total_steps:      int,                 # d
    prior_corrected:  Optional[str],       # corrected trace from step k-1
    config:           CIMRStepConfig,
    sibling_answers:  Optional[List[str]] = None,
    rollout_idx:      int = 0,
) -> Dict[str, Any]:
    """Execute one CIMR step.

    Returns a dict with every artifact from this step:
        reasoning_response          (raw LLM gen)
        reasoning_answer            (extracted answer from the gen)
        reasoning_prm_score         (PRM on the raw gen — diagnostic)
        critique_response           (raw critique agent text)
        critique_parsed             (parse_self_critique output)
        correction_response         (raw correction text, trimmed)
        correction_parsed           (parse_self_correction output)
        corrected_trace             (= correction_response — what gets
                                     passed forward to step k+1)
        corrected_answer            (extracted from the correction)
        corrected_prm_score         (PRM on the corrected trace)
        step_idx, total_steps, elapsed_sec
    """
    from cimr_prompts import (  # type: ignore
        REASONING_SYSTEM_PROMPT,
        build_initial_reasoning_prompt,
        build_continuation_reasoning_prompt,
    )
    from prompts_self_critique_correct import (  # type: ignore
        SELF_CRITIQUE_SYSTEM_PROMPT,
        SELF_CORRECTION_SYSTEM_PROMPT,
        build_self_critique_prompt,
        build_self_correction_prompt,
        parse_self_critique,
        parse_self_correction,
    )

    t0 = time.time()
    is_first = (step_idx == 1)

    # ── Phase A: generate the reasoning trace for this step ────────────
    if is_first:
        user_msg = build_initial_reasoning_prompt(question)
    else:
        user_msg = build_continuation_reasoning_prompt(
            question        = question,
            prior_corrected = prior_corrected or "",
            step_idx        = step_idx,
            total_steps     = total_steps,
            max_prior_chars = config.continuation_max_prior_chars,
        )
    reasoning_prompt = build_chat_prompt(
        model, REASONING_SYSTEM_PROMPT, user_msg,
    )
    reasoning_raw = _generate_one(
        model, reasoning_prompt,
        max_new_tokens=config.reasoning_max_new_tokens,
        do_sample=(config.reasoning_do_sample and is_first),
        temperature=config.reasoning_temperature,
        top_p=config.reasoning_top_p,
    )
    # Trim post-\boxed rambling so the critique sees clean text.
    reasoning_clean = truncate_at_first_boxed(reasoning_raw or "")
    reasoning_answer = extract_answer(reasoning_clean, ev, data_name)
    reasoning_prm = score_one_trace(
        verifier, question, reasoning_clean, gold_answer,
    )

    # ── Phase B: self-critique ─────────────────────────────────────────
    sibling = (
        sibling_answers
        if (config.other_answer_aware and sibling_answers is not None)
        else None
    )
    try:
        critique_user = build_self_critique_prompt(
            question        = question,
            trace_text      = reasoning_clean,
            trace_answer    = reasoning_answer,
            max_trace_chars = config.critique_max_trace_chars,
            other_answers   = sibling,
        )
        critique_prompt = build_chat_prompt(
            model, SELF_CRITIQUE_SYSTEM_PROMPT, critique_user,
        )
    except Exception as e:
        logger.error(f"[CIMR step {step_idx}] critique prompt build failed: {e}")
        critique_prompt = ""
    critique_raw = _generate_one(
        model, critique_prompt,
        max_new_tokens=config.critique_max_tokens,
        do_sample=False,
    )
    critique_parsed = parse_self_critique(critique_raw)

    # ── Phase C: self-correction ───────────────────────────────────────
    crit_text_for_corr = (
        critique_parsed.get("raw") or critique_raw or ""
    )
    try:
        correction_user = build_self_correction_prompt(
            question        = question,
            trace_text      = reasoning_clean,
            critique_text   = crit_text_for_corr,
            trace_answer    = reasoning_answer,
            max_trace_chars = config.correction_max_trace_chars,
            max_crit_chars  = config.correction_max_critique_chars,
            other_answers   = sibling,
        )
        correction_prompt = build_chat_prompt(
            model, SELF_CORRECTION_SYSTEM_PROMPT, correction_user,
        )
    except Exception as e:
        logger.error(f"[CIMR step {step_idx}] correction prompt build failed: {e}")
        correction_prompt = ""
    correction_raw = _generate_one(
        model, correction_prompt,
        max_new_tokens=config.correction_max_tokens,
        do_sample=False,
    )
    correction_clean = truncate_at_first_boxed(correction_raw or "")
    correction_parsed = parse_self_correction(correction_clean)

    # ── Phase D: pick the corrected trace + corrected answer ───────────
    # The corrected trace is what gets passed forward to step k+1.  We
    # keep it as the cleaned correction text (when non-empty), and fall
    # back to the reasoning trace itself when correction is empty (so
    # the next step has SOMETHING to extend).  The corrected answer
    # prefers the correction agent's \boxed{}, then the critique's
    # confirmed \boxed{}, then the reasoning's \boxed{}.
    if correction_clean.strip():
        corrected_trace = correction_clean
    else:
        corrected_trace = reasoning_clean

    cor_ans  = (correction_parsed.get("corrected_answer") or "").strip()
    conf_ans = (critique_parsed.get("confirmed_answer") or "").strip()
    if cor_ans:
        corrected_answer = cor_ans
    elif conf_ans:
        corrected_answer = conf_ans
    else:
        corrected_answer = reasoning_answer

    # ── Phase E: PRM-score the corrected trace ─────────────────────────
    corrected_prm = score_one_trace(
        verifier, question, corrected_trace, gold_answer,
    )

    elapsed = time.time() - t0

    return dict(
        rollout_idx          = int(rollout_idx),
        step_idx             = int(step_idx),
        total_steps          = int(total_steps),
        # Reasoning generation
        reasoning_response   = reasoning_clean,
        reasoning_answer     = reasoning_answer,
        reasoning_prm_score  = float(reasoning_prm),
        # Self-critique
        critique_response    = critique_raw,
        critique_parsed      = critique_parsed,
        # Self-correction
        correction_response  = correction_clean,
        correction_parsed    = correction_parsed,
        # Forward-passed artifacts (used by step k+1)
        corrected_trace      = corrected_trace,
        corrected_answer     = corrected_answer,
        corrected_prm_score  = float(corrected_prm),
        elapsed_sec          = round(elapsed, 3),
    )


__all__ = [
    "CIMRStepConfig",
    "run_one_step",
    "extract_answer",
    "extract_boxed",
    "truncate_at_first_boxed",
    "build_chat_prompt",
    "score_one_trace",
]
