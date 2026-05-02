"""
cimr_step.py
============
One step of the CIMR pipeline, with both single-rollout and batched
N-rollout execution.

A "step" (one depth level) is:
  1. GENERATE  — produce a reasoning trace (CRISP-style numbered steps).
                 Step 1 generates from the question alone.
                 Steps 2..d extend the corrected output of the prior step.
  2. CRITIQUE  — self-critique agent reads (PROBLEM, TRACE) and emits
                 a short error diagnosis (or confirms correctness).
  3. CORRECT   — self-correction agent reads (PROBLEM, TRACE, CRITIQUE)
                 and emits a corrected trace ending in Final answer: \\boxed{}.
  4. SCORE     — corrected trace is PRM-scored → V_step.

The corrected trace is passed forward to the next depth level.

Public symbols
--------------
  CIMRStepConfig   — dataclass of step-level knobs.
  run_one_step()   — single-rollout step (kept for resume compatibility).
  run_step_batch() — batched N-rollout step (main path for new runs).
  extract_answer, extract_boxed, truncate_at_first_boxed
  build_chat_prompt, score_one_trace, score_traces_batch
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers — boxed answer, post-correction trim
# =============================================================================

_BOXED_OPEN_RE = re.compile(r"\\boxed\{")


def extract_boxed(text: str) -> str:
    """Return content of the LAST \\boxed{...}, handling nested braces."""
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
    return text[start : i - 1].strip() if depth == 0 else text[start:].strip()


def truncate_at_first_boxed(text: str) -> str:
    """Return substring up to and including the FIRST \\boxed{...}.
    Removes post-answer rambling for cleaner PRM scoring."""
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
    """Multi-strategy answer extraction.  Priority: \\boxed{} → evaluator → last number."""
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
# Chat-template prompt builder
# =============================================================================

def build_chat_prompt(model, system_content: str, user_content: str) -> str:
    """Format a (system, user) prompt using the model's chat template."""
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
# Generation helpers
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
    """Run the model on ONE prompt and return the raw response text."""
    if not full_prompt:
        return ""
    results = _generate_batch(
        model, [full_prompt],
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )
    return results[0] if results else ""


def _generate_batch(
    model,
    prompts: List[str],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float = 0.7,
    top_p: float = 0.95,
    repetition_penalty: float = 1.10,
    no_repeat_ngram_size: int = 12,
) -> List[str]:
    """Batch-generate for a list of prompts, respecting model.batch_size.

    Prompts are chunked into model.batch_size groups and sent in separate
    _generate_batch_raw calls so GPU memory is not exhausted.  Within each
    chunk the model processes all sequences in parallel.
    """
    if not prompts:
        return []

    kwargs: Dict[str, Any] = dict(max_new_tokens=max_new_tokens)
    if do_sample:
        kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        kwargs.update(do_sample=False)
    kwargs["repetition_penalty"]   = repetition_penalty
    kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

    bs = getattr(model, "batch_size", len(prompts)) or len(prompts)
    results: List[str] = []

    for start in range(0, len(prompts), bs):
        chunk = prompts[start : start + bs]
        try:
            out = model._generate_batch_raw(chunk, **kwargs)
            results.extend(out or [""] * len(chunk))
        except TypeError:
            kw2 = {
                k: v for k, v in kwargs.items()
                if k not in ("repetition_penalty", "no_repeat_ngram_size")
            }
            try:
                out = model._generate_batch_raw(chunk, **kw2)
                results.extend(out or [""] * len(chunk))
            except Exception as e:
                logger.error(f"[CIMR] batch generation failed: {e}")
                results.extend([""] * len(chunk))
        except Exception as e:
            logger.error(f"[CIMR] batch generation failed: {e}")
            results.extend([""] * len(chunk))

    return results


# =============================================================================
# PRM scoring helpers
# =============================================================================

def score_one_trace(
    verifier,
    question:    str,
    trace:       str,
    gold_answer: Optional[str] = None,
) -> float:
    """PRM-score a single trace.  Returns 0.0 on failure."""
    scores = score_traces_batch(verifier, question, [trace], gold_answer)
    return scores[0] if scores else 0.0


def score_traces_batch(
    verifier,
    question:    str,
    traces:      List[str],
    gold_answer: Optional[str] = None,
) -> List[float]:
    """PRM-score a batch of traces in one verifier call.

    Empty/whitespace traces receive 0.0 without hitting the verifier.
    """
    if not traces:
        return []
    if verifier is None:
        return [0.0] * len(traces)

    valid_idxs   = [i for i, t in enumerate(traces) if t and t.strip()]
    valid_traces = [traces[i] for i in valid_idxs]

    scores = [0.0] * len(traces)
    if not valid_traces:
        return scores

    try:
        from verifier import score_candidates  # type: ignore
        raw = score_candidates(valid_traces, verifier, gold_answer, question=question)
        flat = np.asarray(raw, dtype=float).reshape(-1).tolist()
        for out_i, in_i in enumerate(valid_idxs):
            scores[in_i] = float(flat[out_i]) if out_i < len(flat) else 0.0
    except Exception as e:
        logger.warning(f"[CIMR] batch PRM scoring failed: {e}")

    return scores


# =============================================================================
# Step config
# =============================================================================

@dataclass
class CIMRStepConfig:
    """Step-level knobs, shared across all d steps of a rollout."""
    # Reasoning generation
    reasoning_max_new_tokens:      int   = 2048
    reasoning_temperature:         float = 0.7
    reasoning_top_p:               float = 0.95
    reasoning_do_sample:           bool  = True   # True for step 1; greedy for later
    # Critique
    critique_max_tokens:           int   = 512
    critique_max_trace_chars:      int   = 4000
    # Correction
    correction_max_tokens:         int   = 1536
    correction_max_trace_chars:    int   = 4000
    correction_max_critique_chars: int   = 1500
    # Continuation prompt truncation
    continuation_max_prior_chars:  int   = 4000
    # Sibling-answer-aware critique/correction
    other_answer_aware:            bool  = False


# =============================================================================
# Single-rollout step (kept for resume / compatibility)
# =============================================================================

def run_one_step(
    *,
    model,
    verifier,
    ev,
    data_name:       str,
    question:        str,
    gold_answer:     Optional[str],
    step_idx:        int,
    total_steps:     int,
    prior_corrected: Optional[str],
    config:          CIMRStepConfig,
    sibling_answers: Optional[List[str]] = None,
    rollout_idx:     int = 0,
) -> Dict[str, Any]:
    """Execute one CIMR step for a single rollout."""
    records = run_step_batch(
        model                 = model,
        verifier              = verifier,
        ev                    = ev,
        data_name             = data_name,
        question              = question,
        gold_answer           = gold_answer,
        step_idx              = step_idx,
        total_steps           = total_steps,
        prior_corrected_list  = [prior_corrected],
        config                = config,
        rollout_idxs          = [rollout_idx],
        sibling_answers_list  = [sibling_answers] if sibling_answers else None,
    )
    return records[0]


# =============================================================================
# Batched N-rollout step — main path
# =============================================================================

def run_step_batch(
    *,
    model,
    verifier,
    ev,
    data_name:            str,
    question:             str,
    gold_answer:          Optional[str],
    step_idx:             int,
    total_steps:          int,
    prior_corrected_list: List[Optional[str]],
    config:               CIMRStepConfig,
    rollout_idxs:         List[int],
    sibling_answers_list: Optional[List[Optional[List[str]]]] = None,
) -> List[Dict[str, Any]]:
    """Execute one depth level for N rollouts simultaneously.

    Parameters
    ----------
    prior_corrected_list : one entry per rollout — corrected trace from
                           the previous depth (None for depth 1).
    rollout_idxs         : 0-indexed rollout ids, one per entry in
                           prior_corrected_list.
    sibling_answers_list : optional per-rollout list of sibling answers for
                           other-answer-aware critique.

    Returns
    -------
    List of step-record dicts, one per rollout, in the same order as
    rollout_idxs.
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

    n = len(rollout_idxs)
    is_first = (step_idx == 1)
    t0 = time.time()

    # ── Phase A: reasoning generation ────────────────────────────────────
    if is_first:
        user_msgs = [build_initial_reasoning_prompt(question)] * n
    else:
        user_msgs = [
            build_continuation_reasoning_prompt(
                question        = question,
                prior_corrected = prior_corrected_list[k] or "",
                step_idx        = step_idx,
                total_steps     = total_steps,
                max_prior_chars = config.continuation_max_prior_chars,
            )
            for k in range(n)
        ]

    reasoning_prompts = [
        build_chat_prompt(model, REASONING_SYSTEM_PROMPT, u)
        for u in user_msgs
    ]

    reasoning_raws = _generate_batch(
        model, reasoning_prompts,
        max_new_tokens = config.reasoning_max_new_tokens,
        do_sample      = (config.reasoning_do_sample and is_first),
        temperature    = config.reasoning_temperature,
        top_p          = config.reasoning_top_p,
    )

    reasoning_cleans  = [truncate_at_first_boxed(r or "") for r in reasoning_raws]
    reasoning_answers = [extract_answer(r, ev, data_name) for r in reasoning_cleans]
    reasoning_prms    = score_traces_batch(verifier, question, reasoning_cleans, gold_answer)

    # ── Phase B: self-critique ────────────────────────────────────────────
    critique_user_msgs = []
    for k in range(n):
        siblings = (
            sibling_answers_list[k]
            if (config.other_answer_aware and sibling_answers_list is not None)
            else None
        )
        critique_user_msgs.append(build_self_critique_prompt(
            question        = question,
            trace_text      = reasoning_cleans[k],
            trace_answer    = reasoning_answers[k],
            max_trace_chars = config.critique_max_trace_chars,
            other_answers   = siblings,
        ))

    critique_prompts = [
        build_chat_prompt(model, SELF_CRITIQUE_SYSTEM_PROMPT, u)
        for u in critique_user_msgs
    ]

    critique_raws    = _generate_batch(
        model, critique_prompts,
        max_new_tokens = config.critique_max_tokens,
        do_sample      = False,
    )
    critique_parseds = [parse_self_critique(r) for r in critique_raws]

    # ── Phase C: self-correction ──────────────────────────────────────────
    correction_user_msgs = []
    for k in range(n):
        crit_text = (critique_parseds[k].get("raw") or critique_raws[k] or "")
        siblings = (
            sibling_answers_list[k]
            if (config.other_answer_aware and sibling_answers_list is not None)
            else None
        )
        correction_user_msgs.append(build_self_correction_prompt(
            question        = question,
            trace_text      = reasoning_cleans[k],
            critique_text   = crit_text,
            trace_answer    = reasoning_answers[k],
            max_trace_chars = config.correction_max_trace_chars,
            max_crit_chars  = config.correction_max_critique_chars,
            other_answers   = siblings,
        ))

    correction_prompts = [
        build_chat_prompt(model, SELF_CORRECTION_SYSTEM_PROMPT, u)
        for u in correction_user_msgs
    ]

    correction_raws    = _generate_batch(
        model, correction_prompts,
        max_new_tokens = config.correction_max_tokens,
        do_sample      = False,
    )
    correction_cleans  = [truncate_at_first_boxed(r or "") for r in correction_raws]
    correction_parseds = [parse_self_correction(r) for r in correction_cleans]

    # ── Phase D: pick corrected trace and answer ──────────────────────────
    corrected_traces  = []
    corrected_answers = []
    for k in range(n):
        corrected_trace = (
            correction_cleans[k]
            if correction_cleans[k].strip()
            else reasoning_cleans[k]
        )
        corrected_traces.append(corrected_trace)

        cor_ans  = (correction_parseds[k].get("corrected_answer") or "").strip()
        conf_ans = (critique_parseds[k].get("confirmed_answer")   or "").strip()
        corrected_answers.append(cor_ans or conf_ans or reasoning_answers[k])

    # ── Phase E: PRM-score corrected traces ───────────────────────────────
    corrected_prms = score_traces_batch(verifier, question, corrected_traces, gold_answer)

    elapsed          = time.time() - t0
    per_rollout_time = round(elapsed / max(n, 1), 3)

    return [
        dict(
            rollout_idx          = int(rollout_idxs[k]),
            step_idx             = int(step_idx),
            total_steps          = int(total_steps),
            # Reasoning
            reasoning_response   = reasoning_cleans[k],
            reasoning_answer     = reasoning_answers[k],
            reasoning_prm_score  = float(reasoning_prms[k]),
            # Critique
            critique_response    = critique_raws[k],
            critique_parsed      = critique_parseds[k],
            # Correction
            correction_response  = correction_cleans[k],
            correction_parsed    = correction_parseds[k],
            # Forward-passed artifacts
            corrected_trace      = corrected_traces[k],
            corrected_answer     = corrected_answers[k],
            corrected_prm_score  = float(corrected_prms[k]),
            elapsed_sec          = per_rollout_time,
        )
        for k in range(n)
    ]


__all__ = [
    "CIMRStepConfig",
    "run_one_step",
    "run_step_batch",
    "extract_answer",
    "extract_boxed",
    "truncate_at_first_boxed",
    "build_chat_prompt",
    "score_one_trace",
    "score_traces_batch",
    "_generate_batch",
]
