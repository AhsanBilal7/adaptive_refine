"""
cimr_step.py
============
One depth-level step of the CIMR pipeline.

A single step does four things in sequence:
  1. GENERATE  — produce a reasoning trace (CRISP-style numbered steps).
                 Depth 1 generates from the question alone.
                 Depths 2..d extend the corrected trace of the prior depth.
  2. CRITIQUE  — model reads (PROBLEM, TRACE) and identifies errors.
  3. CORRECT   — model reads (PROBLEM, TRACE, CRITIQUE) and fixes errors,
                 producing a new numbered trace ending in \\boxed{answer}.
  4. SCORE     — PRM-score both the raw reasoning trace and the corrected
                 trace.

The corrected trace is passed forward to the next depth level.

Key fixes vs prior version
--------------------------
  * All prompt builders are imported ONLY from cimr_prompts.py (this file).
    The external prompts_self_critique_correct.py is NOT used, eliminating
    the few-shot example leakage that caused critique/correction to generate
    unrelated problems (polar coordinates, base-b divisibility, etc.).
  * Answer selection: the corrected answer is extracted directly from the
    corrected trace via \\boxed{} extraction.  We fall back to the raw
    reasoning answer only if the corrected trace yields nothing — we do NOT
    use critique_parsed["confirmed_answer"] which was pulling from malformed
    parse fields.
  * Greedy decoding (do_sample=False) for all critique and correction
    generations; sampling only for depth-1 reasoning generation to provide
    rollout diversity.
  * The corrected_trace is accepted as-is if non-empty; otherwise the raw
    reasoning trace is kept (never a blank trace forward).

Public symbols
--------------
  CIMRStepConfig
  run_one_step()     — single rollout, one step (wraps run_step_batch)
  run_step_batch()   — batched N-rollout step (main path)
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
# Answer / trace helpers
# =============================================================================

def extract_boxed(text: str) -> str:
    """Return the content of the LAST \\boxed{...} in text, handling nested braces."""
    if not text:
        return ""
    matches = list(re.finditer(r"\\boxed\{", text))
    if not matches:
        # Last resort: grab the last bare number.
        nums = re.findall(r"-?\d+(?:[.,]\d+)?", text.replace(",", ""))
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
    """Return text up to and including the FIRST \\boxed{...}.

    This strips post-answer rambling so PRM scoring is cleaner.
    """
    if not text:
        return text
    m = re.search(r"\\boxed\{", text)
    if not m:
        return text
    start = m.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[:i] if depth == 0 else text


def extract_answer(text: str, ev=None, data_name: str = "") -> str:
    """Multi-strategy answer extraction.

    Priority: \\boxed{} → evaluator _extract_pred → last number.
    """
    ans = extract_boxed(text)
    if ans:
        return ans
    if ev is not None:
        try:
            from inference_methods import _extract_pred  # type: ignore
            ans = _extract_pred(text, ev, data_name)
            if ans:
                return str(ans)
        except Exception:
            pass
    nums = re.findall(r"-?\d+(?:[.,]\d+)?", (text or "").replace(",", ""))
    return nums[-1] if nums else ""


# =============================================================================
# Chat-template prompt builder
# =============================================================================

def build_chat_prompt(model, system_content: str, user_content: str) -> str:
    """Format a (system, user) turn using the model's chat template.

    Falls back to a plain "system\\n\\nuser" string if no chat template is
    found on the tokenizer.
    """
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
# Low-level generation helpers
# =============================================================================

def _generate_one(
    model,
    full_prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float = 0.7,
    top_p: float = 0.95,
    repetition_penalty: float = 1.05,
    no_repeat_ngram_size: int = 0,
) -> str:
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
    repetition_penalty: float = 1.05,
    no_repeat_ngram_size: int = 0,
) -> List[str]:
    """Batch-generate for a list of prompts, respecting model.batch_size.

    Prompts are chunked into model.batch_size groups. Within each chunk
    the model processes all sequences in parallel.
    """
    if not prompts:
        return []

    kwargs: Dict[str, Any] = dict(max_new_tokens=max_new_tokens)
    if do_sample:
        kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        kwargs["do_sample"] = False
    if repetition_penalty != 1.0:
        kwargs["repetition_penalty"] = repetition_penalty
    if no_repeat_ngram_size > 0:
        kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

    bs = getattr(model, "batch_size", len(prompts)) or len(prompts)
    results: List[str] = []

    for start in range(0, len(prompts), bs):
        chunk = prompts[start: start + bs]
        try:
            out = model._generate_batch_raw(chunk, **kwargs)
            results.extend(out if out else [""] * len(chunk))
        except TypeError:
            # Some model wrappers don't accept repetition_penalty.
            kw2 = {k: v for k, v in kwargs.items()
                   if k not in ("repetition_penalty", "no_repeat_ngram_size")}
            try:
                out = model._generate_batch_raw(chunk, **kw2)
                results.extend(out if out else [""] * len(chunk))
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
    question: str,
    trace: str,
    gold_answer: Optional[str] = None,
) -> float:
    scores = score_traces_batch(verifier, question, [trace], gold_answer)
    return scores[0] if scores else 0.0


def score_traces_batch(
    verifier,
    question: str,
    traces: List[str],
    gold_answer: Optional[str] = None,
) -> List[float]:
    """PRM-score a batch of traces. Empty traces get 0.0 without calling verifier."""
    if not traces:
        return []
    if verifier is None:
        return [0.0] * len(traces)

    valid_idxs = [i for i, t in enumerate(traces) if t and t.strip()]
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
    """All per-step generation knobs for CIMR."""
    # Reasoning generation
    reasoning_max_new_tokens:   int   = 2048
    reasoning_temperature:      float = 0.7
    reasoning_top_p:            float = 0.95
    reasoning_do_sample:        bool  = True   # True for depth-1; greedy later

    # Self-critique
    critique_max_tokens:        int   = 512
    critique_max_trace_chars:   int   = 3000

    # Self-correction
    correction_max_tokens:      int   = 2048
    correction_max_trace_chars: int   = 3000
    correction_max_critique_chars: int = 1000

    # Continuation prompt truncation (prior corrected trace shown to depth k≥2)
    continuation_max_prior_chars: int = 3000

    # Unused legacy flag — kept for CLI compat; not applied in any prompt
    other_answer_aware: bool = False


# =============================================================================
# Single-rollout step (wraps run_step_batch for convenience)
# =============================================================================

def run_one_step(
    *,
    model,
    verifier,
    ev,
    data_name: str,
    question: str,
    gold_answer: Optional[str],
    step_idx: int,
    total_steps: int,
    prior_corrected: Optional[str],
    config: CIMRStepConfig,
    sibling_answers: Optional[List[str]] = None,  # kept for compat; not used
    rollout_idx: int = 0,
) -> Dict[str, Any]:
    records = run_step_batch(
        model=model,
        verifier=verifier,
        ev=ev,
        data_name=data_name,
        question=question,
        gold_answer=gold_answer,
        step_idx=step_idx,
        total_steps=total_steps,
        prior_corrected_list=[prior_corrected],
        config=config,
        rollout_idxs=[rollout_idx],
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
    data_name: str,
    question: str,
    gold_answer: Optional[str],
    step_idx: int,
    total_steps: int,
    prior_corrected_list: List[Optional[str]],
    config: CIMRStepConfig,
    rollout_idxs: List[int],
    sibling_answers_list: Optional[List[Optional[List[str]]]] = None,  # compat; unused
) -> List[Dict[str, Any]]:
    """Execute one depth level for N rollouts simultaneously.

    All prompts are built self-contained from cimr_prompts — no external
    few-shot prompt files are consulted, preventing content leakage.

    Parameters
    ----------
    prior_corrected_list : corrected trace from depth k-1 per rollout.
                           None for depth 1 (seed generation).
    rollout_idxs         : 0-based rollout ids, one per entry in
                           prior_corrected_list.

    Returns
    -------
    One step-record dict per rollout, in the same order as rollout_idxs.
    """
    # Import only our own, leakage-free prompt builders.
    from .cimr_prompts import (  # type: ignore
        REASONING_SYSTEM_PROMPT,
        CRITIQUE_SYSTEM_PROMPT,
        CORRECTION_SYSTEM_PROMPT,
        build_initial_reasoning_prompt,
        build_continuation_reasoning_prompt,
        build_self_critique_prompt,
        build_self_correction_prompt,
    )

    n = len(rollout_idxs)
    is_first = (step_idx == 1)
    t0 = time.time()

    # ── Phase A: reasoning generation ────────────────────────────────────
    logger.debug(
        f"[CIMR step] depth={step_idx}/{total_steps}  "
        f"is_first={is_first}  batch_size={n}"
    )

    if is_first:
        # All rollouts get the same user message; diversity comes from
        # temperature-based sampling.
        user_msgs = [build_initial_reasoning_prompt(question)] * n
    else:
        user_msgs = [
            build_continuation_reasoning_prompt(
                question=question,
                prior_corrected=prior_corrected_list[k] or "",
                step_idx=step_idx,
                total_steps=total_steps,
                max_prior_chars=config.continuation_max_prior_chars,
            )
            for k in range(n)
        ]

    reasoning_prompts = [
        build_chat_prompt(model, REASONING_SYSTEM_PROMPT, u)
        for u in user_msgs
    ]

    # Use sampling for depth-1 (provides N-way diversity); greedy after.
    do_sample_reasoning = config.reasoning_do_sample and is_first

    reasoning_raws = _generate_batch(
        model, reasoning_prompts,
        max_new_tokens=config.reasoning_max_new_tokens,
        do_sample=do_sample_reasoning,
        temperature=config.reasoning_temperature,
        top_p=config.reasoning_top_p,
    )

    # Truncate at first \\boxed{} to strip post-answer rambling.
    reasoning_cleans = [truncate_at_first_boxed(r or "") for r in reasoning_raws]
    reasoning_answers = [extract_answer(r, ev, data_name) for r in reasoning_cleans]
    reasoning_prms = score_traces_batch(verifier, question, reasoning_cleans, gold_answer)

    logger.debug(
        f"[CIMR step] depth={step_idx} reasoning done.  "
        f"answers={reasoning_answers}"
    )

    # ── Phase B: self-critique (greedy — deterministic diagnosis) ─────────
    critique_user_msgs = [
        build_self_critique_prompt(
            question=question,
            trace_text=reasoning_cleans[k],
            trace_answer=reasoning_answers[k],
            max_trace_chars=config.critique_max_trace_chars,
        )
        for k in range(n)
    ]

    critique_prompts = [
        build_chat_prompt(model, CRITIQUE_SYSTEM_PROMPT, u)
        for u in critique_user_msgs
    ]

    critique_raws = _generate_batch(
        model, critique_prompts,
        max_new_tokens=config.critique_max_tokens,
        do_sample=False,
    )

    logger.debug(f"[CIMR step] depth={step_idx} critique done.")

    # ── Phase C: self-correction (greedy) ─────────────────────────────────
    correction_user_msgs = [
        build_self_correction_prompt(
            question=question,
            trace_text=reasoning_cleans[k],
            critique_text=critique_raws[k] or "",
            trace_answer=reasoning_answers[k],
            max_trace_chars=config.correction_max_trace_chars,
            max_critique_chars=config.correction_max_critique_chars,
        )
        for k in range(n)
    ]

    correction_prompts = [
        build_chat_prompt(model, CORRECTION_SYSTEM_PROMPT, u)
        for u in correction_user_msgs
    ]

    correction_raws = _generate_batch(
        model, correction_prompts,
        max_new_tokens=config.correction_max_tokens,
        do_sample=False,
    )

    # Truncate corrections at first \\boxed{} as well.
    correction_cleans = [truncate_at_first_boxed(r or "") for r in correction_raws]

    # ── Phase D: pick corrected trace and extract answer ──────────────────
    # Rule: use the corrected trace if it is non-empty; otherwise keep the
    # raw reasoning trace.  Never forward a blank trace.
    corrected_traces: List[str] = []
    corrected_answers: List[str] = []

    for k in range(n):
        corr = correction_cleans[k].strip()
        if corr:
            corrected_traces.append(corr)
            # Extract answer from the corrected output directly.
            ans = extract_answer(corr, ev, data_name)
            # Fall back to raw reasoning answer if extraction fails.
            corrected_answers.append(ans if ans else reasoning_answers[k])
        else:
            # Correction produced nothing — keep the raw trace.
            corrected_traces.append(reasoning_cleans[k])
            corrected_answers.append(reasoning_answers[k])

    logger.debug(
        f"[CIMR step] depth={step_idx} corrected answers={corrected_answers}"
    )

    # ── Phase E: PRM-score corrected traces ───────────────────────────────
    corrected_prms = score_traces_batch(verifier, question, corrected_traces, gold_answer)

    elapsed = time.time() - t0
    per_rollout_time = round(elapsed / max(n, 1), 3)

    return [
        dict(
            rollout_idx=int(rollout_idxs[k]),
            step_idx=int(step_idx),
            total_steps=int(total_steps),
            # Phase A
            reasoning_response=reasoning_cleans[k],
            reasoning_answer=reasoning_answers[k],
            reasoning_prm_score=float(reasoning_prms[k]),
            # Phase B
            critique_response=critique_raws[k],
            # Phase C
            correction_response=correction_cleans[k],
            # Phase D / E
            corrected_trace=corrected_traces[k],
            corrected_answer=corrected_answers[k],
            corrected_prm_score=float(corrected_prms[k]),
            elapsed_sec=per_rollout_time,
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
