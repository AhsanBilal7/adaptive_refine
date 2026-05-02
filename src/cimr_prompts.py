"""
cimr_prompts.py
===============
Prompt builders for step-wise reasoning generation in the CIMR pipeline.

The CIMR pipeline runs d sequential reasoning steps per rollout.
Step 1 generates an initial reasoning trace from the question alone.
Steps 2..d generate a continuation given the question PLUS the corrected
output of the previous step.  Between every generation, we run the
self-critique → self-correction agents from
``prompts_self_critique_correct.py`` (which is left untouched and reused
as-is).

This module provides:

  build_initial_reasoning_prompt(question)
      The user-side prompt for step 1's reasoning generation.
  build_continuation_reasoning_prompt(question, prior_corrected, step_idx, total_steps)
      The user-side prompt for steps 2..d, given the corrected output of
      the previous step.  The model is asked to extend / refine the
      reasoning, not start over, and to produce a final answer at the
      last step.
  REASONING_SYSTEM_PROMPT
      The system message used for both initial and continuation
      generations (kept the same across steps so the model's role is
      stable).

Design notes
------------
The continuation prompt is deliberately PERMISSIVE about what the model
does next:
  - if the prior corrected step already contains a final answer, the
    model is allowed to confirm/refine it
  - otherwise the model extends the reasoning toward a final answer
This makes the d-step pipeline degenerate gracefully: even if the
self-correction at step k already produced a clean ``\\boxed{...}``,
running steps k+1..d cannot HURT (the next step has every incentive to
keep the same answer); but if step k's correction was wrong, the next
step has another chance to fix it.

Few-shot demos are intentionally *not* embedded here — the
``prompts_self_critique_correct.py`` few-shots already cover the
critique / correction format, and adding generation few-shots tends to
push the model toward template repetition.  The base reasoning model
(e.g. Qwen2.5-Math-7B-Instruct) already produces well-formatted math
solutions zero-shot.
"""

from __future__ import annotations

from typing import Optional

REASONING_SYSTEM_PROMPT = (
    "You are a careful mathematical reasoner.  Solve the given problem "
    "step-by-step, showing your work clearly.  At the very end of your "
    "response, write the final answer on its own line in the form:\n"
    "Final answer: \\boxed{<answer>}"
)


def build_initial_reasoning_prompt(question: str) -> str:
    """User message for step 1's reasoning generation (the seed step)."""
    q = (question or "").strip()
    return (
        "Solve the following problem step-by-step.\n\n"
        "PROBLEM\n"
        f"{q}\n\n"
        "Show every intermediate calculation, and end your response with "
        "exactly one line of the form:\n"
        "Final answer: \\boxed{<answer>}"
    )


def build_continuation_reasoning_prompt(
    question:        str,
    prior_corrected: str,
    step_idx:        int,
    total_steps:     int,
    max_prior_chars: int = 4000,
) -> str:
    """User message for steps 2..d.

    Receives:
      question        : the original problem.
      prior_corrected : the CORRECTED OUTPUT from the previous step
                        (i.e. the output of self-correction at step k-1).
      step_idx        : the 1-indexed current step (>=2).
      total_steps     : d, the total number of steps in this rollout.
      max_prior_chars : head/tail truncation budget for the prior trace.
    """
    q = (question or "").strip()
    prior = (prior_corrected or "").strip()

    # Head/tail truncate the prior corrected trace to keep the prompt
    # within the model's context budget.
    if max_prior_chars > 0 and len(prior) > max_prior_chars:
        keep = max_prior_chars // 2
        head = prior[:keep].rstrip()
        tail = prior[-keep:].lstrip()
        omit = len(prior) - max_prior_chars
        prior = f"{head}\n\n[... {omit:,} chars omitted ...]\n\n{tail}"

    is_last = (step_idx >= total_steps)

    instructions = (
        "Below is the original PROBLEM and the CORRECTED REASONING from "
        f"step {step_idx - 1} of a {total_steps}-step refinement "
        "pipeline.  You are step "
        f"{step_idx}{' (the FINAL step)' if is_last else ''}.\n\n"
        "Your job:\n"
        "  - Read the CORRECTED REASONING carefully.\n"
        "  - Extend or refine it: verify each claim, fill in missing "
        "steps, fix any subtle issue you notice.\n"
        "  - If the prior reasoning already reaches a confident final "
        "answer, restate it (with a short justification) — do NOT "
        "introduce a different answer unless you can identify a concrete "
        "error.\n"
        "  - End your response with exactly one line of the form:\n"
        "    Final answer: \\boxed{<answer>}"
    )

    return (
        f"{instructions}\n\n"
        "PROBLEM\n"
        f"{q}\n\n"
        f"CORRECTED REASONING (from step {step_idx - 1})\n"
        f"{prior if prior else '(empty)'}\n\n"
        f"Now produce step {step_idx}'s reasoning, ending with "
        "`Final answer: \\boxed{<answer>}`."
    )


__all__ = [
    "REASONING_SYSTEM_PROMPT",
    "build_initial_reasoning_prompt",
    "build_continuation_reasoning_prompt",
]
