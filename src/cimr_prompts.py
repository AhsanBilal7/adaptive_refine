"""
cimr_prompts.py
===============
Prompt builders for the CIMR pipeline.

All prompts are self-contained — no few-shot examples, no cached content.
Every prompt is built fresh from (question, prior_corrected) only.

Pipeline structure per rollout:
  depth 1  → generate initial reasoning from question alone
  depth 2..d → extend/refine using the corrected trace from the prior depth

Between every generation:
  self-critique  → identify specific errors in the current trace
  self-correction → produce a corrected trace that fixes those errors
"""

from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

REASONING_SYSTEM_PROMPT = (
    "You are a precise mathematical reasoner.\n"
    "Solve problems using clearly numbered steps.\n"
    "Show every calculation inside each step.\n"
    "End your response with exactly one line:\n"
    "Final answer: \\boxed{<answer>}"
)

CRITIQUE_SYSTEM_PROMPT = (
    "You are a careful mathematical critic.\n"
    "You will be given a PROBLEM and a SOLUTION TRACE.\n"
    "Your job: identify any errors in the trace — logical mistakes, "
    "wrong calculations, incorrect assumptions, or faulty reasoning.\n"
    "If the trace is correct, say so explicitly.\n"
    "Be concise and specific. Do NOT solve the problem yourself."
)

CORRECTION_SYSTEM_PROMPT = (
    "You are a careful mathematical editor.\n"
    "You will be given a PROBLEM, a SOLUTION TRACE, and a CRITIQUE.\n"
    "Your job: produce a corrected solution that fixes all errors identified "
    "in the critique while keeping the correct parts of the original trace.\n"
    "Use the same numbered-step format.\n"
    "End your response with exactly one line:\n"
    "Final answer: \\boxed{<answer>}"
)


# ---------------------------------------------------------------------------
# Reasoning prompts
# ---------------------------------------------------------------------------

def build_initial_reasoning_prompt(question: str) -> str:
    """Prompt for depth-1 reasoning generation (seed step)."""
    q = (question or "").strip()
    return (
        "Solve the following problem step by step.\n\n"
        "PROBLEM\n"
        f"{q}\n\n"
        "Use numbered steps (Step 1:, Step 2:, ...). "
        "Show every calculation. "
        "End with exactly one line:\n"
        "Final answer: \\boxed{<answer>}"
    )


def build_continuation_reasoning_prompt(
    question: str,
    prior_corrected: str,
    step_idx: int,
    total_steps: int,
    max_prior_chars: int = 3000,
) -> str:
    """Prompt for depths 2..d — extend/refine the prior corrected trace.

    Parameters
    ----------
    question        : the original problem (unchanged across all depths).
    prior_corrected : the corrected trace from depth k-1.
    step_idx        : current depth index (>= 2).
    total_steps     : total depth d.
    max_prior_chars : truncation budget for prior_corrected.
    """
    q = (question or "").strip()
    prior = (prior_corrected or "").strip()

    # Truncate prior if needed (head + tail).
    if max_prior_chars > 0 and len(prior) > max_prior_chars:
        keep = max_prior_chars // 2
        head = prior[:keep].rstrip()
        tail = prior[-keep:].lstrip()
        omitted = len(prior) - max_prior_chars
        prior = f"{head}\n\n[... {omitted:,} characters omitted ...]\n\n{tail}"

    # Find last step number so we continue numbering naturally.
    step_nums = re.findall(r"^Step\s+(\d+)\s*:", prior, re.MULTILINE | re.IGNORECASE)
    last_num = int(step_nums[-1]) if step_nums else (step_idx - 1)
    next_num = last_num + 1

    is_final = (step_idx >= total_steps)
    final_note = " This is the FINAL refinement depth." if is_final else ""

    return (
        f"You are at refinement depth {step_idx} of {total_steps}.{final_note}\n\n"
        "Below is the ORIGINAL PROBLEM and the CORRECTED REASONING from the "
        "previous depth.\n\n"
        "Your task:\n"
        f"  1. Review the corrected reasoning carefully.\n"
        f"  2. Continue from Step {next_num}: — extend or refine the solution, "
        "verifying each claim.\n"
        "  3. If the prior reasoning already reaches a correct answer with no "
        "errors, confirm it concisely. Do NOT change the answer unless you can "
        "identify a specific concrete error.\n"
        "  4. End with exactly one line:\n"
        "     Final answer: \\boxed{<answer>}\n\n"
        "PROBLEM\n"
        f"{q}\n\n"
        f"CORRECTED REASONING FROM DEPTH {step_idx - 1}\n"
        f"{prior if prior else '(none)'}\n\n"
        f"Continue from Step {next_num}:, ending with "
        "`Final answer: \\boxed{{<answer>}}`."
    )


# ---------------------------------------------------------------------------
# Self-critique prompt
# ---------------------------------------------------------------------------

def build_self_critique_prompt(
    question: str,
    trace_text: str,
    trace_answer: str,
    max_trace_chars: int = 3000,
) -> str:
    """Ask the model to critique the reasoning trace for the given problem.

    Parameters
    ----------
    question        : the original problem.
    trace_text      : the reasoning trace to critique.
    trace_answer    : the answer extracted from the trace.
    max_trace_chars : truncation budget.
    """
    q = (question or "").strip()
    trace = (trace_text or "").strip()

    if max_trace_chars > 0 and len(trace) > max_trace_chars:
        keep = max_trace_chars // 2
        head = trace[:keep].rstrip()
        tail = trace[-keep:].lstrip()
        omitted = len(trace) - max_trace_chars
        trace = f"{head}\n\n[... {omitted:,} characters omitted ...]\n\n{tail}"

    ans_line = f"Extracted answer: {trace_answer}" if trace_answer else "Extracted answer: (none found)"

    return (
        "Review the following SOLUTION TRACE for the PROBLEM below.\n"
        "Identify any errors: wrong calculations, incorrect assumptions, "
        "logical mistakes, or missing steps.\n"
        "If no errors exist, state: 'The solution appears correct.'\n"
        "Be specific — point to the exact step where an error occurs.\n"
        "Do NOT re-solve the problem.\n\n"
        "PROBLEM\n"
        f"{q}\n\n"
        "SOLUTION TRACE\n"
        f"{trace}\n\n"
        f"{ans_line}\n\n"
        "CRITIQUE (list errors specifically, or confirm correctness):"
    )


# ---------------------------------------------------------------------------
# Self-correction prompt
# ---------------------------------------------------------------------------

def build_self_correction_prompt(
    question: str,
    trace_text: str,
    critique_text: str,
    trace_answer: str,
    max_trace_chars: int = 3000,
    max_critique_chars: int = 1000,
) -> str:
    """Produce a corrected solution based on the critique.

    Parameters
    ----------
    question          : the original problem.
    trace_text        : the reasoning trace to correct.
    critique_text     : the critique identifying errors.
    trace_answer      : original answer from the trace.
    max_trace_chars   : truncation budget for trace.
    max_critique_chars: truncation budget for critique.
    """
    q = (question or "").strip()
    trace = (trace_text or "").strip()
    critique = (critique_text or "").strip()

    if max_trace_chars > 0 and len(trace) > max_trace_chars:
        keep = max_trace_chars // 2
        head = trace[:keep].rstrip()
        tail = trace[-keep:].lstrip()
        omitted = len(trace) - max_trace_chars
        trace = f"{head}\n\n[... {omitted:,} characters omitted ...]\n\n{tail}"

    if max_critique_chars > 0 and len(critique) > max_critique_chars:
        critique = critique[:max_critique_chars].rstrip() + "\n[critique truncated]"

    return (
        "You are given a PROBLEM, a SOLUTION TRACE, and a CRITIQUE of that trace.\n"
        "Produce a corrected solution:\n"
        "  - Fix every error identified in the critique.\n"
        "  - Keep the numbered-step format (Step 1:, Step 2:, ...).\n"
        "  - Preserve correct parts of the original trace.\n"
        "  - Do NOT introduce new errors.\n"
        "  - End with exactly one line: Final answer: \\boxed{<answer>}\n\n"
        "PROBLEM\n"
        f"{q}\n\n"
        "ORIGINAL SOLUTION TRACE\n"
        f"{trace}\n\n"
        "CRITIQUE\n"
        f"{critique if critique else '(no specific errors identified — solution may be correct)'}\n\n"
        "CORRECTED SOLUTION (numbered steps, ending with Final answer: \\boxed{<answer>}):"
    )


__all__ = [
    "REASONING_SYSTEM_PROMPT",
    "CRITIQUE_SYSTEM_PROMPT",
    "CORRECTION_SYSTEM_PROMPT",
    "build_initial_reasoning_prompt",
    "build_continuation_reasoning_prompt",
    "build_self_critique_prompt",
    "build_self_correction_prompt",
]
