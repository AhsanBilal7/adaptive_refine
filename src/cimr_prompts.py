"""
cimr_prompts.py
===============
Prompt builders for step-wise reasoning generation in the CIMR pipeline.

The CIMR pipeline runs d sequential reasoning steps per rollout using
CRISP-style numbered steps (Step 1:, Step 2:, ...).

  Step 1  — generates initial reasoning from the question alone.
  Steps 2..d — continues/refines reasoning given the corrected output
               of the previous step, preserving step numbering continuity.

Between every generation, the self-critique → self-correction agents from
``prompts_self_critique_correct.py`` are applied.

Depth d and rollout count N are set via ``CIMRConfig`` in cimr_inference.py.
To change d from a shell script, pass the value when constructing CIMRConfig:
    config = CIMRConfig(depth=int(os.environ.get("CIMR_DEPTH", 4)), ...)
"""

from __future__ import annotations

from typing import Optional

REASONING_SYSTEM_PROMPT = (
    "You are a careful mathematical reasoner.\n"
    "Solve the given problem using clearly numbered steps:\n\n"
    "Step 1: [set up / restate the problem]\n"
    "Step 2: [first calculation or key insight]\n"
    "Step 3: [continue the solution]\n"
    "...\n\n"
    "Show every intermediate calculation inside each step.\n"
    "At the end write exactly one line:\n"
    "Final answer: \\boxed{<answer>}"
)


def build_initial_reasoning_prompt(question: str) -> str:
    """User message for step 1's reasoning generation (the seed step)."""
    q = (question or "").strip()
    return (
        "Solve the following problem using numbered steps.\n\n"
        "PROBLEM\n"
        f"{q}\n\n"
        "Number each step (Step 1:, Step 2:, ...). "
        "Show every calculation. End with exactly one line:\n"
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

    Receives the CORRECTED OUTPUT from the previous step and asks the
    model to extend or refine it, continuing the step numbering from
    where the prior corrected trace left off.

    Parameters
    ----------
    question        : the original problem.
    prior_corrected : the corrected output from step k-1.
    step_idx        : current 1-indexed step (>= 2).
    total_steps     : d, total steps in this rollout.
    max_prior_chars : head/tail truncation budget for the prior trace.
    """
    q = (question or "").strip()
    prior = (prior_corrected or "").strip()

    if max_prior_chars > 0 and len(prior) > max_prior_chars:
        keep = max_prior_chars // 2
        head = prior[:keep].rstrip()
        tail = prior[-keep:].lstrip()
        omit = len(prior) - max_prior_chars
        prior = f"{head}\n\n[... {omit:,} chars omitted ...]\n\n{tail}"

    is_last = (step_idx >= total_steps)

    # Count the last step number in prior so the model continues numbering.
    import re
    step_nums = re.findall(r"^Step\s+(\d+)\s*:", prior, re.MULTILINE | re.IGNORECASE)
    last_num = int(step_nums[-1]) if step_nums else (step_idx - 1)
    next_num = last_num + 1

    role_line = (
        f"You are at refinement depth {step_idx}"
        f"{' (the FINAL depth)' if is_last else ''} of {total_steps}."
    )

    instructions = (
        f"{role_line}\n\n"
        "Below is the PROBLEM and the CORRECTED REASONING from the "
        f"previous depth.\n\n"
        "Your job:\n"
        "  - Review the CORRECTED REASONING carefully.\n"
        f"  - Continue from Step {next_num}: — extend or refine, verifying "
        "each claim and filling any gaps.\n"
        "  - If the prior reasoning already reaches a confident correct "
        "answer, confirm it concisely. Do NOT introduce a different answer "
        "unless you can identify a concrete error.\n"
        "  - End your response with exactly one line:\n"
        "    Final answer: \\boxed{<answer>}"
    )

    return (
        f"{instructions}\n\n"
        "PROBLEM\n"
        f"{q}\n\n"
        f"CORRECTED REASONING (from depth {step_idx - 1})\n"
        f"{prior if prior else '(empty)'}\n\n"
        f"Continue from Step {next_num}:, ending with "
        "`Final answer: \\boxed{{<answer>}}`."
    )


__all__ = [
    "REASONING_SYSTEM_PROMPT",
    "build_initial_reasoning_prompt",
    "build_continuation_reasoning_prompt",
]
