"""Prompts of ARR and of every baseline.

All methods use the same model with its own chat template.  Each prompt is a
``(system, user)`` pair.

  solve_prompt        sigma(x): the initial sampling prompt, shared by every method.
  revise_prompt       rho(x, y): the stationary revision prompt of ARR.  It contains
                      only the question and the current solution: no round index,
                      history, or other roots.  Variant 0 is the prompt of the paper;
                      variants 1 and 2 are paraphrases for the prompt ablation.
  critique_prompt,    the two-stage critique-and-correction update of SETS.
  correction_prompt
  continuation_prompt the continuation update of Sequential Revision.
  conma_*_prompt      the two prompts of the ConMA baseline.
"""

from __future__ import annotations

import re
from typing import Tuple

CONTINUATION_SYSTEM = (
    "You are a precise mathematical reasoner.\n"
    "Solve problems using clearly numbered steps.\n"
    "Show every calculation inside each step.\n"
    "End your response with exactly one line:\n"
    "Final answer: \\boxed{<answer>}"
)

CRITIQUE_SYSTEM = (
    "You are a careful mathematical critic.\n"
    "You will be given a PROBLEM and a SOLUTION TRACE.\n"
    "Your job: identify any errors in the trace — logical mistakes, "
    "wrong calculations, incorrect assumptions, or faulty reasoning.\n"
    "If the trace is correct, say so explicitly.\n"
    "Be concise and specific. Do NOT solve the problem yourself."
)

CORRECTION_SYSTEM = (
    "You are a careful mathematical editor.\n"
    "You will be given a PROBLEM, a SOLUTION TRACE, and a CRITIQUE.\n"
    "Your job: produce a corrected solution that fixes all errors identified "
    "in the critique while keeping the correct parts of the original trace.\n"
    "Use the same numbered-step format.\n"
    "End your response with exactly one line:\n"
    "Final answer: \\boxed{<answer>}"
)


def _continuation_reasoning_prompt(
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


def _self_critique_prompt(
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


def _self_correction_prompt(
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


SOLVE_SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."

REVISE_INSTRUCTIONS = (
    # 0: the revision prompt rho of the paper
    "Check the proposed solution against the question. Return a complete corrected\n"
    "solution. Correct any reasoning or calculation errors, including their downstream\n"
    "consequences. Preserve wording that does not need correction. If no correction\n"
    "is needed, reproduce the proposed solution exactly. Output only the solution.",
    # 1, 2: paraphrases for the prompt-sensitivity ablation
    "Verify the proposed solution against the question and output a complete corrected\n"
    "solution. Fix every reasoning or calculation error together with everything that\n"
    "depends on it. Keep all wording that is already correct. If nothing needs fixing,\n"
    "copy the proposed solution exactly. Output nothing except the solution.",
    "Review the proposed solution for the question below and return the full corrected\n"
    "solution. Correct any mistakes in reasoning or arithmetic and their consequences\n"
    "later in the solution, and leave correct text unchanged. If it is already correct,\n"
    "repeat it exactly as given. Respond with the solution only.",
)

Prompt = Tuple[str, str]      # (system, user)


_same = lambda s: s


def solve_prompt(question: str, system: str = SOLVE_SYSTEM) -> Prompt:
    return system, question.strip()


def revise_prompt(question: str, solution: str, variant: int = 0, system: str = SOLVE_SYSTEM) -> Prompt:
    return system, (
        f"{REVISE_INSTRUCTIONS[variant]}\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Proposed solution:\n{solution.strip()}"
    )


# ``fmt`` rewrites the "\\boxed{}" answer line for benchmarks with another answer format.
def continuation_prompt(question: str, prior: str, depth: int, total: int, max_prior_chars: int, fmt=_same) -> Prompt:
    return fmt(CONTINUATION_SYSTEM), fmt(_continuation_reasoning_prompt(question, prior, depth, total,
                                                                               max_prior_chars))


def critique_prompt(question: str, trace: str, answer: str, max_trace_chars: int, fmt=_same) -> Prompt:
    return fmt(CRITIQUE_SYSTEM), fmt(_self_critique_prompt(question, trace, answer, max_trace_chars))


def correction_prompt(question: str, trace: str, critique: str, answer: str, max_trace_chars: int,
                      max_critique_chars: int, fmt=_same) -> Prompt:
    return fmt(CORRECTION_SYSTEM), fmt(_self_correction_prompt(question, trace, critique, answer,
                                                                           max_trace_chars, max_critique_chars))


# ConMA prompts (Wang et al., 2026, Appendix C), math versions, verbatim.
CONMA_DSE = (
    "Question: {question}\n"
    "We have generated some preliminary solutions (ranked by likelihood): {candidates}\n"
    "Please perform a Critical Review (Sanity Check) on the above solutions: 1. Do they interpret the question "
    "correctly? 2. Are there any calculation errors in the steps? 3. Is there a logic gap or an edge case ignored?\n"
    "If you find that any of the candidate solutions are robust and correct, please verify them and explain why. "
    "If you identify logical flaws in the existing solutions, please propose a new or corrected solution path and "
    "answer."
)
CONMA_MSA = (
    "Please solve the following multiple-choice question. Question: {question}\n"
    "Review the Candidate Options below. They are ordered based on preliminary analysis, but the correct solution "
    "may require careful verification among the candidates. Select the strictly correct option based on rigorous "
    "logic and calculation.\n"
    "Candidate Options: {choices}\n"
    "Please verify each option step-by-step and select the most correct one. Provide detailed steps and the final "
    "answer in the format \\boxed{{X}}."
)


def conma_dse_prompt(question: str, candidates_text: str, system: str = SOLVE_SYSTEM, fmt=_same) -> Prompt:
    return system, fmt(CONMA_DSE.format(question=question.strip(), candidates=candidates_text))


def conma_msa_prompt(question: str, choices_text: str, system: str = SOLVE_SYSTEM, fmt=_same) -> Prompt:
    return system, fmt(CONMA_MSA.format(question=question.strip(), choices=choices_text))


__all__ = ["SOLVE_SYSTEM", "REVISE_INSTRUCTIONS", "solve_prompt", "revise_prompt", "continuation_prompt",
           "critique_prompt", "correction_prompt", "conma_dse_prompt", "conma_msa_prompt"]
