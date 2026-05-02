"""
prompts_self_critique_correct.py
================================

Prompts and parsing utilities for the **self-critique** and
**self-correction** agents used by the enhanced CIMR pipeline.

Methodology
-----------
For each candidate reasoning trace produced by the base sampler:

  1.  Self-Critique Agent
      ----------------------
      Reads (PROBLEM, TRACE).
      Does NOT re-solve.  Identifies whether the trace is correct or has
      an error, the error TYPE (e.g. arithmetic, algebra, missed case,
      logical, substitution, ...) and the LOCATION of the error in the
      trace.  Outputs a short paragraph.  If the trace is judged correct,
      it confirms the final answer in \\boxed{}.

  2.  Self-Correction Agent
      ----------------------
      Reads (PROBLEM, TRACE, CRITIQUE).
      Produces a corrected reasoning trajectory and final answer in the
      form `Final answer: \\boxed{<answer>}`.  If the critique says
      "no correction is needed", the corrector simply restates the
      original final answer.

This decoupled design keeps critique focussed on diagnosis and
correction focussed on synthesis, avoiding interference between the two
roles (Saunders et al. 2022; Madaan et al. 2023; Tyen et al. 2023).

The few-shot examples below intentionally mirror the simple-style
demonstrations supplied in `self_critique_examples_simple.txt` and
`self_correction_examples_simple.txt`.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


# =============================================================================
# Self-Critique Agent
# =============================================================================

SELF_CRITIQUE_SYSTEM_PROMPT = (
    "You are a Self-Critique Agent.\n\n"
    "Your job:\n"
    "Read the PROBLEM and the TRACE.\n"
    "Do NOT solve the problem from scratch.\n"
    "Check whether the TRACE has a reasoning or calculation error.\n\n"
    "Output format:\n"
    "Write one short paragraph only.\n"
    "Include:\n"
    "1. Whether the trace is correct or incorrect.\n"
    "2. The main error type, if any.\n"
    "3. The exact location of the error in the trace.\n"
    "4. Why that error changes the final answer.\n"
    "5. If the trace is correct, confirm the final answer in \\boxed{}.\n\n"
    "Keep the critique simple, direct, and specific."
)


# Few-shot examples — kept consistent with self_critique_examples_simple.txt
_SELF_CRITIQUE_FEW_SHOT = """\
Here are short examples of the exact output format required.

EXAMPLE_START
PROBLEM
Convert the point (0,3) in rectangular coordinates to polar coordinates. Enter your answer in the form (r,theta), where r > 0 and 0 <= theta < 2*pi.

TRACE
To convert (0,3), compute r = sqrt(0^2+3^2)=3. Since x=0 and y>0, the point is on the positive y-axis, so theta = pi/2. Therefore the answer is (3, pi/2).

CRITIQUE
The trace is correct and no correction is needed. The radius is computed correctly as r=sqrt(0^2+3^2)=3, and the angle is correctly identified as theta=pi/2 because the point lies on the positive y-axis. The final answer is \\boxed{(3,\\pi/2)}.
EXAMPLE_END

EXAMPLE_START
PROBLEM
Find the sum of all integer bases b > 9 for which 17_b is a divisor of 97_b.

TRACE
Convert to base 10: 17_b=b+7 and 97_b=9b+7. We need b+7 to divide 9b+7. Set 9b+7=k(b+7), so b=7(k-1)/(9-k). Now list divisors of 7. This gives only b=49, so the sum is 49.

CRITIQUE
The trace is incorrect because of an incomplete case analysis. The error occurs when the solver says to "list divisors of 7." From b=7(k-1)/(9-k), the condition is not simply that 9-k divides 7; after simplification, the relevant divisor condition leads to divisors of 56. This missed the valid base b=21, so the trace only finds b=49 and gives the wrong sum. The corrected final answer should be \\boxed{70}.
EXAMPLE_END

EXAMPLE_START
PROBLEM
Cities A and B are 45 miles apart. Alicia bikes from A toward B at 18 mph. Beth bikes from B toward A at 12 mph, leaving at the same time. How many miles from City A will they be when they meet?

TRACE
Their combined speed is 18+12=30 mph. The meeting time is 45/30=1.5 hours. Alicia travels 18*1.5=27 miles, so they meet 27 miles from City A.

CRITIQUE
The trace is correct and no correction is needed. Since the two cyclists move toward each other, adding their speeds gives the correct closing speed of 30 mph. The time 45/30=1.5 hours is correct, and Alicia's distance from City A is 18*1.5=27 miles. The final answer is \\boxed{27}.
EXAMPLE_END
"""


def _truncate_trace_simple(text: str, max_chars: int) -> str:
    """Head/tail truncation to keep prompts within budget."""
    s = str(text or "").strip()
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    keep = max_chars // 2
    head = s[:keep].rstrip()
    tail = s[-keep:].lstrip()
    omit = len(s) - max_chars
    return f"{head}\n\n[... {omit:,} chars omitted ...]\n\n{tail}"


def build_self_critique_prompt(
    question:        str,
    trace_text:      str,
    trace_answer:    str = "",
    max_trace_chars: int = 4000,
    other_answers:   Optional[List[str]] = None,
) -> str:
    """
    Build the user message body for the self-critique agent.

    Inputs
    ------
    question         : the math problem.
    trace_text       : the full reasoning trace produced by the sampler.
    trace_answer     : the extracted \\boxed{} answer from the trace
                       (shown as a hint, but the critique focuses on the
                       reasoning, not just the answer).
    max_trace_chars  : if >0, head/tail truncate the trace text to this many
                       characters before inserting it into the prompt.
    other_answers    : if not None, a list of the OTHER candidate final
                       answers produced by sibling traces from the same
                       sampler (any duplicates / counts are preserved by
                       the caller).  When supplied, the prompt surfaces
                       a tally of those alternatives so the critique can
                       cross-check against the wider answer distribution.
                       Pass `None` to disable this view (default).
    """
    body  = _truncate_trace_simple(trace_text, max_trace_chars)
    cand  = (trace_answer or "").strip() or "(none)"

    other_block_lines: List[str] = []
    if other_answers is not None:
        # Tally the other answers (preserves multiplicity, sorted by count).
        from collections import Counter
        cleaned = [str(a).strip() for a in other_answers if str(a).strip()]
        if cleaned:
            counts = Counter(cleaned).most_common()
            tally  = ", ".join(
                f"{ans!s} (x{n})" for ans, n in counts
            )
            other_block_lines = [
                "",
                "OTHER CANDIDATE ANSWERS (from sibling traces of the same "
                "sampler, for cross-checking only — do NOT trust them as "
                "ground truth):",
                f"  {tally}",
            ]
        else:
            other_block_lines = [
                "",
                "OTHER CANDIDATE ANSWERS: (none extracted from sibling traces)",
            ]

    lines = [
        "Read the PROBLEM and the TRACE below. Decide whether the TRACE "
        "contains a reasoning or calculation error. Do NOT re-solve the "
        "problem. Output exactly one short paragraph that follows the "
        "five-point structure described in the system prompt.",
        "",
        "PROBLEM",
        question.strip(),
        "",
        "TRACE",
        body if body else "(empty)",
        "",
        f"(The trace's final extracted answer is: {cand})",
    ]
    lines.extend(other_block_lines)
    lines.extend([
        "",
        _SELF_CRITIQUE_FEW_SHOT.rstrip(),
        "",
        "Now produce the CRITIQUE for the actual PROBLEM and TRACE above.",
        "Begin your response with the word `CRITIQUE`.",
    ])
    return "\n".join(lines)


# Regexes used by parse_self_critique
_CRIT_CORRECT_RE = re.compile(
    r"\b("
    r"correct\s+and\s+no\s+correction\s+is\s+needed"
    r"|trace\s+is\s+correct"
    r"|reasoning\s+is\s+correct"
    r"|no\s+(?:reasoning|calculation)?\s*errors?"
    r"|no\s+correction\s+is\s+needed"
    r")\b",
    re.IGNORECASE,
)
_CRIT_INCORRECT_RE = re.compile(
    r"\b("
    r"trace\s+is\s+incorrect"
    r"|reasoning\s+is\s+incorrect"
    r"|the\s+trace\s+has\s+an?\s+error"
    r"|there\s+is\s+an?\s+error"
    r"|contains?\s+an?\s+error"
    r"|the\s+error\s+occurs"
    r"|wrong\s+answer"
    r"|incomplete\s+case\s+analysis"
    r"|miscomputed"
    r"|incorrectly"
    r")\b",
    re.IGNORECASE,
)


def parse_self_critique(raw_response: str) -> Dict[str, Any]:
    """
    Parse the critique agent's response.

    Returns a dict with:
      raw                 : the cleaned critique text.
      verdict_label       : "CORRECT" | "INCORRECT" | "UNCLEAR".
      is_trace_correct    : True iff the critique judged the trace correct.
      valid_probability   : soft confidence ∈ [0, 1] that the trace is
                            correct (used by the soft-weighted selection
                            rules — replaces the old VERIFY+VALID gate).
      error_type          : best-effort extraction of the cited error type
                            ("calculation", "algebra", "case analysis", ...)
                            — empty when the trace is correct or the type
                            cannot be detected.
      error_location      : best-effort extraction of where in the trace
                            the critique says the error occurred.
      confirmed_answer    : if the critique confirmed an answer in
                            \\boxed{}, that answer (else "").
    """
    out = dict(
        raw                 = "",
        verdict_label       = "UNCLEAR",
        is_trace_correct    = False,
        valid_probability   = 0.5,
        error_type          = "",
        error_location      = "",
        confirmed_answer    = "",
    )
    if not raw_response or not isinstance(raw_response, str):
        return out

    text = raw_response.strip()
    # Strip a leading "CRITIQUE" header if the model echoed our cue.
    text = re.sub(r"^\s*CRITIQUE\s*[:\-]?\s*", "", text, count=1, flags=re.IGNORECASE)
    out["raw"] = text

    # Confirmed answer (if any) — the LAST \boxed{...}.
    boxed = _extract_last_boxed(text)
    if boxed:
        out["confirmed_answer"] = boxed.strip()

    # Heuristic verdict.
    has_correct_phrase   = bool(_CRIT_CORRECT_RE.search(text))
    has_incorrect_phrase = bool(_CRIT_INCORRECT_RE.search(text))

    # "incorrect" wins over "correct" (the model often says
    # "the original conclusion is incorrect" while also confirming a
    # corrected boxed answer).
    if has_incorrect_phrase and not has_correct_phrase:
        out["verdict_label"]     = "INCORRECT"
        out["is_trace_correct"]  = False
        out["valid_probability"] = 0.05
    elif has_correct_phrase and not has_incorrect_phrase:
        out["verdict_label"]     = "CORRECT"
        out["is_trace_correct"]  = True
        out["valid_probability"] = 0.95
    elif has_incorrect_phrase and has_correct_phrase:
        # Both keywords present — likely "the trace claims it's correct
        # but the reasoning is incorrect because ...".  Trust INCORRECT.
        out["verdict_label"]     = "INCORRECT"
        out["is_trace_correct"]  = False
        out["valid_probability"] = 0.20
    else:
        # Neither keyword fires.  Fall back to a soft 50/50 prior; if the
        # critique confirmed an answer in \boxed{}, lean slightly toward
        # CORRECT.
        if out["confirmed_answer"]:
            out["verdict_label"]     = "CORRECT"
            out["is_trace_correct"]  = True
            out["valid_probability"] = 0.65
        else:
            out["verdict_label"]     = "UNCLEAR"
            out["valid_probability"] = 0.50

    # Best-effort error type / location extraction (purely heuristic).
    if not out["is_trace_correct"]:
        m = re.search(
            r"main\s+error\s+(?:type)?\s*(?:is)?\s*(?:an?)?\s*([^\.\n]+)\.",
            text, re.IGNORECASE,
        )
        if m:
            out["error_type"] = m.group(1).strip(" ,;-")
        m = re.search(
            r"(?:error|mistake)\s+(?:occurs|is|appears|happens)\s+(?:in|at|when|after)\s+([^\.\n]+)\.",
            text, re.IGNORECASE,
        )
        if m:
            out["error_location"] = m.group(1).strip(" ,;-")

    return out


# =============================================================================
# Self-Correction Agent
# =============================================================================

SELF_CORRECTION_SYSTEM_PROMPT = (
    "You are a Self-Correction Agent.\n\n"
    "Your job:\n"
    "Read the PROBLEM, the original TRACE, and the CRITIQUE.\n"
    "Write a corrected solution that fixes the exact issues found in the "
    "critique.\n\n"
    "Rules:\n"
    "1. Do not repeat the same mistake.\n"
    "2. Keep the solution clear and step-by-step.\n"
    "3. If the critique says the trace is already correct, write one short "
    "sentence saying no correction is needed.\n"
    "4. Always end with exactly this format:\n\n"
    "Final answer: \\boxed{answer}"
)


# Few-shot examples consistent with self_correction_examples_simple.txt
_SELF_CORRECTION_FEW_SHOT = """\
Here are short examples of the exact output format required.

EXAMPLE_START
PROBLEM
Convert the point (0,3) in rectangular coordinates to polar coordinates. Enter your answer in the form (r,theta), where r > 0 and 0 <= theta < 2*pi.

TRACE
To convert (0,3), compute r = sqrt(0^2+3^2)=3. Since x=0 and y>0, the point is on the positive y-axis, so theta = pi/2. Therefore the answer is (3, pi/2).

CRITIQUE
The trace is correct and no correction is needed. The radius is computed correctly as r=sqrt(0^2+3^2)=3, and the angle is correctly identified as theta=pi/2 because the point lies on the positive y-axis. The final answer is \\boxed{(3,\\pi/2)}.

CORRECTED TRACE
The original trace is valid, so no correction is needed.

Final answer: \\boxed{(3,\\pi/2)}
EXAMPLE_END

EXAMPLE_START
PROBLEM
Find the sum of all integer bases b > 9 for which 17_b is a divisor of 97_b.

TRACE
Convert to base 10: 17_b=b+7 and 97_b=9b+7. We need b+7 to divide 9b+7. Set 9b+7=k(b+7), so b=7(k-1)/(9-k). Now list divisors of 7. This gives only b=49, so the sum is 49.

CRITIQUE
The trace is incorrect because of an incomplete case analysis. The error occurs when the solver says to "list divisors of 7." From b=7(k-1)/(9-k), the condition is not simply that 9-k divides 7; after simplification, the relevant divisor condition leads to divisors of 56. This missed the valid base b=21, so the trace only finds b=49 and gives the wrong sum. The corrected final answer should be \\boxed{70}.

CORRECTED TRACE
First convert the base-b numbers to base 10:
17_b=b+7 and 97_b=9b+7.

We need b+7 to divide 9b+7. Rewrite 9b+7 using b+7:
9b+7 = 9(b+7)-56.

Since b+7 divides 9(b+7), it must also divide 56. Therefore b+7 is a positive divisor of 56.

The positive divisors of 56 are 1, 2, 4, 7, 8, 14, 28, and 56. Since b>9, we need b+7>16. The valid choices are b+7=28 and b+7=56.

Thus b=21 or b=49. Their sum is 21+49=70.

Final answer: \\boxed{70}
EXAMPLE_END

EXAMPLE_START
PROBLEM
Cities A and B are 45 miles apart. Alicia bikes from A toward B at 18 mph. Beth bikes from B toward A at 12 mph, leaving at the same time. How many miles from City A will they be when they meet?

TRACE
Their combined speed is 18+12=30 mph. The meeting time is 45/30=1.5 hours. Alicia travels 18*1.5=27 miles, so they meet 27 miles from City A.

CRITIQUE
The trace is correct and no correction is needed. Since the two cyclists move toward each other, adding their speeds gives the correct closing speed of 30 mph. The time 45/30=1.5 hours is correct, and Alicia's distance from City A is 18*1.5=27 miles. The final answer is \\boxed{27}.

CORRECTED TRACE
The original trace is valid, so no correction is needed.

Final answer: \\boxed{27}
EXAMPLE_END
"""


def build_self_correction_prompt(
    question:        str,
    trace_text:      str,
    critique_text:   str,
    trace_answer:    str = "",
    max_trace_chars: int = 4000,
    max_crit_chars:  int = 1500,
    other_answers:   Optional[List[str]] = None,
) -> str:
    """
    Build the user message body for the self-correction agent.

    Inputs
    ------
    question         : the math problem.
    trace_text       : the full original reasoning trace.
    critique_text    : the critique produced by the self-critique agent.
    trace_answer     : the original trace's extracted final answer
                       (shown as a hint).
    max_trace_chars  : truncation budget for the trace.
    max_crit_chars   : truncation budget for the critique.
    other_answers    : if not None, a list of the OTHER candidate final
                       answers produced by sibling traces from the same
                       sampler.  When supplied, the prompt surfaces a
                       tally of those alternatives as additional context
                       (the corrector is reminded NOT to trust them as
                       ground truth — they are only for cross-checking).
                       Pass `None` to disable (default).
    """
    body  = _truncate_trace_simple(trace_text,    max_trace_chars)
    crit  = _truncate_trace_simple(critique_text, max_crit_chars)
    cand  = (trace_answer or "").strip() or "(none)"

    other_block_lines: List[str] = []
    if other_answers is not None:
        from collections import Counter
        cleaned = [str(a).strip() for a in other_answers if str(a).strip()]
        if cleaned:
            counts = Counter(cleaned).most_common()
            tally  = ", ".join(
                f"{ans!s} (x{n})" for ans, n in counts
            )
            other_block_lines = [
                "",
                "OTHER CANDIDATE ANSWERS (from sibling traces of the same "
                "sampler, for cross-checking only — do NOT trust them as "
                "ground truth; rely on the CRITIQUE and your own working):",
                f"  {tally}",
            ]
        else:
            other_block_lines = [
                "",
                "OTHER CANDIDATE ANSWERS: (none extracted from sibling traces)",
            ]

    lines = [
        "Below is a math problem, the original reasoning TRACE, and a "
        "CRITIQUE of that trace produced by a separate critique agent.  "
        "Your job is to write a CORRECTED TRACE that fixes the exact "
        "issues raised in the CRITIQUE, and to end your response with a "
        "single line of the form `Final answer: \\boxed{<answer>}`.",
        "",
        "PROBLEM",
        question.strip(),
        "",
        "TRACE",
        body if body else "(empty)",
        "",
        f"(The original trace's extracted final answer was: {cand})",
        "",
        "CRITIQUE",
        crit if crit else "(empty)",
    ]
    lines.extend(other_block_lines)
    lines.extend([
        "",
        _SELF_CORRECTION_FEW_SHOT.rstrip(),
        "",
        "Now produce the CORRECTED TRACE for the actual PROBLEM, TRACE, "
        "and CRITIQUE above. Be concise and end with exactly one line:",
        "Final answer: \\boxed{<answer>}",
        "",
        "Begin your response with the words `CORRECTED TRACE`.",
    ])
    return "\n".join(lines)


# =============================================================================
# Helpers — boxed-answer extraction
# =============================================================================

def _extract_last_boxed(text: str) -> str:
    """
    Return the content of the LAST ``\\boxed{...}`` in `text`, handling
    nested braces.  Empty string if no \\boxed{} is found.
    """
    if not text:
        return ""
    matches = list(re.finditer(r"\\boxed\{", text))
    if not matches:
        return ""
    start = matches[-1].end()
    depth = 1
    i = start
    n = len(text)
    while i < n and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        i += 1
    return text[start : i - 1].strip() if depth == 0 else text[start:].strip()


# =============================================================================
# Convenience: parse the corrected trace
# =============================================================================

def parse_self_correction(raw_response: str) -> Dict[str, Any]:
    """
    Parse the self-correction agent's response.

    Returns a dict with:
      raw              : the cleaned correction text.
      corrected_answer : the answer extracted from the FINAL \\boxed{}.
      no_correction_needed : True iff the corrector explicitly said the
                             original trace is valid.
    """
    out = dict(
        raw                  = "",
        corrected_answer     = "",
        no_correction_needed = False,
    )
    if not raw_response or not isinstance(raw_response, str):
        return out

    text = raw_response.strip()
    # Strip a leading "CORRECTED TRACE" header if echoed.
    text = re.sub(
        r"^\s*CORRECTED\s+TRACE\s*[:\-]?\s*",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    )
    out["raw"] = text

    # Final \boxed{} answer
    boxed = _extract_last_boxed(text)
    if boxed:
        out["corrected_answer"] = boxed.strip()

    # "no correction is needed" sentinel
    if re.search(
        r"(?:original\s+trace\s+is\s+valid"
        r"|no\s+correction\s+is\s+needed"
        r"|trace\s+is\s+already\s+correct)",
        text,
        re.IGNORECASE,
    ):
        out["no_correction_needed"] = True

    return out


__all__ = [
    "SELF_CRITIQUE_SYSTEM_PROMPT",
    "SELF_CORRECTION_SYSTEM_PROMPT",
    "build_self_critique_prompt",
    "build_self_correction_prompt",
    "parse_self_critique",
    "parse_self_correction",
]
