"""
prompts_self_critique_correct.py
================================

Prompts and parsing utilities for the **self-critique** and
**self-correction** agents used by the CIMR pipeline.

Methodology
-----------
For each candidate reasoning trace produced at a given depth level:

  1.  Self-Critique Agent
      ----------------------
      Reads (PROBLEM, TRACE).
      Does NOT re-solve.  Identifies whether the trace is correct or has
      an error, the error TYPE and LOCATION.  Outputs a short paragraph.

  2.  Self-Correction Agent
      ----------------------
      Reads (PROBLEM, TRACE, CRITIQUE).
      Produces a corrected reasoning trajectory ending in
      ``Final answer: \\boxed{<answer>}``.

Design
------
Few-shot demonstrations are intentionally NOT embedded in the user
message.  Injecting unrelated example problems into the user turn
causes the model to generate content for those examples rather than
the actual problem at hand.  Format guidance lives exclusively in the
system prompt.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


# =============================================================================
# Self-Critique Agent
# =============================================================================

SELF_CRITIQUE_SYSTEM_PROMPT = (
    "You are a Self-Critique Agent.\n\n"
    "Your task: read the PROBLEM and the TRACE. Determine whether the "
    "TRACE contains a reasoning or calculation error.\n\n"
    "Rules:\n"
    "- Do NOT re-solve the problem from scratch.\n"
    "- Do NOT reference any problem other than the one shown.\n"
    "- Focus only on the TRACE provided.\n\n"
    "Output exactly one short paragraph that covers:\n"
    "1. Whether the trace is CORRECT or INCORRECT.\n"
    "2. If INCORRECT: the error type "
    "(arithmetic / algebra / case analysis / substitution / "
    "logical gap / ...).\n"
    "3. If INCORRECT: the exact step or line where the error occurs.\n"
    "4. If INCORRECT: why that error changes the final answer.\n"
    "5. If CORRECT: confirm the final answer as \\boxed{answer}.\n\n"
    "Begin your response with the word CRITIQUE."
)


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
    Build the user message for the self-critique agent.

    Only the actual PROBLEM and TRACE are included — no unrelated
    example problems.  Format guidance lives in the system prompt.
    """
    body = _truncate_trace_simple(trace_text, max_trace_chars)
    cand = (trace_answer or "").strip() or "(none)"

    lines = [
        "PROBLEM",
        question.strip(),
        "",
        "TRACE",
        body if body else "(empty)",
        "",
        f"(Extracted final answer: {cand})",
    ]

    if other_answers is not None:
        from collections import Counter
        cleaned = [str(a).strip() for a in other_answers if str(a).strip()]
        if cleaned:
            counts = Counter(cleaned).most_common()
            tally = ", ".join(f"{ans!s} (x{n})" for ans, n in counts)
            lines.extend([
                "",
                "Other candidate answers from sibling traces "
                "(cross-check only — do NOT treat as ground truth):",
                f"  {tally}",
            ])

    lines.extend([
        "",
        "Produce the CRITIQUE for the PROBLEM and TRACE above.",
        "Begin with the word CRITIQUE.",
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
      raw                 : cleaned critique text
      verdict_label       : "CORRECT" | "INCORRECT" | "UNCLEAR"
      is_trace_correct    : bool
      valid_probability   : soft confidence in [0, 1]
      error_type          : cited error type (empty if correct)
      error_location      : location of error in trace (empty if correct)
      confirmed_answer    : if critique confirmed \\boxed{}, that answer
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
    # Strip leading CRITIQUE header if echoed.
    text = re.sub(r"^\s*CRITIQUE\s*[:\-]?\s*", "", text, count=1, flags=re.IGNORECASE)
    # Strip any stray EXAMPLE_START / EXAMPLE_END blocks that leaked from a cached prompt.
    text = re.sub(
        r"EXAMPLE_START.*?EXAMPLE_END", "", text,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()
    out["raw"] = text

    boxed = _extract_last_boxed(text)
    if boxed:
        out["confirmed_answer"] = boxed.strip()

    has_correct_phrase   = bool(_CRIT_CORRECT_RE.search(text))
    has_incorrect_phrase = bool(_CRIT_INCORRECT_RE.search(text))

    if has_incorrect_phrase and not has_correct_phrase:
        out["verdict_label"]     = "INCORRECT"
        out["is_trace_correct"]  = False
        out["valid_probability"] = 0.05
    elif has_correct_phrase and not has_incorrect_phrase:
        out["verdict_label"]     = "CORRECT"
        out["is_trace_correct"]  = True
        out["valid_probability"] = 0.95
    elif has_incorrect_phrase and has_correct_phrase:
        out["verdict_label"]     = "INCORRECT"
        out["is_trace_correct"]  = False
        out["valid_probability"] = 0.20
    else:
        if out["confirmed_answer"]:
            out["verdict_label"]     = "CORRECT"
            out["is_trace_correct"]  = True
            out["valid_probability"] = 0.65
        else:
            out["verdict_label"]     = "UNCLEAR"
            out["valid_probability"] = 0.50

    if not out["is_trace_correct"]:
        m = re.search(
            r"main\s+error\s+(?:type)?\s*(?:is)?\s*(?:an?)?\s*([^\.\n]+)\.",
            text, re.IGNORECASE,
        )
        if m:
            out["error_type"] = m.group(1).strip(" ,;-")
        m = re.search(
            r"(?:error|mistake)\s+(?:occurs|is|appears|happens)\s+"
            r"(?:in|at|when|after)\s+([^\.\n]+)\.",
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
    "Your task: read the PROBLEM, the original TRACE, and the CRITIQUE. "
    "Write a corrected solution that fixes the exact issues raised.\n\n"
    "Rules:\n"
    "1. Address ONLY the PROBLEM shown — do not solve any other problem.\n"
    "2. Fix the specific errors identified in the CRITIQUE.\n"
    "3. Write the corrected solution step-by-step "
    "(Step 1:, Step 2:, ...).\n"
    "4. If the CRITIQUE says the trace is already correct, write one "
    "sentence confirming this.\n"
    "5. Always end with exactly one line:\n\n"
    "   Final answer: \\boxed{answer}\n\n"
    "Begin your response with the words CORRECTED TRACE."
)


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
    Build the user message for the self-correction agent.

    Only the actual PROBLEM, TRACE, and CRITIQUE are included — no
    unrelated example problems in the user turn.
    """
    body = _truncate_trace_simple(trace_text,    max_trace_chars)
    crit = _truncate_trace_simple(critique_text, max_crit_chars)
    cand = (trace_answer or "").strip() or "(none)"

    lines = [
        "PROBLEM",
        question.strip(),
        "",
        "TRACE",
        body if body else "(empty)",
        "",
        f"(Original extracted answer: {cand})",
        "",
        "CRITIQUE",
        crit if crit else "(empty)",
    ]

    if other_answers is not None:
        from collections import Counter
        cleaned = [str(a).strip() for a in other_answers if str(a).strip()]
        if cleaned:
            counts = Counter(cleaned).most_common()
            tally = ", ".join(f"{ans!s} (x{n})" for ans, n in counts)
            lines.extend([
                "",
                "Other candidate answers from sibling traces "
                "(cross-check only — do NOT treat as ground truth):",
                f"  {tally}",
            ])

    lines.extend([
        "",
        "Write the CORRECTED TRACE for the PROBLEM above.",
        "Fix only the issues from the CRITIQUE. End with:",
        "Final answer: \\boxed{answer}",
        "",
        "Begin with the words CORRECTED TRACE.",
    ])
    return "\n".join(lines)


# =============================================================================
# Helpers — boxed-answer extraction
# =============================================================================

def _extract_last_boxed(text: str) -> str:
    """Return content of the LAST \\boxed{...}, handling nested braces."""
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
# parse_self_correction
# =============================================================================

def parse_self_correction(raw_response: str) -> Dict[str, Any]:
    """
    Parse the self-correction agent's response.

    Returns a dict with:
      raw                  : cleaned correction text
      corrected_answer     : answer from the final \\boxed{}
      no_correction_needed : True iff corrector said original trace is valid
    """
    out = dict(
        raw                  = "",
        corrected_answer     = "",
        no_correction_needed = False,
    )
    if not raw_response or not isinstance(raw_response, str):
        return out

    text = raw_response.strip()
    # Strip leading CORRECTED TRACE header.
    text = re.sub(
        r"^\s*CORRECTED\s+TRACE\s*[:\-]?\s*",
        "", text, count=1, flags=re.IGNORECASE,
    )
    # Strip any stray EXAMPLE_START / EXAMPLE_END blocks.
    text = re.sub(
        r"EXAMPLE_START.*?EXAMPLE_END", "", text,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()
    out["raw"] = text

    boxed = _extract_last_boxed(text)
    if boxed:
        out["corrected_answer"] = boxed.strip()

    if re.search(
        r"(?:original\s+trace\s+is\s+(?:valid|correct)"
        r"|no\s+correction\s+is\s+needed"
        r"|trace\s+is\s+already\s+correct)",
        text, re.IGNORECASE,
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
