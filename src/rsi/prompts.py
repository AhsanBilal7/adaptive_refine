"""
prompts.py
==========
Role prompts P_t for the Procedure-RSI pipeline.

* Initial / RESTART generator prompt and the FULL_REFINE rewrite prompt are
  FIXED (§3: "initial and rewrite generator prompts remain fixed") and reuse
  the paper's CIMR builders in ``src/cimr_prompts.py``.
* Critique and correction prompts have an editable system prompt and an
  editable extra instruction; they are the only prompt text the editor may
  change.
* The critic is always asked (fixed suffix, not editable) to end with an
  ``ERROR CATEGORY:`` line so the critique category feature is parseable.
  The category is a model assertion, never a correctness label.
"""

from __future__ import annotations

import re
from typing import Dict

from src.cimr_prompts import (  # type: ignore
    CORRECTION_SYSTEM_PROMPT,
    CRITIQUE_SYSTEM_PROMPT,
    REASONING_SYSTEM_PROMPT,
    build_continuation_reasoning_prompt,
    build_initial_reasoning_prompt,
    build_self_correction_prompt,
    build_self_critique_prompt,
)

from .state import CRITIQUE_CATEGORIES

EDITABLE_PROMPT_KEYS = ("critique_system", "critique_extra", "correct_system", "correct_extra")

DEFAULT_PROMPTS: Dict[str, str] = {
    "critique_system": CRITIQUE_SYSTEM_PROMPT,
    "critique_extra": "",
    "correct_system": CORRECTION_SYSTEM_PROMPT,
    "correct_extra": "",
}

_CATEGORY_CHOICES = [c for c in CRITIQUE_CATEGORIES if c != "unknown"]
CATEGORY_SUFFIX = (
    "\n\nAfter the critique, output one final line of the form\n"
    "ERROR CATEGORY: <one of " + ", ".join(_CATEGORY_CHOICES) + ">\n"
    "Use no_specific_error if you found no concrete error."
)


def initial_prompt(question: str):
    return REASONING_SYSTEM_PROMPT, build_initial_reasoning_prompt(question)


def rewrite_prompt(question: str, prior: str, step_idx: int, total_steps: int, max_prior_chars: int):
    return REASONING_SYSTEM_PROMPT, build_continuation_reasoning_prompt(
        question=question, prior_corrected=prior, step_idx=step_idx,
        total_steps=max(total_steps, step_idx), max_prior_chars=max_prior_chars,
    )


def critique_prompt(prompts: Dict[str, str], question: str, trace: str, answer: str,
                    max_trace_chars: int):
    user = build_self_critique_prompt(question=question, trace_text=trace,
                                      trace_answer=answer, max_trace_chars=max_trace_chars)
    extra = (prompts.get("critique_extra") or "").strip()
    if extra:
        user = f"{user}\n\nADDITIONAL CHECKING INSTRUCTIONS\n{extra}"
    return prompts.get("critique_system") or CRITIQUE_SYSTEM_PROMPT, user + CATEGORY_SUFFIX


def correct_prompt(prompts: Dict[str, str], question: str, trace: str, critique: str,
                   answer: str, max_trace_chars: int, max_critique_chars: int):
    critique = strip_category_line(critique)
    user = build_self_correction_prompt(question=question, trace_text=trace, critique_text=critique,
                                        trace_answer=answer, max_trace_chars=max_trace_chars,
                                        max_critique_chars=max_critique_chars)
    extra = (prompts.get("correct_extra") or "").strip()
    if extra:
        user = f"{user}\n\nADDITIONAL REVISION INSTRUCTIONS\n{extra}"
    return prompts.get("correct_system") or CORRECTION_SYSTEM_PROMPT, user


_CAT_LINE = re.compile(r"ERROR\s*CATEGORY\s*[:\-]\s*([A-Za-z_ \-]+)", re.IGNORECASE)
_SYNONYMS = {
    "none": "no_specific_error", "no error": "no_specific_error", "no errors": "no_specific_error",
    "correct": "no_specific_error", "no_specific_error": "no_specific_error",
    "no specific error": "no_specific_error", "calculation": "arithmetic", "computation": "arithmetic",
    "arithmetic": "arithmetic", "algebra": "algebra", "algebraic": "algebra", "logic": "logic",
    "logical": "logic", "reasoning": "logic", "assumption": "assumption", "case omission": "case_omission",
    "case_omission": "case_omission", "missing case": "case_omission", "misread": "misread",
    "misinterpretation": "misread", "incomplete": "incomplete", "other": "other",
}


def strip_category_line(critique: str) -> str:
    return _CAT_LINE.sub("", critique or "").strip()


def parse_critique_category(critique: str) -> str:
    """Parse the ERROR CATEGORY line; fall back to a keyword classifier."""
    text = critique or ""
    m = None
    for m in _CAT_LINE.finditer(text):
        pass
    if m is not None:
        raw = m.group(1).strip().lower().replace("-", " ")
        raw = re.sub(r"\s+", " ", raw)
        if raw.replace(" ", "_") in CRITIQUE_CATEGORIES:
            return raw.replace(" ", "_")
        for k, v in _SYNONYMS.items():
            if raw.startswith(k):
                return v
    low = text.lower()
    if not low.strip():
        return "unknown"
    if re.search(r"appears correct|no (apparent |identified |specific )?errors?|is correct\b", low):
        return "no_specific_error"
    for pat, cat in ((r"arithmetic|calculat|computation", "arithmetic"),
                     (r"algebra|simplif|equation", "algebra"),
                     (r"assum", "assumption"), (r"\bcase", "case_omission"),
                     (r"misread|misinterpret|misunderst", "misread"),
                     (r"incomplete|missing step|not finish", "incomplete"),
                     (r"logic|reasoning|invalid|incorrect", "logic")):
        if re.search(pat, low):
            return cat
    return "other"


__all__ = [
    "EDITABLE_PROMPT_KEYS", "DEFAULT_PROMPTS", "CATEGORY_SUFFIX",
    "initial_prompt", "rewrite_prompt", "critique_prompt", "correct_prompt",
    "parse_critique_category", "strip_category_line",
]
