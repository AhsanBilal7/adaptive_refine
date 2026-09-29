"""
voting.py
=========
Answer extraction Φ, normalization η, one-answer-per-rollout plurality vote
and the gold grader — a FIXED extraction and tie-break policy shared by every
method (updated_methodology.md §3.5).

Conventions (fixed for every method and baseline)
-------------------------------------------------
* Φ: content of the LAST ``\\boxed{...}``; optionally the last number when no
  box exists (``extract_fallback_last_number``).  No extraction → abstain.
* η: paper App. A.5 — strip whitespace, drop ``\\, \\! \\;``, unwrap one layer
  of braces, drop trailing decimal zeros, lowercase.  Additionally
  ``\\dfrac/\\tfrac → \\frac``, ``\\left/\\right`` and ``$`` are removed and a
  trailing period is stripped so trivially identical answers share a vote.
* Plurality over normalized answers; ties → the answer supported by the
  LOWEST slot index; all abstain → ``INVALID_ANSWER``.
* Agreement is never used as a truth or preference label.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

INVALID_ANSWER = "[invalid]"

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_answer(text: str, fallback_last_number: bool = True) -> str:
    """Φ(r): last \\boxed{} content (brace matched), else optional last number."""
    if not text:
        return ""
    idx = text.rfind("\\boxed")
    if idx >= 0:
        j = text.find("{", idx)
        if j >= 0:
            depth, k = 1, j + 1
            while k < len(text) and depth > 0:
                if text[k] == "{":
                    depth += 1
                elif text[k] == "}":
                    depth -= 1
                k += 1
            if depth == 0:
                ans = text[j + 1:k - 1].strip()
                if ans:
                    return ans
    if fallback_last_number:
        nums = _NUM_RE.findall(text.replace(",", ""))
        return nums[-1] if nums else ""
    return ""


def normalize_answer(ans: str) -> str:
    """η(a) — returns "" for invalid/empty answers."""
    s = str(ans or "").strip()
    if not s:
        return ""
    s = s.replace("$", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\left", "").replace("\\right", "")
    for sp in ("\\,", "\\!", "\\;", "\\:", "\\ "):
        s = s.replace(sp, "")
    s = re.sub(r"\s+", "", s)
    if s.startswith("{") and s.endswith("}") and s.count("{") == 1:
        s = s[1:-1]
    s = s.rstrip(".")
    if re.fullmatch(r"-?\d+\.\d+", s):
        s = s.rstrip("0").rstrip(".")
    return s.lower()


def plurality_vote(norm_answers: Sequence[str], display: Optional[Sequence[str]] = None
                   ) -> Tuple[str, str, Dict[str, int]]:
    """Return (winner_norm, winner_display, counts) with lowest-slot tie-break.

    ``norm_answers[i]`` is slot i's normalized answer ("" = abstain).
    """
    counts: "OrderedDict[str, int]" = OrderedDict()
    first_slot: Dict[str, int] = {}
    for i, a in enumerate(norm_answers):
        if not a:
            continue
        counts[a] = counts.get(a, 0) + 1
        first_slot.setdefault(a, i)
    if not counts:
        return "", INVALID_ANSWER, {}
    best = max(counts.items(), key=lambda kv: (kv[1], -first_slot[kv[0]]))[0]
    disp = best
    if display is not None:
        disp = display[first_slot[best]]
    return best, disp, dict(counts)


class Grader:
    """Sealed gold grader.  Only used AFTER a solve terminates, never inside
    the controller.  Results are memoised per (prediction, gold)."""

    def __init__(self, data_name: str = "math500"):
        self.data_name = data_name
        self._ev = None
        self._check = None
        self._cache: Dict[Tuple[str, str], bool] = {}
        try:
            from src.inference_methods import _build_evaluator, _check_correct  # type: ignore
            self._ev = _build_evaluator(data_name)
            self._check = _check_correct
        except Exception:
            self._ev = None
            self._check = None

    def is_correct(self, pred: str, gold: Optional[str]) -> Optional[bool]:
        if gold is None:
            return None
        if not pred or pred == INVALID_ANSWER:
            return False
        key = (pred, str(gold))
        if key in self._cache:
            return self._cache[key]
        ok = False
        if normalize_answer(pred) == normalize_answer(gold):
            ok = True
        elif self._check is not None:
            try:
                ok = bool(self._check(pred, str(gold), self._ev, self.data_name))
            except Exception:
                ok = False
        self._cache[key] = ok
        return ok


def sanity_check_grader(grader: Grader) -> List[str]:
    """§9: check extraction on known (in)equivalent answers and that the
    evaluator detects deliberately changed predictions.  Returns failures."""
    failures: List[str] = []
    equiv = [("\\frac{1}{2}", "0.5"), ("3.500", "3.5"), ("{7}", "7"), ("\\dfrac{3}{4}", "\\frac{3}{4}"),
             ("28", "28")]
    inequiv = [("2", "3"), ("\\frac{1}{2}", "\\frac{1}{3}"), ("12", "21"), ("-5", "5")]
    for a, b in equiv:
        if not grader.is_correct(a, b):
            failures.append(f"expected equivalent: {a!r} vs {b!r}")
    for a, b in inequiv:
        if grader.is_correct(a, b):
            failures.append(f"expected NOT equivalent: {a!r} vs {b!r}")
    # extraction checks
    cases = [("Step 1 ... Final answer: \\boxed{\\frac{a}{b}}", "\\frac{a}{b}"),
             ("\\boxed{1} then \\boxed{2}", "2"), ("no box here", "")]
    for text, want in cases:
        got = extract_answer(text, fallback_last_number=False)
        if got != want:
            failures.append(f"extract({text!r}) = {got!r}, want {want!r}")
    # deliberately changed predictions must flip correctness
    for gold in ("28", "\\frac{3}{5}", "12"):
        if not grader.is_correct(gold, gold):
            failures.append(f"gold {gold!r} not graded correct against itself")
        mutated = gold + "1"
        if grader.is_correct(mutated, gold):
            failures.append(f"mutated prediction {mutated!r} graded correct against {gold!r}")
    return failures


__all__ = [
    "INVALID_ANSWER", "extract_answer", "normalize_answer", "plurality_vote",
    "Grader", "sanity_check_grader",
]
