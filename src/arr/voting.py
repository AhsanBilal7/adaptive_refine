"""Answer extraction, normalization, majority vote, and the gold grader.

Every method uses the same rules:

* extraction phi: content of the last ``\\boxed{...}``; if there is no box, the
  last number in the text (``extract_fallback_last_number``).  No answer means
  the root abstains.
* normalization: remove spaces, ``$``, ``\\left``/``\\right``, thin spaces, one
  pair of outer braces, trailing zeros of decimals and a final period; map
  ``\\dfrac``/``\\tfrac`` to ``\\frac``; lowercase.
* vote: the most frequent normalized answer over the voting roots; a tie goes
  to the answer of the lowest root index; no answer at all gives ``[invalid]``.
* ``Grader`` scores the voted answer against the gold label after the solve
  finishes.  The controller never sees it.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

INVALID_ANSWER = "[invalid]"

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def truncate_at_first_boxed(text: str) -> str:
    """Text up to and including the first ``\\boxed{...}`` (drops text after the answer)."""
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


def extract_answer(text: str, fallback_last_number: bool = True) -> str:
    """Content of the last ``\\boxed{}`` (braces matched), else optionally the last number."""
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
    """Normalized form of an answer (``""`` for an empty or invalid answer)."""
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


def majority_vote(norm_answers: Sequence[str], display: Optional[Sequence[str]] = None
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
        from .evaluators import build_evaluator, check_correct
        self._ev = build_evaluator(data_name)
        self._check = check_correct

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
    """Check extraction on known (in)equivalent answers and that the
    evaluator detects deliberately changed predictions.  Returns failures."""
    failures: List[str] = []
    equiv = [("\\frac{1}{2}", "0.5"), ("3.500", "3.5"), ("{7}", "7"), ("\\dfrac{3}{4}", "\\frac{3}{4}"),
             ("28", "28")]
    if grader.data_name == "gsm8k":                 # numeric-only answers and grader
        equiv = [("3.500", "3.5"), ("28", "28")]
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
    "INVALID_ANSWER", "extract_answer", "normalize_answer", "majority_vote", "truncate_at_first_boxed",
    "Grader", "sanity_check_grader",
]
