"""
evaluator_classes.py
--------------------
Evaluator classes for answer extraction and equivalence checking.
Ported from the project's Evaluator.py.

Key improvement over plain string equality
-------------------------------------------
MATHEvaluator._check_answers_equiv uses a four-level strategy:
  1. Exact string match (case-insensitive)
  2. Normalised LaTeX match  — strips \\left/\\right, collapses whitespace,
     normalises comma spacing.  Catches cases like:
       \\left( \\frac{3}{5}, \\frac{8}{3} \\right]  ==  \\left(\\frac{3}{5},\\frac{8}{3}\\right]
  3. latex_equiv  (symbolic, via grader.math_equal fallback when unavailable)
  4. latex_equiv on the normalised forms
"""

import re
from typing import Tuple

# ── latex_equiv with graceful fallback ────────────────────────────────────────
try:
    from help_functions.math_toolkit.latex_answer_check import latex_equiv  # type: ignore
except ImportError:
    def latex_equiv(a: str, b: str) -> bool:  # type: ignore
        try:
            from grader import math_equal  # type: ignore
            return bool(math_equal(a, b, timeout=True))
        except Exception:
            return a.strip() == b.strip()


# ──────────────────────────────────────────────────────────────────────────────
# LaTeX normalisation helper
# ──────────────────────────────────────────────────────────────────────────────

def _normalise_latex(s: str) -> str:
    """
    Normalise a LaTeX string so superficial formatting differences
    do not cause false negatives.

    - Collapse all whitespace runs to one space
    - Remove spaces inside/around braces and parentheses
    - Remove \\left and \\right (decorative)
    - Normalise comma spacing
    """
    # 1. Collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    # 2. Spaces around braces
    s = re.sub(r'\{\s+', '{', s)
    s = re.sub(r'\s+\}', '}', s)
    # 3. Spaces around parentheses
    s = re.sub(r'\(\s+', '(', s)
    s = re.sub(r'\s+\)', ')', s)
    # 4. Remove \left / \right
    s = s.replace('\\left', '').replace('\\right', '')
    # 5. Normalise comma spacing
    s = re.sub(r'\s*,\s*', ',', s)
    # 6. Remove spaces before ] and [ brackets
    s = re.sub(r'\s+\]', ']', s)
    s = re.sub(r'\[\s+', '[', s)
    return s.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Base Evaluator
# ──────────────────────────────────────────────────────────────────────────────

class Evaluator:
    def __init__(self, num_last_chars_for_eval=128) -> None:
        self.num_last_chars_for_eval = num_last_chars_for_eval

    def _is_number(self, s) -> Tuple[bool, str]:
        try:
            res = float(s)
            return True, str(res)
        except Exception:
            pass
        try:
            import unicodedata
            res = unicodedata.numeric(s)
            return True, str(res)
        except Exception:
            pass
        return False, None

    def _last_boxed_only_string(self, string):
        if string is None:
            return None
        assert isinstance(string, str)
        idx = string.rfind("\\boxed")
        if idx < 0:
            idx = string.rfind("\\fbox")
            if idx < 0:
                return None
        i, right_brace_idx, num_open = idx, None, 0
        while i < len(string):
            if string[i] == "{":
                num_open += 1
            if string[i] == "}":
                num_open -= 1
                if num_open == 0:
                    right_brace_idx = i
                    break
            i += 1
        if right_brace_idx is None:
            return None
        return string[idx: right_brace_idx + 1]

    def _remove_boxed(self, s):
        if s is None:
            return None
        assert isinstance(s, str)
        results = re.findall(r'\\boxed\{(.*)\}', s, re.DOTALL)
        return results[-1] if results else None

    def validate_model_completion(self, completion: str) -> bool:
        return True

    def validate_model_answer(self, answer: str) -> bool:
        return answer is not None

    def check_answers_equiv(self, answer_a: str, answer_b: str):
        if not self.validate_model_answer(answer_a) or not self.validate_model_answer(answer_b):
            return None
        return (
            self._check_answers_equiv(answer_a, answer_b)
            or self._check_answers_equiv(answer_b, answer_a)
        )

    def _check_answers_equiv(self, answer_a: str, answer_b: str):
        raise NotImplementedError

    def extract_answer_from_gold_solution(self, solution: str) -> str:
        return self._extract_answer_from_gold_solution(solution)

    def _extract_answer_from_gold_solution(self, solution: str) -> str:
        raise NotImplementedError

    def extract_answer_from_model_completion(self, completion: str) -> str:
        if not self.validate_model_completion(completion):
            return None
        return self._extract_answer_from_model_completion(completion)

    def _extract_answer_from_model_completion(self, completion: str) -> str:
        raise NotImplementedError

    def find_majority_completion_and_answer(self, candidates):
        if not candidates:
            return None, None
        clusters = []
        for candidate in candidates:
            c_ans = self.extract_answer_from_model_completion(
                candidate[-self.num_last_chars_for_eval:]
            )
            placed = False
            for cluster in clusters:
                r_ans = self.extract_answer_from_model_completion(
                    cluster[0][-self.num_last_chars_for_eval:]
                )
                if self.check_answers_equiv(c_ans, r_ans):
                    cluster.append(candidate)
                    placed = True
                    break
            if not placed:
                clusters.append([candidate])
        majority_cluster = max(clusters, key=len)
        majority_completion = majority_cluster[0]
        majority_answer = self.extract_answer_from_model_completion(
            majority_completion[-self.num_last_chars_for_eval:]
        )
        return majority_completion, majority_answer


# ──────────────────────────────────────────────────────────────────────────────
# Dataset-specific evaluators
# ──────────────────────────────────────────────────────────────────────────────

class GSM8KEvaluator(Evaluator):
    def __init__(self) -> None:
        super().__init__()

    def _check_answers_equiv(self, answer_a: str, answer_b: str):
        if answer_a is None or answer_b is None:
            return None
        ok_a, num_a = self._is_number(answer_a)
        ok_b, num_b = self._is_number(answer_b)
        return (num_a == num_b) if (ok_a and ok_b) else False

    def validate_model_completion(self, completion: str) -> bool:
        return isinstance(completion, str) and "\\boxed" in completion

    def _extract_answer_from_gold_solution(self, solution):
        if isinstance(solution, float):
            return str(solution)
        if isinstance(solution, str):
            return solution.split("#### ")[-1].strip()
        raise ValueError(f"Unexpected gold solution type: {type(solution)}")

    def _extract_answer_from_model_completion(self, completion: str):
        ans = self._remove_boxed(completion)
        return ans if ans else None



def _numeric_equiv(a: str, b: str, tol: float = 1e-2) -> bool:
    """Try to evaluate both expressions numerically and compare."""
    def _to_float(s):
        try:
            return float(s)
        except Exception:
            pass
        try:
            from sympy import N as sympyN
            from sympy.parsing.latex import parse_latex
            return float(sympyN(parse_latex(s.replace('\\\\', '\\'))))
        except Exception:
            pass
        try:
            from latex2sympy2 import latex2sympy
            from sympy import N as sympyN
            return float(sympyN(latex2sympy(s)))
        except Exception:
            pass
        return None

    fa, fb = _to_float(a), _to_float(b)
    if fa is not None and fb is not None:
        if fb != 0:
            return abs(fa - fb) / abs(fb) < tol
        return abs(fa - fb) < tol
    return False

class MATHEvaluator(Evaluator):
    def __init__(self) -> None:
        super().__init__()

    def _check_answers_equiv(self, answer_a: str, answer_b: str):
        if answer_a is None or answer_b is None:
            return None
        if answer_a == "" or answer_b == "":
            return False

        a = answer_a.strip()
        b = answer_b.strip()

        # 1. Exact match
        if a.lower() == b.lower():
            return True

        # 2. Normalised LaTeX match (whitespace / \\left\\right tolerant)
        na, nb = _normalise_latex(a), _normalise_latex(b)
        if na.lower() == nb.lower():
            return True

        # 3. Symbolic / numeric equivalence via latex_equiv
        try:
            if latex_equiv(a, b):
                return True
        except Exception:
            pass

        # 4. latex_equiv on normalised forms
        try:
            if latex_equiv(na, nb):
                return True
        except Exception:
            pass

        # 5. Numeric equivalence (handles \frac{1}{12} == 0.0833...)
        if _numeric_equiv(a, b):
            return True

        return False

    def validate_model_completion(self, completion: str) -> bool:
        return isinstance(completion, str) and "\\boxed" in completion

    def _extract_answer_from_gold_solution(self, solution: str):
        box = self._last_boxed_only_string(solution)
        ans = self._remove_boxed(box)
        return ans if ans else None

    def _extract_answer_from_model_completion(self, completion: str):
        box = self._last_boxed_only_string(completion)
        ans = self._remove_boxed(box)
        return ans if ans else None


class OlympiadEvaluator(Evaluator):
    def __init__(self) -> None:
        super().__init__()

    def _check_answers_equiv(self, answer_a: str, answer_b: str):
        if answer_a is None or answer_b is None:
            return None
        if answer_a == "" or answer_b == "":
            return False
        a, b = answer_a.strip(), answer_b.strip()
        if a.lower() == b.lower():
            return True
        na, nb = _normalise_latex(a), _normalise_latex(b)
        if na.lower() == nb.lower():
            return True
        try:
            if latex_equiv(a, b):
                return True
        except Exception:
            pass
        try:
            return latex_equiv(na, nb)
        except Exception:
            return False

    def _extract_answer_from_gold_solution(self, solution: str):
        return solution.strip("$")

    def _extract_answer_from_model_completion(self, completion: str):
        box = self._last_boxed_only_string(completion)
        ans = self._remove_boxed(box)
        return ans if ans else None


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────

# ── Multiple-choice evaluator (MMLU, ARC-Challenge) ───────────────────────────
# Gold answer is a single letter A/B/C/D.
# Model answer is extracted by looking for the last standalone letter
# A/B/C/D in the completion (inside \boxed{} if present, otherwise raw).
# Equivalence: exact letter match after normalisation.

class MCQEvaluator(Evaluator):
    """
    Evaluator for multiple-choice benchmarks where the gold answer is a
    single letter (A, B, C, D).

    Used for: MMLU, ARC-Challenge.

    Answer extraction strategy (in priority order):
      1. Content of \\boxed{} if present.
      2. Last occurrence of a standalone A/B/C/D/E in the completion.
      3. First character of the stripped completion if it is a letter.
    """

    # Letters that are valid single-option answers
    _VALID = set("ABCDE")

    def _check_answers_equiv(self, answer_a: str, answer_b: str):
        if answer_a is None or answer_b is None:
            return None
        # Normalise both to uppercase letter
        a = answer_a.strip().upper()
        b = answer_b.strip().upper()
        # Strip trailing punctuation (e.g. "A." → "A")
        a = a.rstrip(".):").strip()
        b = b.rstrip(".):").strip()
        if not a or not b:
            return False
        return a == b

    def validate_model_completion(self, completion: str) -> bool:
        # Accept any non-empty string; extraction handles missing \boxed{}
        return isinstance(completion, str) and len(completion.strip()) > 0

    def _extract_answer_from_gold_solution(self, solution: str) -> str:
        return solution.strip().upper().rstrip(".):").strip()

    def _extract_answer_from_model_completion(self, completion: str) -> str:
        if not completion:
            return ""

        # 1. Try \boxed{} first
        box = self._last_boxed_only_string(completion)
        if box:
            ans = self._remove_boxed(box)
            if ans:
                return ans.strip().upper().rstrip(".):").strip()

        # 2. Last standalone letter A-E preceded by whitespace / punctuation
        import re as _re
        matches = _re.findall(r'(?<![A-Za-z])([A-E])(?![A-Za-z])', completion.upper())
        if matches:
            return matches[-1]

        # 3. First character of stripped completion
        first = completion.strip()[0].upper()
        if first in self._VALID:
            return first

        return completion.strip()[:1].upper()


# ── Yes/No evaluator (StrategyQA) ──────────────────────────────────────────────
# Gold answer is "yes" or "no".  Model answer is extracted by searching
# for the first occurrence of "yes" or "no" in the completion.

class YesNoEvaluator(Evaluator):
    """
    Evaluator for binary yes/no questions.

    Used for: StrategyQA.

    Answer extraction: search for "yes"/"no" (case-insensitive) in the
    completion, preferring \\boxed{} content first.
    """

    def _check_answers_equiv(self, answer_a: str, answer_b: str):
        if answer_a is None or answer_b is None:
            return None
        return answer_a.strip().lower() == answer_b.strip().lower()

    def validate_model_completion(self, completion: str) -> bool:
        return isinstance(completion, str) and len(completion.strip()) > 0

    def _extract_answer_from_gold_solution(self, solution: str) -> str:
        return solution.strip().lower()

    def _extract_answer_from_model_completion(self, completion: str) -> str:
        if not completion:
            return ""

        # 1. \boxed{} content
        box = self._last_boxed_only_string(completion)
        if box:
            ans = self._remove_boxed(box)
            if ans:
                lower = ans.strip().lower()
                if lower in ("yes", "no"):
                    return lower

        # 2. First occurrence of "yes" or "no" as a whole word
        import re as _re
        m = _re.search(r'\b(yes|no)\b', completion.lower())
        if m:
            return m.group(1)

        return ""


# ── BBH (Big-Bench Hard) evaluator ────────────────────────────────────────────
# BBH tasks have heterogeneous answer formats (letters, words, numbers, dates).
# We use a multi-strategy approach: exact match → normalised string → numeric.

class BBHEvaluator(Evaluator):
    """
    Evaluator for BIG-Bench Hard tasks.

    BBH answers can be:
      - Multiple-choice letters: (A), (B), (C) …
      - Short strings: "valid", "True", "False", specific words
      - Numbers or dates

    Extraction: \\boxed{} first, then last parenthesised letter (A)/(B)…,
    then last quoted string, then full stripped completion.
    """

    def _check_answers_equiv(self, answer_a: str, answer_b: str):
        if answer_a is None or answer_b is None:
            return None
        a = answer_a.strip().lower().strip("().,")
        b = answer_b.strip().lower().strip("().,")
        if a == b:
            return True
        # Numeric fallback
        try:
            return abs(float(a) - float(b)) < 1e-6
        except (ValueError, TypeError):
            pass
        return False

    def validate_model_completion(self, completion: str) -> bool:
        return isinstance(completion, str) and len(completion.strip()) > 0

    def _extract_answer_from_gold_solution(self, solution: str) -> str:
        return solution.strip()

    def _extract_answer_from_model_completion(self, completion: str) -> str:
        if not completion:
            return ""

        # 1. \boxed{} content
        box = self._last_boxed_only_string(completion)
        if box:
            ans = self._remove_boxed(box)
            if ans:
                return ans.strip()

        # 2. Last parenthesised option: (A), (B), …
        import re as _re
        matches = _re.findall(r'\(([A-E])\)', completion.upper())
        if matches:
            return f"({matches[-1]})"

        # 3. Last line of the completion (often the answer in BBH)
        lines = [l.strip() for l in completion.strip().splitlines() if l.strip()]
        if lines:
            return lines[-1]

        return completion.strip()


EvaluatorMapping = {
    "GSM8K":             GSM8KEvaluator,
    "GSM_Hard":          GSM8KEvaluator,
    "MATH":              MATHEvaluator,
    "MATH500":           MATHEvaluator,
    "ScaleQuest":        MATHEvaluator,
    "ScaleQuestLabeled": MATHEvaluator,
    "OpenMathInstruct":  MATHEvaluator,
    "AMC2022":           MATHEvaluator,
    "AMC2023":           MATHEvaluator,
    "AMC":               MATHEvaluator,
    "AIME":              MATHEvaluator,
    "AIME2023":          MATHEvaluator,
    "AIME2024":          MATHEvaluator,
    "OlympiadBench":     OlympiadEvaluator,
    "NuminaMath":        MATHEvaluator,
    # ── New general-reasoning benchmarks ──────────────────────────────────────
    "MMLU":              MCQEvaluator,
    "ARC":               MCQEvaluator,
    "ARC_Challenge":     MCQEvaluator,
    "BBH":               BBHEvaluator,
    "TruthfulQA":        MCQEvaluator,   # gold = correct choice text; compare as string
    "StrategyQA":        YesNoEvaluator,
}


def get_evaluator(task_name: str = "MATH500") -> Evaluator:
    if task_name in EvaluatorMapping:
        return EvaluatorMapping[task_name]()
    if any(x in task_name for x in ["MATH", "ScaleQuest", "NuminaMath", "OpenMathInstruct"]):
        return MATHEvaluator()
    if any(x in task_name for x in ["MMLU", "ARC"]):
        return MCQEvaluator()
    if "BBH" in task_name:
        return BBHEvaluator()
    if "StrategyQA" in task_name or "strategy" in task_name.lower():
        return YesNoEvaluator()
    if "TruthfulQA" in task_name or "truthful" in task_name.lower():
        return MCQEvaluator()
    raise ValueError(f"Task '{task_name}' not in EvaluatorMapping")