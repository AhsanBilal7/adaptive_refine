"""
verifier.py
-----------
Scoring / verification of candidate reasoning traces.

Three verifier back-ends are provided:

1. OutcomeVerifier  – extracts the final \boxed{} answer and uses
                      math_equal (grader.py) for binary correctness.
                      Returns 1.0 (correct) or 0.0 (wrong).

2. LengthHeuristicVerifier  – a lightweight proxy score that rewards
                               shorter, cleaner answers.  Useful when
                               no gold answer is available at inference
                               time (purely for ranking purposes).

3. HFProcessRewardVerifier  – wraps a HuggingFace Process Reward Model
                               (PRM) to get a continuous quality score
                               per trace.  Falls back to OutcomeVerifier
                               when the PRM cannot be loaded.

The main function `score_candidates` accepts a list of traces and
returns a numpy array of float scores (higher = better).
"""

import logging
import re
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: extract the boxed answer from a model trace
# ──────────────────────────────────────────────────────────────────────────────

def extract_boxed_answer(trace: str) -> str:
    """Return the content of the last \\boxed{} in a trace, or ''."""
    if "boxed" not in trace:
        return ""
    ans = trace.split("boxed")[-1]
    if not ans:
        return ""
    if ans[0] == "{":
        stack, a = 1, ""
        for c in ans[1:]:
            if c == "{":
                stack += 1; a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
        return a
    return ans.split("$")[0].strip()


# ──────────────────────────────────────────────────────────────────────────────
# 1. Outcome Verifier  (exact-match with math_equal)
# ──────────────────────────────────────────────────────────────────────────────

class OutcomeVerifier:
    """
    Binary verifier: uses math_equal from grader.py.

    Scores 1.0 if the extracted answer matches the gold answer, else 0.0.
    Requires the ground-truth answer at verification time.
    """

    def __init__(self, data_name: str = "math", timeout: bool = True):
        self.data_name = data_name
        self.timeout = timeout

    def score(self, trace: str, gold_answer: str) -> float:
        try:
            from parser import extract_answer  # type: ignore
            pred = extract_answer(trace, self.data_name)
        except Exception:
            pred = extract_boxed_answer(trace)

        try:
            from grader import math_equal  # type: ignore
            return 1.0 if math_equal(pred, gold_answer, timeout=self.timeout) else 0.0
        except Exception:
            return 1.0 if str(pred).strip() == str(gold_answer).strip() else 0.0

    def score_batch(self, traces: List[str], gold_answer: str) -> np.ndarray:
        return np.array([self.score(t, gold_answer) for t in traces])


# ──────────────────────────────────────────────────────────────────────────────
# 2. Length-Heuristic Verifier  (proxy score – no gold needed)
# ──────────────────────────────────────────────────────────────────────────────

class LengthHeuristicVerifier:
    """
    Proxy verifier that assigns higher score to:
      - traces that contain a \boxed{} answer  (+0.5 bonus)
      - shorter traces  (normalised inverse-length score in [0, 0.5])

    This is useful purely for ranking when no gold answer is available
    at inference time (e.g. real-world deployment).
    """

    def __init__(self, max_chars: int = 4000):
        self.max_chars = max_chars

    def score(self, trace: str) -> float:
        has_box = 1.0 if extract_boxed_answer(trace) else 0.0
        length_score = max(0.0, 1.0 - len(trace) / self.max_chars)
        return 0.5 * has_box + 0.5 * length_score

    def score_batch(self, traces: List[str]) -> np.ndarray:
        return np.array([self.score(t) for t in traces])


# ──────────────────────────────────────────────────────────────────────────────
# 3. HF Process-Reward Verifier
# ──────────────────────────────────────────────────────────────────────────────

class HFProcessRewardVerifier:
    """
    Wraps a HuggingFace PRM (e.g. 'Qwen/Qwen2.5-Math-PRM-7B').

    The PRM scores each reasoning step; we return the min-step score
    (conservative) or mean-step score depending on `aggregation`.

    Falls back to LengthHeuristicVerifier if the model cannot be loaded.

    Parameters
    ----------
    model_name : HF Hub PRM model id
    aggregation: 'min' | 'mean' | 'last'
    device     : 'auto' | 'cuda' | 'cpu'
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Math-PRM-7B",
        aggregation: str = "min",
        device: str = "auto",
        cache_dir: Optional[str] = None,
    ):
        self.aggregation = aggregation
        self._fallback = LengthHeuristicVerifier()

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name, cache_dir=cache_dir
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=device if device == "auto" else None,
                cache_dir=cache_dir,
            )
            if device not in ("auto",):
                self._model = self._model.to(device)
            self._model.eval()
            self._device = next(self._model.parameters()).device
            self._loaded = True
            logger.info(f"PRM loaded: {model_name}")
        except Exception as e:
            logger.warning(f"Could not load PRM '{model_name}': {e}. "
                           f"Falling back to LengthHeuristicVerifier.")
            self._loaded = False

    def _score_one(self, trace: str) -> float:
        if not self._loaded:
            return self._fallback.score(trace)
        import torch
        steps = [s.strip() for s in re.split(r"\n\n+|Step \d+[:.]\s*", trace) if s.strip()]
        if not steps:
            steps = [trace]
        step_scores = []
        for step in steps:
            enc = self._tokenizer(
                step, return_tensors="pt", truncation=True, max_length=512
            ).to(self._device)
            with torch.no_grad():
                logits = self._model(**enc).logits
                score = torch.sigmoid(logits[0, 0]).item()
            step_scores.append(score)
        if not step_scores:
            return 0.5
        if self.aggregation == "min":
            return float(min(step_scores))
        elif self.aggregation == "last":
            return float(step_scores[-1])
        return float(np.mean(step_scores))

    def score_batch(self, traces: List[str]) -> np.ndarray:
        return np.array([self._score_one(t) for t in traces])


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_verifier(
    verifier_type: str = "outcome",
    data_name: str = "math",
    prm_model: Optional[str] = None,
    prm_aggregation: str = "min",
    device: str = "auto",
    cache_dir: Optional[str] = None,
):
    """
    Factory for verifiers.

    verifier_type: 'outcome' | 'length' | 'prm'
    """
    if verifier_type == "outcome":
        return OutcomeVerifier(data_name=data_name)
    elif verifier_type == "length":
        return LengthHeuristicVerifier()
    elif verifier_type == "prm":
        mdl = prm_model or "Qwen/Qwen2.5-Math-PRM-7B"
        return HFProcessRewardVerifier(
            model_name=mdl,
            aggregation=prm_aggregation,
            device=device,
            cache_dir=cache_dir,
        )
    raise ValueError(f"Unknown verifier_type: {verifier_type}")


# ──────────────────────────────────────────────────────────────────────────────
# Unified scoring entry point
# ──────────────────────────────────────────────────────────────────────────────

def score_candidates(
    traces: List[str],
    verifier,
    gold_answer: Optional[str] = None,
) -> np.ndarray:
    """
    Score a list of candidate traces.

    Parameters
    ----------
    traces      : list of model-generated reasoning strings
    verifier    : an instance of any verifier class above
    gold_answer : required for OutcomeVerifier; ignored otherwise

    Returns
    -------
    np.ndarray of float scores, shape (len(traces),)
    """
    if isinstance(verifier, OutcomeVerifier):
        if gold_answer is None:
            raise ValueError("OutcomeVerifier requires gold_answer.")
        return verifier.score_batch(traces, gold_answer)
    elif isinstance(verifier, LengthHeuristicVerifier):
        return verifier.score_batch(traces)
    elif isinstance(verifier, HFProcessRewardVerifier):
        return verifier.score_batch(traces)
    raise TypeError(f"Unsupported verifier type: {type(verifier)}")
