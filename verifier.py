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

    All reasoning steps from ALL traces are collected into one flat list,
    then scored in padded mini-batches of `batch_size` in a single
    model.forward() call per mini-batch — no per-step or per-trace loops
    that touch the GPU individually.

    Parameters
    ----------
    model_name  : HF Hub PRM model id
    aggregation : 'min' | 'mean' | 'last'   — how to collapse step scores
    device      : 'auto' | 'cuda' | 'cpu'
    batch_size  : number of steps per model.forward() call (default 8)
                  tune this to fit PRM VRAM: larger = faster, more VRAM
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Math-PRM-7B",
        aggregation: str = "min",
        device: str = "auto",
        cache_dir: Optional[str] = None,
        batch_size: int = 8,
    ):
        self.aggregation = aggregation
        self.batch_size  = batch_size
        self._fallback   = LengthHeuristicVerifier()

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name, cache_dir=cache_dir
            )
            # left-pad so batches of different-length steps align correctly
            self._tokenizer.padding_side = "left"
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

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
            logger.info(f"PRM loaded: {model_name}  |  batch_size={batch_size}")
        except Exception as e:
            logger.warning(
                f"Could not load PRM '{model_name}': {e}. "
                f"Falling back to LengthHeuristicVerifier."
            )
            self._loaded = False

    # ── batched step scoring  (the only place model.forward() is called) ─────

    def _score_step_batch(self, steps: List[str]) -> List[float]:
        """
        Score a flat list of step strings in mini-batches of self.batch_size.
        Returns a flat list of float scores, one per step.
        All steps in one mini-batch are tokenised together into one padded
        tensor and scored in a single model.forward() call.
        """
        import torch
        scores: List[float] = []

        for i in range(0, len(steps), self.batch_size):
            mini_batch = steps[i : i + self.batch_size]
            enc = self._tokenizer(
                mini_batch,
                return_tensors="pt",
                padding=True,        # pad all steps to same length
                truncation=True,
                max_length=512,
            ).to(self._device)

            with torch.no_grad():
                logits      = self._model(**enc).logits          # (B, num_labels)
                batch_scores = torch.sigmoid(logits[:, 0])       # (B,)
                scores.extend(batch_scores.cpu().tolist())

        return scores

    # ── public API ────────────────────────────────────────────────────────────

    def score_batch(self, traces: List[str]) -> np.ndarray:
        """
        Score N traces in one batched pass.

        Strategy
        --------
        1. Split every trace into reasoning steps.
        2. Collect ALL steps from ALL traces into one flat list.
        3. Score the flat list in mini-batches via _score_step_batch()
           → each mini-batch = one model.forward() call.
        4. Re-group scores back per trace and aggregate (min/mean/last).

        This means: for N=8 traces each with ~5 steps = 40 steps total.
        With batch_size=8 that is 5 model.forward() calls, not 40.
        """
        if not self._loaded:
            return self._fallback.score_batch(traces)

        # Step 1 & 2: flatten all steps from all traces
        all_steps: List[str] = []
        step_counts: List[int] = []

        for trace in traces:
            steps = [
                s.strip()
                for s in re.split(r"\n\n+|Step \d+[:.]\s*", trace)
                if s.strip()
            ]
            if not steps:
                steps = [trace]
            all_steps.extend(steps)
            step_counts.append(len(steps))

        # Step 3: score entire flat list in mini-batches
        all_scores = self._score_step_batch(all_steps)

        # Step 4: re-group and aggregate per trace
        final_scores: List[float] = []
        idx = 0
        for cnt in step_counts:
            step_scores = all_scores[idx : idx + cnt]
            idx += cnt
            if self.aggregation == "min":
                final_scores.append(float(min(step_scores)))
            elif self.aggregation == "last":
                final_scores.append(float(step_scores[-1]))
            else:  # mean
                final_scores.append(float(np.mean(step_scores)))

        return np.array(final_scores)


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
    batch_size: int = 8,
):
    """
    Factory for verifiers.

    verifier_type : 'outcome' | 'length' | 'prm'
    batch_size    : used only by HFProcessRewardVerifier (steps per forward call)
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
            batch_size=batch_size,
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