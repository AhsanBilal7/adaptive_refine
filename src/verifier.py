"""
verifier.py
-----------
Scoring / verification of candidate reasoning traces.

Three verifier back-ends:

1. OutcomeVerifier          — binary correct/wrong vs gold answer (oracle only,
                              leaks gold into selection — do NOT use for fair eval).

2. LengthHeuristicVerifier  — proxy score; no gold needed; deployment baseline.

3. HFProcessRewardVerifier  — Qwen2.5-Math-PRM-7B correct implementation.

Correct PRM usage (Qwen2.5-Math-PRM-7B)
-----------------------------------------
The model is NOT a sequence classifier.  It is an AutoModel whose final hidden
states at special <extra_0> step-separator tokens carry the per-step reward
signal.  The correct procedure is:

  1. Build a FULL chat-formatted string with the system prompt, user question,
     and the assistant's full response, where reasoning steps are joined with
     the <extra_0> separator token.
  2. Run ONE forward pass over the full sequence.
  3. Extract hidden-state logits ONLY at the positions of <extra_0> tokens.
  4. Apply softmax over the 2-class dimension → take the positive-class
     probability (index 1) as the per-step reward score.
  5. Aggregate step scores per trace: min / mean / last.

This preserves full left-context for every step — step 3 is scored knowing
what steps 1 and 2 said, which is the whole point of a Process Reward Model.

The old implementation used AutoModelForSequenceClassification and scored each
step as an isolated string.  This was wrong in two ways:
  - Wrong model class (classification head vs hidden-state rewards).
  - No context: step 3 scored without knowing steps 1-2.
"""

import logging
import re
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Step separator token used by Qwen2.5-Math-PRM-7B
STEP_SEP = "<extra_0>"

# System prompt used during PRM training — must match at scoring time
PRM_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


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
# 1. Outcome Verifier
# ──────────────────────────────────────────────────────────────────────────────

class OutcomeVerifier:
    """
    Binary verifier: uses math_equal from grader.py.

    WARNING: leaks gold answer into selection.
    Use ONLY as an oracle upper bound — never for fair evaluation.
    """

    def __init__(self, data_name: str = "math", timeout: bool = True):
        self.data_name = data_name
        self.timeout   = timeout

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
# 2. Length-Heuristic Verifier
# ──────────────────────────────────────────────────────────────────────────────

class LengthHeuristicVerifier:
    """
    Proxy verifier — no gold, no PRM needed.
    Rewards traces that have a \\boxed{} answer and are concise.
    Use only as a deployment / no-label baseline.
    """

    def __init__(self, max_chars: int = 4000):
        self.max_chars = max_chars

    def score(self, trace: str) -> float:
        has_box      = 1.0 if extract_boxed_answer(trace) else 0.0
        length_score = max(0.0, 1.0 - len(trace) / self.max_chars)
        return 0.5 * has_box + 0.5 * length_score

    def score_batch(self, traces: List[str]) -> np.ndarray:
        return np.array([self.score(t) for t in traces])


# ──────────────────────────────────────────────────────────────────────────────
# 3. HF Process-Reward Verifier  (Qwen2.5-Math-PRM-7B correct implementation)
# ──────────────────────────────────────────────────────────────────────────────

class HFProcessRewardVerifier:
    """
    Correct implementation of Qwen2.5-Math-PRM-7B scoring.

    Key design decisions
    --------------------
    Model class : AutoModel  (NOT AutoModelForSequenceClassification)
      The reward signal comes from hidden-state logits at <extra_0> positions,
      not from a classification head over the [CLS] token.

    Input format : full chat conversation with <extra_0> step separators
      Each trace is formatted as:
        [system] + [user: question] + [assistant: step1<extra_0>step2<extra_0>...]
      The FULL sequence is passed in ONE forward pass so every step is scored
      with complete left-context from all preceding steps.

    Score extraction : make_step_rewards()
      Finds <extra_0> token positions in the input_ids.
      Reads hidden-state logits at those positions.
      Applies softmax over the 2-class dim and takes index 1 (good-step prob).

    Aggregation : min | mean | last
      min  — most conservative; penalises any weak step (recommended)
      mean — average quality across all steps
      last — quality of the final answer step only

    Batching
    --------
    Each trace requires a separate forward pass (different question context).
    Traces for the same question share the same question prefix — they could
    be batched, but only if padded to the same length which is expensive for
    long traces.  We process one trace at a time for correctness, batching
    only within a single trace's steps via the flat-sequence approach.
    For throughput, increase batch_size to process multiple traces at once
    (traces are padded to the same length within the batch).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Math-PRM-7B",
        aggregation: str = "min",
        device: str = "auto",
        cache_dir: Optional[str] = None,
        batch_size: int = 8,        # traces per forward pass (not steps)
        system_prompt: str = PRM_SYSTEM_PROMPT,
    ):
        self.aggregation   = aggregation
        self.batch_size    = batch_size
        self.system_prompt = system_prompt

        import torch
        from transformers import AutoModel, AutoTokenizer

        logger.info(f"Loading PRM tokenizer: {model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        # Padding side: right for causal LM with variable-length sequences.
        # The reward is read from <extra_0> positions inside the sequence,
        # not from a fixed position, so padding side affects only the
        # attention mask — right-padding keeps the real tokens contiguous.
        self._tokenizer.padding_side = "right"
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        # Cache the <extra_0> token id once — used in every forward pass
        self._step_sep_id = self._tokenizer.encode(STEP_SEP)[0]
        logger.info(f"  step_sep_id ({STEP_SEP!r}) = {self._step_sep_id}")

        logger.info(f"Loading PRM model: {model_name}")
        self._model = AutoModel.from_pretrained(
            model_name,
            device_map=device if device == "auto" else None,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        self._model.config.use_cache = False
        if device not in ("auto",):
            self._model = self._model.to(device)
        self._model.eval()
        self._device = next(self._model.parameters()).device
        logger.info(
            f"PRM loaded on {self._device}  |  "
            f"aggregation={aggregation}  batch_size={batch_size}"
        )

    # ── step splitting ────────────────────────────────────────────────────────

    def _split_steps(self, trace: str) -> List[str]:
        """
        Split a model trace into reasoning steps.

        Strategy:
          1. If the trace already contains the PRM step separator, split on it.
          2. Otherwise split on double newlines or 'Step N:' markers.
          3. Always return at least one step (the full trace).
        """
        if STEP_SEP in trace:
            steps = [s.strip() for s in trace.split(STEP_SEP) if s.strip()]
            return steps if steps else [trace]

        steps = [
            s.strip()
            for s in re.split(r"\n\n+|Step \d+[:.]\s*", trace)
            if s.strip()
        ]
        return steps if steps else [trace]

    # ── chat template builder ─────────────────────────────────────────────────

    def _build_conversation_str(self, question: str, steps: List[str]) -> str:
        """
        Build the full chat-formatted string for one trace.

        Format (matches PRM training format):
          [system prompt]
          [user: question]
          [assistant: step1<extra_0>step2<extra_0>...stepN<extra_0>]

        The <extra_0> token is appended after EVERY step including the last.
        """
        assistant_content = STEP_SEP.join(steps) + STEP_SEP
        messages = [
            {"role": "system",    "content": self.system_prompt},
            {"role": "user",      "content": question},
            {"role": "assistant", "content": assistant_content},
        ]
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    # ── per-step reward extraction ────────────────────────────────────────────

    @staticmethod
    def _make_step_rewards(logits, token_masks) -> List[List[float]]:
        """
        Extract per-step reward scores from the model's hidden-state logits.

        Taken directly from the Qwen2.5-Math-PRM-7B official usage example.

        Parameters
        ----------
        logits      : tensor (batch, seq_len, 2)  — model output hidden states
        token_masks : tensor (batch, seq_len)      — True at <extra_0> positions

        Returns
        -------
        List[List[float]] — one list of step scores per sequence in the batch
        """
        import torch.nn.functional as F
        probabilities = F.softmax(logits, dim=-1)                       # (B, L, 2)
        probabilities = probabilities * token_masks.unsqueeze(-1)       # zero non-sep positions
        all_scores = []
        for i in range(probabilities.size(0)):
            sample        = probabilities[i]                            # (L, 2)
            # Extract rows where token_mask was True (non-zero rows)
            positive_probs = sample[sample.sum(-1) != 0][:, 1]         # (n_steps,)
            all_scores.append(positive_probs.cpu().tolist())
        return all_scores

    # ── single-trace scoring ──────────────────────────────────────────────────

    def _score_single_trace(self, question: str, trace: str) -> float:
        """
        Score one trace by running a single forward pass over the full
        conversation sequence.  Returns aggregated step reward (float).
        """
        import torch

        steps   = self._split_steps(trace)
        conv    = self._build_conversation_str(question, steps)
        input_ids = self._tokenizer.encode(
            conv, return_tensors="pt"
        ).to(self._device)

        token_masks = (input_ids == self._step_sep_id)   # (1, seq_len)

        if token_masks.sum() == 0:
            # No step separators found — degenerate trace; return 0.0
            logger.debug("No <extra_0> tokens found in trace; returning 0.0")
            return 0.0

        with torch.no_grad():
            outputs = self._model(input_ids=input_ids)

        # outputs[0] is the last hidden state: (1, seq_len, hidden_dim)
        # For Qwen2.5-Math-PRM-7B, the model head maps hidden_dim → 2 labels
        logits = outputs[0]                                   # (1, seq_len, 2)
        step_scores = self._make_step_rewards(logits, token_masks)[0]

        if not step_scores:
            return 0.0

        if self.aggregation == "min":
            return float(min(step_scores))
        elif self.aggregation == "last":
            return float(step_scores[-1])
        else:  # mean
            return float(np.mean(step_scores))

    # ── batched public API ────────────────────────────────────────────────────

    def score_batch(
        self,
        traces: List[str],
        question: Optional[str] = None,
    ) -> np.ndarray:
        """
        Score a list of candidate traces for ONE question.

        Parameters
        ----------
        traces   : N candidate traces (all answers to the same question)
        question : the question text — required for correct PRM input format.
                   If None (legacy call), traces are scored without question
                   context (less accurate but still works).

        Returns
        -------
        np.ndarray of float scores, shape (N,), higher = better reasoning.
        """
        if question is None:
            logger.warning(
                "score_batch called without question context. "
                "PRM scores will be less accurate. "
                "Pass question= for correct results."
            )
            # Score without question context — build a minimal conversation
            question = ""

        scores = []
        for trace in traces:
            scores.append(self._score_single_trace(question, trace))
        return np.array(scores, dtype=float)


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_verifier(
    verifier_type: str = "prm",
    data_name: str = "math",
    prm_model: Optional[str] = None,
    prm_aggregation: str = "min",
    device: str = "auto",
    cache_dir: Optional[str] = None,
    batch_size: int = 8,
):
    """
    Factory for verifiers.

    verifier_type : 'prm' (default) | 'outcome' | 'length'
    batch_size    : traces per forward pass for HFProcessRewardVerifier
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
    raise ValueError(f"Unknown verifier_type: {verifier_type!r}")


# ──────────────────────────────────────────────────────────────────────────────
# Unified scoring entry point
# ──────────────────────────────────────────────────────────────────────────────

def score_candidates(
    traces: List[str],
    verifier,
    gold_answer: Optional[str] = None,
    question: Optional[str] = None,
) -> np.ndarray:
    """
    Score a list of candidate traces.

    Parameters
    ----------
    traces      : model-generated reasoning strings (all for ONE question)
    verifier    : instance of any verifier class above
    gold_answer : required for OutcomeVerifier; ignored otherwise
    question    : the question text — passed to HFProcessRewardVerifier
                  so each trace is scored in full conversation context.
                  Should always be provided when using PRM.

    Returns
    -------
    np.ndarray of float scores, shape (len(traces),), higher = better
    """
    if isinstance(verifier, OutcomeVerifier):
        if gold_answer is None:
            raise ValueError("OutcomeVerifier requires gold_answer.")
        return verifier.score_batch(traces, gold_answer)

    elif isinstance(verifier, LengthHeuristicVerifier):
        return verifier.score_batch(traces)

    elif isinstance(verifier, HFProcessRewardVerifier):
        return verifier.score_batch(traces, question=question)

    raise TypeError(f"Unsupported verifier type: {type(verifier)}")