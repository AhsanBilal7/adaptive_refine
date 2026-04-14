"""
inference_methods.py
--------------------
Implements the three test-time compute strategies:

  1. GreedyInference  – single greedy (argmax) decode per question.
  2. BestOfN          – sample N traces in ONE batched call, pick best-scored.
  3. DNBestOfN        – Direction-Normalised BoN (proposed method).

Evaluator integration
---------------------
Each inference class creates ONE Evaluator instance at __init__ time
(e.g. MATHEvaluator for math500, GSM8KEvaluator for gsm8k).  That single
instance is stored as self.ev and reused for every question in the run.

  self.ev.extract_answer_from_model_completion(trace)  → predicted answer
  self.ev.check_answers_equiv(pred, gold)              → is_correct

This means the evaluator is never re-imported or re-instantiated per question.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class InferenceResult:
    method:           str
    question:         str
    gold_answer:      Optional[str]
    selected_trace:   str
    predicted_answer: str
    is_correct:       Optional[bool]
    N:                int
    elapsed_sec:      float

    all_traces:        List[str]   = field(default_factory=list)
    raw_scores:        List[float] = field(default_factory=list)
    normalized_scores: List[float] = field(default_factory=list)
    cluster_labels:    List[int]   = field(default_factory=list)
    cluster_sizes:     List[int]   = field(default_factory=list)
    selected_index:    int         = -1
    alpha:             float       = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method":             self.method,
            "gold_answer":        self.gold_answer,
            "predicted_answer":   self.predicted_answer,
            "raw_llm_output":     self.selected_trace,
            "is_correct":         self.is_correct,
            "N":                  self.N,
            "elapsed_sec":        round(self.elapsed_sec, 3),
            "selected_index":     self.selected_index,
            "alpha":              self.alpha,
            # ── Cluster fields (DN-BoN and BestOfNMajority populate these) ──────
            # cluster_labels[i]  : cluster id for trace i  (int, 0-indexed)
            # cluster_sizes[i]   : size of trace i's cluster
            "cluster_labels":     self.cluster_labels,
            "cluster_sizes":      self.cluster_sizes,
            # ── Score fields ────────────────────────────────────────────────────
            # raw_scores[i]       : verifier score for trace i
            # normalized_scores[i]: raw / cluster_size^alpha  (DN-BoN)
            "raw_scores":         [round(s, 4) for s in self.raw_scores],
            "normalized_scores":  [round(s, 4) for s in self.normalized_scores],
            # ── Per-trace predictions (needed for all 5 motivation experiments) ─
            # Saved as character lengths here; full traces saved separately if
            # save_all_traces=True is set in the evaluator call.
            # trace_lengths[i]   : number of characters in trace i
            "trace_lengths":      [len(t) for t in self.all_traces] if self.all_traces else [],
        }


# ──────────────────────────────────────────────────────────────────────────────
# Evaluator factory  — called ONCE per method instance, not per question
# ──────────────────────────────────────────────────────────────────────────────

# dataset name → task key understood by evaluator_classes.get_evaluator()
_EVALUATOR_MAP: Dict[str, str] = {
    "math500":       "MATH500",
    "math":          "MATH500",
    "aime24":        "AIME2024",
    "aime25":        "AIME",
    "amc23":         "AMC2023",
    "gsm8k":         "GSM8K",
    "olympiadbench": "OlympiadBench",
    "college_math":  "MATH500",
    "minerva_math":  "MATH500",
    "custom":        "MATH500",
}


def _build_evaluator(data_name: str):
    """
    Instantiate and return the correct Evaluator for `data_name`.
    Called once in __init__ of each inference class; the result is stored
    as self.ev and reused for the entire evaluation run.
    """
    try:
        from evaluator_classes import get_evaluator  # type: ignore
        task = _EVALUATOR_MAP.get(data_name, "MATH500")
        ev = get_evaluator(task)
        logger.info(f"Evaluator: {ev.__class__.__name__} for dataset='{data_name}'")
        return ev
    except Exception as e:
        logger.warning(
            f"Could not build Evaluator for '{data_name}': {e}. "
            f"Falling back to grader.math_equal."
        )
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Extraction and correctness helpers  (use the pre-built evaluator)
# ──────────────────────────────────────────────────────────────────────────────

def _extract_pred(trace: str, ev, data_name: str) -> str:
    """
    Extract the final answer from a model trace.

    Priority
    --------
    1. ev.extract_answer_from_model_completion()  — dataset-specific, \boxed{}-aware
    2. parser.extract_answer()                    — broad LaTeX parser fallback
    3. Direct \boxed{} brace-matching
    4. Last number / expression in the trace      — handles no-boxed AIME outputs
    """
    # 1. Evaluator (the correct path for all supported datasets).
    #    Skip validate_model_completion so we still extract even without \boxed{}.
    if ev is not None:
        try:
            ans = ev._extract_answer_from_model_completion(trace)
            if ans is not None and ans != "":
                return ans
        except Exception:
            pass

    # 2. parser.py fallback
    try:
        from parser import extract_answer  # type: ignore
        result = extract_answer(trace, data_name)
        if result and result != "NONE":
            return result
    except Exception:
        pass

    # 3. Direct \boxed{} brace-matching
    try:
        from verifier import extract_boxed_answer  # type: ignore
        ans = extract_boxed_answer(trace)
        if ans:
            return ans
    except Exception:
        pass

    # 4. Last-resort: extract the last integer or decimal in the trace.
    #    Useful for AIME-style answers where the model writes the number
    #    at the end without \boxed{} formatting.
    import re as _re
    nums = _re.findall(r"-?\d+(?:\.\d+)?", trace.replace(",", ""))
    if nums:
        return nums[-1]

    return ""


def _check_correct(pred: str, gold: Optional[str], ev, data_name: str) -> Optional[bool]:
    """
    Check answer correctness.

    Priority
    --------
    1. ev.check_answers_equiv()  — latex_equiv + normalisation aware
    2. grader.math_equal()       — sympy-based fallback
    3. Normalised string equality — last resort
    """
    if gold is None:
        return None
    if not pred:
        return False

    # 1. Evaluator (whitespace-tolerant, latex_equiv, numeric)
    if ev is not None:
        try:
            result = ev.check_answers_equiv(pred, gold)
            if result is not None:
                return bool(result)
        except Exception:
            pass

    # 2. grader.math_equal
    try:
        from grader import math_equal  # type: ignore
        return bool(math_equal(pred, gold, timeout=True))
    except Exception:
        pass

    # 3. Normalised string equality
    return pred.strip() == gold.strip()


def _make_error_result(method_name: str, question: str,
                        gold: Optional[str], N: int) -> InferenceResult:
    return InferenceResult(
        method=method_name, question=question, gold_answer=gold,
        selected_trace="", predicted_answer="", is_correct=False,
        N=N, elapsed_sec=0.0,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 1. Greedy Inference
# ──────────────────────────────────────────────────────────────────────────────

class GreedyInference:
    """
    Greedy (argmax) decoding.

    self.ev is the Evaluator instance built once at __init__ time and
    reused for every question.
    """

    def __init__(self, model, data_name: str = "math"):
        self.model     = model
        self.data_name = data_name
        self.N         = 1
        self.ev        = _build_evaluator(data_name)   # built ONCE here

    def run(
        self,
        question: str,
        gold_answer: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> InferenceResult:
        t0      = time.time()
        trace   = self.model.generate_greedy(question, max_new_tokens=max_new_tokens)
        elapsed = time.time() - t0

        pred    = _extract_pred(trace, self.ev, self.data_name)
        correct = _check_correct(pred, gold_answer, self.ev, self.data_name)

        return InferenceResult(
            method="greedy", question=question,
            gold_answer=gold_answer, selected_trace=trace,
            predicted_answer=pred, is_correct=correct,
            N=1, elapsed_sec=elapsed,
            all_traces=[trace], selected_index=0,
        )

    def run_batch(
        self,
        examples: List[Dict[str, Any]],
        max_new_tokens: Optional[int] = None,
    ) -> List[InferenceResult]:
        questions    = [ex["question"]    for ex in examples]
        gold_answers = [ex.get("answer")  for ex in examples]

        t0 = time.time()
        logger.info(
            f"[GreedyInference.run_batch] {len(questions)} questions  "
            f"batch_size={self.model.batch_size}  evaluator={self.ev.__class__.__name__}"
        )

        traces        = self.model.generate_greedy_batch(questions, max_new_tokens=max_new_tokens)
        total_elapsed = time.time() - t0
        per_q         = round(total_elapsed / max(len(questions), 1), 4)

        results: List[InferenceResult] = []
        post_bar = tqdm(
            zip(questions, gold_answers, traces),
            total=len(questions),
            desc="[Greedy] post-process",
            unit="q",
            dynamic_ncols=True,
            colour="green",
        )
        for q, gold, trace in post_bar:
            pred    = _extract_pred(trace, self.ev, self.data_name)   # reuses self.ev
            correct = _check_correct(pred, gold, self.ev, self.data_name)
            results.append(InferenceResult(
                method="greedy", question=q,
                gold_answer=gold, selected_trace=trace,
                predicted_answer=pred, is_correct=correct,
                N=1, elapsed_sec=per_q,
                all_traces=[trace], selected_index=0,
            ))
        post_bar.close()
        return results


# ──────────────────────────────────────────────────────────────────────────────
# 2. Best-of-N
# ──────────────────────────────────────────────────────────────────────────────

class BestOfN:
    """
    Standard Best-of-N.   x* = argmax_i  V(x_i)

    self.ev is the Evaluator instance built once at __init__ time.
    """

    def __init__(
        self,
        model,
        verifier,
        N: int = 8,
        data_name: str = "math",
        temperature: float = 0.7,
        top_p: float = 0.95,
    ):
        self.model       = model
        self.verifier    = verifier
        self.N           = N
        self.data_name   = data_name
        self.temperature = temperature
        self.top_p       = top_p
        self.ev          = _build_evaluator(data_name)   # built ONCE here

    def _select(
        self,
        question: str,
        gold: Optional[str],
        traces: List[str],
        elapsed: float,
    ) -> InferenceResult:
        from verifier import score_candidates  # type: ignore

        scores   = score_candidates(traces, self.verifier, gold, question=question)
        best_idx = int(np.argmax(scores))
        selected = traces[best_idx]

        pred    = _extract_pred(selected, self.ev, self.data_name)   # reuses self.ev
        correct = _check_correct(pred, gold, self.ev, self.data_name)

        return InferenceResult(
            method="bon", question=question,
            gold_answer=gold, selected_trace=selected,
            predicted_answer=pred, is_correct=correct,
            N=self.N, elapsed_sec=elapsed,
            all_traces=traces,
            raw_scores=scores.tolist(),
            normalized_scores=scores.tolist(),
            selected_index=best_idx, alpha=0.0,
        )

    def run(
        self,
        question: str,
        gold_answer: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> InferenceResult:
        t0      = time.time()
        traces  = self.model.generate_samples(
            question, N=self.N,
            temperature=self.temperature, top_p=self.top_p,
            max_new_tokens=max_new_tokens,
        )
        elapsed = time.time() - t0
        return self._select(question, gold_answer, traces, elapsed)

    def run_batch(
        self,
        examples: List[Dict[str, Any]],
        max_new_tokens: Optional[int] = None,
    ) -> List[InferenceResult]:
        results: List[InferenceResult] = []
        logger.info(
            f"[BestOfN.run_batch] {len(examples)} questions  N={self.N}  "
            f"batch_size={self.model.batch_size}  evaluator={self.ev.__class__.__name__}"
        )

        pbar      = tqdm(examples, desc=f"[BoN N={self.N}]", unit="q",
                         dynamic_ncols=True, colour="blue")
        n_correct = 0

        for i, ex in enumerate(pbar):
            q    = ex["question"]
            gold = ex.get("answer")
            try:
                t0      = time.time()
                traces  = self.model.generate_samples(
                    q, N=self.N,
                    temperature=self.temperature, top_p=self.top_p,
                    max_new_tokens=max_new_tokens,
                )
                elapsed = time.time() - t0
                res     = self._select(q, gold, traces, elapsed)
            except Exception as e:
                logger.error(f"  [BestOfN] q[{i}] failed: {e}")
                res = _make_error_result("bon", q, gold, self.N)

            results.append(res)
            if res.is_correct is not None:
                n_correct += int(res.is_correct)
                pbar.set_postfix(acc=f"{n_correct/(i+1):.3f}",
                                 correct=f"{n_correct}/{i+1}", refresh=False)

        pbar.close()
        return results


# ──────────────────────────────────────────────────────────────────────────────
# 3. Direction-Normalised Best-of-N  (DN-BoN)
# ──────────────────────────────────────────────────────────────────────────────

class DNBestOfN:
    """
    DN-BoN.   s_i = V(x_i) / |C(x_i)|^alpha      x* = argmax_i s_i

    self.ev is the Evaluator instance built once at __init__ time.
    """

    def __init__(
        self,
        model,
        verifier,
        cluster_manager,
        N: int = 8,
        alpha: float = 0.5,
        data_name: str = "math",
        temperature: float = 0.7,
        top_p: float = 0.95,
    ):
        self.model           = model
        self.verifier        = verifier
        self.cluster_manager = cluster_manager
        self.N               = N
        self.alpha           = alpha
        self.data_name       = data_name
        self.temperature     = temperature
        self.top_p           = top_p
        self.ev              = _build_evaluator(data_name)   # built ONCE here

    def _select(
        self,
        question: str,
        gold: Optional[str],
        traces: List[str],
        elapsed: float,
    ) -> InferenceResult:
        from verifier import score_candidates  # type: ignore

        raw_scores              = score_candidates(traces, self.verifier, gold, question=question)
        labels, cluster_sizes   = self.cluster_manager.cluster(traces)
        # Clamp to ≥1: cluster_sizes should always be ≥1 from ClusterManager,
        # but guard against any edge case that would produce 0^alpha = NaN/inf.
        safe_sizes  = np.maximum(cluster_sizes.astype(float), 1.0)
        norm_scores = raw_scores / (safe_sizes ** self.alpha)

        best_idx = int(np.argmax(norm_scores))
        selected = traces[best_idx]

        pred    = _extract_pred(selected, self.ev, self.data_name)
        correct = _check_correct(pred, gold, self.ev, self.data_name)

        # Per-trace predictions — needed for Exp 2, 3, 4, 5
        all_preds = [_extract_pred(t, self.ev, self.data_name) for t in traces]
        # Per-trace correctness — needed for Exp 3 (oracle), Exp 5
        all_correct = [
            bool(_check_correct(p, gold, self.ev, self.data_name))
            for p in all_preds
        ] if gold is not None else []

        res = InferenceResult(
            method="dn_bon", question=question,
            gold_answer=gold, selected_trace=selected,
            predicted_answer=pred, is_correct=correct,
            N=self.N, elapsed_sec=elapsed,
            all_traces=traces,
            raw_scores=raw_scores.tolist(),
            normalized_scores=norm_scores.tolist(),
            cluster_labels=labels.tolist(),
            cluster_sizes=cluster_sizes.tolist(),
            selected_index=best_idx, alpha=self.alpha,
        )
        # Attach extra fields for motivation experiments
        res._all_preds   = all_preds
        res._all_correct = all_correct
        return res

    def run(
        self,
        question: str,
        gold_answer: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> InferenceResult:
        t0      = time.time()
        traces  = self.model.generate_samples(
            question, N=self.N,
            temperature=self.temperature, top_p=self.top_p,
            max_new_tokens=max_new_tokens,
        )
        elapsed = time.time() - t0
        return self._select(question, gold_answer, traces, elapsed)

    def run_batch(
        self,
        examples: List[Dict[str, Any]],
        max_new_tokens: Optional[int] = None,
    ) -> List[InferenceResult]:
        results: List[InferenceResult] = []
        logger.info(
            f"[DNBestOfN.run_batch] {len(examples)} questions  N={self.N}  "
            f"alpha={self.alpha}  batch_size={self.model.batch_size}  "
            f"evaluator={self.ev.__class__.__name__}"
        )

        pbar      = tqdm(examples, desc=f"[DN-BoN N={self.N} α={self.alpha}]",
                         unit="q", dynamic_ncols=True, colour="magenta")
        n_correct = 0

        for i, ex in enumerate(pbar):
            q    = ex["question"]
            gold = ex.get("answer")
            try:
                t0      = time.time()
                traces  = self.model.generate_samples(
                    q, N=self.N,
                    temperature=self.temperature, top_p=self.top_p,
                    max_new_tokens=max_new_tokens,
                )
                elapsed = time.time() - t0
                res     = self._select(q, gold, traces, elapsed)
            except Exception as e:
                logger.error(f"  [DNBestOfN] q[{i}] failed: {e}")
                res = _make_error_result("dn_bon", q, gold, self.N)

            results.append(res)
            if res.is_correct is not None:
                n_correct += int(res.is_correct)
                pbar.set_postfix(
                    acc=f"{n_correct/(i+1):.3f}",
                    correct=f"{n_correct}/{i+1}",
                    clusters=len(set(res.cluster_labels)) if res.cluster_labels else "-",
                    refresh=False,
                )

        pbar.close()
        return results


# ──────────────────────────────────────────────────────────────────────────────
# 4. Greedy Pass@N  (select by maximum sequence log-probability)
# ──────────────────────────────────────────────────────────────────────────────

class GreedyPassN:
    """
    Sample N traces with temperature, then select the one with the highest
    mean per-token log-probability under the model.

    This is the model's implicit preference among the N candidates —
    analogous to greedy decoding but with diversity from sampling.
    No reward model or gold answer is required at selection time.

    Selection cost: one batched forward pass over N completions (no generation).
    """

    def __init__(
        self,
        model,
        N: int = 8,
        data_name: str = "math",
        temperature: float = 0.7,
        top_p: float = 0.95,
    ):
        self.model       = model
        self.N           = N
        self.data_name   = data_name
        self.temperature = temperature
        self.top_p       = top_p
        self.ev          = _build_evaluator(data_name)

    def _select(
        self,
        question: str,
        gold: Optional[str],
        traces: List[str],
        elapsed: float,
    ) -> InferenceResult:
        prompt   = self.model._build_prompt(question)
        logprobs = self.model.score_sequences_logprob(prompt, traces)
        best_idx = int(np.argmax(logprobs))
        selected = traces[best_idx]

        pred    = _extract_pred(selected, self.ev, self.data_name)
        correct = _check_correct(pred, gold, self.ev, self.data_name)

        return InferenceResult(
            method="greedy_pass_n", question=question,
            gold_answer=gold, selected_trace=selected,
            predicted_answer=pred, is_correct=correct,
            N=self.N, elapsed_sec=elapsed,
            all_traces=traces,
            raw_scores=logprobs,
            normalized_scores=logprobs,
            selected_index=best_idx, alpha=0.0,
        )

    def run(
        self,
        question: str,
        gold_answer: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> InferenceResult:
        t0      = time.time()
        traces  = self.model.generate_samples(
            question, N=self.N,
            temperature=self.temperature, top_p=self.top_p,
            max_new_tokens=max_new_tokens,
        )
        elapsed = time.time() - t0
        return self._select(question, gold_answer, traces, elapsed)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Best-of-N with Majority Voting  (self-consistency)
# ──────────────────────────────────────────────────────────────────────────────

class BestOfNMajority:
    """
    Sample N traces, then select by majority vote (Wang et al., 2023
    self-consistency).

    Algorithm
    ---------
    1. Sample N traces with temperature sampling.
    2. Extract the final answer from each trace.
    3. Group traces by equivalent answer using ev.check_answers_equiv()
       so LaTeX variants count as the same answer.
    4. Return the first trace from the largest vote group.

    No reward model required — only the model and an equivalence checker.
    """

    def __init__(
        self,
        model,
        N: int = 8,
        data_name: str = "math",
        temperature: float = 0.7,
        top_p: float = 0.95,
    ):
        self.model       = model
        self.N           = N
        self.data_name   = data_name
        self.temperature = temperature
        self.top_p       = top_p
        self.ev          = _build_evaluator(data_name)

    def _majority_vote(self, traces: List[str]):
        """
        Group traces by equivalent final answer.

        Returns
        -------
        best_idx      : index of selected trace (first in majority group)
        majority_ans  : the majority answer string
        vote_counts   : vote_counts[i] = size of trace i's group
        group_labels  : group id for each trace
        """
        answers = [_extract_pred(t, self.ev, self.data_name) for t in traces]

        groups: List[List[int]] = []
        group_rep: List[str]    = []

        for idx, ans in enumerate(answers):
            placed = False
            for g, rep in enumerate(group_rep):
                try:
                    equiv = (
                        self.ev.check_answers_equiv(ans, rep)
                        if self.ev is not None else None
                    )
                except Exception:
                    equiv = None
                if equiv is None:
                    equiv = (ans.strip() == rep.strip())
                if equiv:
                    groups[g].append(idx)
                    placed = True
                    break
            if not placed:
                groups.append([idx])
                group_rep.append(ans)

        majority_g   = max(range(len(groups)), key=lambda g: len(groups[g]))
        best_idx     = groups[majority_g][0]
        majority_ans = group_rep[majority_g]

        vote_counts  = [0] * len(traces)
        group_labels = [0] * len(traces)
        for g, members in enumerate(groups):
            for idx in members:
                vote_counts[idx]  = len(members)
                group_labels[idx] = g

        return best_idx, majority_ans, vote_counts, group_labels

    def _select(
        self,
        question: str,
        gold: Optional[str],
        traces: List[str],
        elapsed: float,
    ) -> InferenceResult:
        best_idx, majority_ans, vote_counts, group_labels = \
            self._majority_vote(traces)

        selected = traces[best_idx]
        pred     = majority_ans or _extract_pred(selected, self.ev, self.data_name)
        correct  = _check_correct(pred, gold, self.ev, self.data_name)

        max_v      = max(vote_counts) if vote_counts else 1
        norm_votes = [v / max_v for v in vote_counts]

        return InferenceResult(
            method="bon_majority", question=question,
            gold_answer=gold, selected_trace=selected,
            predicted_answer=pred, is_correct=correct,
            N=self.N, elapsed_sec=elapsed,
            all_traces=traces,
            raw_scores=[float(v) for v in vote_counts],
            normalized_scores=norm_votes,
            cluster_labels=group_labels,
            cluster_sizes=vote_counts,
            selected_index=best_idx, alpha=0.0,
        )

    def run(
        self,
        question: str,
        gold_answer: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> InferenceResult:
        t0      = time.time()
        traces  = self.model.generate_samples(
            question, N=self.N,
            temperature=self.temperature, top_p=self.top_p,
            max_new_tokens=max_new_tokens,
        )
        elapsed = time.time() - t0
        return self._select(question, gold_answer, traces, elapsed)


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_inference_method(
    method_name: str,
    model,
    verifier,
    cluster_manager=None,
    N: int = 8,
    alpha: float = 0.5,
    data_name: str = "math",
    temperature: float = 0.7,
    top_p: float = 0.95,
):
    """
    Instantiate an inference method by name.

    method_name choices
    -------------------
    greedy         — single argmax decode, no sampling
    bon            — Best-of-N with verifier score
    dn_bon         — Direction-Normalised BoN (requires cluster_manager)
    greedy_pass_n  — N samples, select by max sequence log-probability
    bon_majority   — N samples, select by majority vote (self-consistency)
    """
    if method_name == "greedy":
        return GreedyInference(model=model, data_name=data_name)
    elif method_name == "bon":
        return BestOfN(model=model, verifier=verifier, N=N,
                       data_name=data_name, temperature=temperature, top_p=top_p)
    elif method_name == "dn_bon":
        if cluster_manager is None:
            raise ValueError("DN-BoN requires a ClusterManager.")
        return DNBestOfN(model=model, verifier=verifier, cluster_manager=cluster_manager,
                         N=N, alpha=alpha, data_name=data_name,
                         temperature=temperature, top_p=top_p)
    elif method_name == "greedy_pass_n":
        return GreedyPassN(model=model, N=N, data_name=data_name,
                           temperature=temperature, top_p=top_p)
    elif method_name == "bon_majority":
        return BestOfNMajority(model=model, N=N, data_name=data_name,
                               temperature=temperature, top_p=top_p)
    raise ValueError(
        f"Unknown method '{method_name}'. "
        f"Choose: greedy | bon | dn_bon | greedy_pass_n | bon_majority"
    )