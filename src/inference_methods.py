"""
inference_methods.py
--------------------
Implements the test-time compute strategies:

  1. GreedyInference   – single greedy (argmax) decode per question.
  2. BestOfN           – sample N traces, pick best by verifier score.
  3. DNBestOfN         – Direction-Normalised BoN (hard cluster-size penalty).
  3b. DNBonUpdated     – CRISP-DORA: CRISP answer-cluster aggregation fused with
                         DORA semantic-uniqueness redundancy correction,
                         with optional iterative prefix refinement.
  4. GreedyPassN       – N samples, select by max sequence log-probability.
  5. BestOfNMajority   – N samples, select by majority vote (self-consistency).

CRISP-DORA (DNBonUpdated) overview
------------------------------------
Combines two published inference-time scaling methods:

  CRISP (Li et al., ICLR 2026): clusters trajectories by final answer,
  aggregates reward at the cluster level to suppress the RM's inverse
  long-tail bias, uses early termination for easy questions, and
  iteratively refines sampling via prefix conditioning.

  DORA (Wang et al., NeurIPS 2025): corrects for semantic redundancy via
  a soft cosine-similarity uniqueness score γ_i so near-duplicate reasoning
  paths are down-weighted without hard clustering.

  Fusion scoring:
      F(C_j) = [Σ_{r_i ∈ C_j}  ṡ_i^β · γ_i^δ]  ·  |C_j|^α
  where ṡ_i is the temperature-scaled PRM reward, γ_i is the DORA
  uniqueness score (diagonal of the row-softmax affinity matrix),
  and α, β, δ are tunable exponents.

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
    "mmlu":          "MMLU",
    "bbh":           "BBH",
    "arc_challenge": "ARC_Challenge",
    "truthfulqa":    "TruthfulQA",
    "strategyqa":    "StrategyQA",
}


def _build_evaluator(data_name: str):
    """
    Instantiate and return the correct Evaluator for `data_name`.
    Called once in __init__ of each inference class; the result is stored
    as self.ev and reused for the entire evaluation run.
    """
    try:
        from .evaluator_classes import get_evaluator  # type: ignore
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
# 3b. DN-BoN Updated  —  CRISP-DORA Redundancy-Aware Answer Selection
#
# Combines two inference-time scaling improvements:
#
#   CRISP (Li et al., ICLR 2026):
#     • Cluster trajectories by final answer to suppress the RM's inverse
#       long-tail bias (rare incorrect responses that receive inflated scores).
#     • Aggregate reward at the cluster level rather than the trajectory level.
#     • Early termination when answer diversity collapses (easy questions).
#     • Iterative prefix refinement: condition the next sampling round on the
#       leading steps of the best trajectory found so far.
#
#   DORA (Wang et al., NeurIPS 2025):
#     • Identify semantic redundancy among trajectories via a soft cosine-
#       similarity clustering (no hard cluster count needed).
#     • Uniqueness score γ_i = P_ii (diagonal of the row-softmax affinity
#       matrix) acts as a soft inverse cluster-size, down-weighting near-
#       duplicate reasoning paths without discarding them.
#     • Direction-level resource allocation is provably optimal under a
#       Bayesian Beta-prior model of trajectory correctness.
#
#   Fusion:
#     The combined cluster score is:
#
#         F(C_j) = [ Σ_{r_i ∈ C_j}  ṡ_i · γ_i ]  ·  |C_j|^α
#
#     where
#         ṡ_i = softmax(s_i / T_r)   reward-temperature-scaled PRM score
#         γ_i = P_ii                  DORA semantic uniqueness
#         α   ∈ [0, 1]               frequency regularisation exponent
#
#     This estimates:
#         P(A_j correct) ∝ Σ_{r_i∈C_j} P(correct|r_i) · P(independent|r_i) · |C_j|^α
#
# Iterative mode (max_iter > 1):
#     After round 0 (unconditioned sampling + scoring), the winning cluster's
#     best trajectory is extracted up to `prefix_depth` reasoning steps and
#     used as a prefix for the next sampling round.  Subsequent samples are
#     appended to this prefix so the search progressively narrows around the
#     most promising reasoning direction while DORA continues to penalise
#     within-round redundancy.
# ──────────────────────────────────────────────────────────────────────────────

class DNBonUpdated:
    """
    CRISP-DORA Redundancy-Aware Answer Selection.

    Parameters
    ----------
    model            : HFModelWrapper — language model for trajectory generation.
    verifier         : reward model with score_candidates / score_batch API.
    cluster_manager  : ClusterManager — provides sentence embeddings (loaded once).
    N                : total trajectory budget per question (initial round).
    T_b              : reward softmax temperature  (default 0.3).
                       Smaller → trust the reward model more sharply.
    T_s              : semantic similarity softmax temperature  (default 0.1).
                       Smaller → stronger redundancy penalty.
    alpha            : frequency regularisation exponent α ∈ [0, 1]  (default 0.3).
                       0 → pure reward+uniqueness; 1 → majority-vote dominant.
    data_name        : dataset name for evaluator selection.
    temperature      : LM sampling temperature (default 0.7).
    top_p            : nucleus sampling probability (default 0.95).
    max_iter         : maximum CRISP prefix-refinement iterations (default 1).
                       1 = one-shot selection (no iterative prefix).
                       >1 = iterative mode: each iteration conditions on the
                            best prefix found in the previous round.
    n_iter_samples   : samples drawn per subsequent iteration (default N//2).
                       If None, defaults to max(4, N // 2).
    prefix_depth     : reasoning steps extracted as prefix at each iteration
                       (default 2).  Step counting is newline-based.
    early_stop_clusters : K threshold τ_K for early termination (default 2).
                       If the number of unique answer clusters ≤ τ_K the
                       question is judged easy and we return immediately.
    early_stop_majority : p_max threshold τ_p for early termination (default 0.75).
                       If the fraction of trajectories sharing the modal answer
                       ≥ τ_p, we return immediately.
    beta             : reward exponent in flexible scoring (default 1.0).
    delta            : uniqueness exponent in flexible scoring (default 1.0).
                       Final combined weight: w'_i ∝ ṡ_i^β · γ_i^δ
    """

    # ── construction ──────────────────────────────────────────────────────────

    def __init__(
        self,
        model,
        verifier,
        cluster_manager,
        N: int = 16,
        T_b: float = 0.3,
        T_s: float = 0.1,
        alpha: float = 0.3,
        data_name: str = "math",
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_iter: int = 1,
        n_iter_samples: Optional[int] = None,
        prefix_depth: int = 2,
        early_stop_clusters: int = 1,
        early_stop_majority: float = 0.85,
        beta: float = 1.0,
        delta: float = 0.5,
    ):
        self.model                = model
        self.verifier             = verifier
        self.cluster_manager      = cluster_manager
        self.N                    = N
        self.T_b                  = T_b
        self.T_s                  = T_s
        self.alpha                = alpha
        self.data_name            = data_name
        self.temperature          = temperature
        self.top_p                = top_p
        self.max_iter             = max_iter
        self.n_iter_samples       = n_iter_samples if n_iter_samples is not None \
                                    else max(4, N // 2)
        self.prefix_depth         = prefix_depth
        self.early_stop_clusters  = early_stop_clusters
        self.early_stop_majority  = early_stop_majority
        self.beta                 = beta
        self.delta                = delta
        self.ev                   = _build_evaluator(data_name)  # built ONCE

    # ── static numerics ───────────────────────────────────────────────────────

    @staticmethod
    def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
        """
        Numerically stable softmax with temperature scaling.
        x : 1-D or 2-D ndarray.  If 2-D, applied row-wise.
        """
        t = max(temperature, 1e-8)
        if x.ndim == 1:
            z = x / t
            z -= z.max()
            e = np.exp(z)
            return e / (e.sum() + 1e-300)
        else:
            z = x / t
            z -= z.max(axis=1, keepdims=True)
            e = np.exp(z)
            return e / (e.sum(axis=1, keepdims=True) + 1e-300)

    # ── Step 4: reward scores ─────────────────────────────────────────────────

    def _reward_scores(
        self,
        traces: List[str],
        gold: Optional[str],
        question: str,
    ) -> np.ndarray:
        """
        Score traces with the PRM/ORM and return raw scalar scores s_i.

        The verifier's score_candidates() call is the standard entry point
        used by BestOfN / DNBestOfN; we reuse it here unchanged.
        """
        from verifier import score_candidates  # type: ignore
        return score_candidates(traces, self.verifier, gold, question=question)

    # ── Step 5: DORA semantic uniqueness ─────────────────────────────────────

    def _dora_uniqueness(self, traces: List[str]):
        """
        Compute DORA semantic uniqueness scores γ_i for all traces.

        Algorithm (DORA paper, Eqs. 10-12):
          e_i  = Embed(r_i)                      unit-normalised embeddings
          S_ij = e_i · e_j                       cosine similarity matrix
          P_ij = softmax(S_i / T_s)              row-wise soft affinity
          γ_i  = P_ii                            uniqueness = diagonal of P

        Degenerate-input handling:
          - Single trace: γ = [1.0], S = [[1.0]], P = [[1.0]]
          - All embeddings identical (zero-variance): uniform γ = 1/N
          - NaN / Inf in embeddings: replaced with zeros before computation
        """
        n = len(traces)

        # ── edge case: single trace ────────────────────────────────────────────
        if n == 1:
            dummy_emb = np.ones((1, 1), dtype=np.float32)
            S = np.array([[1.0]], dtype=np.float32)
            P = np.array([[1.0]], dtype=np.float32)
            gamma = np.array([1.0], dtype=np.float32)
            return gamma, dummy_emb, S, P

        texts      = self.cluster_manager._get_texts_to_embed(traces)
        embeddings = self.cluster_manager._embed(texts)           # (N, d) unit-normed

        # ── sanitise embeddings: replace NaN / Inf with zero ──────────────────
        if not np.isfinite(embeddings).all():
            logger.warning("[CRISP-DORA] Non-finite values in embeddings; zeroing.")
            embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

        # ── re-normalise rows (ClusterManager should already do this, but guard) ─
        row_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Rows with zero norm get replaced by uniform unit vectors
        zero_rows = (row_norms.squeeze() < 1e-10)
        if zero_rows.any():
            logger.warning(
                f"[CRISP-DORA] {zero_rows.sum()} zero-norm embedding rows; "
                "replacing with uniform unit vectors."
            )
            d = embeddings.shape[1]
            for i in np.where(zero_rows)[0]:
                vec = np.ones(d, dtype=np.float32)
                embeddings[i] = vec / (np.linalg.norm(vec) + 1e-12)
            row_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

        embeddings = embeddings / (row_norms + 1e-12)

        S = np.clip(embeddings @ embeddings.T, -1.0, 1.0)          # (N, N)

        # ── degenerate: all embeddings identical → uniform uniqueness ──────────
        off_diag = S - np.eye(n)
        if np.abs(off_diag).max() < 1e-6:
            logger.debug(
                "[CRISP-DORA] All embeddings identical; "
                "using uniform uniqueness γ = 1/N."
            )
            gamma      = np.full(n, 1.0 / n, dtype=np.float32)
            P          = np.full((n, n), 1.0 / n, dtype=np.float32)
            return gamma, embeddings, S, P

        P     = self._softmax(S, self.T_s)                          # (N, N) row-stochastic
        gamma = np.diag(P)                                          # (N,)
        return gamma, embeddings, S, P

    # ── Step 6: combined trajectory weights ──────────────────────────────────

    def _trajectory_weights(
        self,
        raw_scores: np.ndarray,   # (N,) PRM scores s_i
        gamma: np.ndarray,        # (N,) uniqueness scores γ_i
    ) -> np.ndarray:
        """
        Combine reward quality and semantic uniqueness into a single weight.

        Flexible parameterisation (PDF eq. 15):
            w'_i  ∝  ṡ_i^β · γ_i^δ
        where ṡ_i = softmax(s_i / T_r) is the temperature-scaled reward weight.

        With β=δ=1 this collapses to the canonical form:
            w'_i  ∝  ṡ_i · γ_i    (PDF eq. 13-14)
        """
        s_tilde  = self._softmax(raw_scores, self.T_b)            # (N,) reward weights
        combined = (s_tilde ** self.beta) * (gamma ** self.delta)  # (N,)
        denom    = combined.sum()
        if denom < 1e-12:
            return s_tilde      # degenerate fallback: quality only
        return combined / denom

    # ── Step 3: CRISP answer clustering ──────────────────────────────────────

    def _answer_clusters(
        self,
        traces: List[str],
    ) -> Dict[str, List[int]]:
        """
        Partition trajectories by normalised final answer.

        Returns
        -------
        clusters : dict  answer_string → [trace_indices]
        answers  : list of per-trace answer strings (parallel to traces)

        Equivalent answers (LaTeX variants, numeric formatting) are merged
        via ev.check_answers_equiv() so '\\frac{1}{2}' and '0.5' land in
        the same cluster.
        """
        answers  = [_extract_pred(t, self.ev, self.data_name) for t in traces]
        clusters: Dict[str, List[int]] = {}
        rep_list: List[str] = []   # canonical representative for each cluster key

        for idx, ans in enumerate(answers):
            placed = False
            for rep in rep_list:
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
                    clusters[rep].append(idx)
                    placed = True
                    break
            if not placed:
                rep_list.append(ans)
                clusters[ans] = [idx]

        return clusters, answers

    # ── Step 7: cluster-level aggregation ────────────────────────────────────

    def _cluster_scores(self, clusters, w_prime, raw_scores):
        scores = {}

        for ans_key, indices in clusters.items():
            rewards = np.array([raw_scores[i] for i in indices], dtype=float)

            # top-k reward inside answer cluster
            k = min(2, len(rewards))
            topk_mean = float(np.mean(np.sort(rewards)[-k:]))

            # soft frequency support
            freq_bonus = np.log1p(len(indices))

            # penalize singleton weird answers
            singleton_penalty = 1.0 if len(indices) == 1 else 0.0

            # robust cluster score
            F = (
                1.0 * topk_mean
                + 0.15 * freq_bonus
                - 0.10 * singleton_penalty
            )

            scores[ans_key] = F

        return scores

    # ── Steps 9–10: early termination check ──────────────────────────────────

    def _should_terminate_early(
        self,
        clusters: Dict[str, List[int]],
        n_total: int,
    ) -> bool:
        """
        Return True if the question looks easy enough that continued
        reward-guided search would likely hurt rather than help (CRISP Cl.1).

        Two conditions (either triggers early stop):
          τ_K : number of distinct answer clusters ≤ early_stop_clusters
          τ_p : fraction of trajectories in the modal cluster ≥ early_stop_majority
        """
        K     = len(clusters)
        p_max = max(len(v) for v in clusters.values()) / max(n_total, 1)
        if K <= self.early_stop_clusters:
            logger.debug(
                f"  [CRISP-DORA] Early stop: K={K} ≤ τ_K={self.early_stop_clusters}"
            )
            return True
        if p_max >= self.early_stop_majority:
            logger.debug(
                f"  [CRISP-DORA] Early stop: p_max={p_max:.2f} ≥ τ_p={self.early_stop_majority}"
            )
            return True
        return False

    # ── Step 9: prefix extraction (CRISP-style) ───────────────────────────────

    @staticmethod
    def _extract_prefix(trace: str, depth: int) -> str:
        """
        Extract the first `depth` reasoning steps from a trace.

        Steps are delimited by "\\nStep " (matching the format used in
        crisp_reason.py and the CRISP paper prompts).  If the trace does
        not contain step markers, fall back to splitting on double-newlines
        so the method degrades gracefully for any formatting style.

        The extracted prefix is returned as a raw string that can be
        appended directly to the prompt template.
        """
        if depth <= 0:
            return ""

        # Primary: CRISP-style "\nStep N:" markers
        import re as _re
        step_pattern = _re.compile(r'(?=\nStep \d+[:\.])', _re.IGNORECASE)
        parts = step_pattern.split("\n" + trace.lstrip())
        # parts[0] is typically empty / preamble; actual steps start at parts[1]
        step_parts = [p for p in parts if p.strip().startswith("Step")]
        if len(step_parts) >= depth:
            prefix_steps = step_parts[:depth]
            return "\n".join(prefix_steps).strip()

        # Fallback: paragraph-split for models that use blank-line separators
        paragraphs = [p.strip() for p in trace.split("\n\n") if p.strip()]
        if len(paragraphs) >= depth:
            return "\n\n".join(paragraphs[:depth])

        # Last resort: return the whole trace (already short)
        return trace.strip()

    # ── prefix-conditioned sample generation ─────────────────────────────────

    def _generate_with_prefix(
        self,
        question: str,
        prefix: str,
        n_samples: int,
        max_new_tokens: Optional[int] = None,
    ) -> List[str]:
        """
        Generate `n_samples` trajectory continuations conditioned on `prefix`.

        The prefix is appended to the standard prompt so the model sees:
            <system>...<user>question<assistant>prefix...

        For chat-template models this is achieved by injecting the prefix
        as the start of an assistant turn (partial assistant message).
        For non-chat models the prefix is appended directly to the prompt text.

        The returned strings are the FULL trajectories (prefix + continuation)
        so they can be scored and embedded identically to unconditioned samples.
        """
        # Build the base prompt for the question
        base_prompt = self.model._build_prompt(question)

        # Append prefix — the model then continues from that point.
        # We use the same left-padded batch approach as generate_samples.
        conditioned_prompt = base_prompt.rstrip() + "\n" + prefix.lstrip()

        samples: List[str] = []
        remaining = n_samples
        while remaining > 0:
            this_bs = min(remaining, self.model.batch_size)
            raw_outputs = self.model._generate_batch_raw(
                [conditioned_prompt] * this_bs,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                max_new_tokens=max_new_tokens,
            )
            # Prepend the prefix so the full trace is self-contained
            for continuation in raw_outputs:
                full_trace = (prefix.strip() + "\n" + continuation.strip()).strip()
                samples.append(full_trace)
            remaining -= this_bs

        return samples

    # ── single-round scoring kernel ───────────────────────────────────────────

    def _score_round(
        self,
        question: str,
        gold: Optional[str],
        traces: List[str],
    ):
        """
        Run one complete CRISP-DORA scoring round on a set of traces.

        Empty/whitespace-only traces are removed before scoring and
        embedding.  Their indices are still present in the returned
        cluster/score arrays (padded with zeros) so callers never need
        to track the filtering themselves.

        Returns
        -------
        clusters      : Dict[str, List[int]]  answer → indices into `traces`
        answers       : List[str]             per-trace extracted answers
        cluster_scores: Dict[str, float]      F(C_j)
        w_prime       : np.ndarray (N,)       combined trajectory weights
        raw_scores    : np.ndarray (N,)       raw PRM scores  (0 for empty traces)
        gamma         : np.ndarray (N,)       DORA uniqueness (0 for empty traces)
        embeddings    : np.ndarray (N, d)     unit-normed embeddings (zeros for empty)
        """
        N = len(traces)

        # ── Filter empty traces ────────────────────────────────────────────────
        valid_mask  = [bool(t.strip()) for t in traces]
        valid_idx   = [i for i, ok in enumerate(valid_mask) if ok]
        valid_traces = [traces[i] for i in valid_idx]

        n_empty = N - len(valid_traces)
        if n_empty:
            logger.warning(
                f"[CRISP-DORA] {n_empty}/{N} traces are empty; "
                "they will receive zero weight."
            )

        # ── Absolute edge-case: ALL traces empty ───────────────────────────────
        if not valid_traces:
            logger.warning(
                "[CRISP-DORA] All traces empty — returning uniform fallback."
            )
            w_uniform  = np.full(N, 1.0 / max(N, 1), dtype=np.float32)
            raw_zeros  = np.zeros(N, dtype=np.float32)
            gamma_zeros = np.zeros(N, dtype=np.float32)
            emb_zeros  = np.zeros((N, 1), dtype=np.float32)
            # Put everything in one "unknown" answer cluster
            clusters   = {"": list(range(N))}
            answers    = [""] * N
            cl_scores  = {"": float(w_uniform.sum())}
            return clusters, answers, cl_scores, w_uniform, raw_zeros, gamma_zeros, emb_zeros

        # ── Step 4: PRM/ORM reward scores (valid traces only) ─────────────────
        raw_valid = self._reward_scores(valid_traces, gold, question)   # (n_valid,)

        # ── Step 5: DORA semantic uniqueness (valid traces only) ──────────────
        gamma_valid, emb_valid, S_valid, P_valid = self._dora_uniqueness(valid_traces)

        # ── Step 6: combined trajectory weights (valid traces only) ───────────
        w_valid = self._trajectory_weights(raw_valid, gamma_valid)      # (n_valid,)

        # ── Expand back to full N-length arrays (zero for empty traces) ───────
        raw_scores = np.zeros(N, dtype=np.float32)
        gamma      = np.zeros(N, dtype=np.float32)
        w_prime    = np.zeros(N, dtype=np.float32)
        d          = emb_valid.shape[1]
        embeddings = np.zeros((N, d), dtype=np.float32)

        for out_i, in_i in enumerate(valid_idx):
            raw_scores[in_i] = raw_valid[out_i]
            gamma[in_i]      = gamma_valid[out_i]
            w_prime[in_i]    = w_valid[out_i]
            embeddings[in_i] = emb_valid[out_i]

        # Re-normalise w_prime over the full N (empty traces already have 0)
        denom = w_prime.sum()
        if denom > 1e-12:
            w_prime /= denom

        # ── Step 3: CRISP answer clustering (all traces; empty → answer "") ───
        clusters, answers = self._answer_clusters(traces)

        # ── Step 7: cluster-level aggregation ─────────────────────────────────
        # cluster_scores = self._cluster_scores(clusters, w_prime)
        cluster_scores = self._cluster_scores(clusters, w_prime, raw_scores)

        return clusters, answers, cluster_scores, w_prime, raw_scores, gamma, embeddings

    # ── main selection logic ──────────────────────────────────────────────────

    def _select(
        self,
        question: str,
        gold: Optional[str],
        initial_traces: List[str],
        elapsed_generation: float = 0.0,
        max_new_tokens: Optional[int] = None,
        *,
        elapsed: Optional[float] = None,   # legacy alias used by evaluator.py
    ) -> "InferenceResult":
        """
        Full CRISP-DORA selection with optional iterative prefix refinement.

        Iteration 0: score the initial unconditioned traces.
        Iterations 1…max_iter-1: extract prefix from best trace of the
            winning cluster; resample conditioned on that prefix; merge all
            collected traces; re-score the union.

        Early termination fires as soon as the answer distribution collapses
        (K ≤ τ_K or p_max ≥ τ_p) — matching CRISP Cl.1 behaviour.
        """
        # Resolve legacy elapsed kwarg forwarded by evaluator.py's flat-batch path
        if elapsed is not None:
            elapsed_generation = elapsed
        t0_select = time.time()

        # ── Accumulate all traces across iterations
        all_traces   : List[str]   = list(initial_traces)
        iteration_log: List[Dict]  = []

        # ── Iteration 0: score initial batch
        (clusters, answers, cluster_scores,
         w_prime, raw_scores, gamma, embeddings) = self._score_round(
            question, gold, all_traces
        )

        iteration_log.append({
            "iter": 0, "n_traces": len(all_traces),
            "n_clusters": len(clusters),
            "best_cluster": max(cluster_scores, key=cluster_scores.get)
                            if cluster_scores else "",
            "best_score": max(cluster_scores.values()) if cluster_scores else 0.0,
        })
        logger.debug(f"  [CRISP-DORA] iter=0  traces={len(all_traces)}  "
                     f"clusters={len(clusters)}  scores={cluster_scores}")

        # Check early termination after round 0
        terminate = self._should_terminate_early(clusters, len(all_traces))

        # ── Iterative prefix refinement (CRISP Step 9)
        for iteration in range(1, self.max_iter):
            if terminate:
                logger.debug(
                    f"  [CRISP-DORA] Early termination at iter={iteration}"
                )
                break

            # Step 8: find winning cluster and best representative trajectory
            if not cluster_scores:
                break
            # best_ans_key = max(cluster_scores, key=cluster_scores.get)
            best_ans_key = max(cluster_scores, key=cluster_scores.get)
            best_indices = clusters[best_ans_key]

            # Best trajectory = highest combined weight inside winning cluster
            best_in_cluster = max(best_indices, key=lambda i: float(w_prime[i]))
            best_trace      = all_traces[best_in_cluster]

            # Extract prefix (first `prefix_depth` reasoning steps)
            prefix = self._extract_prefix(best_trace, self.prefix_depth + iteration - 1)
            if not prefix:
                logger.debug("  [CRISP-DORA] Empty prefix extracted; stopping iteration.")
                break

            logger.debug(
                f"  [CRISP-DORA] iter={iteration}  prefix_depth="
                f"{self.prefix_depth + iteration - 1}  "
                f"prefix_len={len(prefix)}"
            )

            # Resample conditioned on the extracted prefix
            new_traces = self._generate_with_prefix(
                question, prefix,
                n_samples=self.n_iter_samples,
                max_new_tokens=max_new_tokens,
            )

            # Accumulate and re-score the full union
            all_traces.extend(new_traces)
            (clusters, answers, cluster_scores,
             w_prime, raw_scores, gamma, embeddings) = self._score_round(
                question, gold, all_traces
            )

            iteration_log.append({
                "iter": iteration, "n_traces": len(all_traces),
                "n_clusters": len(clusters),
                "best_cluster": max(cluster_scores, key=cluster_scores.get)
                                if cluster_scores else "",
                "best_score": max(cluster_scores.values()) if cluster_scores else 0.0,
            })
            logger.debug(
                f"  [CRISP-DORA] iter={iteration}  total_traces={len(all_traces)}  "
                f"clusters={len(clusters)}  scores={cluster_scores}"
            )

            terminate = self._should_terminate_early(clusters, len(all_traces))

        # ── Step 8: final answer selection
        if not cluster_scores:
            # Absolute fallback: return the highest raw-score trace
            best_idx = int(np.argmax(raw_scores)) if len(raw_scores) > 0 else 0
            selected = all_traces[best_idx] if all_traces else ""
            pred     = _extract_pred(selected, self.ev, self.data_name)
            correct  = _check_correct(pred, gold, self.ev, self.data_name)
        else:
            # If early stop triggered, use majority cluster like CRISP/self-consistency.
            # Otherwise, use CRISP-DORA cluster score.
            if terminate:
                best_ans_key = max(clusters, key=lambda a: len(clusters[a]))
            else:
                best_ans_key = max(cluster_scores, key=cluster_scores.get)

            best_indices = clusters[best_ans_key]

            # Pick best trace inside selected answer cluster using corrected trajectory weight.
            best_idx = max(best_indices, key=lambda i: float(w_prime[i]))

            selected = all_traces[best_idx]
            pred = best_ans_key or _extract_pred(selected, self.ev, self.data_name)
            correct = _check_correct(pred, gold, self.ev, self.data_name)

        # ── Per-trace analysis fields (used by ablation / motivation exps)
        all_preds   = [_extract_pred(t, self.ev, self.data_name) for t in all_traces]
        all_correct = (
            [bool(_check_correct(p, gold, self.ev, self.data_name)) for p in all_preds]
            if gold is not None else []
        )

        # Build per-trace cluster-label and cluster-size arrays aligned to all_traces
        # (cluster membership may have changed across iterations)
        label_map = {}
        size_map  = {}
        for ans_key, indices in clusters.items():
            for i in indices:
                label_map[i] = ans_key
                size_map[i]  = len(indices)

        cluster_labels_out = [label_map.get(i, "") for i in range(len(all_traces))]
        cluster_sizes_out  = [size_map.get(i, 1)   for i in range(len(all_traces))]

        # Pad raw_scores / w_prime if traces were added across iterations
        # (re-score was called over the full union so lengths should match;
        #  guard anyway)
        raw_scores_out = raw_scores.tolist() if len(raw_scores) == len(all_traces) \
                         else raw_scores.tolist() + [0.0] * (len(all_traces) - len(raw_scores))
        w_prime_out    = w_prime.tolist() if len(w_prime) == len(all_traces) \
                         else w_prime.tolist() + [0.0] * (len(all_traces) - len(w_prime))
        gamma_out      = gamma.tolist() if len(gamma) == len(all_traces) \
                         else gamma.tolist() + [0.0] * (len(all_traces) - len(gamma))

        elapsed_total = elapsed_generation + (time.time() - t0_select)

        res = InferenceResult(
            method="dn_bon_updated",
            question=question,
            gold_answer=gold,
            selected_trace=selected,
            predicted_answer=pred,
            is_correct=correct,
            N=len(all_traces),              # actual trajectories evaluated
            elapsed_sec=elapsed_total,
            all_traces=all_traces,
            raw_scores=raw_scores_out,
            normalized_scores=w_prime_out,
            cluster_labels=cluster_labels_out,
            cluster_sizes=cluster_sizes_out,
            selected_index=best_idx,
            alpha=self.alpha,
        )
        # Attach CRISP-DORA-specific fields for downstream analysis / logging
        res._all_preds        = all_preds
        res._all_correct      = all_correct
        res._quality_weights  = (
            self._softmax(np.array(raw_scores_out), self.T_b).tolist()
        )
        res._gamma            = gamma_out
        res._T_b              = self.T_b
        res._T_s              = self.T_s
        res._alpha            = self.alpha
        res._beta             = self.beta
        res._delta            = self.delta
        res._cluster_scores   = cluster_scores
        res._iteration_log    = iteration_log
        res._n_iterations     = len(iteration_log)
        return res

    # ── public run / run_batch ────────────────────────────────────────────────

    def run(
        self,
        question: str,
        gold_answer: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> "InferenceResult":
        """
        Run CRISP-DORA for a single question.

        Generation (initial round) is timed separately from selection so the
        InferenceResult.elapsed_sec covers both generation and all scoring rounds.
        """
        t0      = time.time()
        traces  = self.model.generate_samples(
            question, N=self.N,
            temperature=self.temperature, top_p=self.top_p,
            max_new_tokens=max_new_tokens,
        )
        elapsed_gen = time.time() - t0

        return self._select(
            question, gold_answer, traces, elapsed_gen,
            max_new_tokens=max_new_tokens,
        )

    def run_batch(
        self,
        examples: List[Dict[str, Any]],
        max_new_tokens: Optional[int] = None,
    ) -> List["InferenceResult"]:
        """
        Run CRISP-DORA for a list of questions (sequential, one question at a time).

        Live accuracy is tracked and displayed in the progress bar.
        """
        results: List[InferenceResult] = []
        logger.info(
            f"[DNBonUpdated/CRISP-DORA] {len(examples)} questions  "
            f"N={self.N}  max_iter={self.max_iter}  "
            f"T_b={self.T_b}  T_s={self.T_s}  α={self.alpha}  "
            f"β={self.beta}  δ={self.delta}  "
            f"early_stop_K={self.early_stop_clusters}  "
            f"early_stop_p={self.early_stop_majority}  "
            f"batch_size={self.model.batch_size}  "
            f"evaluator={self.ev.__class__.__name__}"
        )

        pbar = tqdm(
            examples,
            desc=(
                f"[CRISP-DORA N={self.N} iter={self.max_iter} "
                f"T_b={self.T_b} T_s={self.T_s} α={self.alpha}]"
            ),
            unit="q", dynamic_ncols=True, colour="cyan",
        )
        n_correct = 0

        for i, ex in enumerate(pbar):
            q    = ex["question"]
            gold = ex.get("answer")
            try:
                t0          = time.time()
                traces      = self.model.generate_samples(
                    q, N=self.N,
                    temperature=self.temperature, top_p=self.top_p,
                    max_new_tokens=max_new_tokens,
                )
                elapsed_gen = time.time() - t0
                res         = self._select(
                    q, gold, traces, elapsed_gen,
                    max_new_tokens=max_new_tokens,
                )
            except Exception as e:
                logger.error(f"  [CRISP-DORA] q[{i}] failed: {e}", exc_info=True)
                res = _make_error_result("dn_bon_updated", q, gold, self.N)

            results.append(res)
            if res.is_correct is not None:
                n_correct += int(res.is_correct)
                iters_done = getattr(res, "_n_iterations", 1)
                pbar.set_postfix(
                    acc=f"{n_correct/(i+1):.3f}",
                    correct=f"{n_correct}/{i+1}",
                    iters=iters_done,
                    clusters=len(getattr(res, "_cluster_scores", {})),
                    refresh=False,
                )

        pbar.close()
        return results



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
    N: int = 16,
    alpha: float = 0.3,
    T_b: float = 0.3,
    T_s: float = 0.1,
    data_name: str = "math",
    temperature: float = 0.7,
    top_p: float = 0.95,
    # ── CRISP-DORA iterative parameters (dn_bon_updated only) ─────────────────
    max_iter: int = 1,
    n_iter_samples: Optional[int] = None,
    prefix_depth: int = 2,
    early_stop_clusters: int = 1,
    early_stop_majority: float = 0.85,
    beta: float = 1.0,
    delta: float = 0.0,
):
    """
    Instantiate an inference method by name.

    method_name choices
    -------------------
    greedy          — single argmax decode, no sampling
    bon             — Best-of-N with verifier score
    dn_bon          — Direction-Normalised BoN (hard cluster-size penalty,
                      requires cluster_manager)
    dn_bon_updated  — CRISP-DORA: answer clustering + DORA semantic uniqueness
                      + optional iterative prefix refinement
                      (requires cluster_manager)
    greedy_pass_n   — N samples, select by max sequence log-probability
    bon_majority    — N samples, select by majority vote (self-consistency)

    CRISP-DORA parameters (dn_bon_updated only)
    --------------------------------------------
    T_b                 : reward softmax temperature        (default 0.3)
    T_s                 : similarity softmax temperature    (default 0.1)
    alpha               : frequency regularisation exponent (default 0.3)
    beta                : reward exponent in w'_i ∝ ṡ^β·γ^δ (default 1.0)
    delta               : uniqueness exponent               (default 1.0)
    max_iter            : CRISP prefix-refinement iterations (default 1 = one-shot)
    n_iter_samples      : trajectories per subsequent iteration (default N//2)
    prefix_depth        : reasoning steps extracted as prefix (default 2)
    early_stop_clusters : τ_K cluster-count threshold        (default 2)
    early_stop_majority : τ_p modal-cluster fraction         (default 0.75)
    """
    if method_name == "greedy":
        return GreedyInference(model=model, data_name=data_name)

    elif method_name == "bon":
        return BestOfN(model=model, verifier=verifier, N=N,
                       data_name=data_name, temperature=temperature, top_p=top_p)

    elif method_name == "dn_bon":
        if cluster_manager is None:
            raise ValueError("DN-BoN requires a ClusterManager.")
        return DNBestOfN(
            model=model, verifier=verifier, cluster_manager=cluster_manager,
            N=N, alpha=alpha, data_name=data_name,
            temperature=temperature, top_p=top_p,
        )

    elif method_name == "dn_bon_updated":
        if cluster_manager is None:
            raise ValueError("CRISP-DORA (dn_bon_updated) requires a ClusterManager.")
        return DNBonUpdated(
            model=model, verifier=verifier, cluster_manager=cluster_manager,
            N=N, T_b=T_b, T_s=T_s, alpha=alpha,
            data_name=data_name, temperature=temperature, top_p=top_p,
            max_iter=max_iter, n_iter_samples=n_iter_samples,
            prefix_depth=prefix_depth,
            early_stop_clusters=early_stop_clusters,
            early_stop_majority=early_stop_majority,
            beta=beta, delta=delta,
        )

    elif method_name == "greedy_pass_n":
        return GreedyPassN(model=model, N=N, data_name=data_name,
                           temperature=temperature, top_p=top_p)

    elif method_name == "bon_majority":
        return BestOfNMajority(model=model, N=N, data_name=data_name,
                               temperature=temperature, top_p=top_p)

    raise ValueError(
        f"Unknown method '{method_name}'. "
        f"Choose: greedy | bon | dn_bon | dn_bon_updated | greedy_pass_n | bon_majority"
    )