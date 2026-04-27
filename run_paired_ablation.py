"""
run_paired_ablation.py
======================
Paired, sample-reuse ablation harness for DN-BoN (NeurIPS submission).

Key design principle
--------------------
  1. GENERATE ONCE  — for each question, N traces are sampled once and
     cached in memory (and optionally to disk as a JSONL sample cache).
  2. SCORE ONCE     — PRM scores are computed once per question.
  3. CLUSTER ONCE per (tau) value — embeddings are computed once;
     re-clustering is just re-labelling with a new distance threshold.
  4. APPLY ALL METHODS — BoN, Majority, Oracle, DN-BoN(α, τ) — on the
     *same* samples, making every comparison paired and fair.

Outputs
-------
  <output_dir>/
    sample_cache_<tag>.jsonl          — raw traces + PRM scores per question
    results_paired_<tag>.jsonl        — per-question, per-config outcome
    summary_paired_<tag>.json         — accuracy + all metrics, all configs
    analysis_paired_<tag>.json        — per-question win/loss/tie analysis
    latex_table_<tag>.txt             — ready-to-paste NeurIPS table
    hurt_cases_<tag>.jsonl            — questions where DN-BoN hurts vs BoN
    help_cases_<tag>.jsonl            — questions where DN-BoN helps vs BoN
    figures/
      accuracy_vs_alpha_<tag>.png
      accuracy_vs_tau_<tag>.png
      help_vs_hurt_fmax_<tag>.png
      fmax_vs_accuracy_scatter_<tag>.png
      cluster_entropy_hist_<tag>.png
      pairwise_delta_heatmap_<tag>.png

Metrics reported per question (for NeurIPS paper analysis)
----------------------------------------------------------
  fmax          — dominant cluster fraction (redundancy proxy)
  H             — Shannon cluster entropy (bits)
  He            — normalised entropy ∈ [0,1]
  M             — number of unique clusters
  r             — unique-directions ratio M/N
  oracle_in_top  — whether any trace is correct (oracle upper bound)
  bon_rank_of_correct — rank of best correct trace under raw PRM (1-indexed)
  bon_selected_correct — whether BoN selected a correct trace
  dnbon_selected_correct — same for DN-BoN (best α/τ combo)
  multiplicity_bias_fired — whether the selected BoN trace came from the
                             dominant cluster but a correct trace existed
                             outside it (Theorem 1 manifestation)
  correction_succeeded   — whether DN-BoN rescued that case

Usage
-----
  # Quick test (AIME-24, N=8, default α/τ sweeps)
  python run_paired_ablation.py \\
      --model Qwen/Qwen2.5-Math-7B-Instruct \\
      --dataset aime24 \\
      --N 8 --batch_size 8 \\
      --max_samples 30

  # Full paper run
  python run_paired_ablation.py \\
      --model  Qwen/Qwen2.5-Math-7B-Instruct \\
      --verifier_model Qwen/Qwen2.5-Math-PRM-7B \\
      --dataset aime24 \\
      --N 8 --batch_size 8 \\
      --alpha_values 0.0 0.25 0.5 0.75 1.0 \\
      --tau_values   0.70 0.75 0.80 0.85 0.90 0.95 \\
      --max_samples 30 --output_dir ./results_paired

  # Resume (skip regeneration if sample cache exists)
  python run_paired_ablation.py \\
      --model  Qwen/Qwen2.5-Math-7B-Instruct \\
      --dataset math500 --N 8 --batch_size 8 \\
      --load_cache ./results_paired/sample_cache_math500_N8.jsonl
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("paired_ablation")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Paired DN-BoN ablation — generate once, evaluate all configs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    g = p.add_argument_group("Model / Verifier")
    g.add_argument("--model", required=True)
    g.add_argument("--verifier_model",  default="Qwen/Qwen2.5-Math-PRM-7B")
    g.add_argument("--prm_aggregation", default="min",
                   choices=["min", "mean", "last"])
    g.add_argument("--embedding_model", default="all-MiniLM-L6-v2")
    g.add_argument("--batch_size",      type=int,   default=8)
    g.add_argument("--dtype",           default="auto",
                   choices=["auto", "float16", "bfloat16", "float32"])
    g.add_argument("--device",          default="auto")
    g.add_argument("--gpus",            default=None)
    g.add_argument("--load_in_4bit",    action="store_true")
    g.add_argument("--load_in_8bit",    action="store_true")
    g.add_argument("--trust_remote_code", action="store_true")
    g.add_argument("--hf_cache_dir",    default=None)

    # Generation
    g = p.add_argument_group("Generation")
    g.add_argument("--N",              type=int,   default=8)
    g.add_argument("--max_new_tokens", type=int,   default=2048)
    g.add_argument("--temperature",    type=float, default=0.7)
    g.add_argument("--top_p",          type=float, default=0.95)

    # Dataset
    g = p.add_argument_group("Dataset")
    g.add_argument("--dataset",     required=True,
                   choices=["math500", "aime24", "aime25", "amc23", "gsm8k",
                             "olympiadbench", "college_math", "minerva_math",
                             "mmlu", "custom"])
    g.add_argument("--split",       default=None)
    g.add_argument("--max_samples", type=int, default=None)
    g.add_argument("--custom_path", default=None)

    # Ablation grids
    g = p.add_argument_group("Ablation grids")
    g.add_argument("--alpha_values", nargs="+", type=float,
                   default=[0.0, 0.25, 0.5, 0.75, 1.0],
                   help="α values to sweep (0 = pure BoN)")
    g.add_argument("--tau_values",   nargs="+", type=float,
                   default=[0.70, 0.80, 0.85, 0.90, 0.95],
                   help="τ (cosine similarity) clustering thresholds to sweep")
    g.add_argument("--clustering_method", default="agglomerative",
                   choices=["agglomerative", "greedy"])

    # CIMR (Cluster-Informed Meta-Reasoning)
    g = p.add_argument_group("CIMR — LLM meta-judge over clusters")
    g.add_argument("--enable_cimr", action="store_true",
                   help="Enable CIMR meta-judge (adds one LLM forward pass per question)")
    g.add_argument("--cimr_variants", nargs="+", type=str,
                   default=["full", "concise"],
                   choices=["full", "concise", "answer_only"],
                   help="CIMR prompt variants to evaluate")
    g.add_argument("--cimr_tau", type=float, default=0.85,
                   help="τ for CIMR clustering (fixed; not swept)")
    g.add_argument("--cimr_max_trace_chars", type=int, default=3000,
                   help="Max chars per representative trace in CIMR judge prompt")
    g.add_argument("--cimr_max_judge_tokens", type=int, default=2048,
                   help="Max tokens for CIMR judge response")
    g.add_argument("--cimr_judge_batch_size", type=int, default=4,
                   help=(
                       "Number of judge prompts packed into one model.generate() call. "
                       "Increase for higher GPU utilisation (if VRAM allows). "
                       "Recommended: 4 for N=8 full-trace prompts on H100 80GB; "
                       "8-16 for concise/answer_only variants."
                   ))

    # Cache / resume
    g = p.add_argument_group("Cache / resume")
    g.add_argument("--load_cache", default=None,
                   help="Path to existing sample-cache JSONL; skips generation")
    g.add_argument("--save_traces", action="store_true", default=True,
                   help="Save full trace text in sample cache (needed for re-embedding)")
    g.add_argument("--no_save_traces", dest="save_traces", action="store_false")

    # Output
    g = p.add_argument_group("Output")
    g.add_argument("--output_dir", default="./results_paired")
    g.add_argument("--log_every",  type=int, default=10)
    g.add_argument("--no_plots",   action="store_true",
                   help="Skip matplotlib figure generation")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — diversity metrics (mirror paper Eqs. 10–13)
# ─────────────────────────────────────────────────────────────────────────────

def diversity_metrics(labels: np.ndarray) -> Dict[str, float]:
    """
    Compute per-question diversity metrics matching the paper's definitions.

    Parameters
    ----------
    labels : np.ndarray  shape (N,)  integer cluster ids

    Returns
    -------
    dict with keys: H, He, fmax, r, M, N
    """
    N = len(labels)
    if N == 0:
        return dict(H=0.0, He=0.0, fmax=1.0, r=0.0, M=0, N=0)

    unique, counts = np.unique(labels, return_counts=True)
    M  = int(len(unique))
    pm = counts / N

    # Eq. 10: Shannon entropy
    H  = float(-np.sum(pm * np.log2(pm + 1e-12)))
    # Eq. 11: normalised entropy
    He = float(H / math.log2(N)) if N > 1 else 0.0
    # Eq. 12: dominant-cluster fraction
    fmax = float(counts.max() / N)
    # Eq. 13: unique-directions ratio
    r    = float(M / N)

    return dict(H=H, He=He, fmax=fmax, r=r, M=M, N=N)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — majority vote (answer-level, same logic as BestOfNMajority)
# ─────────────────────────────────────────────────────────────────────────────

def _majority_select(
    answers: List[str],
    ev,
    data_name: str,
) -> Tuple[int, str, List[int]]:
    """
    Returns (best_idx, majority_answer, vote_counts_per_trace).
    Uses the evaluator's check_answers_equiv when available.
    """
    groups: List[List[int]] = []
    group_rep: List[str]    = []

    for idx, ans in enumerate(answers):
        placed = False
        for g, rep in enumerate(group_rep):
            try:
                equiv = (
                    ev.check_answers_equiv(ans, rep)
                    if ev is not None else None
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

    if not groups:
        return 0, "", [1]

    majority_g   = max(range(len(groups)), key=lambda g: len(groups[g]))
    best_idx     = groups[majority_g][0]
    majority_ans = group_rep[majority_g]

    vote_counts = [0] * len(answers)
    for g, members in enumerate(groups):
        for idx in members:
            vote_counts[idx] = len(members)

    return best_idx, majority_ans, vote_counts


# ─────────────────────────────────────────────────────────────────────────────
# Core: apply all selection rules to a shared pool of traces + scores
# ─────────────────────────────────────────────────────────────────────────────

def apply_all_methods(
    question:    str,
    gold:        Optional[str],
    traces:      List[str],
    prm_scores:  np.ndarray,       # shape (N,)
    all_correct: List[bool],       # per-trace oracle correctness
    alpha_values: List[float],
    tau_values:   List[float],
    cluster_manager,               # ClusterManager instance (one per tau if needed)
    ev,
    data_name:   str,
    clustering_method: str = "agglomerative",
    # CIMR arguments
    judge_model=None,              # HFModelWrapper for CIMR judge (None = skip CIMR)
    cimr_variants: Optional[List[str]] = None,
    cimr_tau: float = 0.85,
    cimr_max_trace_chars: int = 3000,
    cimr_max_judge_tokens: int = 2048,
    cimr_judge_batch_size: int = 4,
) -> Dict[str, Any]:
    """
    Given a fixed set of N traces and their PRM scores, compute the outcome
    of every method / (α, τ) combination WITHOUT any new model calls.

    Returns a flat dict of results keyed by method label.
    """
    N = len(traces)
    answers = []
    for t in traces:
        try:
            from inference_methods import _extract_pred
            ans = _extract_pred(t, ev, data_name)
        except Exception:
            ans = ""
        answers.append(ans)

    results: Dict[str, Any] = {}

    # ── BoN (α = 0, no clustering needed) ────────────────────────────────────
    bon_idx      = int(np.argmax(prm_scores))
    bon_correct  = bool(all_correct[bon_idx])
    bon_answer   = answers[bon_idx]
    results["bon"] = dict(
        selected_idx=bon_idx,
        predicted_answer=bon_answer,
        is_correct=bon_correct,
        raw_score=float(prm_scores[bon_idx]),
        norm_score=float(prm_scores[bon_idx]),
        cluster_size=1,
        alpha=0.0,
        tau=float("nan"),
        method="bon",
    )

    # ── Oracle (select best correct trace if any) ─────────────────────────────
    correct_idxs = [i for i, c in enumerate(all_correct) if c]
    if correct_idxs:
        oracle_idx    = int(max(correct_idxs, key=lambda i: prm_scores[i]))
        oracle_correct = True
    else:
        oracle_idx    = bon_idx
        oracle_correct = False
    results["oracle"] = dict(
        selected_idx=oracle_idx,
        predicted_answer=answers[oracle_idx],
        is_correct=oracle_correct,
        raw_score=float(prm_scores[oracle_idx]),
        norm_score=float(prm_scores[oracle_idx]),
        cluster_size=1,
        alpha=float("nan"),
        tau=float("nan"),
        method="oracle",
    )

    # ── Majority vote (no verifier) ───────────────────────────────────────────
    maj_idx, maj_ans, vote_counts = _majority_select(answers, ev, data_name)
    maj_correct = bool(all_correct[maj_idx])
    results["majority"] = dict(
        selected_idx=maj_idx,
        predicted_answer=maj_ans,
        is_correct=maj_correct,
        raw_score=float(vote_counts[maj_idx]),
        norm_score=float(vote_counts[maj_idx]),
        cluster_size=vote_counts[maj_idx],
        alpha=float("nan"),
        tau=float("nan"),
        method="majority",
    )

    # ── DN-BoN for every (tau, alpha) pair ────────────────────────────────────
    # Cache embeddings across tau values — re-embed only once
    _embed_cache: Optional[np.ndarray] = None

    for tau in tau_values:
        # Re-cluster with this tau (updates distance_threshold in-place safely)
        try:
            cluster_manager.similarity_threshold = tau
            cluster_manager.distance_threshold   = 1.0 - tau
            labels, cluster_sizes = cluster_manager.cluster(traces)
        except Exception as e:
            logger.warning(f"Clustering failed (tau={tau}): {e}. Using singleton clusters.")
            labels        = np.arange(N, dtype=int)
            cluster_sizes = np.ones(N, dtype=int)

        # Diversity metrics for this tau
        div = diversity_metrics(labels)

        for alpha in alpha_values:
            if alpha == 0.0:
                # Recovers plain BoN; add tau-specific diversity info
                norm_scores = prm_scores.copy()
            else:
                safe_sizes  = np.maximum(cluster_sizes.astype(float), 1.0)
                norm_scores = prm_scores / (safe_sizes ** alpha)

            best_idx   = int(np.argmax(norm_scores))
            is_correct = bool(all_correct[best_idx])
            key        = f"dn_bon_tau{tau:.2f}_alpha{alpha:.2f}"

            results[key] = dict(
                selected_idx=best_idx,
                predicted_answer=answers[best_idx],
                is_correct=is_correct,
                raw_score=float(prm_scores[best_idx]),
                norm_score=float(norm_scores[best_idx]),
                cluster_size=int(cluster_sizes[best_idx]),
                alpha=alpha,
                tau=tau,
                method="dn_bon",
                # Diversity metrics (same for all alpha given same tau)
                fmax=div["fmax"],
                H=div["H"],
                He=div["He"],
                M=div["M"],
                r=div["r"],
                # Per-trace scores for saving
                raw_scores=[round(float(s), 5) for s in prm_scores],
                norm_scores=[round(float(s), 5) for s in norm_scores],
                cluster_labels=labels.tolist(),
                cluster_sizes_all=cluster_sizes.tolist(),
            )

    # ── CIMR: Cluster-Informed Meta-Reasoning ────────────────────────────────
    if judge_model is not None and cimr_variants:
        from cimr_inference import cimr_select_from_cache

        # Cluster once with the CIMR-specific tau
        try:
            cluster_manager.similarity_threshold = cimr_tau
            cluster_manager.distance_threshold   = 1.0 - cimr_tau
            cimr_labels, cimr_sizes = cluster_manager.cluster(traces)
        except Exception as e:
            logger.warning(f"CIMR clustering failed: {e}. Using singleton clusters.")
            cimr_labels = np.arange(N, dtype=int)
            cimr_sizes  = np.ones(N, dtype=int)

        cimr_div = diversity_metrics(cimr_labels)

        for variant in cimr_variants:
            try:
                cimr_result = cimr_select_from_cache(
                    question=question,
                    traces=traces,
                    prm_scores=prm_scores.tolist(),
                    cluster_labels=cimr_labels.tolist(),
                    answers=answers,
                    judge_model=judge_model,
                    ev=ev,
                    data_name=data_name,
                    variant=variant,
                    max_trace_chars=cimr_max_trace_chars,
                    max_judge_tokens=cimr_max_judge_tokens,
                    gold_answer=gold,
                    judge_batch_size=cimr_judge_batch_size,
                )

                key = f"cimr_{variant}"
                results[key] = dict(
                    selected_idx=-1,   # CIMR doesn't select a single trace
                    predicted_answer=cimr_result["predicted_answer"],
                    is_correct=cimr_result["is_correct"],
                    raw_score=float("nan"),
                    norm_score=float("nan"),
                    cluster_size=-1,
                    alpha=float("nan"),
                    tau=cimr_tau,
                    method=f"cimr_{variant}",
                    # Diversity metrics (from CIMR clustering)
                    fmax=cimr_div["fmax"],
                    H=cimr_div["H"],
                    He=cimr_div["He"],
                    M=cimr_div["M"],
                    r=cimr_div["r"],
                    n_clusters=cimr_result["n_clusters"],
                    judge_tokens=cimr_result.get("approx_input_tokens", 0),
                    judge_elapsed_sec=cimr_result.get("elapsed_sec", 0.0),
                    fallback=cimr_result.get("fallback", False),
                    # Cluster details for analysis
                    clusters_shown=cimr_result.get("clusters_meta", []),
                    # ── Full judge trace and extracted answer ─────────────
                    judge_raw_response=cimr_result.get("judge_response", ""),
                    cimr_extracted_answer=cimr_result["predicted_answer"],
                )
            except Exception as e:
                logger.error(f"CIMR variant '{variant}' failed: {e}")
                results[f"cimr_{variant}"] = dict(
                    predicted_answer="",
                    is_correct=False,
                    method=f"cimr_{variant}",
                    error=str(e),
                    judge_raw_response="",
                    cimr_extracted_answer="",
                )

    # ── Per-question cross-method analysis ───────────────────────────────────
    # Multiplicity bias detection (Theorem 1 manifestation)
    # Fires when: BoN selects from the dominant cluster but a correct trace
    # exists outside the dominant cluster (or in a smaller cluster).
    bon_cluster = results["bon"]["selected_idx"]
    # Use default tau (first in list) for this analysis
    default_tau = tau_values[len(tau_values) // 2]  # middle tau
    default_alpha_key = f"dn_bon_tau{default_tau:.2f}_alpha{0.5:.2f}"
    if default_alpha_key in results:
        default_labels = np.array(results[default_alpha_key]["cluster_labels"])
        bon_cluster_id = int(default_labels[bon_idx])
        dominant_cid   = int(np.bincount(default_labels).argmax())
        bon_in_dominant = (bon_cluster_id == dominant_cid)
        # Is there a correct trace outside the dominant cluster?
        correct_outside_dominant = any(
            all_correct[i] and int(default_labels[i]) != dominant_cid
            for i in range(N)
        )
        # Did BoN fail but DN-BoN succeed?
        dnbon_rescued = (
            not bon_correct
            and results.get(default_alpha_key, {}).get("is_correct", False)
        )
        # Did DN-BoN hurt (BoN correct but DN-BoN wrong)?
        dnbon_hurt = (
            bon_correct
            and not results.get(default_alpha_key, {}).get("is_correct", True)
        )
    else:
        bon_in_dominant          = False
        correct_outside_dominant = False
        dnbon_rescued            = False
        dnbon_hurt               = False

    # Did CIMR succeed where BoN failed?
    cimr_rescued = False
    cimr_hurt = False
    if "cimr_full" in results:
        cimr_correct = results["cimr_full"].get("is_correct", None)
        if cimr_correct is not None:
            cimr_rescued = (not bon_correct and cimr_correct)
            cimr_hurt    = (bon_correct and not cimr_correct)

    results["__meta__"] = dict(
        N=N,
        any_correct=len(correct_idxs) > 0,
        n_correct_traces=len(correct_idxs),
        answers=answers,
        all_correct=all_correct,
        bon_idx=bon_idx,
        bon_correct=bon_correct,
        oracle_correct=oracle_correct,
        majority_correct=maj_correct,
        bon_in_dominant_cluster=bon_in_dominant,
        correct_exists_outside_dominant=correct_outside_dominant,
        multiplicity_bias_fired=(bon_in_dominant and correct_outside_dominant and not bon_correct),
        dnbon_rescued=dnbon_rescued,
        dnbon_hurt=dnbon_hurt,
        cimr_rescued=cimr_rescued,
        cimr_hurt=cimr_hurt,
        default_tau=default_tau,
    )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Generation + scoring phase (runs once)
# ─────────────────────────────────────────────────────────────────────────────

def generate_and_score(
    examples:       List[Dict[str, Any]],
    model,
    verifier,
    ev,
    data_name:      str,
    N:              int,
    max_new_tokens: int,
    temperature:    float,
    top_p:          float,
    cache_path:     Optional[str],
    log_every:      int = 10,
) -> List[Dict[str, Any]]:
    """
    For each question: generate N traces, score with verifier, extract answers.
    Saves to cache_path (JSONL) as it goes for crash recovery.
    Returns list of dicts, one per question.
    """
    from verifier import score_candidates  # type: ignore

    records = []
    t_total = time.time()

    cache_f = open(cache_path, "w", encoding="utf-8") if cache_path else None

    for i, ex in enumerate(examples):
        q    = ex["question"]
        gold = ex.get("answer", None)
        qid  = ex.get("id", str(i))

        try:
            t0      = time.time()
            traces  = model.generate_samples(
                q, N=N, temperature=temperature, top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
            gen_sec = time.time() - t0

            t1         = time.time()
            prm_scores = score_candidates(traces, verifier, gold, question=q)
            score_sec  = time.time() - t1

            # Per-trace oracle correctness
            from inference_methods import _extract_pred, _check_correct  # type: ignore
            answers     = [_extract_pred(t, ev, data_name) for t in traces]
            all_correct = [
                bool(_check_correct(a, gold, ev, data_name)) for a in answers
            ]

            rec = dict(
                id=qid,
                question=q,
                gold_answer=gold,
                traces=traces,
                prm_scores=[round(float(s), 6) for s in prm_scores],
                answers=answers,
                all_correct=all_correct,
                gen_sec=round(gen_sec, 3),
                score_sec=round(score_sec, 3),
            )
        except Exception as e:
            logger.error(f"[gen+score] q[{i}] failed: {e}")
            rec = dict(
                id=qid, question=q, gold_answer=gold,
                traces=[], prm_scores=[], answers=[],
                all_correct=[], gen_sec=0.0, score_sec=0.0, error=str(e),
            )

        records.append(rec)

        if cache_f is not None:
            cache_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            cache_f.flush()

        if (i + 1) % log_every == 0 or (i + 1) == len(examples):
            elapsed = time.time() - t_total
            pct_correct = (
                sum(any(r["all_correct"]) for r in records) / len(records) * 100
                if records else 0
            )
            logger.info(
                f"  Generated {i+1}/{len(examples)}  "
                f"oracle_any={pct_correct:.1f}%  "
                f"elapsed={elapsed:.0f}s"
            )

    if cache_f is not None:
        cache_f.close()
        logger.info(f"Sample cache written → {cache_path}")

    return records


def load_cache(cache_path: str) -> List[Dict[str, Any]]:
    records = []
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} cached questions from {cache_path}")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation utilities
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_results(
    all_question_results: List[Dict[str, Any]],
    alpha_values: List[float],
    tau_values:   List[float],
) -> Dict[str, Any]:
    """
    Aggregate per-question results into per-config summary statistics.
    Returns nested dict: config_key → metrics dict.
    """
    # Collect per-config lists
    per_config: Dict[str, List] = defaultdict(list)

    for qr in all_question_results:
        for key, val in qr.items():
            if key.startswith("__"):
                continue
            per_config[key].append(val.get("is_correct", None))

    summary: Dict[str, Any] = {}
    for key, correct_list in per_config.items():
        valid = [c for c in correct_list if c is not None]
        if not valid:
            continue
        acc = sum(valid) / len(valid) * 100
        summary[key] = dict(
            accuracy=round(acc, 2),
            n_correct=sum(valid),
            n_total=len(valid),
        )

    # Add redundancy stats (from meta)
    meta_list = [qr.get("__meta__", {}) for qr in all_question_results]
    if meta_list:
        fmax_list  = []
        H_list     = []
        bias_fired = 0
        rescued    = 0
        hurt       = 0
        cimr_rescued = 0
        cimr_hurt    = 0

        for m in meta_list:
            bias_fired   += int(m.get("multiplicity_bias_fired", False))
            rescued      += int(m.get("dnbon_rescued", False))
            hurt         += int(m.get("dnbon_hurt",    False))
            cimr_rescued += int(m.get("cimr_rescued",  False))
            cimr_hurt    += int(m.get("cimr_hurt",     False))

        # Get fmax / H from the middle-tau DN-BoN configs
        for qr in all_question_results:
            for k, v in qr.items():
                if k.startswith("dn_bon") and "fmax" in v:
                    fmax_list.append(v["fmax"])
                    H_list.append(v["H"])
                    break

        n = len(meta_list)
        summary["__redundancy__"] = dict(
            mean_fmax=round(float(np.mean(fmax_list)) if fmax_list else 0, 4),
            mean_H=round(float(np.mean(H_list)) if H_list else 0, 4),
            multiplicity_bias_fired_count=bias_fired,
            multiplicity_bias_fired_pct=round(bias_fired / n * 100, 1) if n else 0,
            dnbon_rescued_count=rescued,
            dnbon_hurt_count=hurt,
            dnbon_net_help=rescued - hurt,
            cimr_rescued_count=cimr_rescued,
            cimr_hurt_count=cimr_hurt,
            cimr_net_help=cimr_rescued - cimr_hurt,
            n_questions=n,
        )

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX table generator
# ─────────────────────────────────────────────────────────────────────────────

def build_latex_table(
    summary:       Dict[str, Any],
    alpha_values:  List[float],
    tau_values:    List[float],
    dataset:       str,
    N:             int,
) -> str:
    """
    Produce a NeurIPS-style LaTeX table for the ablation.

    Primary table: accuracy vs. (α, τ) for DN-BoN, with BoN / Oracle / Majority
    as fixed rows at top.
    """
    lines = [
        "% ── Auto-generated by run_paired_ablation.py ──────────────────",
        f"% Dataset: {dataset}  N={N}",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Paired ablation: accuracy (\%) on \texttt{" + dataset + r"} at $N=" + str(N) + r"$.",
        r"  All methods use the \emph{same} sampled traces.",
        r"  \textbf{Bold} = best DN-BoN config per column.}",
        r"\label{tab:paired_ablation_" + dataset + r"}",
    ]

    n_tau = len(tau_values)
    col_fmt = "l" + "c" * n_tau
    lines += [
        r"\resizebox{\linewidth}{!}{",
        r"\begin{tabular}{" + col_fmt + r"}",
        r"\toprule",
    ]

    # Header row: τ values
    tau_header = " & ".join([f"$\\tau={t:.2f}$" for t in tau_values])
    lines.append(r"Method / $\alpha$ & " + tau_header + r" \\")
    lines.append(r"\midrule")

    # Fixed baselines
    for method_key, label in [("bon", "BoN (Oracle)"),
                                ("majority", "Majority Voting"),
                                ("oracle", "Oracle Upper Bound")]:
        if method_key in summary:
            acc = summary[method_key]["accuracy"]
            # Same value across all tau columns
            row = label + " & " + " & ".join([f"{acc:.1f}"] * n_tau) + r" \\"
            lines.append(row)

    # CIMR rows (also tau-invariant since CIMR uses its own fixed tau)
    for variant in ["full", "concise", "answer_only"]:
        cimr_key = f"cimr_{variant}"
        if cimr_key in summary:
            acc = summary[cimr_key]["accuracy"]
            label = f"CIMR ({variant})"
            row = label + " & " + " & ".join([f"{acc:.1f}"] * n_tau) + r" \\"
            lines.append(row)

    lines.append(r"\midrule")

    # DN-BoN rows: one row per alpha, columns = tau
    # Find global best for bolding
    all_dnbon_accs = []
    for a in alpha_values:
        for t in tau_values:
            k = f"dn_bon_tau{t:.2f}_alpha{a:.2f}"
            if k in summary:
                all_dnbon_accs.append(summary[k]["accuracy"])
    best_acc = max(all_dnbon_accs) if all_dnbon_accs else -1

    for a in alpha_values:
        if a == 0.0:
            alpha_label = r"DN-BoN ($\alpha=0$, = BoN)"
        else:
            alpha_label = f"DN-BoN ($\\alpha={a}$)"

        cells = []
        for t in tau_values:
            k = f"dn_bon_tau{t:.2f}_alpha{a:.2f}"
            if k in summary:
                acc = summary[k]["accuracy"]
                cell = f"\\textbf{{{acc:.1f}}}" if abs(acc - best_acc) < 0.05 else f"{acc:.1f}"
            else:
                cell = "--"
            cells.append(cell)

        lines.append(alpha_label + " & " + " & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",  # close resizebox
        r"\end{table}",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Help / Hurt analysis
# ─────────────────────────────────────────────────────────────────────────────

def build_help_hurt_analysis(
    records:              List[Dict[str, Any]],
    all_question_results: List[Dict[str, Any]],
    alpha_values:         List[float],
    tau_values:           List[float],
    default_alpha:        float = 0.5,
    default_tau:          float = 0.85,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Compare DN-BoN (default α, τ) vs BoN on each question.
    Returns (help_cases, hurt_cases) — lists of annotated question dicts.
    """
    dnbon_key = f"dn_bon_tau{default_tau:.2f}_alpha{default_alpha:.2f}"
    help_cases, hurt_cases = [], []

    for rec, qr in zip(records, all_question_results):
        bon_correct   = qr.get("bon", {}).get("is_correct", None)
        dnbon_correct = qr.get(dnbon_key, {}).get("is_correct", None)

        if bon_correct is None or dnbon_correct is None:
            continue

        meta = qr.get("__meta__", {})

        case = dict(
            id=rec["id"],
            question=rec["question"][:200],  # truncate for readability
            gold_answer=rec.get("gold_answer"),
            n_traces=len(rec.get("traces", [])),
            n_correct_traces=meta.get("n_correct_traces", 0),
            bon_correct=bon_correct,
            dnbon_correct=dnbon_correct,
            oracle_correct=meta.get("oracle_correct", False),
            majority_correct=qr.get("majority", {}).get("is_correct", False),
            bon_answer=qr.get("bon", {}).get("predicted_answer", ""),
            dnbon_answer=qr.get(dnbon_key, {}).get("predicted_answer", ""),
            bon_raw_score=qr.get("bon", {}).get("raw_score", 0.0),
            dnbon_raw_score=qr.get(dnbon_key, {}).get("raw_score", 0.0),
            dnbon_norm_score=qr.get(dnbon_key, {}).get("norm_score", 0.0),
            dnbon_cluster_size=qr.get(dnbon_key, {}).get("cluster_size", 1),
            fmax=qr.get(dnbon_key, {}).get("fmax", float("nan")),
            H=qr.get(dnbon_key, {}).get("H", float("nan")),
            He=qr.get(dnbon_key, {}).get("He", float("nan")),
            M=qr.get(dnbon_key, {}).get("M", 1),
            r=qr.get(dnbon_key, {}).get("r", 1.0),
            multiplicity_bias_fired=meta.get("multiplicity_bias_fired", False),
            prm_scores=rec.get("prm_scores", []),
            cluster_labels=qr.get(dnbon_key, {}).get("cluster_labels", []),
            cluster_sizes_all=qr.get(dnbon_key, {}).get("cluster_sizes_all", []),
            # ── CIMR judge outputs ────────────────────────────────────────
            cimr_full_correct=qr.get("cimr_full", {}).get("is_correct", None),
            cimr_full_answer=qr.get("cimr_full", {}).get("cimr_extracted_answer", ""),
            cimr_full_raw_response=qr.get("cimr_full", {}).get("judge_raw_response", ""),
            cimr_concise_correct=qr.get("cimr_concise", {}).get("is_correct", None),
            cimr_concise_answer=qr.get("cimr_concise", {}).get("cimr_extracted_answer", ""),
            cimr_concise_raw_response=qr.get("cimr_concise", {}).get("judge_raw_response", ""),
        )

        if not bon_correct and dnbon_correct:
            help_cases.append(case)
        elif bon_correct and not dnbon_correct:
            hurt_cases.append(case)

    # Sort help cases by fmax descending (most redundant first — biggest wins)
    help_cases.sort(key=lambda x: x.get("fmax", 0), reverse=True)
    # Sort hurt cases by H ascending (least diverse — surprising hurts)
    hurt_cases.sort(key=lambda x: x.get("H", 0))

    return help_cases, hurt_cases


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def make_plots(
    summary:              Dict[str, Any],
    all_question_results: List[Dict[str, Any]],
    alpha_values:         List[float],
    tau_values:           List[float],
    help_cases:           List[Dict],
    hurt_cases:           List[Dict],
    dataset:              str,
    N:                    int,
    figures_dir:          str,
    tag:                  str,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        logger.warning("matplotlib not available — skipping plots.")
        return

    os.makedirs(figures_dir, exist_ok=True)

    # ── 1. Accuracy vs. α (one line per τ) ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    colors  = cm.tab10(np.linspace(0, 1, len(tau_values)))
    bon_acc = summary.get("bon", {}).get("accuracy", None)
    if bon_acc is not None:
        ax.axhline(bon_acc, color="black", linestyle="--", linewidth=1.5,
                   label=f"BoN (α=0): {bon_acc:.1f}%")

    for c, tau in zip(colors, tau_values):
        accs = []
        for a in alpha_values:
            k = f"dn_bon_tau{tau:.2f}_alpha{a:.2f}"
            accs.append(summary.get(k, {}).get("accuracy", float("nan")))
        ax.plot(alpha_values, accs, marker="o", color=c, linewidth=2,
                label=f"τ={tau:.2f}")

    ax.set_xlabel("Normalisation strength α", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(f"DN-BoN: Accuracy vs. α  [{dataset}, N={N}]", fontsize=13)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(figures_dir, f"accuracy_vs_alpha_{tag}.pdf")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    fig.savefig(p.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {p}")

    # ── 2. Accuracy vs. τ (one line per α) ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    colors  = cm.tab10(np.linspace(0, 1, len(alpha_values)))
    if bon_acc is not None:
        ax.axhline(bon_acc, color="black", linestyle="--", linewidth=1.5,
                   label=f"BoN (baseline): {bon_acc:.1f}%")

    for c, a in zip(colors, alpha_values):
        accs = []
        for tau in tau_values:
            k = f"dn_bon_tau{tau:.2f}_alpha{a:.2f}"
            accs.append(summary.get(k, {}).get("accuracy", float("nan")))
        ax.plot(tau_values, accs, marker="s", color=c, linewidth=2,
                label=f"α={a:.2f}")

    ax.set_xlabel("Clustering threshold τ", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(f"DN-BoN: Accuracy vs. τ  [{dataset}, N={N}]", fontsize=13)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(figures_dir, f"accuracy_vs_tau_{tag}.pdf")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    fig.savefig(p.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {p}")

    # ── 3. Help vs. Hurt: fmax distribution ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    help_fmax = [c["fmax"] for c in help_cases if not np.isnan(c.get("fmax", float("nan")))]
    hurt_fmax = [c["fmax"] for c in hurt_cases if not np.isnan(c.get("fmax", float("nan")))]
    bins = np.linspace(0, 1, 11)
    if help_fmax:
        ax.hist(help_fmax, bins=bins, alpha=0.65, color="seagreen",
                label=f"DN-BoN helps (n={len(help_fmax)})", density=True)
    if hurt_fmax:
        ax.hist(hurt_fmax, bins=bins, alpha=0.65, color="firebrick",
                label=f"DN-BoN hurts (n={len(hurt_fmax)})", density=True)
    ax.set_xlabel("Dominant-cluster fraction $f_{\\mathrm{max}}$", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Help vs. Hurt — fmax distribution [{dataset}]", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(figures_dir, f"help_vs_hurt_fmax_{tag}.pdf")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    fig.savefig(p.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {p}")

    # ── 4. fmax vs. Accuracy scatter (per-question) ───────────────────────────
    default_tau   = tau_values[len(tau_values) // 2]
    default_alpha = 0.5
    dnbon_key     = f"dn_bon_tau{default_tau:.2f}_alpha{default_alpha:.2f}"

    fmax_vals, bon_corr, dnbon_corr = [], [], []
    for qr in all_question_results:
        if dnbon_key in qr and "bon" in qr:
            fmax_vals.append(qr[dnbon_key].get("fmax", float("nan")))
            bon_corr.append(int(qr["bon"].get("is_correct", False)))
            dnbon_corr.append(int(qr[dnbon_key].get("is_correct", False)))

    if fmax_vals:
        fmax_arr = np.array(fmax_vals)
        delta    = np.array(dnbon_corr, dtype=float) - np.array(bon_corr, dtype=float)

        # Bin by fmax
        bins_edges = np.linspace(0, 1, 6)
        bin_centers, bin_delta = [], []
        for lo, hi in zip(bins_edges[:-1], bins_edges[1:]):
            mask = (fmax_arr >= lo) & (fmax_arr < hi)
            if mask.sum() > 0:
                bin_centers.append((lo + hi) / 2)
                bin_delta.append(float(delta[mask].mean()))

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.scatter(fmax_arr, delta + np.random.normal(0, 0.02, len(delta)),
                   alpha=0.35, s=20, color="steelblue", label="per-question Δ")
        if bin_centers:
            ax.plot(bin_centers, bin_delta, "o-", color="darkorange",
                    linewidth=2, markersize=7, label="bin mean")
        ax.set_xlabel("$f_{\\mathrm{max}}$ (dominant-cluster fraction)", fontsize=12)
        ax.set_ylabel("DN-BoN − BoN correct (Δ)", fontsize=12)
        ax.set_title(f"Correction strength vs. redundancy [{dataset}]", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = os.path.join(figures_dir, f"fmax_vs_accuracy_scatter_{tag}.pdf")
        fig.savefig(p, dpi=200, bbox_inches="tight")
        fig.savefig(p.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Saved: {p}")

    # ── 5. Cluster entropy histogram (DN-BoN vs Majority) ────────────────────
    dn_H_vals, maj_vote_vals = [], []
    for qr in all_question_results:
        if dnbon_key in qr:
            dn_H_vals.append(qr[dnbon_key].get("H", float("nan")))
        if "majority" in qr:
            # Approximate majority entropy from vote counts
            vc = qr["majority"].get("cluster_size", 1)
            dn_H_vals  # already computed above

    if dn_H_vals:
        fig, ax = plt.subplots(figsize=(6, 4))
        valid_H = [h for h in dn_H_vals if not np.isnan(h)]
        if valid_H:
            ax.hist(valid_H, bins=15, alpha=0.7, color="steelblue",
                    label="DN-BoN cluster entropy H")
        ax.set_xlabel("Cluster entropy H (bits)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"Semantic cluster entropy distribution [{dataset}, N={N}]",
                     fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = os.path.join(figures_dir, f"cluster_entropy_hist_{tag}.pdf")
        fig.savefig(p, dpi=200, bbox_inches="tight")
        fig.savefig(p.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Saved: {p}")

    # ── 6. Pairwise delta heatmap: DN-BoN(α, τ) − BoN ────────────────────────
    delta_matrix = np.full((len(alpha_values), len(tau_values)), float("nan"))
    for i, a in enumerate(alpha_values):
        for j, t in enumerate(tau_values):
            k = f"dn_bon_tau{t:.2f}_alpha{a:.2f}"
            if k in summary and "bon" in summary:
                delta_matrix[i, j] = (
                    summary[k]["accuracy"] - summary["bon"]["accuracy"]
                )

    if not np.all(np.isnan(delta_matrix)):
        fig, ax = plt.subplots(figsize=(7, 4))
        vmax = max(abs(np.nanmin(delta_matrix)), abs(np.nanmax(delta_matrix))) + 0.5
        im   = ax.imshow(
            delta_matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax,
            aspect="auto",
        )
        ax.set_xticks(range(len(tau_values)))
        ax.set_xticklabels([f"{t:.2f}" for t in tau_values], fontsize=9)
        ax.set_yticks(range(len(alpha_values)))
        ax.set_yticklabels([f"{a:.2f}" for a in alpha_values], fontsize=9)
        ax.set_xlabel("Clustering threshold τ", fontsize=11)
        ax.set_ylabel("Normalisation strength α", fontsize=11)
        ax.set_title(f"Δ Accuracy = DN-BoN − BoN  [{dataset}, N={N}]", fontsize=13)

        for i in range(len(alpha_values)):
            for j in range(len(tau_values)):
                if not np.isnan(delta_matrix[i, j]):
                    ax.text(j, i, f"{delta_matrix[i,j]:+.1f}",
                            ha="center", va="center", fontsize=8,
                            color="black" if abs(delta_matrix[i, j]) < vmax * 0.6 else "white")

        plt.colorbar(im, ax=ax, label="Δ Accuracy (pp)")
        fig.tight_layout()
        p = os.path.join(figures_dir, f"pairwise_delta_heatmap_{tag}.pdf")
        fig.savefig(p, dpi=200, bbox_inches="tight")
        fig.savefig(p.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Saved: {p}")

    logger.info("All figures saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    figures_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    tag = f"{args.dataset}_N{args.N}"

    # ── 0. GPU setup ──────────────────────────────────────────────────────────
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        logger.info(f"CUDA_VISIBLE_DEVICES={args.gpus}")

    import torch
    if torch.cuda.is_available():
        logger.info(f"GPUs available: {torch.cuda.device_count()}")

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    logger.info(f"Loading dataset: {args.dataset}")
    from dataset_loader import load_dataset_split  # type: ignore
    examples = load_dataset_split(
        data_name=args.dataset, split=args.split,
        max_samples=args.max_samples, custom_path=args.custom_path,
        cache_dir=args.hf_cache_dir,
    )
    data_name = examples[0]["_data_name"] if examples else args.dataset
    logger.info(f"Loaded {len(examples)} examples.")

    # ── 2. Load model ─────────────────────────────────────────────────────────
    logger.info(f"Loading generator: {args.model}")
    from model_wrapper import HFModelWrapper  # type: ignore
    model = HFModelWrapper(
        model_name_or_path=args.model,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.hf_cache_dir,
    )

    # ── 3. Load verifier ──────────────────────────────────────────────────────
    logger.info(f"Loading verifier: {args.verifier_model}")
    from verifier import build_verifier  # type: ignore
    verifier = build_verifier(
        verifier_type="prm",
        data_name=data_name,
        prm_model=args.verifier_model,
        prm_aggregation=args.prm_aggregation,
        device=args.device,
        cache_dir=args.hf_cache_dir,
        batch_size=args.batch_size,
    )

    # ── 4. Build evaluator ────────────────────────────────────────────────────
    from inference_methods import _build_evaluator  # type: ignore
    ev = _build_evaluator(data_name)

    # ── 5. Build ClusterManager (shared; tau updated per call) ────────────────
    logger.info(f"Loading ClusterManager: {args.embedding_model}")
    from clustering import ClusterManager  # type: ignore
    cluster_manager = ClusterManager(
        embedding_model=args.embedding_model,
        similarity_threshold=args.tau_values[0],  # will be overridden per call
        clustering_method=args.clustering_method,
        use_answer_only=False,
        batch_size=args.batch_size,
    )

    # ── 6. Generate traces (or load from cache) ───────────────────────────────
    cache_path = os.path.join(args.output_dir, f"sample_cache_{tag}.jsonl")

    if args.load_cache and os.path.exists(args.load_cache):
        logger.info(f"Loading sample cache from {args.load_cache}")
        records = load_cache(args.load_cache)
    elif os.path.exists(cache_path):
        logger.info(f"Found existing sample cache: {cache_path} — loading.")
        records = load_cache(cache_path)
    else:
        logger.info(
            f"No sample cache found. Generating {args.N} traces × "
            f"{len(examples)} questions..."
        )
        records = generate_and_score(
            examples=examples,
            model=model,
            verifier=verifier,
            ev=ev,
            data_name=data_name,
            N=args.N,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            cache_path=cache_path if args.save_traces else None,
            log_every=args.log_every,
        )
        if args.save_traces and not os.path.exists(cache_path):
            # write manually if not already written
            with open(cache_path, "w") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            logger.info(f"Sample cache written → {cache_path}")

    # ── 7. Apply all methods on shared samples ────────────────────────────────
    logger.info(
        f"Applying all methods on cached traces "
        f"(α: {args.alpha_values}, τ: {args.tau_values})..."
    )

    # ── Prepare CIMR judge model reference ───────────────────────────────────
    judge_model   = model if args.enable_cimr else None
    cimr_variants = args.cimr_variants if args.enable_cimr else None

    # ─────────────────────────────────────────────────────────────────────────
    # PASS 1  — BoN, Majority, Oracle, DN-BoN (all numerical, no LLM call)
    #           Run question-by-question; fast, no GPU needed beyond embeddings.
    # PASS 2  — CIMR (batched LLM judge across all Q questions at once)
    #           All judge prompts assembled, then dispatched in chunks of
    #           cimr_judge_batch_size through _generate_batch_raw.  This gives
    #           full GPU utilisation instead of the one-at-a-time approach that
    #           caused the judge_batch_size=1 bottleneck.
    # ─────────────────────────────────────────────────────────────────────────

    all_question_results: List[Dict[str, Any]] = []

    # Per-question state needed for CIMR pass (stored to avoid re-reading records)
    _cimr_inputs: List[Optional[Dict[str, Any]]] = []   # None for skipped questions

    logger.info(f"  Pass 1: numerical methods  ({len(records)} questions)")
    for i, rec in enumerate(records):
        traces      = rec.get("traces", [])
        prm_scores  = np.array(rec.get("prm_scores", []), dtype=float)
        all_correct = rec.get("all_correct", [])

        if len(traces) == 0 or len(prm_scores) == 0:
            logger.warning(f"  Skipping q[{i}] — empty traces/scores.")
            all_question_results.append({"__meta__": {"error": True}})
            _cimr_inputs.append(None)
            continue

        # Run BoN / Majority / Oracle / DN-BoN only (judge_model=None skips CIMR)
        qr = apply_all_methods(
            question=rec["question"],
            gold=rec.get("gold_answer"),
            traces=traces,
            prm_scores=prm_scores,
            all_correct=all_correct,
            alpha_values=args.alpha_values,
            tau_values=args.tau_values,
            cluster_manager=cluster_manager,
            ev=ev,
            data_name=data_name,
            clustering_method=args.clustering_method,
            judge_model=None,           # CIMR deferred to Pass 2
            cimr_variants=None,
            cimr_tau=args.cimr_tau,
            cimr_max_trace_chars=args.cimr_max_trace_chars,
            cimr_max_judge_tokens=args.cimr_max_judge_tokens,
            cimr_judge_batch_size=args.cimr_judge_batch_size,
        )
        qr["__id__"] = rec.get("id", str(i))
        all_question_results.append(qr)

        # Store inputs for CIMR pass
        if judge_model is not None and cimr_variants:
            # Cluster with CIMR-specific tau (reuse cluster_manager)
            try:
                cluster_manager.similarity_threshold = args.cimr_tau
                cluster_manager.distance_threshold   = 1.0 - args.cimr_tau
                cimr_labels, cimr_sizes = cluster_manager.cluster(traces)
            except Exception as e:
                logger.warning(f"  CIMR pre-cluster failed q[{i}]: {e}")
                cimr_labels = np.arange(len(traces), dtype=int)
                cimr_sizes  = np.ones(len(traces), dtype=int)

            from inference_methods import _extract_pred  # type: ignore
            answers = [_extract_pred(t, ev, data_name) for t in traces]

            _cimr_inputs.append(dict(
                question      = rec["question"],
                gold          = rec.get("gold_answer"),
                traces        = traces,
                prm_scores    = prm_scores.tolist(),
                cluster_labels= cimr_labels.tolist(),
                cluster_sizes = cimr_sizes.tolist(),
                answers       = answers,
                cimr_div      = diversity_metrics(cimr_labels),
            ))
        else:
            _cimr_inputs.append(None)

    # ─────────────────────────────────────────────────────────────────────────
    # PASS 2 — CIMR batched judge inference across all Q questions
    # ─────────────────────────────────────────────────────────────────────────
    if judge_model is not None and cimr_variants:
        from cimr_inference import cimr_judge_batch  # type: ignore

        # Collect valid (non-skipped) questions for each variant
        valid_idxs = [i for i, ci in enumerate(_cimr_inputs) if ci is not None]

        logger.info(
            f"  Pass 2: CIMR batched judge  "
            f"Q_valid={len(valid_idxs)}  "
            f"variants={cimr_variants}  "
            f"judge_batch_size={args.cimr_judge_batch_size}"
        )

        for variant in cimr_variants:
            if not valid_idxs:
                break

            judge_results_valid = cimr_judge_batch(
                questions           = [_cimr_inputs[i]["question"]       for i in valid_idxs],
                traces_list         = [_cimr_inputs[i]["traces"]          for i in valid_idxs],
                prm_scores_list     = [_cimr_inputs[i]["prm_scores"]      for i in valid_idxs],
                cluster_labels_list = [_cimr_inputs[i]["cluster_labels"]  for i in valid_idxs],
                answers_list        = [_cimr_inputs[i]["answers"]         for i in valid_idxs],
                judge_model         = judge_model,
                ev                  = ev,
                data_name           = data_name,
                variant             = variant,
                max_trace_chars     = args.cimr_max_trace_chars,
                max_judge_tokens    = args.cimr_max_judge_tokens,
                judge_batch_size    = args.cimr_judge_batch_size,
                gold_answers        = [_cimr_inputs[i]["gold"]            for i in valid_idxs],
            )

            # Merge CIMR results back into per-question result dicts
            for new_i, orig_i in enumerate(valid_idxs):
                jr       = judge_results_valid[new_i]
                ci       = _cimr_inputs[orig_i]
                cimr_div = ci["cimr_div"]
                key      = f"cimr_{variant}"

                all_question_results[orig_i][key] = dict(
                    selected_idx         = -1,
                    predicted_answer     = jr["predicted_answer"],
                    is_correct           = jr["is_correct"],
                    raw_score            = float("nan"),
                    norm_score           = float("nan"),
                    cluster_size         = -1,
                    alpha                = float("nan"),
                    tau                  = args.cimr_tau,
                    method               = key,
                    fmax                 = cimr_div["fmax"],
                    H                    = cimr_div["H"],
                    He                   = cimr_div["He"],
                    M                    = cimr_div["M"],
                    r                    = cimr_div["r"],
                    n_clusters           = jr["n_clusters"],
                    judge_tokens         = jr.get("approx_input_tokens", 0),
                    judge_elapsed_sec    = jr.get("elapsed_sec", 0.0),
                    fallback             = jr.get("fallback", False),
                    clusters_shown       = jr.get("clusters_meta", []),
                    judge_raw_response   = jr.get("judge_response", ""),
                    cimr_extracted_answer= jr["predicted_answer"],
                )

        # Fill in error placeholders for any variant on skipped questions
        skipped_idxs = [i for i, ci in enumerate(_cimr_inputs) if ci is None]
        for orig_i in skipped_idxs:
            for variant in cimr_variants:
                key = f"cimr_{variant}"
                if key not in all_question_results[orig_i]:
                    all_question_results[orig_i][key] = dict(
                        predicted_answer="", is_correct=False,
                        method=key, error="skipped",
                        judge_raw_response="", cimr_extracted_answer="",
                    )

    # ─────────────────────────────────────────────────────────────────────────
    # Write results JSONL + per-question detailed log
    # ─────────────────────────────────────────────────────────────────────────
    results_path = os.path.join(args.output_dir, f"results_paired_{tag}.jsonl")
    results_f    = open(results_path, "w", encoding="utf-8")

    _STRIP_KEYS    = {"raw_scores", "norm_scores", "cluster_labels", "cluster_sizes_all"}
    _default_dn_key = (
        f"dn_bon_tau"
        f"{args.tau_values[len(args.tau_values)//2]:.2f}"
        f"_alpha0.50"
    )
    # Running accuracy counters for progress log
    _acc: Dict[str, int] = {}

    for i, qr in enumerate(all_question_results):
        rec = records[i] if i < len(records) else {}

        # ── Per-question detailed log ─────────────────────────────────────────
        q_short  = rec.get("question", "")[:80].replace("\n", " ")
        gold_ans = rec.get("gold_answer", "?")
        traces_q = rec.get("traces", [])
        prm_q    = rec.get("prm_scores", [])

        log_lines = [
            "",
            f"  ┌─ Q[{i+1}/{len(all_question_results)}]  gold={gold_ans}",
            f"  │  {q_short}{'...' if len(rec.get('question','')) > 80 else ''}",
            f"  │",
            f"  │  N={len(traces_q)} traces:",
        ]

        # All N answers with PRM scores
        from inference_methods import _extract_pred  # type: ignore
        n_answers = qr.get("__meta__", {}).get("answers", [])
        if not n_answers and traces_q:
            n_answers = [_extract_pred(t, ev, data_name) for t in traces_q]
        for j, (ans, prm) in enumerate(
            zip(n_answers, prm_q) if n_answers else []
        ):
            correct_flag = ""
            if qr.get("__meta__", {}).get("all_correct"):
                correct_flag = " ✓" if qr["__meta__"]["all_correct"][j] else " ✗"
            log_lines.append(
                f"  │    [{j:2d}] prm={prm:.4f}  ans={ans!r:<20s}{correct_flag}"
            )

        log_lines.append(f"  │")
        log_lines.append(f"  │  Method selections:")

        # BoN
        bon = qr.get("bon", {})
        _acc["bon"] = _acc.get("bon", 0) + int(bon.get("is_correct", False))
        log_lines.append(
            f"  │    BoN          → ans={bon.get('predicted_answer','?')!r:<20s}"
            f"  prm={bon.get('raw_score', float('nan')):.4f}"
            f"  correct={bon.get('is_correct','?')}"
        )

        # Majority
        maj = qr.get("majority", {})
        _acc["majority"] = _acc.get("majority", 0) + int(maj.get("is_correct", False))
        log_lines.append(
            f"  │    Majority     → ans={maj.get('predicted_answer','?')!r:<20s}"
            f"  votes={maj.get('raw_score', '?')}"
            f"  correct={maj.get('is_correct','?')}"
        )

        # Oracle
        orc = qr.get("oracle", {})
        log_lines.append(
            f"  │    Oracle       → ans={orc.get('predicted_answer','?')!r:<20s}"
            f"  correct={orc.get('is_correct','?')}"
        )

        # DN-BoN default
        dn = qr.get(_default_dn_key, {})
        _acc[_default_dn_key] = _acc.get(_default_dn_key, 0) + int(dn.get("is_correct", False))
        log_lines.append(
            f"  │    DN-BoN(def)  → ans={dn.get('predicted_answer','?')!r:<20s}"
            f"  norm={dn.get('norm_score', float('nan')):.4f}"
            f"  k={dn.get('cluster_size','?')}"
            f"  correct={dn.get('is_correct','?')}"
        )

        # CIMR variants
        for variant in (cimr_variants or []):
            ckey = f"cimr_{variant}"
            cr   = qr.get(ckey, {})
            _acc[ckey] = _acc.get(ckey, 0) + int(cr.get("is_correct", False))
            fb   = " [fallback]" if cr.get("fallback") else ""
            log_lines.append(
                f"  │    CIMR-{variant:<8s} → ans={cr.get('cimr_extracted_answer','?')!r:<20s}"
                f"  correct={cr.get('is_correct','?')}"
                f"{fb}"
            )

        # Running accuracy line
        n_done = i + 1
        acc_parts = [f"BoN={_acc.get('bon',0)}/{n_done}"]
        acc_parts.append(f"Maj={_acc.get('majority',0)}/{n_done}")
        acc_parts.append(f"DN-BoN={_acc.get(_default_dn_key,0)}/{n_done}")
        for variant in (cimr_variants or []):
            ckey = f"cimr_{variant}"
            acc_parts.append(f"CIMR-{variant}={_acc.get(ckey,0)}/{n_done}")
        log_lines.append(f"  │")
        log_lines.append(f"  │  Running accuracy:  {' | '.join(acc_parts)}")
        log_lines.append(f"  └{'─' * 66}")

        for line in log_lines:
            logger.info(line)

        # ── Write to JSONL ────────────────────────────────────────────────────
        lightweight = {
            k: {kk: vv for kk, vv in v.items() if kk not in _STRIP_KEYS}
            if isinstance(v, dict) else v
            for k, v in qr.items()
        }
        results_f.write(json.dumps(lightweight, ensure_ascii=False) + "\n")
        results_f.flush()

    results_f.close()
    logger.info(f"Per-question results → {results_path}")

    # ── 8. Aggregate ──────────────────────────────────────────────────────────
    logger.info("Aggregating results...")
    summary = aggregate_results(
        all_question_results, args.alpha_values, args.tau_values
    )

    # ── 9. Help / Hurt analysis ───────────────────────────────────────────────
    default_alpha = 0.5 if 0.5 in args.alpha_values else args.alpha_values[0]
    default_tau   = 0.85 if 0.85 in args.tau_values  else args.tau_values[len(args.tau_values) // 2]

    help_cases, hurt_cases = build_help_hurt_analysis(
        records, all_question_results, args.alpha_values, args.tau_values,
        default_alpha=default_alpha, default_tau=default_tau,
    )

    # ── 10. Save outputs ──────────────────────────────────────────────────────

    # Summary JSON
    summary_path = os.path.join(args.output_dir, f"summary_paired_{tag}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary JSON → {summary_path}")

    # Analysis JSON (meta per question)
    analysis = [
        {"id": qr.get("__id__"), **qr.get("__meta__", {})}
        for qr in all_question_results
    ]
    analysis_path = os.path.join(args.output_dir, f"analysis_paired_{tag}.json")
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    logger.info(f"Analysis JSON → {analysis_path}")

    # LaTeX table
    latex = build_latex_table(summary, args.alpha_values, args.tau_values,
                               args.dataset, args.N)
    latex_path = os.path.join(args.output_dir, f"latex_table_{tag}.txt")
    with open(latex_path, "w") as f:
        f.write(latex)
    logger.info(f"LaTeX table → {latex_path}")

    # Help / hurt case JSONL
    help_path = os.path.join(args.output_dir, f"help_cases_{tag}.jsonl")
    hurt_path = os.path.join(args.output_dir, f"hurt_cases_{tag}.jsonl")
    with open(help_path, "w") as f:
        for c in help_cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(hurt_path, "w") as f:
        for c in hurt_cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    logger.info(f"Help cases ({len(help_cases)}) → {help_path}")
    logger.info(f"Hurt cases ({len(hurt_cases)}) → {hurt_path}")

    # ── 11. Console summary table ─────────────────────────────────────────────
    logger.info("\n" + "=" * 72)
    logger.info(f" PAIRED ABLATION SUMMARY  |  {args.dataset}  |  N={args.N}")
    logger.info("=" * 72)
    for method in ["bon", "majority", "oracle"]:
        if method in summary:
            s = summary[method]
            logger.info(f"  {method:<25s}  acc={s['accuracy']:6.2f}%  "
                        f"({s['n_correct']}/{s['n_total']})")
    # CIMR results
    for variant in (args.cimr_variants if args.enable_cimr else []):
        cimr_key = f"cimr_{variant}"
        if cimr_key in summary:
            s = summary[cimr_key]
            logger.info(f"  CIMR ({variant}){' '*(17 - len(variant))}  "
                        f"acc={s['accuracy']:6.2f}%  "
                        f"({s['n_correct']}/{s['n_total']})")
    logger.info("  " + "─" * 60)
    logger.info(f"  {'Method / (α, τ)':<25s}  {'Accuracy':>10}  Δ vs BoN")
    logger.info("  " + "─" * 60)
    bon_acc = summary.get("bon", {}).get("accuracy", 0.0)
    best_dn = None
    best_acc_val = -1.0
    for a in args.alpha_values:
        for t in args.tau_values:
            k = f"dn_bon_tau{t:.2f}_alpha{a:.2f}"
            if k in summary:
                s = summary[k]
                delta = s["accuracy"] - bon_acc
                marker = " ◀ BEST" if s["accuracy"] > best_acc_val else ""
                if s["accuracy"] > best_acc_val:
                    best_acc_val = s["accuracy"]
                    best_dn = k
                logger.info(f"  DN-BoN α={a:.2f} τ={t:.2f}   "
                             f"acc={s['accuracy']:6.2f}%  Δ={delta:+.2f}pp{marker}")

    if "__redundancy__" in summary:
        r = summary["__redundancy__"]
        logger.info("  " + "─" * 60)
        logger.info(f"  Mean fmax:          {r['mean_fmax']:.3f}")
        logger.info(f"  Mean H (entropy):   {r['mean_H']:.3f} bits")
        logger.info(f"  Multiplicity bias fired: {r['multiplicity_bias_fired_count']} "
                    f"({r['multiplicity_bias_fired_pct']:.1f}%)")
        logger.info(f"  DN-BoN rescued:     {r['dnbon_rescued_count']} questions")
        logger.info(f"  DN-BoN hurt:        {r['dnbon_hurt_count']} questions")
        logger.info(f"  Net help:           {r['dnbon_net_help']:+d} questions")
        if r.get("cimr_rescued_count", 0) > 0 or r.get("cimr_hurt_count", 0) > 0:
            logger.info(f"  CIMR rescued:       {r['cimr_rescued_count']} questions")
            logger.info(f"  CIMR hurt:          {r['cimr_hurt_count']} questions")
            logger.info(f"  CIMR net help:      {r['cimr_net_help']:+d} questions")
    logger.info("=" * 72)
    if best_dn:
        logger.info(f"  Best config: {best_dn}  ({best_acc_val:.2f}%)")
    logger.info("=" * 72 + "\n")

    # ── 12. Plots ─────────────────────────────────────────────────────────────
    if not args.no_plots:
        logger.info("Generating figures...")
        make_plots(
            summary=summary,
            all_question_results=all_question_results,
            alpha_values=args.alpha_values,
            tau_values=args.tau_values,
            help_cases=help_cases,
            hurt_cases=hurt_cases,
            dataset=args.dataset,
            N=args.N,
            figures_dir=figures_dir,
            tag=tag,
        )

    logger.info("Done.  All outputs in: " + args.output_dir)


if __name__ == "__main__":
    main()