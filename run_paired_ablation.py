"""
run_paired_ablation.py
======================
Paired, sample-reuse ablation harness for DN-BoN with the new
**rollout-based step-wise CIMR** pipeline.

Two-pass design
---------------
  PASS 1 — BASELINES from CACHE
    BoN, Majority, Oracle, DN-BoN(α, τ) all consume the cached samples
    written by ``generate_and_score()``.  No new LLM calls.

  PASS 2 — CIMR LIVE (with on-the-go save + resume)
    For each question, the orchestrator runs N independent rollouts.
    Each rollout chains d steps where each step does
        generate reasoning  →  self-critique  →  self-correction
    and the corrected output of step k is fed as input context to
    step k+1's reasoning generator.  Across the N rollouts, the
    orchestrator computes BOTH:
        * majority vote over the final answers
        * max PRM (Best-of-N over the corrected traces)
    and saves both predictions / accuracies.

    CIMR results are persisted after EVERY step into:
        cimr_full_results_<dataset>_N<N>_d<D>.json
    so any rerun automatically resumes the question (and rollout) at
    the first un-finished step.

CLI knobs (CIMR-specific)
-------------------------
  --enable_cimr
  --cimr_n_rollouts N             (default 8 — independent rollouts per q)
  --cimr_depth      D             (default 4 — refinement steps per rollout)
  --cimr_other_answer_aware
  --cimr_reasoning_max_new_tokens / temperature / top_p
  --cimr_critique_max_tokens / max_trace_chars
  --cimr_correction_max_tokens / max_trace_chars / max_critique_chars
  --cimr_continuation_max_prior_chars
"""

import argparse
import json
import logging
import math
import os
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
        description=(
            "Paired DN-BoN ablation — generate baseline samples once, "
            "evaluate cache-driven baselines + run live CIMR rollouts."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    g = p.add_argument_group("Model / Verifier")
    g.add_argument("--model", required=True)
    g.add_argument("--verifier_model",  default="Qwen/Qwen2.5-Math-PRM-7B")
    g.add_argument("--prm_aggregation", default="min",
                   choices=["min", "mean", "last"])
    g.add_argument("--embedding_model", default="all-MiniLM-L6-v2")
    g.add_argument("--batch_size",      type=int, default=8)
    g.add_argument("--dtype",           default="auto",
                   choices=["auto", "float16", "bfloat16", "float32"])
    g.add_argument("--device",          default="auto")
    g.add_argument("--gpus",            default=None)
    g.add_argument("--load_in_4bit",    action="store_true")
    g.add_argument("--load_in_8bit",    action="store_true")
    g.add_argument("--trust_remote_code", action="store_true")
    g.add_argument("--hf_cache_dir",    default=None)

    # Generation
    g = p.add_argument_group("Generation (baselines)")
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
                   default=[0.0, 0.25, 0.5, 0.75, 1.0])
    g.add_argument("--tau_values",   nargs="+", type=float,
                   default=[0.70, 0.80, 0.85, 0.90, 0.95])
    g.add_argument("--clustering_method", default="agglomerative",
                   choices=["agglomerative", "greedy"])

    # CIMR — rollout-based step-wise pipeline
    g = p.add_argument_group("CIMR — rollouts × steps (self-critique + correction)")
    g.add_argument("--enable_cimr", action="store_true",
                   help="Enable CIMR rollout-based step-wise meta-reasoning")
    g.add_argument("--cimr_n_rollouts", type=int, default=8,
                   help="N — independent rollouts per question.")
    g.add_argument("--cimr_depth", type=int, default=4,
                   help=(
                       "d — number of sequential refinement steps per "
                       "rollout.  Each step does (generate, critique, "
                       "correct, score), and the corrected output of "
                       "step k is fed as input context to step k+1."
                   ))
    g.add_argument("--cimr_other_answer_aware", action="store_true",
                   help=(
                       "Surface sibling rollouts' current corrected answers "
                       "to the critique + correction agents."
                   ))

    # CIMR — generation knobs
    g.add_argument("--cimr_reasoning_max_new_tokens", type=int, default=2048)
    g.add_argument("--cimr_reasoning_temperature",    type=float, default=0.7)
    g.add_argument("--cimr_reasoning_top_p",          type=float, default=0.95)

    # CIMR — critique knobs
    g.add_argument("--cimr_critique_max_tokens",      type=int, default=384)
    g.add_argument("--cimr_critique_max_trace_chars", type=int, default=4000)

    # CIMR — correction knobs
    g.add_argument("--cimr_correction_max_tokens",    type=int, default=1024)
    g.add_argument("--cimr_correction_max_trace_chars",    type=int, default=4000)
    g.add_argument("--cimr_correction_max_critique_chars", type=int, default=1500)

    # CIMR — continuation prompt knob
    g.add_argument("--cimr_continuation_max_prior_chars", type=int, default=4000,
                   help=(
                       "Head/tail truncation budget (chars) for the "
                       "prior-step CORRECTED REASONING shown to step k≥2."
                   ))

    # Cache / resume
    g = p.add_argument_group("Cache / resume")
    g.add_argument("--load_cache", default=None,
                   help="Path to existing sample-cache JSONL; skips generation")
    g.add_argument("--save_traces", action="store_true", default=True)
    g.add_argument("--no_save_traces", dest="save_traces", action="store_false")
    g.add_argument("--skip_baselines", action="store_true",
                   help="Skip cache-driven baseline pass and run CIMR-only")

    # Output
    g = p.add_argument_group("Output")
    g.add_argument("--output_dir", default="./results_paired")
    g.add_argument("--log_every",  type=int, default=10)
    g.add_argument("--no_plots",   action="store_true")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Diversity metrics + majority helper
# ─────────────────────────────────────────────────────────────────────────────

def diversity_metrics(labels: np.ndarray) -> Dict[str, float]:
    N = len(labels)
    if N == 0:
        return dict(H=0.0, He=0.0, fmax=1.0, r=0.0, M=0, N=0)
    unique, counts = np.unique(labels, return_counts=True)
    M = int(len(unique))
    pm = counts / N
    H = float(-np.sum(pm * np.log2(pm + 1e-12)))
    He = float(H / math.log2(N)) if N > 1 else 0.0
    fmax = float(counts.max() / N)
    r = float(M / N)
    return dict(H=H, He=He, fmax=fmax, r=r, M=M, N=N)


def _majority_select(
    answers: List[str], ev, data_name: str,
) -> Tuple[int, str, List[int]]:
    groups: List[List[int]] = []
    group_rep: List[str] = []
    for idx, ans in enumerate(answers):
        placed = False
        for g, rep in enumerate(group_rep):
            try:
                equiv = ev.check_answers_equiv(ans, rep) if ev is not None else None
            except Exception:
                equiv = None
            if equiv is None:
                equiv = (str(ans).strip() == str(rep).strip())
            if equiv:
                groups[g].append(idx)
                placed = True
                break
        if not placed:
            groups.append([idx])
            group_rep.append(ans)
    if not groups:
        return 0, "", [1]
    majority_g = max(range(len(groups)), key=lambda g: len(groups[g]))
    best_idx = groups[majority_g][0]
    majority_ans = group_rep[majority_g]
    vote_counts = [0] * len(answers)
    for g, members in enumerate(groups):
        for idx in members:
            vote_counts[idx] = len(members)
    return best_idx, majority_ans, vote_counts


# ─────────────────────────────────────────────────────────────────────────────
# Cache-driven baseline methods (BoN / Majority / Oracle / DN-BoN)
# ─────────────────────────────────────────────────────────────────────────────

def apply_baseline_methods(
    question:    str,
    gold:        Optional[str],
    traces:      List[str],
    prm_scores:  np.ndarray,
    all_correct: List[bool],
    alpha_values: List[float],
    tau_values:   List[float],
    cluster_manager,
    ev,
    data_name:   str,
) -> Dict[str, Any]:
    N = len(traces)
    answers: List[str] = []
    for t in traces:
        try:
            from src.inference_methods import _extract_pred  # type: ignore
            ans = _extract_pred(t, ev, data_name)
        except Exception:
            ans = ""
        answers.append(ans)

    results: Dict[str, Any] = {}

    # BoN
    bon_idx = int(np.argmax(prm_scores))
    bon_correct = bool(all_correct[bon_idx]) if bon_idx < len(all_correct) else False
    results["bon"] = dict(
        method="bon", selected_idx=bon_idx, predicted_answer=answers[bon_idx],
        is_correct=bon_correct,
        raw_score=float(prm_scores[bon_idx]),
        norm_score=float(prm_scores[bon_idx]),
        cluster_size=1, alpha=0.0, tau=float("nan"),
    )

    # Oracle
    correct_idxs = [i for i, c in enumerate(all_correct) if c]
    if correct_idxs:
        oracle_idx = int(max(correct_idxs, key=lambda i: prm_scores[i]))
        oracle_correct = True
    else:
        oracle_idx = bon_idx
        oracle_correct = False
    results["oracle"] = dict(
        method="oracle", selected_idx=oracle_idx,
        predicted_answer=answers[oracle_idx], is_correct=oracle_correct,
        raw_score=float(prm_scores[oracle_idx]),
        norm_score=float(prm_scores[oracle_idx]),
        cluster_size=1, alpha=float("nan"), tau=float("nan"),
    )

    # Majority
    maj_idx, maj_ans, vote_counts = _majority_select(answers, ev, data_name)
    maj_correct = bool(all_correct[maj_idx]) if maj_idx < len(all_correct) else False
    results["majority"] = dict(
        method="majority", selected_idx=maj_idx, predicted_answer=maj_ans,
        is_correct=maj_correct,
        raw_score=float(vote_counts[maj_idx]),
        norm_score=float(vote_counts[maj_idx]),
        cluster_size=int(vote_counts[maj_idx]),
        alpha=float("nan"), tau=float("nan"),
    )

    # DN-BoN sweep
    for tau in tau_values:
        try:
            cluster_manager.similarity_threshold = tau
            cluster_manager.distance_threshold = 1.0 - tau
            labels, cluster_sizes = cluster_manager.cluster(traces)
        except Exception as e:
            logger.warning(f"Clustering failed (τ={tau}): {e}")
            labels = np.arange(N, dtype=int)
            cluster_sizes = np.ones(N, dtype=int)
        div = diversity_metrics(labels)

        for alpha in alpha_values:
            if alpha == 0.0:
                norm_scores = prm_scores.copy()
            else:
                safe_sizes = np.maximum(cluster_sizes.astype(float), 1.0)
                norm_scores = prm_scores / (safe_sizes ** alpha)
            best_idx = int(np.argmax(norm_scores))
            is_correct = (
                bool(all_correct[best_idx])
                if best_idx < len(all_correct) else False
            )
            key = f"dn_bon_tau{tau:.2f}_alpha{alpha:.2f}"
            results[key] = dict(
                method=key, selected_idx=best_idx,
                predicted_answer=answers[best_idx], is_correct=is_correct,
                raw_score=float(prm_scores[best_idx]),
                norm_score=float(norm_scores[best_idx]),
                cluster_size=int(cluster_sizes[best_idx]),
                alpha=alpha, tau=tau,
                fmax=div["fmax"], H=div["H"], He=div["He"],
                M=div["M"], r=div["r"], n_clusters=div["M"],
                cluster_labels=labels.tolist(),
            )

    # Multiplicity-bias diagnostic
    default_tau = tau_values[len(tau_values) // 2]
    default_key = f"dn_bon_tau{default_tau:.2f}_alpha0.50"
    if default_key in results:
        labels = np.array(results[default_key]["cluster_labels"])
        bon_cluster_id = int(labels[bon_idx]) if bon_idx < len(labels) else -1
        dominant_cid = int(np.bincount(labels).argmax()) if labels.size > 0 else -1
        bon_in_dominant = (bon_cluster_id == dominant_cid)
        correct_outside_dominant = any(
            (i < len(all_correct) and all_correct[i]
             and i < len(labels) and int(labels[i]) != dominant_cid)
            for i in range(N)
        )
        dnbon_rescued = (
            not bon_correct
            and results.get(default_key, {}).get("is_correct", False)
        )
        dnbon_hurt = (
            bon_correct
            and not results.get(default_key, {}).get("is_correct", True)
        )
    else:
        bon_in_dominant = False
        correct_outside_dominant = False
        dnbon_rescued = False
        dnbon_hurt = False

    results["__meta__"] = dict(
        N=N, any_correct=len(correct_idxs) > 0,
        n_correct_traces=len(correct_idxs),
        answers=answers, all_correct=all_correct,
        bon_idx=bon_idx, bon_correct=bon_correct,
        oracle_correct=oracle_correct, majority_correct=maj_correct,
        bon_in_dominant_cluster=bon_in_dominant,
        correct_exists_outside_dominant=correct_outside_dominant,
        multiplicity_bias_fired=(bon_in_dominant
                                 and correct_outside_dominant
                                 and not bon_correct),
        dnbon_rescued=dnbon_rescued, dnbon_hurt=dnbon_hurt,
        default_tau=default_tau,
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Generation + scoring (one-time)
# ─────────────────────────────────────────────────────────────────────────────

def generate_and_score(
    examples: List[Dict[str, Any]],
    model, verifier, ev,
    data_name: str, N: int,
    max_new_tokens: int, temperature: float, top_p: float,
    cache_path: Optional[str],
    log_every: int = 10,
) -> List[Dict[str, Any]]:
    from src.verifier import score_candidates  # type: ignore
    records = []
    t_total = time.time()
    cache_f = open(cache_path, "w", encoding="utf-8") if cache_path else None
    for i, ex in enumerate(examples):
        q    = ex["question"]
        gold = ex.get("answer", None)
        qid  = ex.get("id", str(i))
        try:
            t0 = time.time()
            traces = model.generate_samples(
                q, N=N, temperature=temperature, top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
            gen_elapsed = time.time() - t0
            prm_scores = (
                score_candidates(traces, verifier, gold, question=q).tolist()
                if traces else []
            )
            from src.inference_methods import _extract_pred, _check_correct  # type: ignore
            answers = [_extract_pred(t, ev, data_name) for t in traces]
            all_correct = (
                [bool(_check_correct(a, gold, ev, data_name)) for a in answers]
                if gold is not None else []
            )
            rec = dict(
                id=qid, question=q, gold_answer=gold,
                traces=traces, prm_scores=prm_scores,
                answers=answers, all_correct=all_correct,
                gen_elapsed_sec=gen_elapsed,
            )
        except Exception as e:
            logger.error(f"Generation failed q[{i}]: {e}")
            rec = dict(
                id=qid, question=q, gold_answer=gold,
                traces=[], prm_scores=[], answers=[], all_correct=[],
                gen_elapsed_sec=0.0, error=str(e),
            )
        records.append(rec)
        if cache_f is not None:
            cache_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            cache_f.flush()
        if (i + 1) % log_every == 0 or (i + 1) == len(examples):
            elapsed = time.time() - t_total
            rate = (i + 1) / max(elapsed, 1e-6)
            logger.info(
                f"  generated {i + 1}/{len(examples)}  "
                f"elapsed={elapsed:.1f}s  rate={rate:.2f} q/s"
            )
    if cache_f is not None:
        cache_f.close()
    return records


def load_cache(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation across questions
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_results(all_qr: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    method_keys = set()
    for qr in all_qr:
        for k in qr.keys():
            if k.startswith("__"):
                continue
            method_keys.add(k)
    for k in sorted(method_keys):
        n_total = 0; n_correct = 0
        for qr in all_qr:
            entry = qr.get(k)
            if not isinstance(entry, dict):
                continue
            ic = entry.get("is_correct")
            if ic is None:
                continue
            n_total += 1
            n_correct += int(bool(ic))
        if n_total > 0:
            summary[k] = dict(
                accuracy=100.0 * n_correct / n_total,
                n_correct=n_correct, n_total=n_total,
            )
    # CIMR rescue/hurt vs BoN
    cimr_rescued = 0; cimr_hurt = 0
    multiplicity_fired = 0
    dnbon_rescued = 0; dnbon_hurt = 0
    for qr in all_qr:
        meta = qr.get("__meta__", {}) or {}
        if meta.get("multiplicity_bias_fired"):
            multiplicity_fired += 1
        if meta.get("dnbon_rescued"):
            dnbon_rescued += 1
        if meta.get("dnbon_hurt"):
            dnbon_hurt += 1
        bon_correct = (qr.get("bon", {}) or {}).get("is_correct")
        # Use majority as the primary CIMR rule for rescue analysis
        cimr_maj = (qr.get("cimr_majority", {}) or {}).get("is_correct")
        if bon_correct is not None and cimr_maj is not None:
            if (not bon_correct) and cimr_maj:
                cimr_rescued += 1
            if bon_correct and (not cimr_maj):
                cimr_hurt += 1
    n_q = max(len(all_qr), 1)
    summary["__redundancy__"] = dict(
        multiplicity_bias_fired_count=multiplicity_fired,
        multiplicity_bias_fired_pct=100.0 * multiplicity_fired / n_q,
        dnbon_rescued_count=dnbon_rescued, dnbon_hurt_count=dnbon_hurt,
        dnbon_net_help=dnbon_rescued - dnbon_hurt,
        cimr_rescued_count=cimr_rescued, cimr_hurt_count=cimr_hurt,
        cimr_net_help=cimr_rescued - cimr_hurt,
    )
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    figures_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    tag = f"{args.dataset}_N{args.N}"

    if args.enable_cimr:
        from src.cimr_inference import (  # type: ignore
            SELECTION_RULE_NAMES, CIMRConfig, CIMRInferencer,
            run_cimr_for_question,
        )
        from src.cimr_step import CIMRStepConfig  # type: ignore
    else:
        SELECTION_RULE_NAMES = []

    # ── 0. GPU setup ───────────────────────────────────────────────────────
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        logger.info(f"CUDA_VISIBLE_DEVICES={args.gpus}")
    import torch  # noqa: E402
    if torch.cuda.is_available():
        logger.info(f"GPUs available: {torch.cuda.device_count()}")

    # ── 1. Load dataset ────────────────────────────────────────────────────
    logger.info(f"Loading dataset: {args.dataset}")
    from src.dataset_loader import load_dataset_split  # type: ignore
    examples = load_dataset_split(
        data_name=args.dataset, split=args.split,
        max_samples=args.max_samples, custom_path=args.custom_path,
        cache_dir=args.hf_cache_dir,
    )
    data_name = examples[0]["_data_name"] if examples else args.dataset
    logger.info(f"Loaded {len(examples)} examples.")

    # ── 2. Load model ──────────────────────────────────────────────────────
    logger.info(f"Loading generator: {args.model}")
    from src.model_wrapper import HFModelWrapper  # type: ignore
    model = HFModelWrapper(
        model_name_or_path=args.model,
        batch_size=args.batch_size,
        device=args.device, dtype=args.dtype,
        load_in_4bit=args.load_in_4bit, load_in_8bit=args.load_in_8bit,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_p=args.top_p,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.hf_cache_dir,
    )

    # ── 3. Load verifier ───────────────────────────────────────────────────
    logger.info(f"Loading verifier: {args.verifier_model}")
    from src.verifier import build_verifier  # type: ignore
    verifier = build_verifier(
        verifier_type="prm", data_name=data_name,
        prm_model=args.verifier_model,
        prm_aggregation=args.prm_aggregation,
        device=args.device, cache_dir=args.hf_cache_dir,
        batch_size=args.batch_size,
    )

    # ── 4. Build evaluator ─────────────────────────────────────────────────
    from src.inference_methods import _build_evaluator  # type: ignore
    ev = _build_evaluator(data_name)

    # ── 5. ClusterManager (shared, tau updated per call) ───────────────────
    logger.info(f"Loading ClusterManager: {args.embedding_model}")
    from src.clustering import ClusterManager  # type: ignore
    cluster_manager = ClusterManager(
        embedding_model=args.embedding_model,
        similarity_threshold=args.tau_values[0],
        clustering_method=args.clustering_method,
        use_answer_only=False, batch_size=args.batch_size,
    )

    # ── 6. CIMR-only mode (skip cache-driven baselines) ───────────────────
    # Build minimal per-question records directly from the loaded examples.
    # This removes the one-time baseline generation / PRM scoring pass
    # and runs CIMR live (N rollouts × d refinement steps) for each
    # question.
    records = []
    for i, ex in enumerate(examples):
        rec: Dict[str, Any] = {
            "id": ex.get("id", str(i)),
            "question": ex.get("question"),
            "gold_answer": ex.get("answer", None),
        }
        records.append(rec)

    # Pre-initialize the per-question results list so CIMR can inject
    # its aggregated predictions (majority / max_prm) into each entry.
    all_question_results: List[Dict[str, Any]] = []
    for i, rec in enumerate(records):
        all_question_results.append({
            "__id__": rec.get("id", str(i)),
            "__meta__": {"question": rec.get("question")},
        })

    # Paths for incremental saving of accuracies and per-question results
    summary_path = os.path.join(args.output_dir, f"summary_paired_{tag}.json")
    results_path = os.path.join(args.output_dir, f"results_paired_{tag}.jsonl")

    # ── 8. PASS 2 — CIMR (live, on-the-go save + resume) ───────────────────
    if args.enable_cimr:
        N_roll = int(args.cimr_n_rollouts)
        D     = int(args.cimr_depth)
        logger.info(
            f"Pass 2: CIMR rollouts × steps   "
            f"N_rollouts={N_roll}  depth={D}  "
            f"other_aware={args.cimr_other_answer_aware}"
        )

        step_cfg = CIMRStepConfig(
            reasoning_max_new_tokens=args.cimr_reasoning_max_new_tokens,
            reasoning_temperature=args.cimr_reasoning_temperature,
            reasoning_top_p=args.cimr_reasoning_top_p,
            reasoning_do_sample=True,
            critique_max_tokens=args.cimr_critique_max_tokens,
            critique_max_trace_chars=args.cimr_critique_max_trace_chars,
            correction_max_tokens=args.cimr_correction_max_tokens,
            correction_max_trace_chars=args.cimr_correction_max_trace_chars,
            correction_max_critique_chars=args.cimr_correction_max_critique_chars,
            continuation_max_prior_chars=args.cimr_continuation_max_prior_chars,
            other_answer_aware=args.cimr_other_answer_aware,
        )
        cimr_cfg = CIMRConfig(
            n_rollouts=N_roll, depth=D, step_config=step_cfg,
            rollout_temperature=args.cimr_reasoning_temperature,
            rollout_top_p=args.cimr_reasoning_top_p,
        )
        inferencer = CIMRInferencer(
            model=model, verifier=verifier, ev=ev,
            data_name=data_name, config=cimr_cfg,
            output_dir=args.output_dir, dataset=args.dataset,
        )

        n_resumed = 0; n_run_now = 0
        for i, rec in enumerate(records):
            qid = str(rec.get("id", i))
            qrec_pre = inferencer.store.get(qid)
            was_complete = (
                qrec_pre is not None and bool(qrec_pre.get("completed", False))
            )

            qrec = run_cimr_for_question(
                inferencer=inferencer,
                question=rec["question"],
                gold_answer=rec.get("gold_answer"),
                qid=qid,
            )

            if was_complete:
                n_resumed += 1
            else:
                n_run_now += 1

            agg = qrec.get("aggregation", {}) or {}
            # Inject majority + max_prm into the per-question results so
            # they show up in the summary alongside the cache baselines.
            all_question_results[i]["cimr_majority"] = dict(
                method="cimr_majority",
                selection_rule="majority",
                predicted_answer=agg.get("majority_pred", ""),
                is_correct=agg.get("majority_is_correct"),
                rollout_idx=agg.get("majority_rollout_idx", -1),
                vote_count=agg.get("majority_vote_count", 0),
                n_rollouts=agg.get("n_rollouts", N_roll),
                depth=D,
            )
            all_question_results[i]["cimr_max_prm"] = dict(
                method="cimr_max_prm",
                selection_rule="max_prm",
                predicted_answer=agg.get("max_prm_pred", ""),
                is_correct=agg.get("max_prm_is_correct"),
                rollout_idx=agg.get("max_prm_rollout_idx", -1),
                prm_score=agg.get("max_prm_score", 0.0),
                n_rollouts=agg.get("n_rollouts", N_roll),
                depth=D,
            )

            if (n_run_now + n_resumed) % max(args.log_every, 1) == 0 \
               or (i + 1) == len(records):
                logger.info(
                    f"  CIMR  q={i+1}/{len(records)}  "
                    f"freshly_done={n_run_now}  resumed={n_resumed}  "
                    f"id={qid}  "
                    f"maj_pred={agg.get('majority_pred', '')!r}  "
                    f"maj_correct={agg.get('majority_is_correct')}  "
                    f"max_prm_pred={agg.get('max_prm_pred', '')!r}  "
                    f"max_prm_correct={agg.get('max_prm_is_correct')}"
                )
                # Incremental save: recompute summary and persist results
                try:
                    cur_summary = aggregate_results(all_question_results)
                    tmp = summary_path + ".tmp"
                    with open(tmp, "w") as sf:
                        json.dump(cur_summary, sf, indent=2)
                    os.replace(tmp, summary_path)
                    # Persist per-question JSONL (overwrite atomically)
                    tmp2 = results_path + ".tmp"
                    with open(tmp2, "w", encoding="utf-8") as rf:
                        for qr in all_question_results:
                            rf.write(json.dumps(qr, ensure_ascii=False, default=str) + "\n")
                    os.replace(tmp2, results_path)
                except Exception as e:
                    logger.warning(f"Failed to write incremental summary/results: {e}")

                # Log concise accuracy summary
                try:
                    for m, v in sorted(cur_summary.items()):
                        if m.startswith("__"):
                            continue
                        logger.info(
                            f"    {m:<20s} acc={v['accuracy']:6.2f}%  "
                            f"({v['n_correct']}/{v['n_total']})"
                        )
                except Exception:
                    pass
                

        logger.info(
            f"  CIMR pass complete.  total={len(records)}  "
            f"freshly_run={n_run_now}  resumed={n_resumed}  "
            f"store={inferencer.store_path}"
        )

    # ── 9. Aggregate ───────────────────────────────────────────────────────
    logger.info("Aggregating results...")
    summary = aggregate_results(all_question_results)

    # ── 10. Save outputs ───────────────────────────────────────────────────
    summary_path = os.path.join(args.output_dir, f"summary_paired_{tag}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary JSON → {summary_path}")

    analysis = [
        {"id": qr.get("__id__"), **qr.get("__meta__", {})}
        for qr in all_question_results
    ]
    with open(os.path.join(args.output_dir, f"analysis_paired_{tag}.json"), "w") as f:
        json.dump(analysis, f, indent=2)

    results_path = os.path.join(args.output_dir, f"results_paired_{tag}.jsonl")
    with open(results_path, "w", encoding="utf-8") as f:
        for qr in all_question_results:
            f.write(json.dumps(qr, ensure_ascii=False, default=str) + "\n")
    logger.info(f"Per-question results → {results_path}")

    # ── 11. Console summary ────────────────────────────────────────────────
    logger.info("\n" + "=" * 72)
    logger.info(
        f" PAIRED ABLATION SUMMARY  |  {args.dataset}  |  N_baseline={args.N}"
    )
    if args.enable_cimr:
        logger.info(
            f"   CIMR: N_rollouts={args.cimr_n_rollouts}  depth={args.cimr_depth}"
        )
    logger.info("=" * 72)

    for method in ["bon", "majority", "oracle"]:
        if method in summary:
            s = summary[method]
            logger.info(
                f"  {method:<25s}  acc={s['accuracy']:6.2f}%  "
                f"({s['n_correct']}/{s['n_total']})"
            )

    if args.enable_cimr:
        logger.info("  " + "·" * 60)
        logger.info(
            f"  CIMR rollouts × steps  "
            f"(N={args.cimr_n_rollouts}, d={args.cimr_depth}):"
        )
        for rule in ["majority", "max_prm"]:
            key = f"cimr_{rule}"
            if key in summary:
                s = summary[key]
                logger.info(
                    f"  CIMR  {rule:<10s}  "
                    f"acc={s['accuracy']:6.2f}%  "
                    f"({s['n_correct']}/{s['n_total']})"
                )

    logger.info("  " + "─" * 60)
    bon_acc = summary.get("bon", {}).get("accuracy", 0.0)
    best_dn, best_acc_val = None, -1.0
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
                logger.info(
                    f"  DN-BoN α={a:.2f} τ={t:.2f}   "
                    f"acc={s['accuracy']:6.2f}%  Δ={delta:+.2f}pp{marker}"
                )

    if "__redundancy__" in summary:
        r = summary["__redundancy__"]
        logger.info("  " + "─" * 60)
        logger.info(
            f"  Multiplicity bias fired: {r['multiplicity_bias_fired_count']} "
            f"({r['multiplicity_bias_fired_pct']:.1f}%)"
        )
        logger.info(f"  DN-BoN net help:  {r['dnbon_net_help']:+d} questions")
        if r.get("cimr_rescued_count", 0) or r.get("cimr_hurt_count", 0):
            logger.info(
                f"  CIMR (majority) rescued: {r['cimr_rescued_count']}  "
                f"hurt: {r['cimr_hurt_count']}  "
                f"net help: {r['cimr_net_help']:+d}"
            )
    logger.info("=" * 72)
    if best_dn:
        logger.info(f"  Best DN-BoN: {best_dn}  ({best_acc_val:.2f}%)")
    logger.info("=" * 72 + "\n")
    logger.info("Done.  All outputs in: " + args.output_dir)


if __name__ == "__main__":
    main()
