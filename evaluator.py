"""
evaluator.py
------------
Evaluation loop that drives batched inference.

The primary entry point is evaluate_batch() which:
  1. Collects all examples into a list.
  2. Calls method.run_batch(examples) so the model processes questions
     with the maximum possible batch utilisation.
  3. Computes accuracy and writes per-question JSONL + summary JSON.

A legacy evaluate() (question-by-question) is kept for compatibility.

Progress bars
-------------
tqdm is used at two levels:
  - Outer bar  : one tick per question as results are aggregated.
  - Inner bars : managed inside inference_methods.run_batch() for
                 per-question generation progress (BoN / DN-BoN).
  - Greedy     : a single tqdm bar in run_batch() wrapping the
                 post-processing loop after all traces are generated.
"""

import json
import logging
import os
import time
from typing import List, Dict, Any, Optional

import numpy as np
from tqdm import tqdm

from inference_methods import InferenceResult

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Primary batched evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_batch(
    method,
    examples: List[Dict[str, Any]],
    max_new_tokens: int = 1024,
    save_path: Optional[str] = None,
    resume: bool = True,
    log_every: int = 10,
) -> Dict[str, Any]:
    """
    Run inference over `examples` using method.run_batch().

    For Greedy: all questions are batched into chunks of model.batch_size
    in a single pass through generate_greedy_batch().

    For BoN / DN-BoN: each question gets one generate() call with batch
    size = N (all N samples produced in parallel).  Questions are processed
    sequentially but each question's compute is fully batched.

    Parameters
    ----------
    method        : GreedyInference / BestOfN / DNBestOfN
    examples      : list of {'question', 'answer', '_data_name', ...}
    max_new_tokens: generation length cap
    save_path     : JSONL file to append results to
    resume        : skip already-done questions when save_path exists
    log_every     : progress log interval (kept for structured logs alongside tqdm)

    Returns
    -------
    dict: accuracy, n_correct, n_total, avg_elapsed_sec, total_wall_sec, results
    """
    method_name = method.__class__.__name__

    # ── resume: find already-completed questions ───────────────────────────
    done_keys: set = set()
    if resume and save_path and os.path.isfile(save_path):
        with open(save_path, "r") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_keys.add(r.get("question", "")[:200])
                except Exception:
                    pass
        logger.info(f"Resume: {len(done_keys)} questions already done.")

    pending = [ex for ex in examples
               if ex["question"][:200] not in done_keys]

    logger.info(
        f"Evaluating {len(pending)}/{len(examples)} questions  "
        f"(method={method_name})"
    )

    if not pending:
        logger.info("All questions already completed.")
        return _empty_summary()

    # ── generation phase (tqdm lives inside run_batch) ─────────────────────
    t_total_start = time.time()
    batch_results: List[InferenceResult] = method.run_batch(
        pending, max_new_tokens=max_new_tokens
    )
    total_wall = time.time() - t_total_start

    # ── aggregation + save phase (tqdm bar here) ───────────────────────────
    f_out = None
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        f_out = open(save_path, "a", encoding="utf-8")

    rows: List[Dict] = []
    n_correct = 0
    n_total   = 0

    agg_bar = tqdm(
        total=len(pending),
        desc=f"Aggregating [{method_name}]",
        unit="q",
        dynamic_ncols=True,
        colour="cyan",
    )

    try:
        for i, (ex, res) in enumerate(zip(pending, batch_results)):
            row = res.to_dict()
            row["question"] = ex["question"][:300]
            rows.append(row)

            if res.is_correct is not None:
                n_correct += int(res.is_correct)
                n_total   += 1

            if f_out:
                f_out.write(json.dumps(row) + "\n")
                f_out.flush()

            # Update tqdm postfix with running accuracy
            if n_total > 0:
                agg_bar.set_postfix(
                    acc=f"{n_correct/n_total:.3f}",
                    correct=f"{n_correct}/{n_total}",
                    refresh=False,
                )
            agg_bar.update(1)

            # Structured log at interval (useful when redirecting to file)
            if (i + 1) % log_every == 0:
                acc = n_correct / max(n_total, 1)
                logger.info(
                    f"  [{i+1}/{len(pending)}] "
                    f"acc={acc:.3f} ({n_correct}/{n_total})"
                )
    finally:
        agg_bar.close()
        if f_out:
            f_out.close()

    accuracy    = n_correct / max(n_total, 1)
    avg_elapsed = total_wall / max(len(rows), 1)

    logger.info(
        f"Done. accuracy={accuracy:.4f} ({n_correct}/{n_total})  "
        f"total_wall={total_wall:.1f}s  avg={avg_elapsed:.2f}s/q"
    )

    return {
        "accuracy":        round(accuracy, 4),
        "n_correct":       n_correct,
        "n_total":         n_total,
        "avg_elapsed_sec": round(avg_elapsed, 3),
        "total_wall_sec":  round(total_wall, 2),
        "results":         rows,
    }


def _empty_summary() -> Dict[str, Any]:
    return {"accuracy": 0.0, "n_correct": 0, "n_total": 0,
            "avg_elapsed_sec": 0.0, "total_wall_sec": 0.0, "results": []}


# ──────────────────────────────────────────────────────────────────────────────
# Legacy single-question loop  (kept for backward compat / debugging)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    method,
    examples: List[Dict[str, Any]],
    max_new_tokens: int = 1024,
    save_path: Optional[str] = None,
    resume: bool = True,
    log_every: int = 10,
) -> Dict[str, Any]:
    """
    Legacy: call method.run() once per question.
    Prefer evaluate_batch() for real workloads.
    """
    method_name = method.__class__.__name__

    done_keys: set = set()
    if resume and save_path and os.path.isfile(save_path):
        with open(save_path, "r") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_keys.add(r.get("question", "")[:200])
                except Exception:
                    pass

    pending = [ex for ex in examples
               if ex["question"][:200] not in done_keys]

    rows, n_correct, n_total, total_elapsed = [], 0, 0, 0.0
    f_out = None
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        f_out = open(save_path, "a", encoding="utf-8")

    pbar = tqdm(
        pending,
        desc=f"[{method_name}]",
        unit="q",
        dynamic_ncols=True,
        colour="green",
    )

    try:
        for i, ex in enumerate(pbar):
            q    = ex["question"]
            gold = ex.get("answer")
            try:
                res: InferenceResult = method.run(
                    question=q, gold_answer=gold,
                    max_new_tokens=max_new_tokens,
                )
            except Exception as e:
                logger.error(f"[{i}] failed: {e}")
                from inference_methods import _make_error_result
                res = _make_error_result(
                    method_name, q, gold, getattr(method, "N", 1)
                )

            row = res.to_dict()
            row["question"] = q[:300]
            rows.append(row)

            if res.is_correct is not None:
                n_correct += int(res.is_correct)
                n_total   += 1
            total_elapsed += res.elapsed_sec

            if f_out:
                f_out.write(json.dumps(row) + "\n")
                f_out.flush()

            if n_total > 0:
                pbar.set_postfix(
                    acc=f"{n_correct/n_total:.3f}",
                    correct=f"{n_correct}/{n_total}",
                    refresh=False,
                )

            if (i + 1) % log_every == 0:
                logger.info(
                    f"[{i+1}/{len(pending)}] "
                    f"acc={n_correct/max(n_total,1):.3f} ({n_correct}/{n_total})"
                )
    finally:
        pbar.close()
        if f_out:
            f_out.close()

    accuracy    = n_correct / max(n_total, 1)
    avg_elapsed = total_elapsed / max(len(rows), 1)
    logger.info(f"Final acc={accuracy:.4f} ({n_correct}/{n_total})")
    return {
        "accuracy":        round(accuracy, 4),
        "n_correct":       n_correct,
        "n_total":         n_total,
        "avg_elapsed_sec": round(avg_elapsed, 3),
        "results":         rows,
    }


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_summary(summary: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {path}")


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Comparison and redundancy
# ──────────────────────────────────────────────────────────────────────────────

def compare_methods(summaries: Dict[str, Dict[str, Any]]) -> str:
    lines = [
        f"\n{'Method':<20} {'Accuracy':>10} {'Correct':>10} "
        f"{'Total':>8} {'Avg time':>10} {'Wall time':>12}",
        "-" * 74,
    ]
    for name, s in summaries.items():
        wall = s.get("total_wall_sec", 0.0)
        lines.append(
            f"{name:<20} {s['accuracy']:>10.4f} {s['n_correct']:>10} "
            f"{s['n_total']:>8} {s['avg_elapsed_sec']:>9.2f}s"
            f" {wall:>11.1f}s"
        )
    return "\n".join(lines)


def redundancy_stats(results: List[Dict]) -> Dict[str, float]:
    n_cl, sel_sz, max_sz, top_sel = [], [], [], []
    for r in results:
        sizes = r.get("cluster_sizes", [])
        if not sizes:
            continue
        size_arr = np.array(sizes)
        unique_clusters = len(set(r.get("cluster_labels", range(len(sizes)))))
        n_cl.append(unique_clusters)
        sel_idx = r.get("selected_index", 0)
        sel_sz.append(sizes[sel_idx] if sel_idx < len(sizes) else 0)
        max_sz.append(int(np.max(size_arr)))
        top_sel.append(1 if sizes[sel_idx] == np.max(size_arr) else 0)
    if not n_cl:
        return {}
    return {
        "avg_num_clusters":               round(float(np.mean(n_cl)), 3),
        "avg_selected_cluster_size":      round(float(np.mean(sel_sz)), 3),
        "avg_max_cluster_size":           round(float(np.mean(max_sz)), 3),
        "frac_selected_from_top_cluster": round(float(np.mean(top_sel)), 3),
    }