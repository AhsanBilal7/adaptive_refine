"""
evaluator.py
------------
Evaluation loop with full on-the-go persistence and crash-safe resumption.

Persistence guarantees
----------------------
JSONL  : written and flushed after EVERY question.
         If the process crashes at question 400/500, questions 0-399 are safe.

Summary JSON : atomically rewritten after every question via write-to-tmp
         then os.replace().  The summary always reflects the exact set of
         questions saved in the JSONL, so they are always consistent.

Resumption
----------
On restart, evaluate_batch() reads the existing JSONL to find which
questions are already done and skips them.  The summary JSON is rebuilt
from the JSONL before continuing so statistics are always accurate.

Architecture change vs previous version
----------------------------------------
Previous: run_batch(all_pending) → returns list → then save
New     : for each question → generate → save JSONL → update summary

This means BoN/DN-BoN (which were already per-question loops inside
run_batch) now yield results one at a time via a generator, and the
evaluator saves each result immediately.  Greedy (which truly batches N
questions per generate() call) saves after each chunk.
"""

import json
import logging
import os
import time
from typing import List, Dict, Any, Optional, Iterator

import numpy as np
from tqdm import tqdm

from .inference_methods import InferenceResult

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Atomic summary writer
# ──────────────────────────────────────────────────────────────────────────────

def _write_summary_atomic(summary: Dict[str, Any], path: str) -> None:
    """
    Write summary JSON atomically via a temp file + os.replace().
    Guarantees the summary file is never partially written.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    os.replace(tmp, path)  # atomic on POSIX systems


# ──────────────────────────────────────────────────────────────────────────────
# JSONL reader  (used for resume)
# ──────────────────────────────────────────────────────────────────────────────

def _load_completed(jsonl_path: str):
    """
    Read an existing JSONL file and return:
      done_keys  : set of question[:200] strings already completed
      rows       : list of all completed row dicts
      n_correct  : count of correct answers in completed rows
      n_total    : count of rows with is_correct not None
      total_time : sum of elapsed_sec across completed rows
    """
    done_keys  = set()
    rows       = []
    n_correct  = 0
    n_total    = 0
    total_time = 0.0

    if not os.path.isfile(jsonl_path):
        return done_keys, rows, n_correct, n_total, total_time

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                key = r.get("question", "")[:200]
                if key in done_keys:
                    continue  # skip duplicate lines from interrupted writes
                done_keys.add(key)
                rows.append(r)
                if r.get("is_correct") is not None:
                    n_correct += int(r["is_correct"])
                    n_total   += 1
                total_time += r.get("elapsed_sec", 0.0)
            except Exception:
                pass  # skip malformed lines

    return done_keys, rows, n_correct, n_total, total_time


# ──────────────────────────────────────────────────────────────────────────────
# Per-question result generators
# ──────────────────────────────────────────────────────────────────────────────

def _greedy_results(method, pending, max_new_tokens) -> Iterator[tuple]:
    """
    Yield (example, InferenceResult) one at a time for GreedyInference.

    Greedy batches GREEDY_BATCH questions per generate() call, so we run
    the full batch, then yield individual results immediately so each can
    be saved before the next chunk starts.
    """
    from inference_methods import GreedyInference  # type: ignore
    bs        = method.model.batch_size
    n         = len(pending)
    per_chunk = round(0.0, 4)   # updated per chunk

    chunk_bar = tqdm(
        range(0, n, bs),
        total=max(1, -(-n // bs)),  # ceil division
        desc=f"[Greedy] generate (bs={bs})",
        unit="chunk",
        dynamic_ncols=True,
        colour="green",
    )

    for start in chunk_bar:
        chunk_ex  = pending[start : start + bs]
        questions = [ex["question"] for ex in chunk_ex]
        golds     = [ex.get("answer") for ex in chunk_ex]

        t0     = time.time()
        traces = method.model.generate_greedy_batch(
            questions, max_new_tokens=max_new_tokens
        )
        elapsed = time.time() - t0
        per_q   = round(elapsed / max(len(chunk_ex), 1), 4)

        chunk_bar.set_postfix(
            questions=f"{min(start + bs, n)}/{n}", refresh=False
        )

        for ex, gold, trace in zip(chunk_ex, golds, traces):
            from inference_methods import _extract_pred, _check_correct  # type: ignore
            pred    = _extract_pred(trace, method.ev, method.data_name)
            correct = _check_correct(pred, gold, method.ev, method.data_name)
            res = InferenceResult(
                method="greedy", question=ex["question"],
                gold_answer=gold, selected_trace=trace,
                predicted_answer=pred, is_correct=correct,
                N=1, elapsed_sec=per_q,
                all_traces=[trace], selected_index=0,
            )
            yield ex, res

    chunk_bar.close()


def _sampling_results(method, pending, max_new_tokens, label, colour) -> Iterator[tuple]:
    """
    Interleaved generation + saving via generate_samples_flat generator.

    generate_samples_flat() yields (q_idx, traces) as each question's
    N samples become available — after every chunk of batch_size sequences.

    With batch_size=64, N=8:
      → 8 questions saved per generate() call
    With batch_size=128, N=8:
      → 16 questions saved per generate() call

    This function receives each (q_idx, traces) immediately, runs _select(),
    and yields (example, InferenceResult) to the evaluator for saving.
    No waiting for all chunks to finish.
    """
    bs = method.model.batch_size
    N  = method.N
    questions = [ex["question"] for ex in pending]
    golds     = [ex.get("answer") for ex in pending]

    logger.info(
        f"  Flat-batch generation: Q={len(questions)}  N={N}  "
        f"batch_size={bs}  →  {len(questions)*N} total seqs  "
        f"→  {-(-len(questions)*N // bs)} generate() calls  "
        f"→  saving every {bs // N} questions"
    )

    n_correct = 0
    n_done    = 0

    save_bar = tqdm(
        total=len(pending),
        desc=f"{label}",
        unit="q",
        dynamic_ncols=True,
        colour=colour,
    )

    try:
        for q_idx, traces in method.model.generate_samples_flat(
            questions, N=N,
            temperature=method.temperature, top_p=method.top_p,
            max_new_tokens=max_new_tokens,
        ):
            ex   = pending[q_idx]
            gold = golds[q_idx]

            try:
                res = method._select(ex["question"], gold, traces, elapsed=0.0)
            except Exception as e:
                logger.error(f"  _select q[{q_idx}] failed: {e}")
                from inference_methods import _make_error_result  # type: ignore
                res = _make_error_result(
                    method.__class__.__name__.lower(), ex["question"], gold, N
                )

            n_done += 1
            if res.is_correct is not None:
                n_correct += int(res.is_correct)
                save_bar.set_postfix(
                    acc=f"{n_correct/n_done:.3f}",
                    correct=f"{n_correct}/{n_done}",
                    refresh=False,
                )
            save_bar.update(1)
            yield ex, res   # → evaluator writes JSONL + summary immediately

    except Exception as e:
        logger.error(f"  generate_samples_flat failed: {e}. Falling back to per-question.")
        # Fallback: per-question loop
        for i, (ex, gold) in enumerate(zip(pending[n_done:], golds[n_done:]), start=n_done):
            try:
                traces = method.model.generate_samples(
                    ex["question"], N=N,
                    temperature=method.temperature, top_p=method.top_p,
                    max_new_tokens=max_new_tokens,
                )
                res = method._select(ex["question"], gold, traces, elapsed=0.0)
            except Exception as e2:
                logger.error(f"  fallback q[{i}] failed: {e2}")
                from inference_methods import _make_error_result  # type: ignore
                res = _make_error_result(
                    method.__class__.__name__.lower(), ex["question"], gold, N
                )
            n_done += 1
            save_bar.update(1)
            yield ex, res
    finally:
        save_bar.close()


# ──────────────────────────────────────────────────────────────────────────────
# Primary evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_batch(
    method,
    examples: List[Dict[str, Any]],
    max_new_tokens: int = 1024,
    save_path: Optional[str] = None,
    summary_path: Optional[str] = None,
    resume: bool = True,
    log_every: int = 10,
    summary_every: int = 1,    # rewrite summary every N questions (default: every question)
) -> Dict[str, Any]:
    """
    Run inference over `examples`, saving every result immediately.

    Parameters
    ----------
    method         : GreedyInference / BestOfN / DNBestOfN
    examples       : list of {'question', 'answer', '_data_name', ...}
    max_new_tokens : generation length cap
    save_path      : JSONL file — one line appended per question, immediately
    summary_path   : summary JSON — atomically rewritten every `summary_every`
                     questions so it always reflects current progress
    resume         : skip already-done questions by reading existing JSONL
    log_every      : structured logger interval
    summary_every  : how often to rewrite the summary JSON (default 1 = always)

    Returns
    -------
    dict: accuracy, n_correct, n_total, avg_elapsed_sec, total_wall_sec, results
    """
    from inference_methods import GreedyInference, BestOfN, DNBestOfN  # type: ignore
    from inference_methods import GreedyPassN, BestOfNMajority          # type: ignore
    from inference_methods import DNBonUpdated                           # type: ignore
    method_name = method.__class__.__name__

    # ── load already-completed results from previous run ──────────────────────
    done_keys  = set()
    rows       = []
    n_correct  = 0
    n_total    = 0
    total_time = 0.0

    if resume and save_path:
        done_keys, rows, n_correct, n_total, total_time = _load_completed(save_path)
        if done_keys:
            logger.info(
                f"Resuming: {len(done_keys)} questions already done  "
                f"(acc so far: {n_correct}/{n_total})"
            )

    pending = [ex for ex in examples
               if ex["question"][:200] not in done_keys]

    logger.info(
        f"Evaluating {len(pending)}/{len(examples)} pending questions  "
        f"(method={method_name})"
    )

    if not pending:
        logger.info("All questions already completed — nothing to do.")
        return _build_return(rows, n_correct, n_total, total_time)

    # ── open JSONL for appending ───────────────────────────────────────────────
    f_out = None
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        f_out = open(save_path, "a", encoding="utf-8")

    # ── choose the result generator ───────────────────────────────────────────
    if isinstance(method, GreedyInference):
        result_gen = _greedy_results(method, pending, max_new_tokens)
    elif isinstance(method, BestOfN):
        result_gen = _sampling_results(
            method, pending, max_new_tokens,
            label=f"[BoN N={method.N}]", colour="blue",
        )
    elif isinstance(method, DNBestOfN):
        result_gen = _sampling_results(
            method, pending, max_new_tokens,
            label=f"[DN-BoN N={method.N} α={method.alpha}]", colour="magenta",
        )
    elif isinstance(method, GreedyPassN):
        result_gen = _sampling_results(
            method, pending, max_new_tokens,
            label=f"[GreedyPass@N N={method.N}]", colour="cyan",
        )
    elif isinstance(method, BestOfNMajority):
        result_gen = _sampling_results(
            method, pending, max_new_tokens,
            label=f"[BoN-Majority N={method.N}]", colour="yellow",
        )
    elif isinstance(method, DNBonUpdated):
        # DORA: soft quality × uniqueness scoring.
        # Uses the same _sampling_results generator as other sampling methods —
        # the selection logic is fully self-contained in DNBonUpdated._select().
        result_gen = _sampling_results(
            method, pending, max_new_tokens,
            label=f"[DN-BoN-Updated N={method.N} T_b={method.T_b} T_s={method.T_s}]",
            colour="cyan",
        )
    else:
        raise TypeError(f"Unknown method type: {type(method)}")

    # ── main loop: generate → save JSONL → update summary ────────────────────
    t_wall_start = time.time()
    n_done_this_run = 0

    try:
        for ex, res in result_gen:
            # Build the row
            row = res.to_dict()
            row["question"] = ex["question"][:300]

            # ── Motivation experiment extra fields ─────────────────────────────
            # DNBestOfN._select attaches _all_preds and _all_correct as
            # ad-hoc attributes so they can be serialised here without
            # changing the InferenceResult dataclass signature.
            if hasattr(res, "_all_preds"):
                row["all_predictions"] = res._all_preds
            if hasattr(res, "_all_correct"):
                row["all_correct"] = res._all_correct
            # DNBonUpdated (DORA) attaches additional scoring fields:
            #   _quality_weights : softmax quality weights w_i  (list of N floats)
            #   _gamma           : uniqueness scores γ_i = P_ii  (list of N floats)
            #   _T_b / _T_s      : temperature hyperparameters used
            if hasattr(res, "_quality_weights"):
                row["quality_weights"] = [round(v, 6) for v in res._quality_weights]
            if hasattr(res, "_gamma"):
                row["gamma"] = [round(v, 6) for v in res._gamma]
            if hasattr(res, "_T_b"):
                row["T_b"] = res._T_b
            if hasattr(res, "_T_s"):
                row["T_s"] = res._T_s

            rows.append(row)

            # Update running counts
            if res.is_correct is not None:
                n_correct += int(res.is_correct)
                n_total   += 1
            total_time += res.elapsed_sec
            n_done_this_run += 1

            # ── Save JSONL immediately ─────────────────────────────────────────
            if f_out:
                f_out.write(json.dumps(row) + "\n")
                f_out.flush()                    # guarantee OS write

            # ── Rewrite summary JSON atomically ───────────────────────────────
            if summary_path and (n_done_this_run % summary_every == 0):
                summary = _build_return(rows, n_correct, n_total, total_time)
                _write_summary_atomic(summary, summary_path)

            # ── Structured log ────────────────────────────────────────────────
            total_done = len(done_keys) + n_done_this_run
            if total_done % log_every == 0:
                acc = n_correct / max(n_total, 1)
                logger.info(
                    f"  [{total_done}/{len(examples)}] "
                    f"acc={acc:.3f} ({n_correct}/{n_total})"
                )
    finally:
        if f_out:
            f_out.close()
        # Always write final summary even if we crashed mid-loop
        if summary_path and rows:
            summary = _build_return(rows, n_correct, n_total, total_time)
            _write_summary_atomic(summary, summary_path)
            logger.info(f"Summary (final) saved → {summary_path}")

    total_wall = time.time() - t_wall_start
    result = _build_return(rows, n_correct, n_total, total_time)
    result["total_wall_sec"] = round(total_wall, 2)

    logger.info(
        f"Done. acc={result['accuracy']:.4f} ({n_correct}/{n_total})  "
        f"wall={total_wall:.1f}s  avg={result['avg_elapsed_sec']:.2f}s/q"
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Summary builder
# ──────────────────────────────────────────────────────────────────────────────

def _build_return(
    rows: List[Dict],
    n_correct: int,
    n_total: int,
    total_time: float,
) -> Dict[str, Any]:
    accuracy    = n_correct / max(n_total, 1)
    avg_elapsed = total_time / max(len(rows), 1)
    return {
        "accuracy":        round(accuracy, 4),
        "n_correct":       n_correct,
        "n_total":         n_total,
        "avg_elapsed_sec": round(avg_elapsed, 3),
        "total_wall_sec":  0.0,   # filled in by caller
        "results":         rows,
    }


def _empty_summary() -> Dict[str, Any]:
    return {"accuracy": 0.0, "n_correct": 0, "n_total": 0,
            "avg_elapsed_sec": 0.0, "total_wall_sec": 0.0, "results": []}


# ──────────────────────────────────────────────────────────────────────────────
# Legacy single-question loop  (kept for debugging)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    method,
    examples: List[Dict[str, Any]],
    max_new_tokens: int = 1024,
    save_path: Optional[str] = None,
    resume: bool = True,
    log_every: int = 10,
) -> Dict[str, Any]:
    """Legacy: one method.run() call per question.  Prefer evaluate_batch()."""
    method_name = method.__class__.__name__

    done_keys, rows, n_correct, n_total, total_elapsed = \
        _load_completed(save_path) if (resume and save_path) else \
        (set(), [], 0, 0, 0.0)

    pending = [ex for ex in examples
               if ex["question"][:200] not in done_keys]

    f_out = None
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        f_out = open(save_path, "a", encoding="utf-8")

    pbar = tqdm(pending, desc=f"[{method_name}]", unit="q",
                dynamic_ncols=True, colour="green")
    try:
        for i, ex in enumerate(pbar):
            q, gold = ex["question"], ex.get("answer")
            try:
                res: InferenceResult = method.run(
                    question=q, gold_answer=gold, max_new_tokens=max_new_tokens
                )
            except Exception as e:
                logger.error(f"[{i}] failed: {e}")
                from inference_methods import _make_error_result
                res = _make_error_result(method_name, q, gold, getattr(method,"N",1))

            row = res.to_dict()
            row["question"] = q[:300]
            rows.append(row)
            if res.is_correct is not None:
                n_correct += int(res.is_correct); n_total += 1
            total_elapsed += res.elapsed_sec
            if f_out:
                f_out.write(json.dumps(row) + "\n"); f_out.flush()
            if n_total > 0:
                pbar.set_postfix(acc=f"{n_correct/n_total:.3f}",
                                 correct=f"{n_correct}/{n_total}", refresh=False)
            if (i + 1) % log_every == 0:
                logger.info(f"[{i+1}] acc={n_correct/max(n_total,1):.3f}")
    finally:
        pbar.close()
        if f_out: f_out.close()

    accuracy    = n_correct / max(n_total, 1)
    avg_elapsed = total_elapsed / max(len(rows), 1)
    logger.info(f"Final acc={accuracy:.4f} ({n_correct}/{n_total})")
    return {"accuracy": round(accuracy,4), "n_correct": n_correct,
            "n_total": n_total, "avg_elapsed_sec": round(avg_elapsed,3),
            "results": rows}


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_summary(summary: Dict[str, Any], path: str):
    """Write summary JSON atomically."""
    _write_summary_atomic(summary, path)
    logger.info(f"Summary saved to {path}")


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
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
        size_arr       = np.array(sizes)
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