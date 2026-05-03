"""
cimr_aggregator.py
==================
Aggregate the N completed rollouts for a single question into two final
predictions:

  1. Majority vote — modal answer across the N final corrected answers.
     Ties broken by: larger total PRM score in the tied group → lower
     rollout index.

  2. Max-PRM (Best-of-N) — answer of the rollout with the highest
     final corrected-trace PRM score.

Both predictions + correctness flags are returned so callers can compare
them side-by-side.

Normalisation
-------------
``_normalize_answer_key`` does light canonicalisation for grouping:
  - strip whitespace and LaTeX spacing commands
  - unwrap single-level surrounding braces
  - strip trailing zeros from decimals
Mathematical equality (e.g. 1/2 vs 0.5) is left to the evaluator's
``_check_correct`` at scoring time.
"""

from __future__ import annotations

import logging
import re
from collections import OrderedDict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Canonicalisation
# =============================================================================

def _normalize_answer_key(ans: str) -> str:
    s = str(ans or "").strip()
    if not s:
        return ""
    s = s.replace(" ", "")
    # Drop LaTeX thin/negative spaces.
    s = s.replace("\\,", "").replace("\\!", "").replace("\\;", "")
    # Unwrap a single wrapping pair of braces.
    if s.startswith("{") and s.endswith("}") and s.count("{") == 1:
        s = s[1:-1]
    # Strip trailing zeros from decimals like "3.500" → "3.5".
    if re.fullmatch(r"-?\d+\.\d+", s):
        s = s.rstrip("0").rstrip(".")
    return s


# =============================================================================
# Aggregation
# =============================================================================

def aggregate_rollouts(
    rollouts: List[Dict[str, Any]],
    gold_answer: Optional[str] = None,
    ev=None,
    data_name: str = "math",
) -> Dict[str, Any]:
    """Aggregate N rollout records into majority-vote and max-PRM predictions.

    Parameters
    ----------
    rollouts    : list of rollout dicts from ``_build_rollout_record``.
    gold_answer : ground-truth answer string (optional).
    ev          : evaluator object with ``_check_correct`` accessible via
                  ``inference_methods._check_correct``.
    data_name   : dataset name passed to ``_check_correct``.

    Returns
    -------
    dict with keys:
      majority_pred, majority_is_correct, majority_rollout_idx,
      majority_vote_count,
      max_prm_pred, max_prm_is_correct, max_prm_rollout_idx, max_prm_score,
      per_rollout_summary, n_rollouts
    """
    if not rollouts:
        return dict(
            majority_pred="", majority_is_correct=None,
            majority_rollout_idx=-1, majority_vote_count=0,
            max_prm_pred="", max_prm_is_correct=None,
            max_prm_rollout_idx=-1, max_prm_score=0.0,
            per_rollout_summary=[], n_rollouts=0,
        )

    # ── Per-rollout summary ─────────────────────────────────────────────
    per_rollout: List[Dict[str, Any]] = []
    for r in rollouts:
        per_rollout.append(dict(
            rollout_idx=int(r.get("rollout_idx", -1)),
            final_answer=str(r.get("final_answer", "")),
            final_prm_score=float(r.get("final_prm_score", 0.0)),
        ))

    # ── 1) Majority vote ────────────────────────────────────────────────
    bucket: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for r in per_rollout:
        key = _normalize_answer_key(r["final_answer"])
        if not key:
            continue  # Skip empty answers — they don't vote.
        if key not in bucket:
            bucket[key] = dict(
                key=key,
                display=r["final_answer"].strip(),
                count=0,
                sum_prm=0.0,
                first_idx=int(r["rollout_idx"]),
                rollout_idxs=[],
            )
        bucket[key]["count"] += 1
        bucket[key]["sum_prm"] += float(r["final_prm_score"])
        bucket[key]["rollout_idxs"].append(int(r["rollout_idx"]))

    if bucket:
        # Sort by (count DESC, sum_prm DESC, first_idx ASC).
        ranked = sorted(
            bucket.values(),
            key=lambda b: (b["count"], b["sum_prm"], -b["first_idx"]),
            reverse=True,
        )
        winner = ranked[0]
        majority_pred = winner["display"]
        majority_rollout_idx = int(winner["first_idx"])
        majority_vote_count = int(winner["count"])
    else:
        majority_pred = ""
        majority_rollout_idx = -1
        majority_vote_count = 0

    # ── 2) Max-PRM (Best-of-N over corrected traces) ────────────────────
    best = max(per_rollout, key=lambda r: r["final_prm_score"])
    max_prm_pred = best["final_answer"]
    max_prm_rollout_idx = int(best["rollout_idx"])
    max_prm_score = float(best["final_prm_score"])

    # ── Correctness checks ──────────────────────────────────────────────
    def _is_correct(pred: str) -> Optional[bool]:
        if gold_answer is None or ev is None or not pred:
            return None
        try:
            try:
                from src.inference_methods import _check_correct  # type: ignore
            except Exception:
                from inference_methods import _check_correct  # type: ignore
            return bool(_check_correct(pred, gold_answer, ev, data_name))
        except Exception as e:
            logger.debug(f"[CIMR aggregator] correctness check failed: {e}")
            return None

    majority_is_correct = _is_correct(majority_pred)
    max_prm_is_correct = _is_correct(max_prm_pred)

    logger.debug(
        f"[CIMR aggregator] "
        f"majority={majority_pred!r} (correct={majority_is_correct}, "
        f"votes={majority_vote_count})  "
        f"max_prm={max_prm_pred!r} (correct={max_prm_is_correct}, "
        f"score={max_prm_score:.4f})"
    )

    return dict(
        majority_pred=majority_pred,
        majority_is_correct=majority_is_correct,
        majority_rollout_idx=majority_rollout_idx,
        majority_vote_count=majority_vote_count,
        max_prm_pred=max_prm_pred,
        max_prm_is_correct=max_prm_is_correct,
        max_prm_rollout_idx=max_prm_rollout_idx,
        max_prm_score=max_prm_score,
        per_rollout_summary=per_rollout,
        n_rollouts=len(rollouts),
    )


__all__ = ["aggregate_rollouts", "_normalize_answer_key"]
