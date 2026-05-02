"""
cimr_aggregator.py
==================
Aggregate the N rollouts of a single question into the two final
predictions:

  1. **Majority vote** over the N final answers
       — pick the modal answer.  Ties broken by larger sum of PRM
         scores within the tied group, then by lower rollout index.

  2. **Max-PRM** (a.k.a. Best-of-N over corrected traces)
       — pick the answer of the rollout whose final corrected trace has
         the highest PRM score.

Both predictions are reported per question, and BOTH accuracies
(majority-vote accuracy, max-PRM accuracy) are written to the summary,
so the user can compare them directly.

Notes on equality
-----------------
We use a light canonicalisation (``_normalize_answer_key``) for grouping
duplicates — strip whitespace, drop trailing zeros, drop wrapping
braces.  Mathematical equality (e.g. ``1/2`` vs ``0.5``) is left to the
evaluator's ``_check_correct`` at scoring time, not used for grouping.
"""

from __future__ import annotations

import logging
import re
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Light canonicalisation for cluster-key equality
# =============================================================================

def _normalize_answer_key(ans: str) -> str:
    s = str(ans or "").strip()
    if not s:
        return ""
    s = s.replace(" ", "")
    s = s.replace("\\,", "").replace("\\!", "")
    if s.startswith("{") and s.endswith("}") and s.count("{") == 1:
        s = s[1:-1]
    if re.fullmatch(r"-?\d+\.\d+", s):
        s = s.rstrip("0").rstrip(".")
    return s


# =============================================================================
# Aggregation
# =============================================================================

def aggregate_rollouts(
    rollouts:     List[Dict[str, Any]],
    gold_answer:  Optional[str]  = None,
    ev                          = None,
    data_name:    str            = "math",
) -> Dict[str, Any]:
    """Aggregate a list of completed rollouts (each from
    ``cimr_rollout.run_one_rollout``) into final majority-vote and
    max-PRM predictions.

    Returns
    -------
    dict with:
      majority_pred         : str
      majority_is_correct   : bool | None
      majority_rollout_idx  : int     (the rollout that "represents" the modal group)
      majority_vote_count   : int
      max_prm_pred          : str
      max_prm_is_correct    : bool | None
      max_prm_rollout_idx   : int
      max_prm_score         : float
      per_rollout_summary   : list of {rollout_idx, final_answer, final_prm_score}
      n_rollouts            : int
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
            rollout_idx     = int(r.get("rollout_idx", -1)),
            final_answer    = str(r.get("final_answer", "")),
            final_prm_score = float(r.get("final_prm_score", 0.0)),
        ))

    # ── 1) Majority vote ────────────────────────────────────────────────
    #     Tie-break order: count desc, sum_prm desc, lowest rollout idx asc.
    bucket: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for r in per_rollout:
        key = _normalize_answer_key(r["final_answer"])
        if not key:
            # Empty answers don't vote.
            continue
        if key not in bucket:
            bucket[key] = dict(
                key            = key,
                display        = r["final_answer"].strip(),
                count          = 0,
                sum_prm        = 0.0,
                first_idx      = int(r["rollout_idx"]),
                rollout_idxs   = [],
            )
        bucket[key]["count"]   += 1
        bucket[key]["sum_prm"] += float(r["final_prm_score"])
        bucket[key]["rollout_idxs"].append(int(r["rollout_idx"]))

    if bucket:
        ranked = sorted(
            bucket.values(),
            key=lambda b: (b["count"], b["sum_prm"], -b["first_idx"]),
            reverse=True,
        )
        winner = ranked[0]
        majority_pred         = winner["display"]
        majority_rollout_idx  = int(winner["first_idx"])
        majority_vote_count   = int(winner["count"])
    else:
        majority_pred        = ""
        majority_rollout_idx = -1
        majority_vote_count  = 0

    # ── 2) Max-PRM (Best-of-N over corrected traces) ────────────────────
    best = max(per_rollout, key=lambda r: r["final_prm_score"])
    max_prm_pred         = best["final_answer"]
    max_prm_rollout_idx  = int(best["rollout_idx"])
    max_prm_score        = float(best["final_prm_score"])

    # ── Correctness checks ──────────────────────────────────────────────
    def _is_correct(pred: str) -> Optional[bool]:
        if gold_answer is None or ev is None:
            return None
        try:
            from inference_methods import _check_correct  # type: ignore
            return bool(_check_correct(pred, gold_answer, ev, data_name))
        except Exception as e:
            logger.warning(f"[CIMR aggregator] correctness check failed: {e}")
            return None

    majority_is_correct = _is_correct(majority_pred)
    max_prm_is_correct  = _is_correct(max_prm_pred)

    return dict(
        majority_pred         = majority_pred,
        majority_is_correct   = majority_is_correct,
        majority_rollout_idx  = majority_rollout_idx,
        majority_vote_count   = majority_vote_count,
        max_prm_pred          = max_prm_pred,
        max_prm_is_correct    = max_prm_is_correct,
        max_prm_rollout_idx   = max_prm_rollout_idx,
        max_prm_score         = max_prm_score,
        per_rollout_summary   = per_rollout,
        n_rollouts            = len(rollouts),
    )


__all__ = ["aggregate_rollouts", "_normalize_answer_key"]
