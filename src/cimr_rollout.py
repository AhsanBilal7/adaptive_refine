"""
cimr_rollout.py
===============
N-rollout × d-step CIMR execution with batched generation.

Architecture
------------
The outer loop iterates over depth steps (1 → d) sequentially, because
each step requires the corrected output of the previous step as input.

Within each depth level, ALL N rollouts are processed simultaneously via
batched model calls (_generate_batch in cimr_step.py), giving N× GPU
utilisation compared to running rollouts one at a time.

    depth 1:  batch-generate reasoning for rollouts 0..N-1
              batch-critique   rollouts 0..N-1
              batch-correct    rollouts 0..N-1
              PRM-score all corrected traces
                         │
                         ▼
    depth 2:  batch-generate reasoning (given depth-1 corrected traces)
              batch-critique
              batch-correct
              PRM-score
                         │
                      ...│...
                         ▼
    depth d:  ... → final corrected traces and answers

Resume on the go
----------------
``run_all_rollouts_batched`` accepts a ``resume_states`` list (one per
rollout).  Rollouts that already have k completed steps re-enter at step
k+1 without re-running prior steps.  Staggered resume states are handled
correctly: at each depth level only the rollouts that are missing that
step are included in the batch.

The ``on_rollout_step_complete`` callback is invoked after every step for
every rollout that was processed, giving the caller an opportunity to
persist incrementally.

``run_one_rollout`` is kept for single-rollout / compatibility use.

Public symbols
--------------
  run_all_rollouts_batched — main entry point (N rollouts, batched per depth)
  run_one_rollout          — single-rollout entry point (sequential depths)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from .cimr_step import CIMRStepConfig, run_one_step, run_step_batch

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================

def _error_step_record(
    rollout_idx: int,
    step_idx:    int,
    total_steps: int,
    prior:       Optional[str],
    error:       str = "",
) -> Dict[str, Any]:
    return dict(
        rollout_idx          = int(rollout_idx),
        step_idx             = int(step_idx),
        total_steps          = int(total_steps),
        reasoning_response   = "",
        reasoning_answer     = "",
        reasoning_prm_score  = 0.0,
        critique_response    = "",
        critique_parsed      = {},
        correction_response  = "",
        correction_parsed    = {},
        corrected_trace      = prior or "",
        corrected_answer     = "",
        corrected_prm_score  = 0.0,
        elapsed_sec          = 0.0,
        error                = error,
    )


def _build_rollout_record(
    *,
    rollout_idx: int,
    total_steps: int,
    steps:       List[Dict[str, Any]],
    completed:   bool,
    elapsed_sec: float,
) -> Dict[str, Any]:
    """Build the per-rollout summary record from its completed steps."""
    if steps:
        last = steps[-1]
        final_answer    = last.get("corrected_answer", "")
        final_prm_score = float(last.get("corrected_prm_score", 0.0))
        final_corrected = last.get("corrected_trace", "")
    else:
        final_answer    = ""
        final_prm_score = 0.0
        final_corrected = ""

    return dict(
        rollout_idx           = int(rollout_idx),
        total_steps           = int(total_steps),
        completed             = bool(completed),
        n_steps_done          = len(steps),
        steps                 = list(steps),
        final_answer          = final_answer,
        final_prm_score       = final_prm_score,
        final_corrected_trace = final_corrected,
        elapsed_sec           = round(float(elapsed_sec), 3),
    )


# =============================================================================
# Batched N-rollout execution — main entry point
# =============================================================================

def run_all_rollouts_batched(
    *,
    model,
    verifier,
    ev,
    data_name:                str,
    question:                 str,
    gold_answer:              Optional[str],
    n_rollouts:               int,
    total_steps:              int,
    config:                   CIMRStepConfig,
    on_rollout_step_complete: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    resume_states:            Optional[List[Optional[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """Run N rollouts × D steps with batched GPU generation per depth level.

    Parameters
    ----------
    n_rollouts               : N, number of independent rollouts.
    total_steps              : d, depth of the CIMR pipeline.
    on_rollout_step_complete : callback(rollout_i, running_record) invoked
                               after every step for every rollout that was
                               processed in that step's batch.  Used by the
                               orchestrator to persist incrementally.
    resume_states            : list of length n_rollouts; each entry is either
                               None (fresh rollout) or a previously-saved
                               rollout dict with a "steps" key.

    Returns
    -------
    List of n_rollouts rollout-record dicts.
    """
    t0 = time.time()

    # ── Initialise from resume states ──────────────────────────────────────
    all_steps: List[List[Dict[str, Any]]] = [[] for _ in range(n_rollouts)]

    if resume_states:
        for i in range(n_rollouts):
            rs = resume_states[i] if i < len(resume_states) else None
            if rs and isinstance(rs, dict):
                steps = list(rs.get("steps", []))
                steps.sort(key=lambda s: int(s.get("step_idx", 0)))
                steps = [s for s in steps if int(s.get("step_idx", 0)) <= total_steps]
                all_steps[i] = steps

    # Number of completed steps per rollout (used to decide which rollouts
    # need each depth level).
    completed = [len(all_steps[i]) for i in range(n_rollouts)]

    # Prior corrected trace for each rollout (seed for the next depth's reasoning).
    prior_corrected: List[Optional[str]] = [
        (all_steps[i][-1].get("corrected_trace") if all_steps[i] else None)
        for i in range(n_rollouts)
    ]

    # ── Sequential depth loop — batched across rollouts ────────────────────
    for step_idx in range(1, total_steps + 1):
        # Rollouts that haven't yet reached this depth level.
        # A rollout with k completed steps needs step k+1 next, so it
        # should be included when step_idx == k+1, i.e. completed[i] < step_idx.
        to_run = [i for i in range(n_rollouts) if completed[i] < step_idx]

        if not to_run:
            logger.debug(
                f"[CIMR batch] depth {step_idx}: all {n_rollouts} rollouts "
                "already done, skipping."
            )
            continue

        prior_for_batch = [prior_corrected[i] for i in to_run]

        logger.debug(
            f"[CIMR batch] depth {step_idx}/{total_steps}  "
            f"rollouts={to_run}  batch_size={len(to_run)}"
        )

        try:
            step_records = run_step_batch(
                model                = model,
                verifier             = verifier,
                ev                   = ev,
                data_name            = data_name,
                question             = question,
                gold_answer          = gold_answer,
                step_idx             = step_idx,
                total_steps          = total_steps,
                prior_corrected_list = prior_for_batch,
                config               = config,
                rollout_idxs         = to_run,
            )
        except Exception as e:
            logger.error(
                f"[CIMR batch] depth {step_idx} batch failed: {e}", exc_info=True
            )
            step_records = [
                _error_step_record(
                    rollout_idx = to_run[k],
                    step_idx    = step_idx,
                    total_steps = total_steps,
                    prior       = prior_corrected[to_run[k]],
                    error       = str(e),
                )
                for k in range(len(to_run))
            ]

        # Update state and fire callbacks.
        for record, rollout_i in zip(step_records, to_run):
            all_steps[rollout_i].append(record)
            completed[rollout_i]     = len(all_steps[rollout_i])
            prior_corrected[rollout_i] = (
                record.get("corrected_trace") or prior_corrected[rollout_i]
            )

            if on_rollout_step_complete is not None:
                running = _build_rollout_record(
                    rollout_idx = rollout_i,
                    total_steps = total_steps,
                    steps       = all_steps[rollout_i],
                    completed   = (completed[rollout_i] >= total_steps),
                    elapsed_sec = time.time() - t0,
                )
                try:
                    on_rollout_step_complete(rollout_i, running)
                except Exception as cb_err:
                    logger.warning(
                        f"[CIMR batch] on_rollout_step_complete failed "
                        f"(rollout {rollout_i}, depth {step_idx}): {cb_err}"
                    )

    # ── Build and return final rollout records ─────────────────────────────
    total_elapsed = time.time() - t0
    return [
        _build_rollout_record(
            rollout_idx = i,
            total_steps = total_steps,
            steps       = all_steps[i],
            completed   = True,
            elapsed_sec = total_elapsed,
        )
        for i in range(n_rollouts)
    ]


# =============================================================================
# Single-rollout entry point (sequential depths, for resume compatibility)
# =============================================================================

def run_one_rollout(
    *,
    model,
    verifier,
    ev,
    data_name:                str,
    question:                 str,
    gold_answer:              Optional[str],
    rollout_idx:              int,
    total_steps:              int,
    config:                   CIMRStepConfig,
    sibling_answers_provider: Optional[Callable[[int], List[str]]] = None,
    on_step_complete:         Optional[Callable[[Dict[str, Any]], None]] = None,
    resume_state:             Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run one d-step rollout sequentially.

    Kept for single-rollout / compatibility use.  For new runs, prefer
    ``run_all_rollouts_batched`` which processes N rollouts in parallel.
    """
    t0 = time.time()

    if resume_state and isinstance(resume_state, dict):
        steps: List[Dict[str, Any]] = list(resume_state.get("steps", []))
        steps.sort(key=lambda s: int(s.get("step_idx", 0)))
        steps = [s for s in steps if int(s.get("step_idx", 0)) <= total_steps]
    else:
        steps = []

    next_step_idx = (steps[-1]["step_idx"] + 1) if steps else 1

    prior_corrected: Optional[str] = (
        steps[-1].get("corrected_trace", "") if steps else None
    )

    for step_idx in range(next_step_idx, total_steps + 1):
        sibling = (
            sibling_answers_provider(rollout_idx)
            if sibling_answers_provider is not None else None
        )
        try:
            step_record = run_one_step(
                model           = model,
                verifier        = verifier,
                ev              = ev,
                data_name       = data_name,
                question        = question,
                gold_answer     = gold_answer,
                step_idx        = step_idx,
                total_steps     = total_steps,
                prior_corrected = prior_corrected,
                config          = config,
                sibling_answers = sibling,
                rollout_idx     = rollout_idx,
            )
        except Exception as e:
            logger.error(f"[CIMR rollout {rollout_idx}] step {step_idx} failed: {e}")
            step_record = _error_step_record(
                rollout_idx, step_idx, total_steps, prior_corrected, str(e)
            )

        steps.append(step_record)
        prior_corrected = step_record.get("corrected_trace", "") or prior_corrected

        if on_step_complete is not None:
            running = _build_rollout_record(
                rollout_idx = rollout_idx,
                total_steps = total_steps,
                steps       = steps,
                completed   = (step_idx >= total_steps),
                elapsed_sec = time.time() - t0,
            )
            try:
                on_step_complete(running)
            except Exception as e:
                logger.warning(
                    f"[CIMR rollout {rollout_idx}] on_step_complete "
                    f"failed at step {step_idx}: {e}"
                )

    return _build_rollout_record(
        rollout_idx = rollout_idx,
        total_steps = total_steps,
        steps       = steps,
        completed   = True,
        elapsed_sec = time.time() - t0,
    )


__all__ = [
    "run_all_rollouts_batched",
    "run_one_rollout",
    "_build_rollout_record",
]
