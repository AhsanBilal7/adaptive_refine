"""
cimr_rollout.py
===============
One rollout = d sequential CIMR steps.

A rollout takes a question and threads d steps in a chain:

    step 1   →  reasoning_1  →  critique_1  →  correction_1
                                                     │
                                                     ▼
    step 2   →  reasoning_2  →  critique_2  →  correction_2
                                                     │
                                                     ▼
        ...
                                                     │
                                                     ▼
    step d   →  reasoning_d  →  critique_d  →  correction_d
                                                     │
                                                     ▼
                                          ROLLOUT FINAL ANSWER

The corrected output of step k is passed as input context to step k+1's
reasoning generation (see ``cimr_prompts.build_continuation_reasoning_prompt``).

The rollout's final answer is taken from step d's CORRECTED ANSWER.
The rollout's final PRM score is the PRM on step d's CORRECTED TRACE.

Resume on the go
----------------
This module is the level at which we save and resume:
  * After EVERY step, the rollout's running record is handed to the
    on_step_complete() callback supplied by the orchestrator.  The
    orchestrator persists it to disk immediately, so a crashed run can
    resume the rollout at the next un-finished step.
  * If the rollout is invoked with a previously-saved partial state,
    it picks up at the first step that is not yet complete.

This file does NOT know about majority voting or aggregation across
rollouts — that lives in ``cimr_aggregator.py`` and is driven by
``cimr_inference.py``.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from .cimr_step import CIMRStepConfig, run_one_step

logger = logging.getLogger(__name__)


def run_one_rollout(
    *,
    model,
    verifier,
    ev,
    data_name:        str,
    question:         str,
    gold_answer:      Optional[str],
    rollout_idx:      int,
    total_steps:      int,                                  # d
    config:           CIMRStepConfig,
    sibling_answers_provider: Optional[Callable[[int], List[str]]] = None,
    on_step_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
    resume_state:     Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run one d-step rollout for the given question.

    Parameters
    ----------
    model, verifier, ev, data_name :
        Generator / PRM verifier / evaluator / dataset name.
    question, gold_answer :
        The problem and (optional) ground-truth answer.
    rollout_idx :
        0-indexed rollout id (one of N).
    total_steps :
        d, the number of refinement steps in this rollout.
    config :
        CIMRStepConfig — generation/critique/correction knobs.
    sibling_answers_provider :
        Optional callable: rollout_idx -> List[str] of sibling rollouts'
        current corrected answers.  Surfaced to critique + correction
        when ``config.other_answer_aware`` is True.  Pass None to
        disable.
    on_step_complete :
        Optional callback invoked AFTER every step finishes, with the
        rollout's running record dict.  Used by the orchestrator to
        persist on-the-go.
    resume_state :
        Optional previously-saved rollout dict (produced by a prior run
        of this function).  When supplied, we skip any step whose index
        is already present in ``resume_state["steps"]``.

    Returns
    -------
    rollout_record : dict
      Keys:
        rollout_idx, total_steps, completed (bool), elapsed_sec,
        steps                  : list of step dicts (one per completed step)
        final_answer           : answer from the final completed step
        final_prm_score        : PRM on the final corrected trace
        final_corrected_trace  : the final corrected trace text
    """
    t0 = time.time()

    # Initialise / resume
    if resume_state and isinstance(resume_state, dict):
        steps: List[Dict[str, Any]] = list(resume_state.get("steps", []))
        # Sanity: keep only steps with sequential 1..k indices.
        steps.sort(key=lambda s: int(s.get("step_idx", 0)))
        # Drop anything beyond total_steps in case d shrank between runs.
        steps = [s for s in steps if int(s.get("step_idx", 0)) <= total_steps]
    else:
        steps = []

    # Index of the next step to run (1-indexed)
    next_step_idx = (steps[-1]["step_idx"] + 1) if steps else 1

    # Carry forward the prior corrected trace
    if steps:
        prior_corrected = steps[-1].get("corrected_trace", "")
    else:
        prior_corrected = None

    # Run remaining steps
    for step_idx in range(next_step_idx, total_steps + 1):
        sibling = (
            sibling_answers_provider(rollout_idx)
            if sibling_answers_provider is not None else None
        )
        try:
            step_record = run_one_step(
                model            = model,
                verifier         = verifier,
                ev               = ev,
                data_name        = data_name,
                question         = question,
                gold_answer      = gold_answer,
                step_idx         = step_idx,
                total_steps      = total_steps,
                prior_corrected  = prior_corrected,
                config           = config,
                sibling_answers  = sibling,
                rollout_idx      = rollout_idx,
            )
        except Exception as e:
            logger.error(
                f"[CIMR rollout {rollout_idx}] step {step_idx} failed: {e}"
            )
            step_record = dict(
                rollout_idx=rollout_idx, step_idx=step_idx,
                total_steps=total_steps,
                reasoning_response="", reasoning_answer="",
                reasoning_prm_score=0.0,
                critique_response="", critique_parsed={},
                correction_response="", correction_parsed={},
                corrected_trace=prior_corrected or "",
                corrected_answer="",
                corrected_prm_score=0.0,
                elapsed_sec=0.0,
                error=str(e),
            )

        steps.append(step_record)
        prior_corrected = step_record.get("corrected_trace", "") or prior_corrected

        # Persist incrementally
        if on_step_complete is not None:
            running_record = _build_rollout_record(
                rollout_idx=rollout_idx,
                total_steps=total_steps,
                steps=steps,
                completed=(step_idx >= total_steps),
                elapsed_sec=time.time() - t0,
            )
            try:
                on_step_complete(running_record)
            except Exception as e:
                logger.warning(
                    f"[CIMR rollout {rollout_idx}] on_step_complete "
                    f"failed at step {step_idx}: {e}"
                )

    return _build_rollout_record(
        rollout_idx=rollout_idx,
        total_steps=total_steps,
        steps=steps,
        completed=True,
        elapsed_sec=time.time() - t0,
    )


def _build_rollout_record(
    *,
    rollout_idx: int,
    total_steps: int,
    steps:       List[Dict[str, Any]],
    completed:   bool,
    elapsed_sec: float,
) -> Dict[str, Any]:
    """Pack the per-rollout record.  The final-answer and final-PRM
    fields are derived from the LAST completed step."""
    if steps:
        last = steps[-1]
        final_answer        = last.get("corrected_answer", "")
        final_prm_score     = float(last.get("corrected_prm_score", 0.0))
        final_corrected     = last.get("corrected_trace", "")
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


__all__ = ["run_one_rollout"]
