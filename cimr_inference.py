"""
cimr_inference.py
=================
CIMR orchestrator — runs N rollouts × d steps per question, with
self-critique + self-correction in every step, and aggregates via
both **majority vote** and **max PRM**.  Saves and resumes on the go
at the (question, rollout, step) granularity.

Pipeline (per question)
-----------------------

  for rollout i in 1..N:                                     # N independent
      for step k in 1..d:                                    # d sequential
          generate reasoning  given (question, prev_corrected)
          self-critique
          self-correction
          PRM-score the corrected trace
      rollout_i.final_answer    = step d's corrected answer
      rollout_i.final_prm_score = step d's corrected PRM
  # Aggregation across N rollouts
  pred_majority = mode of [rollout_i.final_answer for i in 1..N]
  pred_max_prm  = answer of argmax_i rollout_i.final_prm_score

Both `pred_majority` and `pred_max_prm` are reported per question, and
both their accuracies are recorded.

Resume / on-the-go save
-----------------------
After every step finishes (yes — every step, not just every rollout),
the orchestrator persists the question's running state to disk at
``<output_dir>/cimr_full_results_<dataset>_N<N>_d<D>.json``.  On rerun,
the orchestrator:
  1. Loads any previously-completed state for the question.
  2. Skips any rollout that is already fully finished (n_steps_done==d).
  3. Resumes a partially-finished rollout at the first step that was
     not yet saved.

Each rollout is FRESHLY GENERATED — there is no dependency on the
sample cache used by the baseline methods.

Module layout
-------------
This file orchestrates; the heavy lifting lives in:
  cimr_step.py        — one (generate, critique, correct, score) cycle
  cimr_rollout.py     — chains d steps for one rollout
  cimr_aggregator.py  — majority vote + max PRM across N rollouts
  cimr_prompts.py     — step-1 + step-k continuation prompts

Public symbols
--------------
  SELECTION_RULE_NAMES         — selection rules registered with the harness
  CIMRConfig                   — bag of all CIMR knobs
  CIMRInferencer               — main orchestrator class
  run_cimr_for_question        — thin per-question wrapper
  open_cimr_store / save_cimr_store
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from cimr_step import CIMRStepConfig
from cimr_rollout import run_one_rollout
from cimr_aggregator import aggregate_rollouts

logger = logging.getLogger(__name__)


# =============================================================================
# Selection rules registered with the run_paired_ablation harness
# =============================================================================

SELECTION_RULE_NAMES: List[str] = [
    "majority",   # majority vote across N rollouts
    "max_prm",    # argmax PRM across N rollouts (Best-of-N over corrected)
]


# =============================================================================
# Config
# =============================================================================

@dataclass
class CIMRConfig:
    """All CIMR knobs in one place.

    The step-level knobs (max-tokens / max-chars) live on the nested
    ``step_config`` for separation of concerns; everything else (N, D,
    seeds, output dir) lives on this object.
    """
    # Question-level
    n_rollouts:          int = 8                  # N
    depth:               int = 4                  # d
    # Step-level knobs
    step_config:         CIMRStepConfig = field(default_factory=CIMRStepConfig)

    # Optional model-level overrides (mostly needed for the very first step
    # of each rollout where we want sampling temperature > 0 to give
    # different rollouts different starting points).
    rollout_temperature: float = 0.7
    rollout_top_p:       float = 0.95


# =============================================================================
# Persistence (per-question store)
# =============================================================================

def cimr_store_path(
    output_dir: str, dataset: str, N: int, D: int,
) -> str:
    """Path for the per-(dataset, N, D) results store.  N and D are
    embedded in the filename so different sweeps don't collide."""
    return os.path.join(
        output_dir, f"cimr_full_results_{dataset}_N{N}_d{D}.json"
    )


def open_cimr_store(
    output_dir: str, dataset: str, N: int, D: int,
) -> Tuple[str, Dict[str, Any]]:
    """Open (or create) the CIMR results store and return (path, store)."""
    path = cimr_store_path(output_dir, dataset, N, D)
    store: Dict[str, Any] = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                store = data
            elif isinstance(data, list):
                store = {str(r.get("id", i)): r for i, r in enumerate(data)}
            logger.info(
                f"[CIMR] resumed {len(store)} previously-saved questions "
                f"from {path}"
            )
        except Exception as e:
            logger.warning(f"[CIMR] could not load existing store {path}: {e}")
    else:
        logger.info(f"[CIMR] starting a fresh CIMR store at {path}")
    return path, store


def save_cimr_store(path: str, store: Dict[str, Any]) -> None:
    """Atomic save of the CIMR store."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


# =============================================================================
# Per-question record helpers
# =============================================================================

def _empty_question_record(qid: str, question: str, gold: Optional[str],
                           N: int, D: int) -> Dict[str, Any]:
    return dict(
        id           = qid,
        question     = question,
        gold_answer  = gold,
        N            = N,
        depth        = D,
        rollouts     = [None] * N,    # filled in incrementally
        completed    = False,
        aggregation  = None,
        elapsed_sec  = 0.0,
    )


def _question_is_complete(qrec: Dict[str, Any]) -> bool:
    """A question is complete iff every rollout slot is non-None and
    each has the full count of finished steps."""
    rollouts = qrec.get("rollouts") or []
    if not rollouts or len(rollouts) < int(qrec.get("N", 0)):
        return False
    D = int(qrec.get("depth", 0))
    for r in rollouts:
        if r is None:
            return False
        if int(r.get("n_steps_done", 0)) < D:
            return False
        if not r.get("completed", False):
            return False
    return True


# =============================================================================
# Orchestrator
# =============================================================================

class CIMRInferencer:
    """Main orchestrator.  Holds references to the model + verifier +
    evaluator and runs N rollouts × d steps for one question at a
    time, persisting after every step."""

    def __init__(
        self,
        model,
        verifier,
        ev,
        data_name:        str,
        config:           CIMRConfig,
        output_dir:       str,
        dataset:          str,
    ):
        self.model      = model
        self.verifier   = verifier
        self.ev         = ev
        self.data_name  = data_name
        self.config     = config
        self.output_dir = output_dir
        self.dataset    = dataset

        # Resume state
        self.store_path, self.store = open_cimr_store(
            output_dir=output_dir, dataset=dataset,
            N=config.n_rollouts, D=config.depth,
        )

    # ------------------------------------------------------------------
    # Public: run one question end-to-end
    # ------------------------------------------------------------------
    def run_question(
        self,
        question:    str,
        gold_answer: Optional[str],
        qid:         str,
    ) -> Dict[str, Any]:
        """Run N rollouts × D steps for the given question.

        If the question is already fully complete in the store, return
        the cached record without doing any work.  Otherwise, run any
        missing rollouts (or resume partially-finished ones), persist
        after every step, and finally aggregate.
        """
        N = int(self.config.n_rollouts)
        D = int(self.config.depth)

        qrec = self.store.get(str(qid))
        if qrec is None:
            qrec = _empty_question_record(str(qid), question, gold_answer, N, D)
            self.store[str(qid)] = qrec
            save_cimr_store(self.store_path, self.store)
        else:
            # Defensive: pad rollout list if N grew across runs
            if len(qrec.get("rollouts", [])) < N:
                qrec["rollouts"] = (
                    list(qrec.get("rollouts", []))
                    + [None] * (N - len(qrec.get("rollouts", [])))
                )

        # Already done?  Just return.
        if _question_is_complete(qrec):
            logger.debug(
                f"[CIMR q={qid}] all {N} rollouts × {D} steps already done; "
                f"skipping."
            )
            # Make sure aggregation is up to date (e.g. if we changed the
            # gold answer or evaluator since saving).
            if qrec.get("aggregation") is None:
                qrec["aggregation"] = aggregate_rollouts(
                    rollouts    = qrec.get("rollouts", []),
                    gold_answer = qrec.get("gold_answer"),
                    ev          = self.ev,
                    data_name   = self.data_name,
                )
                qrec["completed"] = True
                save_cimr_store(self.store_path, self.store)
            return qrec

        t_q = time.time()
        # Snapshot pre-existing elapsed time once — we'll add this run's
        # wall-clock to it at the end.  This keeps the cumulative time
        # accurate across resumes.
        qrec["elapsed_sec_baseline"] = float(qrec.get("elapsed_sec", 0.0))

        # The "sibling answer" provider lets each rollout's critique +
        # correction agents see the OTHER rollouts' current best answers.
        # Used only when config.step_config.other_answer_aware is True.
        def sibling_provider(self_rollout_idx: int) -> List[str]:
            sibs: List[str] = []
            rollouts = qrec.get("rollouts") or []
            for j, r in enumerate(rollouts):
                if j == self_rollout_idx or r is None:
                    continue
                ans = str(r.get("final_answer", "")).strip()
                # If a rollout is mid-flight, fall back to its latest
                # corrected answer from the most recent saved step.
                if not ans and r.get("steps"):
                    ans = str(r["steps"][-1].get("corrected_answer", "")).strip()
                if ans:
                    sibs.append(ans)
            return sibs

        # Iterate over the N rollout slots.
        for i in range(N):
            existing = qrec["rollouts"][i]
            already_done = (
                existing is not None
                and int(existing.get("n_steps_done", 0)) >= D
                and bool(existing.get("completed", False))
            )
            if already_done:
                continue

            # Persist after every step inside this rollout
            baseline = float(qrec["elapsed_sec_baseline"])
            def on_step_complete(
                running_record: Dict[str, Any],
                _i:          int = i,
                _baseline:   float = baseline,
            ) -> None:
                qrec["rollouts"][_i] = running_record
                qrec["elapsed_sec"]  = round(
                    _baseline + (time.time() - t_q), 3
                )
                save_cimr_store(self.store_path, self.store)

            rollout_record = run_one_rollout(
                model       = self.model,
                verifier    = self.verifier,
                ev          = self.ev,
                data_name   = self.data_name,
                question    = question,
                gold_answer = gold_answer,
                rollout_idx = i,
                total_steps = D,
                config      = self.config.step_config,
                sibling_answers_provider = (
                    sibling_provider
                    if self.config.step_config.other_answer_aware else None
                ),
                on_step_complete = on_step_complete,
                resume_state     = existing,
            )
            qrec["rollouts"][i] = rollout_record
            save_cimr_store(self.store_path, self.store)

        # ── Aggregate across the N rollouts ──────────────────────────────
        agg = aggregate_rollouts(
            rollouts    = qrec["rollouts"],
            gold_answer = gold_answer,
            ev          = self.ev,
            data_name   = self.data_name,
        )
        qrec["aggregation"] = agg
        qrec["completed"]   = True
        # Final elapsed time = pre-existing baseline + this invocation's time
        prior = qrec.get("elapsed_sec_baseline", 0.0)
        qrec["elapsed_sec"] = round(float(prior) + (time.time() - t_q), 3)
        qrec.pop("elapsed_sec_baseline", None)
        save_cimr_store(self.store_path, self.store)

        return qrec


# =============================================================================
# Per-question wrapper (used by the harness)
# =============================================================================

def run_cimr_for_question(
    inferencer:  CIMRInferencer,
    question:    str,
    gold_answer: Optional[str],
    qid:         str,
) -> Dict[str, Any]:
    """Thin wrapper that delegates to ``CIMRInferencer.run_question``."""
    return inferencer.run_question(
        question=question, gold_answer=gold_answer, qid=str(qid),
    )


__all__ = [
    "SELECTION_RULE_NAMES",
    "CIMRConfig",
    "CIMRInferencer",
    "run_cimr_for_question",
    "open_cimr_store",
    "save_cimr_store",
    "cimr_store_path",
]
