"""
cimr_inference.py
=================
CIMR orchestrator — runs N rollouts × d depth-steps per question.

Pipeline (per question)
-----------------------
  For each of N independent rollouts:
    depth 1:
      generate initial reasoning  (sampled — gives diversity)
      self-critique the trace
      self-correct the trace
      PRM-score the corrected trace

    depth 2:
      generate continuation reasoning given depth-1 corrected trace
      self-critique
      self-correct
      PRM-score

    ...

    depth d:
      → final corrected trace + final answer for this rollout

  Aggregation across N rollouts:
    majority  — modal final answer (ties broken by sum PRM then rollout idx)
    max_prm   — answer of the rollout with the highest final PRM score

Both are reported per question.

Batching
--------
Within each depth level ALL N rollouts are processed simultaneously in one
batched model call (generate / critique / correct / score all at once).
Different questions are processed sequentially; the per-depth batching gives
N× GPU utilisation compared to fully-sequential rollouts.

Save / Resume
-------------
State is persisted after every (rollout, depth) step to:
    <output_dir>/cimr_full_results_<dataset>_N<N>_d<D>.json

On re-run:
  - Fully-completed questions are skipped.
  - Partially-completed rollouts resume from their last saved depth.

Configuration
-------------
All knobs live on ``CIMRConfig``.  The depth d and rollout count N are
typically set from the shell script via CLI arguments:

    python run_paired_ablation.py --cimr_depth 4 --cimr_n_rollouts 8 ...

The output filename encodes both: cimr_full_results_<dataset>_N8_d4.json
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from tqdm import tqdm

from .cimr_step import CIMRStepConfig
from .cimr_rollout import run_all_rollouts_batched
from .cimr_aggregator import aggregate_rollouts

logger = logging.getLogger(__name__)

# Kept for backward compatibility with harnesses that import this name.
SELECTION_RULE_NAMES: List[str] = ["majority", "max_prm"]


# =============================================================================
# Config
# =============================================================================

@dataclass
class CIMRConfig:
    """All CIMR knobs in one dataclass.

    n_rollouts          : N — independent rollouts per question.
    depth               : d — self-critique/correct refinement depths per
                              rollout.  Encoded in the output filename as
                              ``_N{N}_d{D}.json``.
    step_config         : CIMRStepConfig — generation / critique / correction
                              token budgets and sampling knobs.
    rollout_temperature : sampling temperature used for depth-1 reasoning
                          generation (provides diversity across the N rollouts).
    rollout_top_p       : nucleus sampling p for depth-1 reasoning.
    """
    n_rollouts: int = 8
    depth: int = 4
    step_config: CIMRStepConfig = field(default_factory=CIMRStepConfig)
    rollout_temperature: float = 0.7
    rollout_top_p: float = 0.95


# =============================================================================
# Persistence helpers
# =============================================================================

def cimr_store_path(output_dir: str, dataset: str, N: int, D: int) -> str:
    return os.path.join(
        output_dir, f"cimr_full_results_{dataset}_N{N}_d{D}.json"
    )


def open_cimr_store(
    output_dir: str, dataset: str, N: int, D: int,
) -> Tuple[str, Dict[str, Any]]:
    """Open (or create) the CIMR results store. Returns (path, store_dict)."""
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
                f"[CIMR] Resumed {len(store)} questions from {path}"
            )
        except Exception as e:
            logger.warning(f"[CIMR] Could not load store {path}: {e}")
    else:
        logger.info(f"[CIMR] Starting fresh store at {path}")

    return path, store


def save_cimr_store(path: str, store: Dict[str, Any]) -> None:
    """Atomic save: write to .tmp then rename."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


# =============================================================================
# Per-question record helpers
# =============================================================================

def _empty_question_record(
    qid: str,
    question: str,
    gold: Optional[str],
    N: int,
    D: int,
) -> Dict[str, Any]:
    return dict(
        id=qid,
        question=question,
        gold_answer=gold,
        N=N,
        depth=D,
        rollouts=[None] * N,
        completed=False,
        aggregation=None,
        elapsed_sec=0.0,
    )


def _question_is_complete(qrec: Dict[str, Any]) -> bool:
    """True iff every rollout slot is non-None and has all D steps done."""
    rollouts = qrec.get("rollouts") or []
    if len(rollouts) < int(qrec.get("N", 0)):
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
    """Main CIMR orchestrator.

    Runs N rollouts × d steps per question.  Within each question all N
    rollouts are batched together at each depth level (one GPU pass per
    phase per depth).  State is persisted after every (rollout, depth)
    step for crash-safe resumption.
    """

    def __init__(
        self,
        model,
        verifier,
        ev,
        data_name: str,
        config: CIMRConfig,
        output_dir: str,
        dataset: str,
    ):
        self.model = model
        self.verifier = verifier
        self.ev = ev
        self.data_name = data_name
        self.config = config
        self.output_dir = output_dir
        self.dataset = dataset

        os.makedirs(output_dir, exist_ok=True)
        self.store_path, self.store = open_cimr_store(
            output_dir=output_dir,
            dataset=dataset,
            N=config.n_rollouts,
            D=config.depth,
        )

    # ------------------------------------------------------------------
    # Single-question execution
    # ------------------------------------------------------------------

    def run_question(
        self,
        question: str,
        gold_answer: Optional[str],
        qid: str,
    ) -> Dict[str, Any]:
        """Run N rollouts × D depths for one question.

        If the question is already fully complete in the store the cached
        record is returned immediately.  Otherwise missing/partial rollouts
        are computed, saved after every step, and aggregated.
        """
        N = int(self.config.n_rollouts)
        D = int(self.config.depth)

        qrec = self.store.get(str(qid))
        if qrec is None:
            qrec = _empty_question_record(str(qid), question, gold_answer, N, D)
            self.store[str(qid)] = qrec
            save_cimr_store(self.store_path, self.store)
        else:
            # Pad rollout list if N increased across runs.
            existing = list(qrec.get("rollouts", []))
            if len(existing) < N:
                qrec["rollouts"] = existing + [None] * (N - len(existing))

        # Fast-path: already fully done.
        if _question_is_complete(qrec):
            logger.debug(f"[CIMR q={qid}] already complete ({N}×{D}); skipping.")
            if qrec.get("aggregation") is None:
                qrec["aggregation"] = aggregate_rollouts(
                    rollouts=qrec.get("rollouts", []),
                    gold_answer=qrec.get("gold_answer"),
                    ev=self.ev,
                    data_name=self.data_name,
                )
                qrec["completed"] = True
                save_cimr_store(self.store_path, self.store)
            return qrec

        t_q = time.time()
        elapsed_baseline = float(qrec.get("elapsed_sec", 0.0))

        # Callback: persist after every (rollout_i, depth) step.
        def on_rollout_step_complete(
            rollout_i: int,
            running_record: Dict[str, Any],
        ) -> None:
            qrec["rollouts"][rollout_i] = running_record
            qrec["elapsed_sec"] = round(
                elapsed_baseline + (time.time() - t_q), 3
            )
            save_cimr_store(self.store_path, self.store)

        resume_states = [qrec["rollouts"][i] for i in range(N)]

        rollout_records = run_all_rollouts_batched(
            model=self.model,
            verifier=self.verifier,
            ev=self.ev,
            data_name=self.data_name,
            question=question,
            gold_answer=gold_answer,
            n_rollouts=N,
            total_steps=D,
            config=self.config.step_config,
            on_rollout_step_complete=on_rollout_step_complete,
            resume_states=resume_states,
        )

        qrec["rollouts"] = rollout_records

        agg = aggregate_rollouts(
            rollouts=qrec["rollouts"],
            gold_answer=gold_answer,
            ev=self.ev,
            data_name=self.data_name,
        )
        qrec["aggregation"] = agg
        qrec["completed"] = True
        qrec["elapsed_sec"] = round(
            elapsed_baseline + (time.time() - t_q), 3
        )
        save_cimr_store(self.store_path, self.store)

        return qrec

    # ------------------------------------------------------------------
    # Batch execution over a list of questions
    # ------------------------------------------------------------------

    def run_questions_batch(
        self,
        examples: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Run CIMR for every question in ``examples``.

        Questions are processed sequentially; within each question all N
        rollouts are batched together at each depth level.

        Each example dict should have:
            question  (or "problem")
            answer    (or "gold_answer")  — optional
            id        (or "idx")          — optional; falls back to index

        Returns a list of question-record dicts in the same order as
        ``examples``.
        """
        N = self.config.n_rollouts
        D = self.config.depth
        results: List[Dict[str, Any]] = []

        pbar = tqdm(
            examples,
            desc=f"[CIMR N={N} d={D} dataset={self.dataset}]",
            unit="q",
            dynamic_ncols=True,
            colour="cyan",
        )
        n_maj_correct = 0
        n_prm_correct = 0

        for i, ex in enumerate(pbar):
            q = ex.get("question") or ex.get("problem") or ""
            gold = ex.get("answer") or ex.get("gold_answer")
            qid = str(ex.get("id") or ex.get("idx") or i)

            try:
                qrec = self.run_question(question=q, gold_answer=gold, qid=qid)
            except Exception as e:
                logger.error(f"[CIMR] question {qid} failed: {e}", exc_info=True)
                qrec = dict(
                    id=qid, question=q, gold_answer=gold,
                    completed=False, error=str(e),
                )

            results.append(qrec)

            agg = qrec.get("aggregation") or {}
            if agg.get("majority_is_correct") is not None:
                n_maj_correct += int(bool(agg["majority_is_correct"]))
            if agg.get("max_prm_is_correct") is not None:
                n_prm_correct += int(bool(agg["max_prm_is_correct"]))

            pbar.set_postfix(
                maj=f"{n_maj_correct / (i + 1):.3f}",
                prm=f"{n_prm_correct / (i + 1):.3f}",
                done=f"{i + 1}/{len(examples)}",
                refresh=False,
            )

        pbar.close()
        return results


# =============================================================================
# Thin wrapper (used by external harnesses like run_paired_ablation.py)
# =============================================================================

def run_cimr_for_question(
    inferencer: CIMRInferencer,
    question: str,
    gold_answer: Optional[str],
    qid: str,
) -> Dict[str, Any]:
    """Thin wrapper delegating to ``CIMRInferencer.run_question``."""
    return inferencer.run_question(
        question=question,
        gold_answer=gold_answer,
        qid=str(qid),
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