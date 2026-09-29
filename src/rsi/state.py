"""
state.py
========
Slot / history store (§9 module 2) and the observable decision state S_d (§4.2).

* Exactly N slots; only a slot's CURRENT trace votes.  Earlier versions are
  kept in ``history`` for analysis only.  RESTART replaces a slot in place.
* The state never contains the gold answer; the grader lives outside.
* ``FEATURE_SCHEMA`` is the fixed vocabulary for rule / plan guards.  It
  contains only observable, problem-agnostic quantities — no problem id, no
  gold, no lookup of solved questions.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .budget import CostMeter
from .voting import plurality_vote

RECENT_WINDOW = 2

CRITIQUE_CATEGORIES = (
    "unknown", "no_specific_error", "arithmetic", "algebra", "logic",
    "assumption", "case_omission", "misread", "incomplete", "other",
)

# name -> (scope, type).  scope: slot | global.  type: num | cat
FEATURE_SCHEMA: Dict[str, tuple] = {
    # per-slot
    "depth":            ("slot", "num"),   # refinement ops applied since the slot's last (re)start
    "answer_freq":      ("slot", "num"),   # fraction of N slots sharing this slot's answer (0 if invalid)
    "answer_changes":   ("slot", "num"),   # total normalized-answer changes of this slot
    "recent_changes":   ("slot", "num"),   # answer changes over the last RECENT_WINDOW ops
    "agrees_plurality": ("slot", "num"),   # 1 if slot answer == current plurality answer
    "valid":            ("slot", "num"),   # 1 if an answer was extracted
    "critique":         ("slot", "cat"),   # latest critique category (model assertion) or "unknown"
    # global
    "budget_frac":      ("global", "num"),  # remaining / B
    "top_frac":         ("global", "num"),  # plurality count / N
    "margin":           ("global", "num"),  # (top − runner-up) / N
    "n_distinct_frac":  ("global", "num"),  # distinct valid answers / N
    "invalid_frac":     ("global", "num"),  # abstaining slots / N
}
NUM_OPS = ("<", "<=", ">", ">=", "==", "!=")
CAT_OPS = ("==", "!=", "in", "not_in")


@dataclass
class SlotVersion:
    op: str
    trace: str
    answer: str
    norm: str
    critique: str = ""
    category: str = "unknown"
    decision_idx: int = -1
    intermediate: str = ""       # g_i for FULL_REFINE


@dataclass
class Slot:
    idx: int
    trace: str = ""
    answer: str = ""
    norm: str = ""
    depth: int = 0
    answer_changes: int = 0
    change_log: List[int] = field(default_factory=list)
    frozen: bool = False
    critique_category: str = "unknown"
    restarts: int = 0
    n_ops: int = 0
    history: List[SlotVersion] = field(default_factory=list)

    def commit(self, op: str, trace: str, answer: str, norm: str, *, critique: str = "",
               category: Optional[str] = None, decision_idx: int = -1, intermediate: str = "",
               restart: bool = False) -> None:
        changed = int(norm != self.norm) if self.history else 0
        self.answer_changes += changed
        if self.history:
            self.change_log.append(changed)
        self.trace, self.answer, self.norm = trace, answer, norm
        self.n_ops += 1 if self.history else 0
        if restart:
            self.depth = 0
            self.restarts += 1
            self.critique_category = "unknown"
        elif op != "INIT":
            self.depth += 1
        if category is not None:
            self.critique_category = category
        self.history.append(SlotVersion(op, trace, answer, norm, critique,
                                        self.critique_category, decision_idx, intermediate))


@dataclass
class SolveState:
    qid: str
    question: str
    n_slots: int
    meter: CostMeter
    slots: List[Slot] = field(default_factory=list)
    initialized: bool = False
    stopped: bool = False
    stop_reason: str = ""
    n_decisions: int = 0
    call_counter: int = 0
    seed_base: int = 0
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)     # restartable per-method scratch (checkpointed)

    def __post_init__(self):
        if not self.slots:
            self.slots = [Slot(i) for i in range(self.n_slots)]

    # -- vote -------------------------------------------------------------
    def norm_answers(self) -> List[str]:
        return [s.norm for s in self.slots]

    def vote(self):
        return plurality_vote(self.norm_answers(), [s.answer for s in self.slots])

    # -- features ---------------------------------------------------------
    def global_features(self) -> Dict[str, Any]:
        N = max(self.n_slots, 1)
        _, _, counts = self.vote()
        ordered = sorted(counts.values(), reverse=True)
        top = ordered[0] if ordered else 0
        second = ordered[1] if len(ordered) > 1 else 0
        return dict(
            budget_frac=round(self.meter.remaining_frac, 6),
            top_frac=top / N,
            margin=(top - second) / N,
            n_distinct_frac=len(counts) / N,
            invalid_frac=sum(1 for s in self.slots if not s.norm) / N,
        )

    def slot_features(self, i: int, g: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        s = self.slots[i]
        N = max(self.n_slots, 1)
        winner, _, counts = self.vote()
        feats = dict(
            depth=s.depth,
            answer_freq=(counts.get(s.norm, 0) / N) if s.norm else 0.0,
            answer_changes=s.answer_changes,
            recent_changes=sum(s.change_log[-RECENT_WINDOW:]),
            agrees_plurality=int(bool(s.norm) and s.norm == winner),
            valid=int(bool(s.norm)),
            critique=s.critique_category,
        )
        feats.update(g if g is not None else self.global_features())
        return feats

    def eligible_slots(self) -> List[int]:
        return [s.idx for s in self.slots if not s.frozen]

    def copy(self) -> "SolveState":
        return copy.deepcopy(self)

    def summary(self) -> Dict[str, Any]:
        winner, disp, counts = self.vote()
        return dict(
            final_norm=winner, final_answer=disp, counts=counts,
            slot_answers=[s.answer for s in self.slots],
            slot_norms=[s.norm for s in self.slots],
            slot_depths=[s.depth for s in self.slots],
            slot_frozen=[s.frozen for s in self.slots],
            n_decisions=self.n_decisions, stop_reason=self.stop_reason,
            cost=self.meter.snapshot(),
        )


def check_condition(feats: Dict[str, Any], cond) -> bool:
    name, op, value = cond
    if name not in feats:
        return False
    x = feats[name]
    if op == "==":
        return x == value
    if op == "!=":
        return x != value
    if op == "in":
        return x in value
    if op == "not_in":
        return x not in value
    try:
        xf, vf = float(x), float(value)
    except (TypeError, ValueError):
        return False
    return {"<": xf < vf, "<=": xf <= vf, ">": xf > vf, ">=": xf >= vf}[op]


def match_all(feats: Dict[str, Any], conds) -> bool:
    return all(check_condition(feats, c) for c in conds)


__all__ = [
    "Slot", "SlotVersion", "SolveState", "FEATURE_SCHEMA", "CRITIQUE_CATEGORIES",
    "NUM_OPS", "CAT_OPS", "RECENT_WINDOW", "check_condition", "match_all",
]
