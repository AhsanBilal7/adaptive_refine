"""
state.py
========
Per-question state: N independent rollouts, each holding its LAST COMPLETE
solution (only that version votes; older versions are kept for analysis).
Rollouts never see each other (§3.2).
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .budget import CostMeter
from .prompts import Feedback
from .structure import Step
from .voting import plurality_vote


@dataclass
class Rollout:
    idx: int
    steps: Optional[List[Step]] = None     # None → no valid structured solution yet (malformed)
    text: str = ""                         # last complete solution (rendered steps or CIMR trace)
    answer: str = ""
    norm: str = ""
    active: bool = True
    freeze_reason: str = ""
    pending: Optional[Feedback] = None     # shared first critique (identical across direct variants)
    seen: List[str] = field(default_factory=list)   # hashes of complete states (cycle detection)
    n_turns: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class QState:
    qid: str
    question: str
    n: int
    meter: CostMeter
    rollouts: List[Rollout] = field(default_factory=list)
    rr: int = 0                            # round-robin pointer
    call_counter: int = 0
    seed_base: int = 0
    events: List[Dict[str, Any]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)
    stop_reason: str = ""
    t_start: float = field(default_factory=time.time)
    elapsed: float = 0.0                   # accumulated wall time (survives resume)

    def __post_init__(self):
        if not self.rollouts:
            self.rollouts = [Rollout(i) for i in range(self.n)]

    def vote(self):
        return plurality_vote([r.norm for r in self.rollouts], [r.answer for r in self.rollouts])

    def copy(self) -> "QState":
        return copy.deepcopy(self)


__all__ = ["Rollout", "QState"]
