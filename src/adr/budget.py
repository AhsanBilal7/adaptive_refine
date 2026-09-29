"""
budget.py
=========
Per-problem compute meter with a hard cap (§4.3).

* One cost measure is predeclared (``cost_measure``: tflops | tokens | calls);
  it defines the hard cap B and the utility term λ·C/B.  The other measures
  are always recorded alongside it.
* FLOPs follow the paper's §3.5 accounting: 2·P·(T_in + T_out) per call.
* A multi-call operation must ``reserve`` its worst-case cost (capped inputs +
  max_new_tokens for every call) before starting.  Spent compute is charged
  even when an operation turns out incomplete, and restarting a slot never
  restores budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class CallCost:
    n_in: int
    n_out: int
    calls: int = 1

    def measure(self, measure: str, num_params: float) -> float:
        if measure == "calls":
            return float(self.calls)
        if measure == "tokens":
            return float(self.n_in + self.n_out)
        return 2.0 * num_params * (self.n_in + self.n_out) / 1e12  # TFLOPs


@dataclass
class CostMeter:
    measure: str
    num_params: float
    budget: float                      # B, in `measure` units
    spent: float = 0.0
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    by_role: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # -- accounting ---------------------------------------------------------
    def cost_of(self, n_in: int, n_out: int, calls: int = 1) -> float:
        return CallCost(n_in, n_out, calls).measure(self.measure, self.num_params)

    def charge(self, role: str, n_in: int, n_out: int) -> float:
        c = self.cost_of(n_in, n_out)
        self.spent += c
        self.calls += 1
        self.tokens_in += int(n_in)
        self.tokens_out += int(n_out)
        r = self.by_role.setdefault(role, dict(calls=0, tokens_in=0, tokens_out=0, cost=0.0))
        r["calls"] += 1
        r["tokens_in"] += int(n_in)
        r["tokens_out"] += int(n_out)
        r["cost"] += c
        return c

    @property
    def remaining(self) -> float:
        return max(0.0, self.budget - self.spent)

    @property
    def remaining_frac(self) -> float:
        return self.remaining / self.budget if self.budget > 0 else 0.0

    def can_afford(self, reserve: float) -> bool:
        return reserve <= self.remaining + 1e-12

    @property
    def tflops(self) -> float:
        return 2.0 * self.num_params * (self.tokens_in + self.tokens_out) / 1e12

    def snapshot(self) -> Dict[str, float]:
        return dict(
            measure=self.measure, budget=self.budget, spent=self.spent,
            calls=self.calls, tokens_in=self.tokens_in, tokens_out=self.tokens_out,
            tflops=self.tflops,
        )


__all__ = ["CostMeter", "CallCost"]
