"""
AdaptSci – Early stopping and efficiency-ratio logic.

Paper reference: Section 4.3.

    β = calls / max(NMSE_t − ε_target, ε)

Stop when:
  1. NMSE_t < ε_target       (convergence)
  2. β > β*                  (marginal returns too small for remaining budget)

β* and ε_target are calibrated on a held-out validation split, or supplied
directly as constructor arguments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# Safe denominator floor (ε in the paper)
_EPS_FLOOR = 1e-10


@dataclass
class EarlyStoppingState:
    """
    Tracks per-problem stopping state.

    Parameters
    ----------
    epsilon_target : NMSE convergence threshold ε_target.
    beta_star      : efficiency ratio threshold β*.
    budget         : total call budget B for this problem.
    """
    epsilon_target: float = 1e-4
    beta_star: float = 5_000.0   # calibrated on validation split
    budget: int = 1000

    # Runtime state
    calls: int = field(default=0, init=False)
    nmse_history: List[float] = field(default_factory=list, init=False)

    def record_call(self, nmse: float) -> None:
        """Register one LLM call and the resulting NMSE."""
        self.calls += 1
        self.nmse_history.append(nmse)

    @property
    def current_nmse(self) -> float:
        if self.nmse_history:
            return min(self.nmse_history)
        return float("inf")

    @property
    def efficiency_ratio(self) -> float:
        """β = calls / max(NMSE_t − ε_target, ε)"""
        denom = max(self.current_nmse - self.epsilon_target, _EPS_FLOOR)
        return self.calls / denom

    def should_stop(self) -> bool:
        """
        Returns True when convergence or budget exhaustion is detected.
        """
        if self.current_nmse < self.epsilon_target:
            logger.debug(
                f"Early stop: convergence (NMSE={self.current_nmse:.2e} "
                f"< ε_target={self.epsilon_target:.2e})"
            )
            return True
        if self.efficiency_ratio > self.beta_star:
            logger.debug(
                f"Early stop: efficiency β={self.efficiency_ratio:.1f} "
                f"> β*={self.beta_star:.1f}"
            )
            return True
        if self.calls >= self.budget:
            logger.debug(f"Early stop: budget exhausted ({self.calls} calls).")
            return True
        return False

    def remaining_budget(self) -> int:
        return max(0, self.budget - self.calls)


def make_early_stopping(
    epsilon_target: float = 1e-4,
    beta_star: float = 5_000.0,
    budget: int = 1000,
) -> EarlyStoppingState:
    """Factory for consistent creation."""
    return EarlyStoppingState(
        epsilon_target=epsilon_target,
        beta_star=beta_star,
        budget=budget,
    )
