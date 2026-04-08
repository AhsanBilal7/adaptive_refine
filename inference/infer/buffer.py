"""
AdaptSci – Experience Buffer with stratified 70/30 seed selection.

Replaces SR-Scientist's pure top-K exploitation buffer with:

  • 70% exploitation pool  – lowest training NMSE (maintains quality)
  • 30% frontier pool      – highest frontier score s_fr(e) (prevents monoculture)

Frontier score (Section 4.2):
    s_fr(e) = α · |ΔNMSE|_w / calls_w  +  (1−α) · min_{j∈S} ‖C_e − C_j‖_F

When the skeleton is degenerate (Section 4.2 "diversity-only mode"):
    α ← 0   →  s_fr = C-diversity only

Paper reference: AdaptSci, Section 4.2.
"""

from __future__ import annotations

import math
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .hessian import HessianDiag, coupling_distance

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EquationRecord – one entry in the buffer
# ---------------------------------------------------------------------------

@dataclass
class EquationRecord:
    """All information stored per discovered equation."""
    equation: str                    # Python source string
    nmse: float                      # NMSE on training data
    mape: float                      # MAPE on training data
    mse: float                       # MSE on training data
    hdiag: HessianDiag               # Hessian diagnostics at θ*
    call_index: int = 0              # global call number when this was added
    # Rolling NMSE history for momentum computation
    nmse_history: deque = field(default_factory=lambda: deque(maxlen=5))
    calls_history: deque = field(default_factory=lambda: deque(maxlen=5))

    def __post_init__(self):
        self.nmse_history.append(self.nmse)
        self.calls_history.append(self.call_index)

    def update(self, new_nmse: float, new_call_index: int) -> None:
        self.nmse = new_nmse
        self.call_index = new_call_index
        self.nmse_history.append(new_nmse)
        self.calls_history.append(new_call_index)

    @property
    def coupling_matrix(self) -> np.ndarray:
        return self.hdiag.C

    def recent_improvement_rate(self, window: int = 3) -> float:
        """
        |ΔNMSE|_w / calls_w  – NMSE reduction per call over last w iterations.
        Higher ⇒ more momentum, worth continuing to explore this family.
        """
        hist = list(self.nmse_history)
        c_hist = list(self.calls_history)
        if len(hist) < 2:
            return 0.0
        w_hist = hist[-min(window, len(hist)):]
        w_calls = c_hist[-min(window, len(c_hist)):]
        delta_nmse = abs(w_hist[0] - w_hist[-1])
        delta_calls = max(abs(w_calls[-1] - w_calls[0]), 1)
        return delta_nmse / delta_calls


# ---------------------------------------------------------------------------
# AdaptSciBuffer
# ---------------------------------------------------------------------------

class AdaptSciBuffer:
    """
    Stratified 70/30 experience buffer.

    Parameters
    ----------
    k          : total seed size returned by `select_seeds()`.
    alpha      : weight on improvement-rate vs C-diversity in frontier score.
                 α=0 → diversity only;  α=1 → improvement only.
    window     : number of recent iterations for momentum computation.
    exploit_frac : fraction of K devoted to exploitation pool (default 0.70).
    """

    def __init__(
        self,
        k: int = 3,
        alpha: float = 0.5,
        window: int = 3,
        exploit_frac: float = 0.70,
    ):
        self.k = k
        self.alpha = alpha
        self.window = window
        self.exploit_frac = exploit_frac
        self._records: List[EquationRecord] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, record: EquationRecord) -> None:
        """Add a new record.  Duplicate equations are updated in place."""
        for existing in self._records:
            if existing.equation.strip() == record.equation.strip():
                existing.update(record.nmse, record.call_index)
                existing.hdiag = record.hdiag
                return
        self._records.append(record)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def top_k_by_nmse(self, k: int) -> List[EquationRecord]:
        """Pure exploitation – lowest NMSE."""
        valid = [r for r in self._records if np.isfinite(r.nmse)]
        return sorted(valid, key=lambda r: r.nmse)[:k]

    def select_seeds(
        self,
        alpha_override: Optional[float] = None,
    ) -> List[EquationRecord]:
        """
        Stratified 70/30 seed selection (Section 4.2).

        Returns up to `self.k` records:
          • ⌊0.7K⌋ exploitation seeds (lowest NMSE)
          • ⌈0.3K⌉ frontier seeds     (highest s_fr)
        """
        if len(self._records) == 0:
            return []

        alpha = alpha_override if alpha_override is not None else self.alpha
        n_exploit = max(1, math.floor(self.exploit_frac * self.k))
        n_frontier = max(0, self.k - n_exploit)

        # --- Exploitation pool ---
        exploit_pool = self.top_k_by_nmse(n_exploit)

        if n_frontier == 0 or len(self._records) <= n_exploit:
            return exploit_pool

        # --- Frontier pool ---
        exploit_set = set(id(r) for r in exploit_pool)
        candidates = [r for r in self._records if id(r) not in exploit_set and np.isfinite(r.nmse)]

        if not candidates:
            return exploit_pool

        # Compute frontier scores
        frontier_scores: List[Tuple[float, EquationRecord]] = []
        for cand in candidates:
            rate = cand.recent_improvement_rate(self.window)
            # C-diversity: min Frobenius distance to any seed in exploit pool
            div = min(
                (coupling_distance(cand.coupling_matrix, seed.coupling_matrix) for seed in exploit_pool),
                default=0.0,
            )
            score = alpha * rate + (1.0 - alpha) * div
            frontier_scores.append((score, cand))

        frontier_scores.sort(key=lambda x: x[0], reverse=True)
        frontier_pool = [r for _, r in frontier_scores[:n_frontier]]

        return exploit_pool + frontier_pool

    def best(self) -> Optional[EquationRecord]:
        """Single best record by NMSE."""
        tops = self.top_k_by_nmse(1)
        return tops[0] if tops else None

    def __len__(self) -> int:
        return len(self._records)

    def formatted_context(
        self,
        k: int,
        alpha_override: Optional[float] = None,
        include_hessian_info: bool = False,
    ) -> str:
        """
        Returns a formatted string of top-K seeds for injection into the
        LLM prompt (mirrors SR-Scientist's `_create_previous_turn_context`).
        """
        seeds = self.select_seeds(alpha_override=alpha_override)[:k]
        if not seeds:
            return ""

        lines = [
            "\n\nIn the previous turn(s), some equations have been explored.",
            "Their performance (MAPE) is listed below, sorted from lowest to highest error.",
            "Use these as inspiration to design a better equation.",
            "\n--- Previously Explored Equations ---",
        ]
        for rec in seeds:
            hline = ""
            if include_hessian_info and rec.hdiag.valid:
                hline = (
                    f"  [κ={rec.hdiag.kappa:.1f}, "
                    f"n_eff={rec.hdiag.n_eff:.1f}, "
                    f"degenerate={rec.hdiag.is_degenerate()}]"
                )
            lines.append(f"\n# MAPE: {rec.mape:.4%}{hline}\n{rec.equation.strip()}")

        lines.append("\n-------------------------------------\n")
        return "\n".join(lines)
