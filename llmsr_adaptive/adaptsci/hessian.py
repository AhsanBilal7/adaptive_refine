"""Hessian-guided optimization allocation for AdaptSci (§1.3–1.4).

Three diagnostics are computed from the Hessian H(f) of the fitting loss
at the optimum θ⋆:

  (1) Condition number   κ(f)  = λ_max(H) / (λ_min(H) + ε)
  (2) Parameter coupling C_ij  = H_ij / √(H_ii · H_jj + ε), averaged
  (3) Effective dim.    n_eff  = tr(H)² / (tr(H²) + ε)

These three signals feed the adaptive optimization policy:

  n_eff/p < eff_dim_ratio_threshold  →  REDIRECT_EXPLORE  (over-parameterized)
  coupling > coupling_threshold      →  REDIRECT_EXPLORE  (weak identifiability)
  κ > condition_number_threshold     →  EXTRA_RESTARTS    (ill-conditioned)
  otherwise                          →  WELL_CONDITIONED  (no extra cost)

The Hessian is approximated via central finite differences applied to the
same numpy/scipy loss path used during parameter fitting, so no autograd
or model access is required.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Hessian estimation
# ─────────────────────────────────────────────────────────────────────────────

def _finite_diff_hessian(
    loss_fn: Callable[[np.ndarray], float],
    theta: np.ndarray,
    h: float = 1e-4,
) -> np.ndarray:
    """Central-difference Hessian approximation at θ.

    H_ij ≈ [f(θ+hei+hej) - f(θ+hei-hej) - f(θ-hei+hej) + f(θ-hei-hej)] / 4h²
    """
    p = len(theta)
    H = np.zeros((p, p), dtype=np.float64)
    for i in range(p):
        for j in range(i, p):
            ei = np.zeros(p); ei[i] = h
            ej = np.zeros(p); ej[j] = h
            try:
                val = (loss_fn(theta + ei + ej)
                       - loss_fn(theta + ei - ej)
                       - loss_fn(theta - ei + ej)
                       + loss_fn(theta - ei - ej)) / (4.0 * h * h)
            except Exception as exc:
                logger.debug('Hessian finite-diff failed at (%d,%d): %s', i, j, exc)
                val = 0.0
            H[i, j] = val
            H[j, i] = val
    return H


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def condition_number(H: np.ndarray, eps: float = 1e-8) -> float:
    """κ(f) = λ_max / (|λ_min| + ε)."""
    try:
        ev = np.linalg.eigvalsh(H)
        return float(ev[-1]) / (abs(float(ev[0])) + eps)
    except np.linalg.LinAlgError:
        return float('inf')


def mean_coupling(H: np.ndarray, eps: float = 1e-8) -> float:
    """Mean absolute off-diagonal normalized coupling.

    C_ij = H_ij / √(H_ii · H_jj + ε).  Returns mean |C_ij| for i ≠ j.
    """
    p = H.shape[0]
    if p < 2:
        return 0.0
    sqrt_diag = np.sqrt(np.abs(np.diag(H)) + eps)
    denom = np.outer(sqrt_diag, sqrt_diag) + eps
    C     = H / denom
    idx   = np.triu_indices(p, k=1)
    return float(np.mean(np.abs(C[idx])))


def effective_dimensionality(H: np.ndarray, eps: float = 1e-8) -> float:
    """n_eff(f) = tr(H)² / (tr(H²) + ε)."""
    return float(np.trace(H)) ** 2 / (float(np.trace(H @ H)) + eps)


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics bundle
# ─────────────────────────────────────────────────────────────────────────────

class HessianDiagnostics:
    """Container for all three Hessian signals for one candidate."""

    def __init__(self, H: np.ndarray, eps: float = 1e-8):
        self.H        = H
        self.n_params = H.shape[0]
        self.kappa    = condition_number(H, eps)
        self.coupling = mean_coupling(H, eps)
        self.n_eff    = effective_dimensionality(H, eps)
        self._eps     = eps

    def eff_dim_ratio(self) -> float:
        """n_eff / p  (∈ (0, 1] for well-identified models)."""
        return self.n_eff / self.n_params if self.n_params > 0 else 1.0

    def to_dict(self) -> dict:
        return {
            'kappa':         self.kappa,
            'coupling':      self.coupling,
            'n_eff':         self.n_eff,
            'n_params':      self.n_params,
            'eff_dim_ratio': self.eff_dim_ratio(),
        }

    def __repr__(self) -> str:
        return (f'HessianDiagnostics('
                f'κ={self.kappa:.2f}, '
                f'coupling={self.coupling:.3f}, '
                f'n_eff/p={self.eff_dim_ratio():.3f})')


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive policy
# ─────────────────────────────────────────────────────────────────────────────

class OptimizationDecision:
    """Output of the adaptive policy for one candidate."""

    WELL_CONDITIONED = 'well_conditioned'
    EXTRA_RESTARTS   = 'extra_restarts'
    REDIRECT_EXPLORE = 'redirect_explore'

    def __init__(
        self,
        action: str,
        extra_restarts: int = 0,
        reason: str = '',
        diagnostics: Optional[HessianDiagnostics] = None,
    ):
        self.action         = action
        self.extra_restarts = extra_restarts
        self.reason         = reason
        self.diagnostics    = diagnostics

    def __repr__(self) -> str:
        return (f'OptimizationDecision('
                f'action={self.action}, '
                f'restarts={self.extra_restarts}, '
                f'reason={self.reason!r})')


def compute_hessian_diagnostics(
    loss_fn: Callable[[np.ndarray], float],
    theta_star: np.ndarray,
    hessian_cfg,
) -> Optional[HessianDiagnostics]:
    """Estimate Hessian and return diagnostics; returns None on failure."""
    if len(theta_star) == 0:
        return None
    try:
        H = _finite_diff_hessian(
            loss_fn,
            theta_star.astype(np.float64),
            h=hessian_cfg.finite_diff_step,
        )
        return HessianDiagnostics(H, eps=hessian_cfg.hessian_epsilon)
    except Exception as exc:
        logger.debug('Hessian computation failed: %s', exc)
        return None


def adaptive_optimization_policy(
    diag: Optional[HessianDiagnostics],
    base_score: float,
    hessian_cfg,
) -> OptimizationDecision:
    """Decide how much optimization effort to allocate to a candidate.

    Priority order
    ──────────────
    1. Diagnostics unavailable             → WELL_CONDITIONED
    2. n_eff/p < eff_dim_ratio_threshold   → REDIRECT_EXPLORE
    3. coupling > coupling_threshold       → REDIRECT_EXPLORE
    4. κ > condition_number_threshold      → EXTRA_RESTARTS (k ∝ log severity)
    5. Otherwise                           → WELL_CONDITIONED
    """
    if diag is None:
        return OptimizationDecision(
            OptimizationDecision.WELL_CONDITIONED,
            reason='diagnostics unavailable')

    if diag.eff_dim_ratio() < hessian_cfg.eff_dim_ratio_threshold:
        return OptimizationDecision(
            OptimizationDecision.REDIRECT_EXPLORE,
            reason=(f'eff_dim_ratio={diag.eff_dim_ratio():.3f} < '
                    f'threshold {hessian_cfg.eff_dim_ratio_threshold}'),
            diagnostics=diag)

    if diag.coupling > hessian_cfg.coupling_threshold:
        return OptimizationDecision(
            OptimizationDecision.REDIRECT_EXPLORE,
            reason=(f'coupling={diag.coupling:.3f} > '
                    f'threshold {hessian_cfg.coupling_threshold}'),
            diagnostics=diag)

    if diag.kappa > hessian_cfg.condition_number_threshold:
        severity  = min(
            np.log10(diag.kappa / hessian_cfg.condition_number_threshold + 1),
            np.log10(hessian_cfg.max_extra_restarts + 1))
        max_log   = np.log10(hessian_cfg.max_extra_restarts + 1)
        n_restarts = max(1, int(round(severity / max_log * hessian_cfg.max_extra_restarts)))
        n_restarts = min(n_restarts, hessian_cfg.max_extra_restarts)
        return OptimizationDecision(
            OptimizationDecision.EXTRA_RESTARTS,
            extra_restarts=n_restarts,
            reason=(f'kappa={diag.kappa:.1f} > '
                    f'threshold {hessian_cfg.condition_number_threshold}'),
            diagnostics=diag)

    return OptimizationDecision(
        OptimizationDecision.WELL_CONDITIONED,
        reason=(f'kappa={diag.kappa:.1f}, '
                f'coupling={diag.coupling:.3f}, '
                f'n_eff/p={diag.eff_dim_ratio():.3f}'),
        diagnostics=diag)
