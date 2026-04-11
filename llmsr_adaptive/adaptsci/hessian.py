"""More stable Hessian diagnostics for AdaptSci."""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _finite_diff_hessian(
    loss_fn: Callable[[np.ndarray], float],
    theta: np.ndarray,
    h: float = 1e-4,
    relative_step: bool = True,
) -> np.ndarray:
    """Central-difference Hessian with optional parameter-relative step sizes."""
    p = len(theta)
    H = np.zeros((p, p), dtype=np.float64)

    steps = np.full(p, h, dtype=np.float64)
    if relative_step:
        steps = h * np.maximum(1.0, np.abs(theta))

    for i in range(p):
        for j in range(i, p):
            ei = np.zeros(p, dtype=np.float64)
            ej = np.zeros(p, dtype=np.float64)
            ei[i] = steps[i]
            ej[j] = steps[j]
            denom = 4.0 * steps[i] * steps[j]
            try:
                val = (
                    loss_fn(theta + ei + ej)
                    - loss_fn(theta + ei - ej)
                    - loss_fn(theta - ei + ej)
                    + loss_fn(theta - ei - ej)
                ) / denom
            except Exception as exc:
                logger.debug("Hessian finite-diff failed at (%d,%d): %s", i, j, exc)
                val = 0.0
            H[i, j] = val
            H[j, i] = val
    return H


def condition_number(H: np.ndarray, eps: float = 1e-8) -> float:
    """Stable condition number using absolute eigenvalues."""
    try:
        ev = np.linalg.eigvalsh(H)
        abs_ev = np.abs(ev)
        return float(abs_ev[-1]) / (float(abs_ev[0]) + eps)
    except np.linalg.LinAlgError:
        return float("inf")


def mean_coupling(H: np.ndarray, eps: float = 1e-8) -> float:
    p = H.shape[0]
    if p < 2:
        return 0.0
    sqrt_diag = np.sqrt(np.abs(np.diag(H)) + eps)
    denom = np.outer(sqrt_diag, sqrt_diag) + eps
    C = H / denom
    idx = np.triu_indices(p, k=1)
    return float(np.mean(np.abs(C[idx])))


def effective_dimensionality(H: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.trace(H)) ** 2 / (float(np.trace(H @ H)) + eps)


class HessianDiagnostics:
    def __init__(self, H: np.ndarray, eps: float = 1e-8):
        self.H = H
        self.n_params = H.shape[0]
        self.kappa = condition_number(H, eps)
        self.coupling = mean_coupling(H, eps)
        self.n_eff = effective_dimensionality(H, eps)
        self._eps = eps

    def eff_dim_ratio(self) -> float:
        return self.n_eff / self.n_params if self.n_params > 0 else 1.0

    def to_dict(self) -> dict:
        return {
            "kappa": self.kappa,
            "coupling": self.coupling,
            "n_eff": self.n_eff,
            "n_params": self.n_params,
            "eff_dim_ratio": self.eff_dim_ratio(),
        }

    def __repr__(self) -> str:
        return (
            f"HessianDiagnostics(κ={self.kappa:.2f}, "
            f"coupling={self.coupling:.3f}, "
            f"n_eff/p={self.eff_dim_ratio():.3f})"
        )


class OptimizationDecision:
    WELL_CONDITIONED = "well_conditioned"
    EXTRA_RESTARTS = "extra_restarts"
    REDIRECT_EXPLORE = "redirect_explore"

    def __init__(
        self,
        action: str,
        extra_restarts: int = 0,
        reason: str = "",
        diagnostics: Optional[HessianDiagnostics] = None,
    ):
        self.action = action
        self.extra_restarts = extra_restarts
        self.reason = reason
        self.diagnostics = diagnostics


def compute_hessian_diagnostics(
    loss_fn: Callable[[np.ndarray], float],
    theta_star: np.ndarray,
    hessian_cfg,
) -> Optional[HessianDiagnostics]:
    if len(theta_star) == 0:
        return None
    try:
        H = _finite_diff_hessian(
            loss_fn,
            theta_star.astype(np.float64),
            h=hessian_cfg.finite_diff_step,
            relative_step=True,
        )
        return HessianDiagnostics(H, eps=hessian_cfg.hessian_epsilon)
    except Exception as exc:
        logger.debug("Hessian computation failed: %s", exc)
        return None


def adaptive_optimization_policy(
    diag: Optional[HessianDiagnostics],
    base_score: float,
    hessian_cfg,
) -> OptimizationDecision:
    """More conservative policy than the original.

    Important change:
    - Never use REDIRECT_EXPLORE just because diagnostics are slightly bad.
    - Use redirection only for clearly degenerate cases.
    """
    if diag is None:
        return OptimizationDecision(
            OptimizationDecision.WELL_CONDITIONED,
            reason="diagnostics unavailable",
        )

    if diag.eff_dim_ratio() < 0.5 * hessian_cfg.eff_dim_ratio_threshold and diag.coupling > 1.1 * hessian_cfg.coupling_threshold:
        return OptimizationDecision(
            OptimizationDecision.REDIRECT_EXPLORE,
            reason=(
                f"very low eff_dim_ratio={diag.eff_dim_ratio():.3f} and "
                f"high coupling={diag.coupling:.3f}"
            ),
            diagnostics=diag,
        )

    if diag.kappa > hessian_cfg.condition_number_threshold:
        severity = min(
            np.log10(diag.kappa / hessian_cfg.condition_number_threshold + 1.0),
            np.log10(hessian_cfg.max_extra_restarts + 1.0),
        )
        max_log = np.log10(hessian_cfg.max_extra_restarts + 1.0)
        n_restarts = max(1, int(round(severity / max_log * hessian_cfg.max_extra_restarts)))
        n_restarts = min(n_restarts, hessian_cfg.max_extra_restarts)
        return OptimizationDecision(
            OptimizationDecision.EXTRA_RESTARTS,
            extra_restarts=n_restarts,
            reason=f"kappa={diag.kappa:.1f} > threshold {hessian_cfg.condition_number_threshold}",
            diagnostics=diag,
        )

    return OptimizationDecision(
        OptimizationDecision.WELL_CONDITIONED,
        reason=(
            f"kappa={diag.kappa:.1f}, "
            f"coupling={diag.coupling:.3f}, "
            f"n_eff/p={diag.eff_dim_ratio():.3f}"
        ),
        diagnostics=diag,
    )
