"""
AdaptSci – Hessian-based adaptive compute diagnostics.

For a candidate equation skeleton fitted to training data via BFGS, this module
computes three geometry-grounded signals from the parameter Hessian at θ*:

    κ      – condition number        (drives adaptive BFGS restart count)
    C      – coupling matrix         (drives buffer diversity / monoculture detection)
    n_eff  – effective param count   (over-parameterisation signal)

All computations are purely numerical (p×p ops) and are NOT counted against
the LLM-call budget.

Reference: AdaptSci paper, Sections 3.2-3.4.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HessianDiag container
# ---------------------------------------------------------------------------

class HessianDiag:
    """Container for all three Hessian-derived signals plus raw eigenvalues."""

    __slots__ = ("kappa", "C", "n_eff", "eigenvalues", "theta_star", "valid")

    def __init__(
        self,
        kappa: float,
        C: np.ndarray,
        n_eff: float,
        eigenvalues: np.ndarray,
        theta_star: np.ndarray,
        valid: bool = True,
    ):
        self.kappa: float = kappa
        self.C: np.ndarray = C              # shape (p, p)
        self.n_eff: float = n_eff
        self.eigenvalues: np.ndarray = eigenvalues
        self.theta_star: np.ndarray = theta_star
        self.valid: bool = valid

    @classmethod
    def invalid(cls, p: int) -> "HessianDiag":
        """Safe sentinel when Hessian computation fails."""
        return cls(
            kappa=float("inf"),
            C=np.zeros((p, p)),
            n_eff=float(p),
            eigenvalues=np.zeros(p),
            theta_star=np.ones(p),
            valid=False,
        )

    # ------------------------------------------------------------------
    # Signals → decisions
    # ------------------------------------------------------------------

    def is_degenerate(
        self,
        coupling_threshold: float = 0.95,
        n_eff_ratio_threshold: float = 0.5,
        p: Optional[int] = None,
    ) -> bool:
        """
        True if structurally degenerate:
          • any |C_ij| > coupling_threshold for i≠j  (entangled params), OR
          • n_eff < n_eff_ratio_threshold * p         (too many flat directions).

        Paper reference: Section 4.2 "When neff ≪ p or |Cij| > 0.95 … force exploration".
        """
        if not self.valid:
            return True
        _p = p if p is not None else self.C.shape[0]
        # Off-diagonal coupling
        p_size = self.C.shape[0]
        for i in range(p_size):
            for j in range(p_size):
                if i != j and abs(self.C[i, j]) > coupling_threshold:
                    return True
        if self.n_eff < n_eff_ratio_threshold * _p:
            return True
        return False

    def adaptive_restart_count(
        self,
        tau1: float = 100.0,
        tau2: float = 1_000.0,
    ) -> int:
        """
        Map κ to number of BFGS restarts (Section 3.2):
            κ < τ1          → 1
            τ1 ≤ κ < τ2    → 3
            κ ≥ τ2          → 5
        Thresholds τ1, τ2 may be calibrated on a validation split; defaults
        are sensible priors from the AdaptSci paper.
        """
        if self.kappa < tau1:
            return 1
        elif self.kappa < tau2:
            return 3
        else:
            return 5

    def to_dict(self) -> Dict:
        """Serialisable summary for logging / JSON output."""
        return {
            "kappa": float(self.kappa),
            "n_eff": float(self.n_eff),
            "C_frobenius_norm": float(np.linalg.norm(self.C, "fro")),
            "valid": bool(self.valid),
        }


# ---------------------------------------------------------------------------
# Numerical Hessian via central finite differences
# ---------------------------------------------------------------------------

def _finite_diff_hessian(
    loss_fn: Callable[[np.ndarray], float],
    theta: np.ndarray,
    eps: float = 1e-4,
) -> np.ndarray:
    """
    Second-order central finite-difference Hessian (symmetric, O(eps²)):

      H_ij = [L(θ+eᵢ+eⱼ) − L(θ+eᵢ−eⱼ) − L(θ−eᵢ+eⱼ) + L(θ−eᵢ−eⱼ)] / (4·eps²)
    """
    p = len(theta)
    H = np.zeros((p, p))
    for i in range(p):
        ei = np.zeros(p)
        ei[i] = eps
        for j in range(i, p):
            ej = np.zeros(p)
            ej[j] = eps
            try:
                fpp = loss_fn(theta + ei + ej)
                fpm = loss_fn(theta + ei - ej)
                fmp = loss_fn(theta - ei + ej)
                fmm = loss_fn(theta - ei - ej)
                h_ij = (fpp - fpm - fmp + fmm) / (4.0 * eps ** 2)
            except Exception:
                h_ij = 0.0
            H[i, j] = h_ij
            H[j, i] = h_ij
    return H


# ---------------------------------------------------------------------------
# Core public function
# ---------------------------------------------------------------------------

def compute_hessian_diag(
    equation_fn: Callable,
    data: list,
    theta_star: Optional[np.ndarray] = None,
    max_params: int = 10,
    eps: float = 1e-4,
    regularisation: float = 1e-8,
) -> HessianDiag:
    """
    Compute κ, C, n_eff for a fitted equation skeleton.

    Parameters
    ----------
    equation_fn    : Python callable `equation(*inputs.T, params) → np.ndarray`.
    data           : list of rows [[output, x1, x2, ...], ...] (training split only).
    theta_star     : pre-fitted parameter vector.  If None, one BFGS run is done.
    max_params     : MAX_NPARAMS used in the SR system (default 10).
    eps            : finite-difference step size.
    regularisation : ridge added to H diagonal for numerical stability.

    Returns
    -------
    HessianDiag
    """
    try:
        outputs = np.array([row[0] for row in data], dtype=float)
        inputs  = np.array([row[1:] for row in data], dtype=float)
        p = max_params

        def loss(params: np.ndarray) -> float:
            try:
                y_pred = equation_fn(*inputs.T, params)
                return float(np.mean((y_pred - outputs) ** 2))
            except Exception:
                return 1e30

        # Fit θ* if not provided
        if theta_star is None:
            res = minimize(loss, np.ones(p), method="BFGS")
            theta_star = res.x

        # Hessian via finite differences
        H = _finite_diff_hessian(loss, theta_star, eps=eps)

        # Symmetrise + regularise
        H = 0.5 * (H + H.T) + regularisation * np.eye(p)

        # Eigenvalues (ascending order, non-negative)
        eigenvalues = np.linalg.eigvalsh(H)
        eigenvalues = np.clip(eigenvalues, 0.0, None)

        lambda_max = float(eigenvalues[-1])
        lambda_min = float(eigenvalues[0])

        # κ: condition number
        kappa = (lambda_max / lambda_min) if lambda_min > 1e-12 else float("inf")

        # n_eff: participation ratio  (tr H)² / tr(H²)
        tr_H  = float(np.trace(H))
        tr_H2 = float(np.trace(H @ H))
        n_eff = (tr_H ** 2) / tr_H2 if tr_H2 > 1e-20 else float(p)

        # C: normalised off-diagonal Hessian (coupling matrix)
        diag_vals = np.diag(H)
        # Safe outer product of sqrt diagonal
        dv_sqrt = np.sqrt(np.maximum(diag_vals, 0.0))
        outer = np.outer(dv_sqrt, dv_sqrt)
        outer = np.where(outer < 1e-20, 1.0, outer)
        C = H / outer
        C = np.clip(C, -1.0, 1.0)

        return HessianDiag(
            kappa=kappa,
            C=C,
            n_eff=n_eff,
            eigenvalues=eigenvalues,
            theta_star=theta_star,
            valid=True,
        )

    except Exception as e:
        logger.warning(f"Hessian computation failed: {e}")
        return HessianDiag.invalid(max_params)


# ---------------------------------------------------------------------------
# Multi-restart BFGS with integrated Hessian computation  (T2 wrapper)
# ---------------------------------------------------------------------------

def evaluate_with_hessian(
    equation_fn: Callable,
    data: list,
    n_restarts: int = 1,
    max_params: int = 10,
    eps: float = 1e-4,
) -> Tuple[float, float, float, HessianDiag, np.ndarray]:
    """
    Run `n_restarts` BFGS initialisations, keep the best result,
    compute Hessian at θ*.  (Section 4.1 "Adaptive Optimisation Effort")

    Returns
    -------
    (nmse, mape, best_mse, HessianDiag, best_theta)
    """
    outputs = np.array([row[0] for row in data], dtype=float)
    inputs  = np.array([row[1:] for row in data], dtype=float)

    def loss(params: np.ndarray) -> float:
        try:
            y_pred = equation_fn(*inputs.T, params)
            return float(np.mean((y_pred - outputs) ** 2))
        except Exception:
            return 1e30

    best_mse   = float("inf")
    best_theta = np.ones(max_params)
    rng = np.random.default_rng(42)

    for r in range(n_restarts):
        x0 = np.ones(max_params) if r == 0 else rng.normal(0, 3.0, max_params)
        try:
            res = minimize(loss, x0, method="BFGS")
            if np.isfinite(res.fun) and res.fun < best_mse:
                best_mse   = res.fun
                best_theta = res.x
        except Exception:
            continue

    # NMSE
    var_out = float(np.var(outputs))
    nmse = (best_mse / var_out) if var_out > 1e-20 else (0.0 if best_mse < 1e-20 else float("inf"))

    # MAPE
    try:
        y_pred = equation_fn(*inputs.T, best_theta)
        zm = np.isclose(outputs, 0)
        nzm = ~zm
        if np.any(nzm):
            mape = float(np.mean(np.abs((outputs[nzm] - y_pred[nzm]) / outputs[nzm])))
        elif np.allclose(y_pred, 0):
            mape = 0.0
        else:
            mape = float("inf")
    except Exception:
        mape = float("inf")

    hdiag = compute_hessian_diag(
        equation_fn, data, theta_star=best_theta,
        max_params=max_params, eps=eps,
    )

    return nmse, mape, best_mse, hdiag, best_theta


# ---------------------------------------------------------------------------
# Buffer diversity helper
# ---------------------------------------------------------------------------

def coupling_distance(C1: np.ndarray, C2: np.ndarray) -> float:
    """Frobenius distance ‖C_e − C_j‖_F between two coupling matrices."""
    return float(np.linalg.norm(C1 - C2, "fro"))
