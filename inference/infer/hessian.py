"""
AdaptSci – Hessian-based adaptive compute diagnostics.

Computes three geometry-grounded signals from the parameter Hessian at θ*:

    κ      – condition number        (drives adaptive BFGS restart count)
    C      – coupling matrix         (drives buffer diversity)
    n_eff  – effective param count   (over-parameterisation signal)

CRITICAL: The Hessian is computed only over the *active* parameter subspace.
SR-Scientist passes MAX_NPARAMS=10 to every equation, but most equations only
use params[0]..params[k-1] for k << 10. The unused parameters sit in perfectly
flat loss directions, giving λ_min=0, κ→∞, n_eff→1 for every equation
regardless of actual complexity.

Fix: after BFGS, detect which params moved significantly from their initial
value (|θ*_i - 1| > threshold) and restrict the Hessian to that subspace.

Reference: AdaptSci paper, Sections 3.2-3.4.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

_ACTIVE_PARAM_THRESHOLD = 1e-3
_MIN_ACTIVE = 1


# ---------------------------------------------------------------------------
# HessianDiag container
# ---------------------------------------------------------------------------

class HessianDiag:
    __slots__ = ("kappa", "C", "n_eff", "eigenvalues", "theta_star",
                 "valid", "n_active", "active_indices")

    def __init__(
        self,
        kappa: float,
        C: np.ndarray,
        n_eff: float,
        eigenvalues: np.ndarray,
        theta_star: np.ndarray,
        valid: bool = True,
        n_active: int = 0,
        active_indices: Optional[List[int]] = None,
    ):
        self.kappa = kappa
        self.C = C
        self.n_eff = n_eff
        self.eigenvalues = eigenvalues
        self.theta_star = theta_star
        self.valid = valid
        self.n_active = n_active
        self.active_indices: List[int] = active_indices or []

    @classmethod
    def invalid(cls, p: int) -> "HessianDiag":
        return cls(
            kappa=float("inf"),
            C=np.zeros((1, 1)),
            n_eff=1.0,
            eigenvalues=np.zeros(1),
            theta_star=np.ones(p),
            valid=False,
            n_active=0,
            active_indices=[],
        )

    def is_degenerate(
        self,
        coupling_threshold: float = 0.95,
        n_eff_ratio_threshold: float = 0.5,
        p: Optional[int] = None,
    ) -> bool:
        """
        Check degeneracy within the ACTIVE subspace only.
        Reference dimension is n_active, not MAX_NPARAMS.
        """
        if not self.valid:
            return True
        ref = self.n_active if self.n_active > 0 else (p or 1)
        # Off-diagonal coupling
        sz = self.C.shape[0]
        for i in range(sz):
            for j in range(sz):
                if i != j and abs(self.C[i, j]) > coupling_threshold:
                    return True
        if self.n_eff < n_eff_ratio_threshold * ref:
            return True
        return False

    def adaptive_restart_count(self, tau1: float = 100.0, tau2: float = 1_000.0) -> int:
        if not self.valid or not np.isfinite(self.kappa):
            return 1
        if self.kappa < tau1:
            return 1
        elif self.kappa < tau2:
            return 3
        else:
            return 5

    def to_dict(self) -> Dict:
        return {
            "kappa": float(self.kappa),
            "n_eff": float(self.n_eff),
            "n_active": int(self.n_active),
            "active_indices": list(self.active_indices),
            "C_frobenius_norm": float(np.linalg.norm(self.C, "fro")),
            "valid": bool(self.valid),
        }


# ---------------------------------------------------------------------------
# Active parameter detection
# ---------------------------------------------------------------------------

def _detect_active_params(
    theta_star: np.ndarray,
    x0: Optional[np.ndarray] = None,
    threshold: float = _ACTIVE_PARAM_THRESHOLD,
    min_active: int = _MIN_ACTIVE,
) -> List[int]:
    """Return indices of params that moved from x0 by more than threshold."""
    if x0 is None:
        x0 = np.ones_like(theta_star)
    displacement = np.abs(theta_star - x0)
    active = [i for i, d in enumerate(displacement) if d > threshold]
    if len(active) < min_active:
        active = list(np.argsort(displacement)[-min_active:])
    return sorted(active)


# ---------------------------------------------------------------------------
# Finite-difference Hessian in active subspace only
# ---------------------------------------------------------------------------

def _finite_diff_hessian_subspace(
    loss_fn: Callable[[np.ndarray], float],
    theta: np.ndarray,
    active_indices: List[int],
    eps: float = 1e-4,
) -> np.ndarray:
    """(n_active x n_active) Hessian via central finite differences."""
    n = len(active_indices)
    p = len(theta)
    H = np.zeros((n, n))
    for ii, gi in enumerate(active_indices):
        ei = np.zeros(p)
        ei[gi] = eps
        for jj, gj in enumerate(active_indices[ii:], start=ii):
            ej = np.zeros(p)
            ej[gj] = eps
            try:
                with np.errstate(all="ignore"):
                    fpp = loss_fn(theta + ei + ej)
                    fpm = loss_fn(theta + ei - ej)
                    fmp = loss_fn(theta - ei + ej)
                    fmm = loss_fn(theta - ei - ej)
                    h_ij = (fpp - fpm - fmp + fmm) / (4.0 * eps ** 2)
                if not np.isfinite(h_ij):
                    h_ij = 0.0
            except Exception:
                h_ij = 0.0
            H[ii, jj] = h_ij
            H[jj, ii] = h_ij
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
    """Compute κ, C, n_eff in the active parameter subspace."""
    try:
        outputs = np.array([row[0] for row in data], dtype=float)
        inputs  = np.array([row[1:] for row in data], dtype=float)
        p = max_params

        def loss(params: np.ndarray) -> float:
            try:
                with np.errstate(all="ignore"):
                    y_pred = equation_fn(*inputs.T, params)
                    val = float(np.mean((y_pred - outputs) ** 2))
                return val if np.isfinite(val) else 1e30
            except Exception:
                return 1e30

        if theta_star is None:
            res = minimize(loss, np.ones(p), method="BFGS")
            theta_star = res.x

        active_indices = _detect_active_params(theta_star)
        n_active = len(active_indices)
        logger.debug(f"Active params: {active_indices} ({n_active}/{p})")

        H = _finite_diff_hessian_subspace(loss, theta_star, active_indices, eps=eps)
        H = 0.5 * (H + H.T) + regularisation * np.eye(n_active)

        eigenvalues = np.clip(np.linalg.eigvalsh(H), 0.0, None)
        lambda_max = float(eigenvalues[-1])
        lambda_min = float(eigenvalues[0])
        kappa = (lambda_max / lambda_min) if lambda_min > 1e-12 else float("inf")

        tr_H  = float(np.trace(H))
        tr_H2 = float(np.trace(H @ H))
        n_eff = (tr_H ** 2) / tr_H2 if tr_H2 > 1e-20 else float(n_active)

        dv_sqrt = np.sqrt(np.maximum(np.diag(H), 0.0))
        outer = np.outer(dv_sqrt, dv_sqrt)
        outer = np.where(outer < 1e-20, 1.0, outer)
        C = np.clip(H / outer, -1.0, 1.0)

        return HessianDiag(
            kappa=kappa, C=C, n_eff=n_eff,
            eigenvalues=eigenvalues, theta_star=theta_star,
            valid=True, n_active=n_active, active_indices=active_indices,
        )

    except Exception as e:
        logger.warning(f"Hessian computation failed: {e}")
        return HessianDiag.invalid(max_params)


# ---------------------------------------------------------------------------
# Multi-restart BFGS + Hessian
# ---------------------------------------------------------------------------

def evaluate_with_hessian(
    equation_fn: Callable,
    data: list,
    n_restarts: int = 1,
    max_params: int = 10,
    eps: float = 1e-4,
) -> Tuple[float, float, float, "HessianDiag", np.ndarray]:
    outputs = np.array([row[0] for row in data], dtype=float)
    inputs  = np.array([row[1:] for row in data], dtype=float)

    def loss(params: np.ndarray) -> float:
        try:
            with np.errstate(all="ignore"):
                y_pred = equation_fn(*inputs.T, params)
                val = float(np.mean((y_pred - outputs) ** 2))
            return val if np.isfinite(val) else 1e30
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

    var_out = float(np.var(outputs))
    nmse = (best_mse / var_out) if var_out > 1e-20 else (0.0 if best_mse < 1e-20 else float("inf"))

    try:
        with np.errstate(all="ignore"):
            y_pred = equation_fn(*inputs.T, best_theta)
        zm  = np.isclose(outputs, 0)
        nzm = ~zm
        if np.any(nzm) and np.all(np.isfinite(y_pred[nzm])):
            mape = float(np.mean(np.abs((outputs[nzm] - y_pred[nzm]) / outputs[nzm])))
        elif np.allclose(y_pred, 0):
            mape = 0.0
        else:
            mape = float("inf")
    except Exception:
        mape = float("inf")

    hdiag = compute_hessian_diag(equation_fn, data, theta_star=best_theta,
                                  max_params=max_params, eps=eps)
    return nmse, mape, best_mse, hdiag, best_theta


# ---------------------------------------------------------------------------
# Buffer diversity helper
# ---------------------------------------------------------------------------

def coupling_distance(C1: np.ndarray, C2: np.ndarray) -> float:
    """Frobenius distance ‖C_e − C_j‖_F, zero-padding if sizes differ."""
    if C1.shape == C2.shape:
        return float(np.linalg.norm(C1 - C2, "fro"))
    n = max(C1.shape[0], C2.shape[0])
    def _pad(C: np.ndarray) -> np.ndarray:
        if C.shape[0] < n:
            padded = np.zeros((n, n))
            padded[:C.shape[0], :C.shape[0]] = C
            return padded
        return C
    return float(np.linalg.norm(_pad(C1) - _pad(C2), "fro"))