"""
stats.py
========
Paired acceptance tests for the outer loop (§6).

* Unit of analysis = question.  Repeated seeds are averaged WITHIN a question
  first (clustered), then candidate − incumbent differences are paired across
  questions.
* Utility improvement: one-sided sign-flip permutation test of mean d_U > 0
  (exact enumeration for n ≤ 16, Monte Carlo otherwise) at α_t = α/R.
* Accuracy noninferiority: one-sided (1−α_t) lower confidence bound of the
  mean accuracy difference (percentile bootstrap over questions) must exceed
  −margin.
* ``required_block_size`` sizes a confirmation block from the pilot
  discordance rate — a tiny block cannot validate a one-point gain.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def paired_question_diffs(cand: Sequence[Dict[str, Any]], inc: Sequence[Dict[str, Any]], key: str
                          ) -> Tuple[np.ndarray, List[str]]:
    a, b = defaultdict(list), defaultdict(list)
    for r in cand:
        if r.get(key) is not None:
            a[r["qid"]].append(float(r[key]))
    for r in inc:
        if r.get(key) is not None:
            b[r["qid"]].append(float(r[key]))
    qids = sorted(set(a) & set(b))
    d = np.array([np.mean(a[q]) - np.mean(b[q]) for q in qids], dtype=float)
    return d, qids


def sign_flip_pvalue(d: np.ndarray, n_samples: int = 10000, seed: int = 0) -> float:
    """One-sided p-value for H1: mean(d) > 0."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n == 0:
        return 1.0
    obs = d.mean()
    if n <= 16:
        signs = np.array(list(itertools.product([1.0, -1.0], repeat=n)))
        null = (signs * d).mean(axis=1)
        return float(np.mean(null >= obs - 1e-12))
    rng = np.random.default_rng(seed)
    signs = rng.choice([1.0, -1.0], size=(n_samples, n))
    null = (signs * d).mean(axis=1)
    return float((1 + np.sum(null >= obs - 1e-12)) / (n_samples + 1))


def bootstrap_lower(d: np.ndarray, alpha: float, n_samples: int = 10000, seed: int = 0) -> float:
    d = np.asarray(d, dtype=float)
    if len(d) == 0:
        return -math.inf
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_samples, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.quantile(means, alpha))


def acceptance_test(cand: Sequence[Dict[str, Any]], inc: Sequence[Dict[str, Any]], alpha_t: float,
                    margin: float, min_gain: float = 0.0, n_samples: int = 10000, seed: int = 0
                    ) -> Dict[str, Any]:
    dU, qids = paired_question_diffs(cand, inc, "utility")
    dA, _ = paired_question_diffs(cand, inc, "correct")
    p = sign_flip_pvalue(dU, n_samples, seed)
    lbA = bootstrap_lower(dA, alpha_t, n_samples, seed + 1)
    utility_pass = bool(len(dU) > 0 and dU.mean() > min_gain and p < alpha_t)
    noninf_pass = bool(lbA > -margin)
    return dict(
        n_questions=len(qids), alpha_t=alpha_t, margin=margin,
        mean_utility_diff=float(dU.mean()) if len(dU) else None,
        utility_p_value=p, utility_pass=utility_pass,
        mean_accuracy_diff=float(dA.mean()) if len(dA) else None,
        accuracy_lower_bound=lbA, noninferiority_pass=noninf_pass,
        discordance_rate=float(np.mean(np.abs(dA) > 1e-9)) if len(dA) else None,
        accept=utility_pass and noninf_pass,
    )


def required_block_size(discordance: float, effect: float, alpha: float, power: float = 0.8) -> int:
    """McNemar-style n for a paired accuracy difference ``effect`` given the
    fraction of discordant questions ``discordance`` (one-sided)."""
    from statistics import NormalDist
    if effect <= 0 or discordance <= 0:
        return 0
    z = NormalDist().inv_cdf
    za, zb = z(1 - alpha), z(power)
    return int(math.ceil((za + zb) ** 2 * discordance / effect ** 2))


__all__ = ["paired_question_diffs", "sign_flip_pvalue", "bootstrap_lower", "acceptance_test",
           "required_block_size"]
