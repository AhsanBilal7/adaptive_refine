"""
stats.py
========
Paired question-level uncertainty for method comparisons (updated_methodology.md §7).

Unit of analysis = question: repeated seeds are averaged within a question
first, then method A − method B differences are paired across questions.
  * two-sided sign-flip permutation p-value (exact for n ≤ 16, Monte Carlo otherwise)
  * 95% percentile-bootstrap CI of the mean paired difference
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def paired_question_diffs(a: Sequence[Dict[str, Any]], b: Sequence[Dict[str, Any]], key: str
                          ) -> Tuple[np.ndarray, List[str]]:
    va, vb = defaultdict(list), defaultdict(list)
    for r in a:
        if r.get(key) is not None:
            va[r["qid"]].append(float(r[key]))
    for r in b:
        if r.get(key) is not None:
            vb[r["qid"]].append(float(r[key]))
    qids = sorted(set(va) & set(vb))
    return np.array([np.mean(va[q]) - np.mean(vb[q]) for q in qids], dtype=float), qids


def sign_flip_pvalue(d: np.ndarray, n_samples: int = 10000, seed: int = 0) -> float:
    """Two-sided p-value for H0: mean paired difference = 0."""
    d = np.asarray(d, dtype=float)
    if len(d) == 0 or np.allclose(d, 0):
        return 1.0
    obs = abs(d.mean())
    if len(d) <= 16:
        signs = np.array(list(itertools.product([1.0, -1.0], repeat=len(d))))
        return float(np.mean(np.abs((signs * d).mean(axis=1)) >= obs - 1e-12))
    rng = np.random.default_rng(seed)
    null = np.abs((rng.choice([1.0, -1.0], size=(n_samples, len(d))) * d).mean(axis=1))
    return float((1 + np.sum(null >= obs - 1e-12)) / (n_samples + 1))


def bootstrap_ci(d: np.ndarray, level: float = 0.95, n_samples: int = 10000, seed: int = 0) -> Tuple[float, float]:
    d = np.asarray(d, dtype=float)
    if len(d) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_samples, len(d)))].mean(axis=1)
    lo = (1 - level) / 2
    return float(np.quantile(means, lo)), float(np.quantile(means, 1 - lo))


def paired_compare(a: Sequence[Dict[str, Any]], b: Sequence[Dict[str, Any]], key: str = "correct") -> Dict[str, Any]:
    d, qids = paired_question_diffs(a, b, key)
    lo, hi = bootstrap_ci(d)
    return dict(key=key, n_questions=len(qids), mean_diff=float(d.mean()) if len(d) else None,
                ci95=[lo, hi], p_value=sign_flip_pvalue(d), a_better=int((d > 1e-9).sum()),
                b_better=int((d < -1e-9).sum()), ties=int((np.abs(d) <= 1e-9).sum()))


__all__ = ["paired_question_diffs", "sign_flip_pvalue", "bootstrap_ci", "paired_compare"]
