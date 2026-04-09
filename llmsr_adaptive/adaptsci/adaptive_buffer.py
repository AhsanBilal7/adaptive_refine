"""Adaptive buffer allocation for AdaptSci (§1.2).

AdaptiveBufferScheduler maintains a sliding window of recent best scores
and adjusts an exploit_weight ∈ [w_min, w_max] online:

  • Strong recent progress  → high exploit_weight (favour top-scoring islands)
  • Progress stalled        → low exploit_weight  (favour structurally diverse islands)

Island sampling probabilities are computed as a convex combination:

  p_island = w · p_exploit + (1 - w) · p_explore

where p_exploit is a softmax over island best scores and p_explore weights
islands by their average pairwise structural distance from all others.

Three diversity functions are supported (controlled by AdaptiveBufferConfig.diversity_fn):
  'token_jaccard'  – Jaccard similarity on Python token sets (default, fast)
  'edit_distance'  – normalised Levenshtein on characters (more precise)
  'length_bucket'  – coarse bucket by program length (cheapest)
"""
from __future__ import annotations

import collections
import logging
import re
from typing import Deque, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Diversity functions
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize_simple(code: str) -> set:
    return set(re.findall(
        r'[A-Za-z_]\w*|[0-9]*\.?[0-9]+|[+\-*/=<>!()\[\]{},:.^%]', code))


def token_jaccard_distance(a: str, b: str) -> float:
    """1 – Jaccard(tokens(a), tokens(b)).  ∈ [0, 1]."""
    ta, tb = _tokenize_simple(a), _tokenize_simple(b)
    if not ta and not tb:
        return 0.0
    union = len(ta | tb)
    return 1.0 - len(ta & tb) / union if union > 0 else 0.0


def edit_distance_normalized(a: str, b: str) -> float:
    """Normalised Levenshtein distance on characters.  ∈ [0, 1]."""
    if a == b:
        return 0.0
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 1.0
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost     = 0 if a[i - 1] == b[j - 1] else 1
            curr[j]  = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[n] / max(m, n)


def length_bucket_distance(a: str, b: str, bucket_size: int = 50) -> float:
    """Coarse distance based on program length buckets.  ∈ [0, 1]."""
    return min(abs(len(a) // bucket_size - len(b) // bucket_size) / 2.0, 1.0)


_DIVERSITY_FNS = {
    'token_jaccard': token_jaccard_distance,
    'edit_distance': edit_distance_normalized,
    'length_bucket': length_bucket_distance,
}


def mean_pairwise_diversity(codes: Sequence[str], fn_name: str = 'token_jaccard') -> float:
    """Average pairwise distance among a list of code strings.  ∈ [0, 1]."""
    dist_fn = _DIVERSITY_FNS.get(fn_name, token_jaccard_distance)
    if len(codes) < 2:
        return 1.0
    total, count = 0.0, 0
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            total += dist_fn(codes[i], codes[j])
            count += 1
    return total / count if count > 0 else 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveBufferScheduler:
    """Online exploit/explore scheduler.

    Usage
    ─────
    scheduler = AdaptiveBufferScheduler(cfg)

    # after each evaluation:
    scheduler.record_score(best_score_so_far, code=str(new_function))

    # inside buffer.get_prompt():
    weights = scheduler.compute_island_weights(island_best_scores, island_top_codes)
    island_id = np.random.choice(n, p=weights)
    """

    def __init__(self, cfg) -> None:
        """Args:
            cfg: AdaptiveBufferConfig instance.
        """
        self._cfg             = cfg
        self._score_window:  Deque[float] = collections.deque(maxlen=cfg.window_size)
        self._code_window:   Deque[str]   = collections.deque(maxlen=cfg.window_size)
        self._exploit_weight: float = (cfg.exploit_weight_max + cfg.exploit_weight_min) / 2.0
        self._iteration:      int   = 0

    # ── public ────────────────────────────────────────────────────────────

    def record_score(self, score: float, code: str = '') -> None:
        """Record the best score achieved after the latest evaluation."""
        self._score_window.append(score)
        if code:
            self._code_window.append(code)
        self._iteration += 1
        self._update_exploit_weight()

    def exploit_weight(self) -> float:
        """Current exploitation weight ∈ [w_min, w_max]."""
        return self._exploit_weight

    def compute_island_weights(
        self,
        island_best_scores: List[float],
        island_top_codes: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Return a probability vector over islands.

        Args:
            island_best_scores: Best score per island (higher = better).
            island_top_codes:   Best code string per island (for diversity).
                                May be None; diversity bonus is then skipped.

        Returns:
            1-D numpy array of sampling probabilities (sums to 1).
        """
        n = len(island_best_scores)
        if n == 0:
            return np.array([])

        scores = np.array(island_best_scores, dtype=np.float64)

        # exploitation component: softmax over scores
        rng = scores.max() - scores.min()
        if rng > 0:
            exploit_prob = np.exp((scores - scores.max()) / (rng + 1e-9))
        else:
            exploit_prob = np.ones(n)
        exploit_prob /= exploit_prob.sum()

        # exploration component: diversity-weighted uniform
        if (island_top_codes
                and len(island_top_codes) == n
                and self._cfg.diversity_weight > 0):
            div_scores = self._diversity_scores(island_top_codes)
        else:
            div_scores = np.ones(n)
        explore_prob = div_scores / (div_scores.sum() + 1e-12)

        # blend
        w       = self._exploit_weight
        blended = w * exploit_prob + (1.0 - w) * explore_prob
        blended = np.clip(blended, 1e-9, None)
        blended /= blended.sum()
        return blended

    def state_dict(self) -> dict:
        """Serialisable state for logging."""
        return {
            'iteration':      self._iteration,
            'exploit_weight': self._exploit_weight,
            'window_size':    len(self._score_window),
            'recent_scores':  list(self._score_window)[-5:],
        }

    # ── private ───────────────────────────────────────────────────────────

    def _update_exploit_weight(self) -> None:
        if len(self._score_window) < 2:
            return
        window = list(self._score_window)
        half   = max(1, len(window) // 2)
        old_best = max(window[:half])
        new_best = max(window[half:])
        rel_improvement = (new_best - old_best) / (abs(old_best) + 1e-12)
        progress = float(np.clip(rel_improvement, -1.0, 1.0))
        cfg = self._cfg
        if progress >= cfg.progress_threshold:
            alpha = min(1.0, progress / cfg.progress_threshold)
            self._exploit_weight = (
                cfg.exploit_weight_min
                + alpha * (cfg.exploit_weight_max - cfg.exploit_weight_min))
        else:
            self._exploit_weight = cfg.exploit_weight_min
            logger.debug(
                'Progress stalled (Δ=%.4f) → exploit_weight=%.3f',
                progress, self._exploit_weight)

    def _diversity_scores(self, codes: List[str]) -> np.ndarray:
        """Per-island average distance from all other islands."""
        n       = len(codes)
        dist_fn = _DIVERSITY_FNS.get(self._cfg.diversity_fn, token_jaccard_distance)
        div     = np.zeros(n)
        for i in range(n):
            dists  = [dist_fn(codes[i], codes[j]) for j in range(n) if j != i]
            div[i] = np.mean(dists) if dists else 0.0
        return div + 1e-9
