"""Smoother adaptive buffer scheduler for AdaptSci."""

from __future__ import annotations

import collections
import logging
import re
from typing import Deque, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def _tokenize_simple(code: str) -> set:
    return set(re.findall(r"[A-Za-z_]\w*|[0-9]*\.?[0-9]+|[+\-*/=<>!()\[\]{},:.^%]", code))


def token_jaccard_distance(a: str, b: str) -> float:
    ta, tb = _tokenize_simple(a), _tokenize_simple(b)
    if not ta and not tb:
        return 0.0
    union = len(ta | tb)
    return 1.0 - len(ta & tb) / union if union > 0 else 0.0


def edit_distance_normalized(a: str, b: str) -> float:
    if a == b:
        return 0.0
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 1.0
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[n] / max(m, n)


def length_bucket_distance(a: str, b: str, bucket_size: int = 50) -> float:
    return min(abs(len(a) // bucket_size - len(b) // bucket_size) / 2.0, 1.0)


_DIVERSITY_FNS = {
    "token_jaccard": token_jaccard_distance,
    "edit_distance": edit_distance_normalized,
    "length_bucket": length_bucket_distance,
}


class AdaptiveBufferScheduler:
    """Online exploit/explore scheduler with smoother updates."""

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._score_window: Deque[float] = collections.deque(maxlen=cfg.window_size)
        self._code_window: Deque[str] = collections.deque(maxlen=cfg.window_size)
        self._exploit_weight: float = (cfg.exploit_weight_max + cfg.exploit_weight_min) / 2.0
        self._iteration: int = 0
        self._policy_redirect_counter: int = 0

    def record_score(self, score: float, code: str = "") -> None:
        self._score_window.append(score)
        if code:
            self._code_window.append(code)
        self._iteration += 1
        self._update_exploit_weight()

    def record_policy_signal(self, signal: str) -> None:
        if signal == "redirect_explore":
            self._policy_redirect_counter += 1

    def exploit_weight(self) -> float:
        return self._exploit_weight

    def compute_island_weights(
        self,
        island_best_scores: List[float],
        island_top_codes: Optional[List[str]] = None,
    ) -> np.ndarray:
        n = len(island_best_scores)
        if n == 0:
            return np.array([])

        scores = np.array(island_best_scores, dtype=np.float64)

        rng = scores.max() - scores.min()
        if rng > 0:
            exploit_prob = np.exp((scores - scores.max()) / (rng + 1e-9))
        else:
            exploit_prob = np.ones(n)
        exploit_prob /= exploit_prob.sum()

        if island_top_codes and len(island_top_codes) == n and self._cfg.diversity_weight > 0:
            div_scores = self._diversity_scores(island_top_codes)
        else:
            div_scores = np.ones(n)
        explore_prob = div_scores / (div_scores.sum() + 1e-12)

        w = self._exploit_weight
        blended = w * exploit_prob + (1.0 - w) * explore_prob
        blended = np.clip(blended, 1e-9, None)
        blended /= blended.sum()
        return blended

    def state_dict(self) -> dict:
        return {
            "iteration": self._iteration,
            "exploit_weight": self._exploit_weight,
            "window_size": len(self._score_window),
            "recent_scores": list(self._score_window)[-5:],
            "redirect_explore_count": self._policy_redirect_counter,
        }

    def _update_exploit_weight(self) -> None:
        if len(self._score_window) < 4:
            return

        window = list(self._score_window)
        half = max(2, len(window) // 2)
        old_best = max(window[:half])
        new_best = max(window[half:])
        rel_improvement = (new_best - old_best) / (abs(old_best) + 1e-12)
        progress = float(np.clip(rel_improvement, -1.0, 1.0))

        cfg = self._cfg
        w_min = cfg.exploit_weight_min
        w_max = cfg.exploit_weight_max
        w_mid = 0.5 * (w_min + w_max)

        # Smooth interpolation instead of hard collapse to w_min.
        if progress >= cfg.progress_threshold:
            alpha = min(1.0, progress / max(cfg.progress_threshold, 1e-12))
            target = w_mid + alpha * (w_max - w_mid)
        elif progress >= 0.0:
            alpha = progress / max(cfg.progress_threshold, 1e-12)
            target = w_min + alpha * (w_mid - w_min)
        else:
            target = max(w_min, w_mid + progress * (w_mid - w_min))

        # If many redirect signals happen, gently encourage exploration.
        if self._policy_redirect_counter > 0:
            target = max(w_min, target - 0.05 * min(self._policy_redirect_counter, 4))
            self._policy_redirect_counter = 0

        # EMA smoothing
        self._exploit_weight = float(np.clip(0.8 * self._exploit_weight + 0.2 * target, w_min, w_max))

    def _diversity_scores(self, codes: List[str]) -> np.ndarray:
        n = len(codes)
        dist_fn = _DIVERSITY_FNS.get(self._cfg.diversity_fn, token_jaccard_distance)
        div = np.zeros(n)
        for i in range(n):
            dists = [dist_fn(codes[i], codes[j]) for j in range(n) if j != i]
            div[i] = np.mean(dists) if dists else 0.0
        return div + 1e-9
