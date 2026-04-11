"""Multi-island experience buffer with adaptive island sampling for AdaptSci.

Implements the islands model from LLM-SR §2.4 / App. B:
  • Boltzmann cluster sampling (temperature-annealed)
  • Short-program bias within clusters
  • Periodic reset of the weakest half of islands

AdaptSci extension:
  get_prompt(scheduler=…) accepts an optional AdaptiveBufferScheduler.
  When provided, island selection weights come from the scheduler instead
  of uniform random, enabling the exploit/explore adaptive balance (§1.2).
"""
from __future__ import annotations

import dataclasses
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from adaptsci import code_manipulation
from adaptsci import config as config_lib
from adaptsci.adaptive_buffer import AdaptiveBufferScheduler


# ─────────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class Prompt:
    """A versioned LLM prompt ready for dispatch."""
    code: str
    version_generated: int
    island_id: int


# ─────────────────────────────────────────────────────────────────────────────
# Island
# ─────────────────────────────────────────────────────────────────────────────

class _Island:
    """One island: Boltzmann cluster sampling + short-program bias."""

    def __init__(self) -> None:
        # clusters: score_bucket → [(score, Function), …]
        self._clusters: Dict[float, List[Tuple[float, code_manipulation.Function]]] = {}
        self._best_score: float = -1e9
        self._best_fn: Optional[code_manipulation.Function] = None
        self._total: int = 0

    # ── population management ─────────────────────────────────────────────

    def register(self, fn: code_manipulation.Function, score: float) -> None:
        bucket = round(score, 3)
        self._clusters.setdefault(bucket, []).append((score, fn))
        self._total += 1
        if score > self._best_score:
            self._best_score = score
            self._best_fn    = fn

    def best_score(self) -> float:
        return self._best_score

    def best_function(self) -> Optional[code_manipulation.Function]:
        return self._best_fn

    def top_code(self) -> str:
        return str(self._best_fn) if self._best_fn else ''

    def num_programs(self) -> int:
        return self._total

    def reset(self, seed_fn: Optional[code_manipulation.Function] = None) -> None:
        self._clusters = {}
        self._total    = 0
        self._best_score = -1e9
        self._best_fn    = None
        if seed_fn is not None:
            self.register(seed_fn, score=0.0)

    # ── sampling ──────────────────────────────────────────────────────────

    def sample_functions(
        self,
        k: int,
        temperature_init: float,
        temperature_period: int,
    ) -> List[code_manipulation.Function]:
        if not self._clusters:
            return []
        buckets = list(self._clusters.keys())
        means   = np.array([
            np.mean([s for s, _ in self._clusters[b]]) for b in buckets
        ], dtype=np.float64)
        lp = means / (temperature_init + 1e-12)
        lp -= lp.max()
        probs = np.exp(lp)
        probs /= probs.sum()
        out = []
        for _ in range(k):
            idx = np.random.choice(len(buckets), p=probs)
            fn  = self._sample_short(self._clusters[buckets[idx]])
            if fn is not None:
                out.append(fn)
        return out

    @staticmethod
    def _sample_short(
        candidates: List[Tuple[float, code_manipulation.Function]],
        tau: float = 2.5,
    ) -> Optional[code_manipulation.Function]:
        if not candidates:
            return None
        lengths = np.array([len(str(f)) for _, f in candidates], dtype=np.float64)
        rng = lengths.max() - lengths.min()
        norm = (lengths - lengths.min()) / (rng + 1e-9) if rng > 0 else np.zeros_like(lengths)
        lp = -0.35 * norm / (tau + 1e-12)  # much weaker simplicity bias
        lp -= lp.max()
        probs = np.exp(lp)
        probs /= probs.sum()
        return candidates[np.random.choice(len(candidates), p=probs)][1]



# ─────────────────────────────────────────────────────────────────────────────
# ExperienceBuffer
# ─────────────────────────────────────────────────────────────────────────────

class ExperienceBuffer:
    """Multi-island experience buffer with adaptive island sampling support."""

    def __init__(
        self,
        config: config_lib.ExperienceBufferConfig,
        template: code_manipulation.Program,
        function_to_evolve: str,
    ) -> None:
        self._config            = config
        self._template          = template
        self._function_to_evolve = function_to_evolve
        self._function_version  = 0
        self._lock              = threading.Lock()
        self._last_reset        = time.time()

        self._islands: List[_Island] = [
            _Island() for _ in range(config.num_islands)
        ]
        # seed all islands with the initial template body
        init_fn = template.get_function(function_to_evolve)
        for isl in self._islands:
            isl.register(init_fn, score=0.0)

    # ── public ────────────────────────────────────────────────────────────

    def get_prompt(
        self,
        scheduler: Optional[AdaptiveBufferScheduler] = None,
    ) -> Prompt:
        """Build a versioned few-shot prompt.

        Args:
            scheduler: When provided, island selection uses adaptive
                exploit/explore weights from the scheduler.  When None,
                a uniformly random island is chosen (plain LLM-SR).
        """
        with self._lock:
            self._maybe_reset_islands()
            island_id = self._choose_island(scheduler)
            island    = self._islands[island_id]
            examples  = island.sample_functions(
                k=self._config.functions_per_prompt,
                temperature_init=self._config.cluster_sampling_temperature_init,
                temperature_period=self._config.cluster_sampling_temperature_period,
            )
            self._function_version += 1
            prompt_str = self._build_prompt(examples, self._function_version)

        return Prompt(
            code=prompt_str,
            version_generated=self._function_version,
            island_id=island_id,
        )

    def register_program(
        self,
        function: code_manipulation.Function,
        island_id: Optional[int],
        scores_per_test: Dict[str, float],
        **kwargs,
    ) -> None:
        """Score and register a function into the buffer."""
        if not scores_per_test:
            return
        score = float(np.mean(list(scores_per_test.values())))

        function.global_sample_nums = kwargs.get('global_sample_nums', None)
        function.score              = score
        function.sample_time        = kwargs.get('sample_time', None)
        function.evaluate_time      = kwargs.get('evaluate_time', None)

        with self._lock:
            target = island_id if island_id is not None else self._best_island_id()
            self._islands[target].register(function, score)

        profiler = kwargs.get('profiler', None)
        if profiler is not None:
            profiler.register_function(function)

    def best_score_per_island(self) -> List[float]:
        return [isl.best_score() for isl in self._islands]

    def top_code_per_island(self) -> List[str]:
        return [isl.top_code() for isl in self._islands]

    def global_best(self) -> Tuple[float, Optional[code_manipulation.Function]]:
        return max(
            ((isl.best_score(), isl.best_function()) for isl in self._islands),
            key=lambda x: x[0])

    # ── private ───────────────────────────────────────────────────────────

    def _choose_island(
        self,
        scheduler: Optional[AdaptiveBufferScheduler],
    ) -> int:
        n = len(self._islands)
        if scheduler is not None:
            weights = scheduler.compute_island_weights(
                self.best_score_per_island(),
                self.top_code_per_island(),
            )
            return int(np.random.choice(n, p=weights))
        return int(np.random.randint(n))

    def _best_island_id(self) -> int:
        return int(np.argmax([isl.best_score() for isl in self._islands]))

    def _maybe_reset_islands(self) -> None:
        if time.time() - self._last_reset < self._config.reset_period:
            return
        self._last_reset = time.time()
        n    = len(self._islands)
        half = n // 2
        ranked   = sorted(range(n), key=lambda i: self._islands[i].best_score())
        survivors = ranked[half:]
        for weak_id in ranked[:half]:
            donor_id = np.random.choice(survivors)
            self._islands[weak_id].reset(
                seed_fn=self._islands[donor_id].best_function())

    def _build_prompt(
        self,
        examples: List[code_manipulation.Function],
        version: int,
    ) -> str:
        lines = [str(self._template.preface)] if self._template.preface else []
        for i, ex in enumerate(examples):
            versioned = code_manipulation.Function(
                name=f'{self._function_to_evolve}_v{i}',
                args=ex.args,
                body=ex.body,
                return_type=ex.return_type,
                docstring=ex.docstring,
            )
            lines.append(str(versioned))
        tf = self._template.get_function(self._function_to_evolve)
        lines.append(
            f'def {self._function_to_evolve}_v{version}({tf.args}):\n'
            f'    """{self._function_to_evolve} (improved version)"""\n')
        return '\n'.join(lines)
