"""AdaptSci configuration.

All configuration dataclasses for one AdaptSci / LLM-SR experiment.

Hierarchy
─────────
  ExperienceBufferConfig  – multi-island experience buffer (LLM-SR §2.4)
  HessianConfig           – Hessian-guided optimization allocation (§1.3–1.4)
  AdaptiveBufferConfig    – adaptive exploit/explore scheduling (§1.2)
  Config                  – top-level config, includes all three above
  ClassConfig             – dependency injection (LLM + Sandbox classes)

ablation_mode controls which AdaptSci components are active:
  'full'         → AdaptSci  (adaptive buffer + Hessian)
  'buffer_only'  → AdaptSci-Buffer ablation
  'hessian_only' → AdaptSci-Hessian ablation
  'none'         → plain LLM-SR (both components disabled)
"""
from __future__ import annotations

import dataclasses
from typing import Type


# ─────────────────────────────────────────────────────────────────────────────
# Experience buffer  (LLM-SR base, unchanged)
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class ExperienceBufferConfig:
    """Parameters for the multi-island experience buffer.

    Args:
        functions_per_prompt: In-context examples per LLM prompt (k).
        num_islands: Number of independently evolving islands (m).
        reset_period: Seconds between weakest-island resets.
        cluster_sampling_temperature_init: Initial Boltzmann temperature τ_c.
        cluster_sampling_temperature_period: Period for temperature schedule.
    """
    functions_per_prompt: int = 2
    num_islands: int = 10
    reset_period: int = 4 * 60 * 60           # 4 hours
    cluster_sampling_temperature_init: float = 0.1
    cluster_sampling_temperature_period: int = 30_000


# ─────────────────────────────────────────────────────────────────────────────
# Hessian-guided optimization config  (AdaptSci §1.3–1.4)
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class HessianConfig:
    """Controls Hessian-guided optimization allocation.

    Args:
        enabled: Master switch – False reproduces plain LLM-SR.
        condition_number_threshold: κ(f) above this → extra BFGS restarts.
        coupling_threshold: Mean |C_ij| above this → redirect to exploration.
        eff_dim_ratio_threshold: n_eff/p below this → redirect to exploration.
        max_extra_restarts: Max additional BFGS restarts per candidate.
        hessian_epsilon: Numerical stability constant ε in all diagnostics.
        finite_diff_step: Finite-difference step h for Hessian approximation.
        use_autograd: Use torch.autograd instead of finite differences
            (set True only when the skeleton uses torch tensor operations).
    """
    enabled: bool = True
    condition_number_threshold: float = 50.0
    coupling_threshold: float = 0.85
    eff_dim_ratio_threshold: float = 0.3
    max_extra_restarts: int = 3
    hessian_epsilon: float = 1e-8
    finite_diff_step: float = 1e-4
    use_autograd: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive buffer config  (AdaptSci §1.2)
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class AdaptiveBufferConfig:
    """Controls adaptive exploit/explore scheduling in the buffer.

    Args:
        enabled: Master switch – False reproduces plain LLM-SR.
        window_size: Sliding window length for progress estimation.
        progress_threshold: Minimum relative improvement for 'strong progress'.
            Below this the system shifts toward exploration.
        exploit_weight_max: Maximum exploitation weight (strong progress → 1.0).
        exploit_weight_min: Minimum exploitation weight (stalled → uniform).
        diversity_fn: Structural diversity measure for exploration island bias.
            Options: 'token_jaccard' | 'edit_distance' | 'length_bucket'.
        diversity_weight: Contribution of diversity to exploration probabilities.
    """
    enabled: bool = True
    window_size: int = 50
    progress_threshold: float = 0.01
    exploit_weight_max: float = 0.9
    exploit_weight_min: float = 0.1
    diversity_fn: str = 'token_jaccard'
    diversity_weight: float = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Top-level config
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class Config:
    """Full configuration for one AdaptSci experiment.

    Args:
        experience_buffer: Multi-island evolution settings.
        hessian: Hessian-guided optimization settings.
        adaptive_buffer: Adaptive exploit/explore settings.
        num_samplers: Parallel samplers (>1 requires external parallelism).
        num_evaluators: Parallel evaluators.
        samples_per_prompt: Equation skeletons per LLM call (b in paper).
        evaluate_timeout_seconds: Hard per-candidate timeout.
        use_api: Use external API (OpenAI / Anthropic) instead of local server.
        api_model: Model name for API mode.
            OpenAI:    'gpt-3.5-turbo' | 'gpt-4o' | 'o1-mini' | …
            Anthropic: 'claude-3-5-sonnet-20241022' | 'claude-3-opus-20240229' | …
        ablation_mode: Which AdaptSci components to activate.
            'full'         → AdaptSci (adaptive buffer + Hessian)
            'buffer_only'  → AdaptSci-Buffer ablation
            'hessian_only' → AdaptSci-Hessian ablation
            'none'         → plain LLM-SR (both disabled)
        llm_backend: Local routing for non-API mode.
            'local' → Flask engine.py server at local_server_url
            'vllm'  → vLLM OpenAI-compatible server at vllm_url
        local_server_url: URL of the local engine.py Flask server.
        vllm_url: URL of a vLLM OpenAI-compatible server.
    """
    experience_buffer: ExperienceBufferConfig = dataclasses.field(
        default_factory=ExperienceBufferConfig)
    hessian: HessianConfig = dataclasses.field(
        default_factory=HessianConfig)
    adaptive_buffer: AdaptiveBufferConfig = dataclasses.field(
        default_factory=AdaptiveBufferConfig)
    num_samplers: int = 1
    num_evaluators: int = 1
    samples_per_prompt: int = 4
    evaluate_timeout_seconds: int = 30
    use_api: bool = False
    api_model: str = 'gpt-3.5-turbo'
    ablation_mode: str = 'full'
    llm_backend: str = 'local'
    local_server_url: str = 'http://127.0.0.1:5000'
    vllm_url: str = 'http://127.0.0.1:8000'

    def __post_init__(self) -> None:
        valid = {'full', 'buffer_only', 'hessian_only', 'none'}
        if self.ablation_mode not in valid:
            raise ValueError(
                f'ablation_mode must be one of {valid}, got {self.ablation_mode!r}')
        valid_backends = {'local', 'vllm'}
        if self.llm_backend not in valid_backends:
            raise ValueError(
                f'llm_backend must be one of {valid_backends}, '
                f'got {self.llm_backend!r}')


# ─────────────────────────────────────────────────────────────────────────────
# Dependency injection
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass()
class ClassConfig:
    """Dependency-injection container for runtime class references.

    Args:
        llm_class: Concrete LLM subclass to instantiate (e.g. LocalLLM).
        sandbox_class: Concrete Sandbox subclass for candidate evaluation.
    """
    llm_class: type
    sandbox_class: type
