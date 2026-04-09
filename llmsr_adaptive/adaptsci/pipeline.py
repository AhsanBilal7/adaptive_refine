"""AdaptSci pipeline.

Wires all AdaptSci components together into the iterative LLM-SR loop:
  1. Resolves ablation_mode into (hessian_enabled, buffer_enabled) flags.
  2. Creates AdaptiveBufferScheduler when buffer is enabled.
  3. Seeds the experience buffer with the initial template body.
  4. Runs samplers in a continuous loop until the sample budget is exhausted.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Sequence, Tuple

from adaptsci import code_manipulation
from adaptsci import config as config_lib
from adaptsci.buffer import ExperienceBuffer
from adaptsci.evaluator import Evaluator
from adaptsci.sampler import Sampler
from adaptsci.profile import Profiler
from adaptsci.adaptive_buffer import AdaptiveBufferScheduler

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_function_names(specification: str) -> Tuple[str, str]:
    """Return (function_to_evolve, function_to_run) from the spec."""
    run_fns    = list(code_manipulation.yield_decorated(specification, 'evaluate', 'run'))
    evolve_fns = list(code_manipulation.yield_decorated(specification, 'equation',  'evolve'))
    if len(run_fns) != 1:
        raise ValueError('Expected exactly 1 function decorated with @evaluate.run.')
    if len(evolve_fns) != 1:
        raise ValueError('Expected exactly 1 function decorated with @equation.evolve.')
    return evolve_fns[0], run_fns[0]


def _resolve_ablation(mode: str) -> Tuple[bool, bool]:
    """Return (hessian_enabled, buffer_enabled) for ablation_mode."""
    hessian = mode in ('full', 'hessian_only')
    buf     = mode in ('full', 'buffer_only')
    logger.info('ablation_mode=%s  →  hessian=%s  adaptive_buffer=%s',
                mode, hessian, buf)
    return hessian, buf


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(
    specification: str,
    inputs: Sequence[Any],
    config: config_lib.Config,
    max_sample_nums: int | None,
    class_config: config_lib.ClassConfig,
    **kwargs,
) -> None:
    """Launch an AdaptSci experiment.

    Args:
        specification:   Spec file contents (problem boilerplate with
                         @equation.evolve and @evaluate.run decorators).
        inputs:          Dataset dict {input_name: data_dict}.
        config:          AdaptSci Config (includes Hessian + AdaptiveBuffer).
        max_sample_nums: Stop after this many LLM samples (None = unlimited).
        class_config:    LLM and Sandbox class references.
        **kwargs:        'log_dir' triggers Profiler creation.
    """
    function_to_evolve, function_to_run = _extract_function_names(specification)
    template = code_manipulation.text_to_program(specification)

    # ── resolve ablation flags ────────────────────────────────────────────
    hessian_enabled, buffer_enabled = _resolve_ablation(config.ablation_mode)

    effective_hessian_cfg = config_lib.HessianConfig(
        enabled=hessian_enabled and config.hessian.enabled,
        condition_number_threshold=config.hessian.condition_number_threshold,
        coupling_threshold=config.hessian.coupling_threshold,
        eff_dim_ratio_threshold=config.hessian.eff_dim_ratio_threshold,
        max_extra_restarts=config.hessian.max_extra_restarts,
        hessian_epsilon=config.hessian.hessian_epsilon,
        finite_diff_step=config.hessian.finite_diff_step,
        use_autograd=config.hessian.use_autograd,
    )

    effective_ab_cfg = config_lib.AdaptiveBufferConfig(
        enabled=buffer_enabled and config.adaptive_buffer.enabled,
        window_size=config.adaptive_buffer.window_size,
        progress_threshold=config.adaptive_buffer.progress_threshold,
        exploit_weight_max=config.adaptive_buffer.exploit_weight_max,
        exploit_weight_min=config.adaptive_buffer.exploit_weight_min,
        diversity_fn=config.adaptive_buffer.diversity_fn,
        diversity_weight=config.adaptive_buffer.diversity_weight,
    )

    # ── adaptive buffer scheduler ─────────────────────────────────────────
    scheduler: Optional[AdaptiveBufferScheduler] = None
    if effective_ab_cfg.enabled:
        scheduler = AdaptiveBufferScheduler(effective_ab_cfg)
        logger.info('AdaptSci: Adaptive buffer scheduler ENABLED.')
    else:
        logger.info('AdaptSci: Adaptive buffer scheduler DISABLED.')

    # ── experience buffer ─────────────────────────────────────────────────
    database = ExperienceBuffer(
        config.experience_buffer, template, function_to_evolve)

    # ── profiler ──────────────────────────────────────────────────────────
    log_dir  = kwargs.get('log_dir', None)
    profiler: Optional[Profiler] = Profiler(log_dir) if log_dir else None

    # ── evaluators ────────────────────────────────────────────────────────
    evaluators = [
        Evaluator(
            database, template, function_to_evolve, function_to_run,
            inputs,
            timeout_seconds=config.evaluate_timeout_seconds,
            sandbox_class=class_config.sandbox_class,
        )
        for _ in range(config.num_evaluators)
    ]

    # ── seed: evaluate initial template body ─────────────────────────────
    initial_body = template.get_function(function_to_evolve).body
    evaluators[0].analyse(
        initial_body,
        island_id=None,
        version_generated=None,
        profiler=profiler,
        hessian_cfg=effective_hessian_cfg,
        adaptive_scheduler=scheduler,
        ablation_mode=config.ablation_mode,
    )

    # ── samplers ──────────────────────────────────────────────────────────
    samplers = [
        Sampler(
            database, evaluators, config.samples_per_prompt,
            config=config,
            max_sample_nums=max_sample_nums,
            llm_class=class_config.llm_class,
            hessian_cfg=effective_hessian_cfg,
            adaptive_scheduler=scheduler,
        )
        for _ in range(config.num_samplers)
    ]

    # ── main loop ─────────────────────────────────────────────────────────
    for s in samplers:
        s.sample(
            profiler=profiler,
            hessian_cfg=effective_hessian_cfg,
            adaptive_scheduler=scheduler,
            ablation_mode=config.ablation_mode,
        )
