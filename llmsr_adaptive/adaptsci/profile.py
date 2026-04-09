"""AdaptSci profiler.

Logs per-sample statistics to TensorBoard scalars and per-sample JSON files.

New vs. plain LLM-SR
─────────────────────
  • Per-sample Hessian diagnostics (κ, coupling, n_eff, eff_dim_ratio)
  • Policy action cumulative counters (well_conditioned / extra_restarts /
    redirect_explore)
  • Adaptive buffer scheduler state (exploit_weight)
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, Optional

from adaptsci import code_manipulation

logger = logging.getLogger(__name__)

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB = True
except ImportError:
    _TB = False
    logger.warning('TensorBoard not available – scalar logging disabled.')


class Profiler:
    """Records and logs AdaptSci experiment metrics."""

    def __init__(
        self,
        log_dir: Optional[str] = None,
        max_log_nums: Optional[int] = None,
    ) -> None:
        self._log_dir      = log_dir
        self._json_dir     = os.path.join(log_dir, 'samples') if log_dir else None
        if self._json_dir:
            os.makedirs(self._json_dir, exist_ok=True)
        self._max_log_nums = max_log_nums
        self._num_samples  = 0

        # ── standard LLM-SR metrics ───────────────────────────────────────
        self._cur_best_score: float  = -99999999.0
        self._cur_best_sample: int   = 0
        self._cur_best_str: str      = ''
        self._evaluate_success: int  = 0
        self._evaluate_failed:  int  = 0
        self._tot_sample_time:  float = 0.0
        self._tot_evaluate_time: float = 0.0
        self._all_functions: Dict[int, code_manipulation.Function] = {}

        # ── AdaptSci metrics ──────────────────────────────────────────────
        self._policy_counts: Dict[str, int] = {
            'well_conditioned': 0,
            'extra_restarts':   0,
            'redirect_explore': 0,
            'unknown':          0,
        }
        self._hessian_kappa_sum:    float = 0.0
        self._hessian_coupling_sum: float = 0.0
        self._hessian_neff_sum:     float = 0.0
        self._hessian_samples:      int   = 0
        self._latest_exploit_weight: float = 0.5

        self._writer = SummaryWriter(log_dir=log_dir) if (log_dir and _TB) else None

    # ── public ────────────────────────────────────────────────────────────

    def register_function(self, fn: code_manipulation.Function) -> None:
        """Record a sampled function and update all metrics."""
        if self._max_log_nums and self._num_samples >= self._max_log_nums:
            return
        order = fn.global_sample_nums or 0
        if order in self._all_functions:
            return
        self._num_samples += 1
        self._all_functions[order] = fn
        self._update_stats(fn)
        self._write_tensorboard(order, fn)
        self._write_json(fn)
        self._print(fn)

    def log_scheduler_state(self, state: dict, global_step: int) -> None:
        """Record adaptive buffer scheduler state (called by Sampler)."""
        self._latest_exploit_weight = state.get('exploit_weight', 0.5)
        if self._writer:
            self._writer.add_scalar(
                'AdaptSci/exploit_weight',
                self._latest_exploit_weight,
                global_step=global_step)

    # ── private ───────────────────────────────────────────────────────────

    def _update_stats(self, fn: code_manipulation.Function) -> None:
        score = fn.score
        if score is not None and score > self._cur_best_score:
            self._cur_best_score  = score
            self._cur_best_sample = fn.global_sample_nums or 0
            self._cur_best_str    = str(fn).strip()
        if score is not None:
            self._evaluate_success += 1
        else:
            self._evaluate_failed  += 1
        self._tot_sample_time   += fn.sample_time   or 0.0
        self._tot_evaluate_time += fn.evaluate_time or 0.0

        hessian_info = getattr(fn, 'hessian_info', {}) or {}
        if hessian_info:
            action = hessian_info.get('policy_action', 'unknown')
            self._policy_counts[action] = self._policy_counts.get(action, 0) + 1
            kappa = hessian_info.get('kappa')
            if kappa is not None:
                self._hessian_kappa_sum    += float(kappa)
                self._hessian_coupling_sum += float(hessian_info.get('coupling') or 0)
                self._hessian_neff_sum     += float(hessian_info.get('n_eff') or 0)
                self._hessian_samples      += 1
        else:
            self._policy_counts['unknown'] += 1

    def _write_tensorboard(self, step: int, fn: code_manipulation.Function) -> None:
        if not self._writer:
            return
        # standard
        self._writer.add_scalar('Best Score', self._cur_best_score, step)
        self._writer.add_scalars('Legal/Illegal Functions', {
            'legal': self._evaluate_success, 'illegal': self._evaluate_failed}, step)
        self._writer.add_scalars('Timing', {
            'sample_time': self._tot_sample_time,
            'evaluate_time': self._tot_evaluate_time}, step)
        if self._cur_best_str:
            self._writer.add_text('Best Function', self._cur_best_str, step)
        # AdaptSci
        self._writer.add_scalars('AdaptSci/Policy Actions',
                                 dict(self._policy_counts), step)
        if self._hessian_samples > 0:
            n = self._hessian_samples
            self._writer.add_scalar('AdaptSci/Mean Kappa',    self._hessian_kappa_sum / n,    step)
            self._writer.add_scalar('AdaptSci/Mean Coupling', self._hessian_coupling_sum / n, step)
            self._writer.add_scalar('AdaptSci/Mean n_eff',    self._hessian_neff_sum / n,     step)
        self._writer.add_scalar('AdaptSci/Exploit Weight', self._latest_exploit_weight, step)
        hessian_info = getattr(fn, 'hessian_info', {}) or {}
        if hessian_info.get('kappa') is not None:
            self._writer.add_scalar('Hessian/kappa',         float(hessian_info['kappa']),          step)
            self._writer.add_scalar('Hessian/coupling',      float(hessian_info.get('coupling', 0)), step)
            self._writer.add_scalar('Hessian/eff_dim_ratio', float(hessian_info.get('eff_dim_ratio', 0)), step)

    def _write_json(self, fn: code_manipulation.Function) -> None:
        if not self._json_dir:
            return
        order        = fn.global_sample_nums or 0
        hessian_info = getattr(fn, 'hessian_info', {}) or {}
        content = {
            'sample_order':  order,
            'function':      str(fn),
            'score':         fn.score,
            'sample_time':   fn.sample_time,
            'evaluate_time': fn.evaluate_time,
            'hessian':       hessian_info,
        }
        with open(os.path.join(self._json_dir, f'samples_{order}.json'), 'w') as f:
            json.dump(content, f, indent=2, default=str)

    def _print(self, fn: code_manipulation.Function) -> None:
        print('================= Evaluated Function =================')
        print(str(fn).strip())
        print('------------------------------------------------------')
        print(f'Score        : {fn.score}')
        print(f'Sample time  : {fn.sample_time}')
        print(f'Evaluate time: {fn.evaluate_time}')
        print(f'Sample order : {fn.global_sample_nums}')
        hessian_info = getattr(fn, 'hessian_info', {}) or {}
        if hessian_info:
            print(f'[AdaptSci]  policy={hessian_info.get("policy_action","n/a")}  '
                  f'κ={hessian_info.get("kappa","n/a")}  '
                  f'coupling={hessian_info.get("coupling","n/a")}  '
                  f'n_eff/p={hessian_info.get("eff_dim_ratio","n/a")}')
            if hessian_info.get('policy_reason'):
                print(f'            reason: {hessian_info["policy_reason"]}')
        print(f'[AdaptSci]  exploit_weight={self._latest_exploit_weight:.3f}')
        print('======================================================\n')
