"""Evaluator with Hessian-guided adaptive optimization allocation (AdaptSci §1.3–1.4).

After the initial parameter fit the analyse() method:
  1. Estimates the Hessian H(f) of the fitting loss at θ⋆ via finite differences.
  2. Computes diagnostics: κ(f), mean coupling, n_eff/p.
  3. Applies the adaptive policy:
       WELL_CONDITIONED  → register candidate as-is.
       EXTRA_RESTARTS    → re-run BFGS from k perturbed starting points.
       REDIRECT_EXPLORE  → skip registration; signal the scheduler to
                           increase exploration weight.
  4. Attaches hessian_info dict to the Function for TensorBoard / JSON logging.

When Hessian is disabled (ablation_mode='none'/'buffer_only' or
hessian_cfg.enabled=False) the evaluator behaves identically to plain LLM-SR.
"""
from __future__ import annotations

import ast
import copy
import logging
import multiprocessing
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Dict, Optional, Type

import numpy as np
from scipy.optimize import minimize as scipy_minimize

from adaptsci import code_manipulation
from adaptsci import hessian as hessian_module
from adaptsci.adaptive_buffer import AdaptiveBufferScheduler

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Numba accelerator helper (optional)
# ─────────────────────────────────────────────────────────────────────────────

def _add_numba_decorator(program: str, function_to_evolve: str) -> str:
    """Add @numba.jit(nopython=True) to the target function if available."""
    try:
        import ast as _ast
        tree = _ast.parse(program)
        imported = any(
            isinstance(n, _ast.Import) and any(a.name == 'numba' for a in n.names)
            for n in tree.body)
        if not imported:
            tree.body.insert(0, _ast.Import(
                names=[_ast.alias(name='numba', asname=None)]))
        for node in _ast.walk(tree):
            if isinstance(node, _ast.FunctionDef) and node.name == function_to_evolve:
                dec = _ast.Call(
                    func=_ast.Attribute(
                        value=_ast.Name(id='numba', ctx=_ast.Load()),
                        attr='jit', ctx=_ast.Load()),
                    args=[],
                    keywords=[_ast.keyword(
                        arg='nopython',
                        value=_ast.NameConstant(value=True))])
                node.decorator_list.append(dec)
        return _ast.unparse(tree)
    except Exception:
        return program


# ─────────────────────────────────────────────────────────────────────────────
# Program assembly helpers
# ─────────────────────────────────────────────────────────────────────────────

class _FunctionEndVisitor(ast.NodeVisitor):
    def __init__(self, name: str) -> None:
        self._name = name
        self.end_line: Optional[int] = None

    def visit_FunctionDef(self, node: Any) -> None:
        if node.name == self._name:
            self.end_line = node.end_lineno
        self.generic_visit(node)


def _trim_body(generated_code: str) -> str:
    """Trim generated code to the valid function body, removing syntax tail."""
    if not generated_code:
        return ''
    code = f'def _header():\n{generated_code}'
    tree = None
    while tree is None:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            if e.lineno is None:
                return ''
            code = '\n'.join(code.splitlines()[:e.lineno - 1])
    v = _FunctionEndVisitor('_header')
    v.visit(tree)
    return '\n'.join(code.splitlines()[1:v.end_line]) + '\n\n'


def _sample_to_program(
    generated_code: str,
    version_generated: Optional[int],
    template: code_manipulation.Program,
    function_to_evolve: str,
) -> tuple[code_manipulation.Function, str]:
    body = _trim_body(generated_code)
    if version_generated is not None:
        body = code_manipulation.rename_function_calls(
            code=body,
            source_name=f'{function_to_evolve}_v{version_generated}',
            target_name=function_to_evolve,
        )
    program = copy.deepcopy(template)
    fn      = program.get_function(function_to_evolve)
    fn.body = body
    return fn, str(program)


def _calls_ancestor(program: str, function_to_evolve: str) -> bool:
    return any(
        name.startswith(f'{function_to_evolve}_v')
        for name in code_manipulation.get_functions_called(program))


# ─────────────────────────────────────────────────────────────────────────────
# Sandbox
# ─────────────────────────────────────────────────────────────────────────────

class Sandbox(ABC):
    """Abstract execution sandbox."""

    @abstractmethod
    def run(
        self,
        program: str,
        function_to_run: str,
        function_to_evolve: str,
        inputs: Any,
        test_input: str,
        timeout_seconds: int,
        **kwargs,
    ) -> tuple[Any, bool]:
        raise NotImplementedError


class LocalSandbox(Sandbox):
    """Subprocess-isolated execution sandbox (identical to LLM-SR original)."""

    def __init__(self, verbose: bool = False, numba_accelerate: bool = False):
        self._verbose          = verbose
        self._numba_accelerate = numba_accelerate

    def run(self, program, function_to_run, function_to_evolve,
            inputs, test_input, timeout_seconds, **kwargs) -> tuple[Any, bool]:
        dataset = inputs[test_input]
        q       = multiprocessing.Queue()
        p       = multiprocessing.Process(
            target=self._run_subprocess,
            args=(program, function_to_run, function_to_evolve,
                  dataset, self._numba_accelerate, q))
        p.start()
        p.join(timeout=timeout_seconds)
        if p.is_alive():
            p.terminate(); p.join()
            result = None, False
        else:
            result = self._dequeue(q)
        if self._verbose:
            self._print(program, result, **kwargs)
        return result

    @staticmethod
    def _dequeue(q: multiprocessing.Queue) -> tuple[Any, bool]:
        for _ in range(5):
            if not q.empty():
                return q.get_nowait()
            time.sleep(0.1)
        return None, False

    def _print(self, program: str, result: tuple, **kwargs) -> None:
        print('=' * 53)
        fn = code_manipulation.text_to_program(program).get_function(
            kwargs.get('func_to_evolve', 'equation'))
        print(str(fn).strip())
        print('-' * 53)
        print(f'Score: {result}')
        print('=' * 53 + '\n')

    @staticmethod
    def _run_subprocess(program, function_to_run, function_to_evolve,
                        dataset, numba_accelerate, q) -> None:
        try:
            if numba_accelerate:
                program = _add_numba_decorator(program, function_to_evolve)
            ns = {}
            exec(program, ns)  # noqa: S102
            result = ns[function_to_run](dataset)
            if not isinstance(result, (int, float)):
                q.put((None, False)); return
            q.put((result, True))
        except Exception as e:
            print(f'Execution Error: {e}')
            q.put((None, False))


# ─────────────────────────────────────────────────────────────────────────────
# Hessian estimation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_loss_fn(program_str: str, function_to_run: str, dataset: dict):
    """Build a scalar loss callable from the compiled program.  Returns (fn, None) or (None, None)."""
    try:
        ns = {}
        exec(program_str, ns)  # noqa: S102
        evaluate_fn = ns.get(function_to_run)
        if evaluate_fn is None:
            return None, None

        def loss_fn(params):  # noqa: ANN001
            try:
                return -float(evaluate_fn(dataset))
            except Exception:
                return 1e9

        return loss_fn, evaluate_fn
    except Exception as exc:
        logger.debug('_build_loss_fn: %s', exc)
        return None, None


def _extract_params(program_str: str) -> Optional[np.ndarray]:
    """Heuristic: extract the last numeric list from the program text."""
    matches = re.findall(r'\[\s*([0-9eE+\-.,\s]+)\]', program_str)
    if not matches:
        return None
    best = max(matches, key=len)
    try:
        vals = [float(x) for x in re.split(r'[,\s]+', best.strip()) if x]
        return np.array(vals, dtype=np.float64) if vals else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class Evaluator:
    """Evaluates LLM-generated skeletons with optional Hessian-guided refit.

    Additional kwargs accepted by analyse():
        hessian_cfg        – HessianConfig (None → Hessian disabled)
        adaptive_scheduler – AdaptiveBufferScheduler (None → no scheduler)
        ablation_mode      – str from Config.ablation_mode
    """

    def __init__(
        self,
        database,
        template: code_manipulation.Program,
        function_to_evolve: str,
        function_to_run: str,
        inputs: Sequence[Any],
        timeout_seconds: int = 30,
        sandbox_class: Type[Sandbox] = LocalSandbox,
    ) -> None:
        self._database           = database
        self._template           = template
        self._function_to_evolve = function_to_evolve
        self._function_to_run    = function_to_run
        self._inputs             = inputs
        self._timeout_seconds    = timeout_seconds
        self._sandbox            = sandbox_class()

    def analyse(
        self,
        sample: str,
        island_id: Optional[int],
        version_generated: Optional[int],
        **kwargs,
    ) -> None:
        """Compile → run → Hessian guidance → register."""

        hessian_cfg        = kwargs.get('hessian_cfg',        None)
        adaptive_scheduler = kwargs.get('adaptive_scheduler', None)
        ablation_mode      = kwargs.get('ablation_mode',      'full')

        hessian_enabled = (
            hessian_cfg is not None
            and hessian_cfg.enabled
            and ablation_mode in ('full', 'hessian_only'))

        # ── compile + initial sandbox run ─────────────────────────────────
        new_function, program = _sample_to_program(
            sample, version_generated, self._template, self._function_to_evolve)

        scores_per_test: Dict[str, float] = {}
        t0 = time.time()

        for inp in self._inputs:
            output, ok = self._sandbox.run(
                program, self._function_to_run, self._function_to_evolve,
                self._inputs, inp, self._timeout_seconds)
            if ok and not _calls_ancestor(program, self._function_to_evolve) and output is not None:
                if not isinstance(output, (int, float)):
                    raise ValueError('@evaluate.run must return int or float.')
                scores_per_test[inp] = output

        # ── Hessian-guided adaptive optimization ─────────────────────────
        hessian_info: dict = {}

        if hessian_enabled and scores_per_test:
            first_inp  = list(self._inputs)[0]
            dataset    = self._inputs[first_inp]
            loss_fn, _ = _build_loss_fn(program, self._function_to_run, dataset)
            theta_star = _extract_params(program)

            if loss_fn is not None and theta_star is not None and len(theta_star) > 0:
                diag     = hessian_module.compute_hessian_diagnostics(
                    loss_fn, theta_star, hessian_cfg)
                decision = hessian_module.adaptive_optimization_policy(
                    diag, max(scores_per_test.values()), hessian_cfg)

                hessian_info = diag.to_dict() if diag is not None else {}
                hessian_info['policy_action'] = decision.action
                hessian_info['policy_reason'] = decision.reason

                logger.debug('Hessian decision: %s', decision)

                if decision.action == hessian_module.OptimizationDecision.EXTRA_RESTARTS:
                    scores_per_test = self._extra_restarts(
                        loss_fn, theta_star, scores_per_test,
                        decision.extra_restarts, first_inp)

                elif decision.action == hessian_module.OptimizationDecision.REDIRECT_EXPLORE:
                    if adaptive_scheduler is not None:
                        best = max(scores_per_test.values()) if scores_per_test else -1e9
                        adaptive_scheduler.record_score(best, code=str(new_function))
                    self._log_skipped(new_function, hessian_info, kwargs)
                    return

        evaluate_time = time.time() - t0

        # ── register to buffer ────────────────────────────────────────────
        if scores_per_test:
            new_function.hessian_info = hessian_info  # type: ignore[attr-defined]
            self._database.register_program(
                new_function, island_id, scores_per_test,
                **kwargs, evaluate_time=evaluate_time)
            if adaptive_scheduler is not None:
                adaptive_scheduler.record_score(
                    max(scores_per_test.values()), code=str(new_function))
        else:
            profiler = kwargs.get('profiler', None)
            if profiler:
                new_function.global_sample_nums = kwargs.get('global_sample_nums', None)
                new_function.score              = None
                new_function.sample_time        = kwargs.get('sample_time', None)
                new_function.evaluate_time      = evaluate_time
                new_function.hessian_info       = hessian_info  # type: ignore[attr-defined]
                profiler.register_function(new_function)

    # ── extra restarts ────────────────────────────────────────────────────

    def _extra_restarts(
        self,
        loss_fn,
        theta_star: np.ndarray,
        scores_per_test: dict,
        n_restarts: int,
        first_inp: str,
    ) -> dict:
        best_score = max(scores_per_test.values())
        p = len(theta_star)
        for k in range(n_restarts):
            scale = np.abs(theta_star).mean() + 1e-3
            theta_init = theta_star + np.random.randn(p) * scale * 0.3
            try:
                res   = scipy_minimize(loss_fn, theta_init, method='L-BFGS-B',
                                       options={'maxiter': 200, 'ftol': 1e-12})
                score = -float(res.fun)
                if score > best_score:
                    best_score = score
                    scores_per_test = dict(scores_per_test)
                    scores_per_test[first_inp] = score
                    logger.debug('Restart %d → score=%.6f', k, score)
            except Exception as exc:
                logger.debug('Restart %d failed: %s', k, exc)
        return scores_per_test

    # ── helpers ───────────────────────────────────────────────────────────

    def _log_skipped(
        self,
        fn: code_manipulation.Function,
        hessian_info: dict,
        kwargs: dict,
    ) -> None:
        profiler = kwargs.get('profiler', None)
        if profiler:
            fn.global_sample_nums = kwargs.get('global_sample_nums', None)
            fn.score              = None
            fn.sample_time        = kwargs.get('sample_time', None)
            fn.evaluate_time      = 0.0
            fn.hessian_info       = hessian_info  # type: ignore[attr-defined]
            profiler.register_function(fn)
