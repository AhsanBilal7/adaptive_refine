"""Evaluator with robust Hessian-guided adaptive optimization allocation.

Main fixes vs. the original AdaptSci evaluator:
1. Build a real parameter-aware loss from the evolved equation(..., params).
2. Fit theta_star directly rather than extracting numbers from source text.
3. Never discard a candidate solely because the policy says REDIRECT_EXPLORE.
4. Use all available training/evaluation splits for Hessian fitting when possible.
5. Persist fitted parameters in hessian_info for logging/debugging.
6. Extra restarts improve the true fitted point, not only a single score entry.
"""

from __future__ import annotations

import ast
import copy
import logging
import multiprocessing
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Dict, Optional, Type

import numpy as np
from scipy.optimize import minimize as scipy_minimize

from adaptsci import code_manipulation
from adaptsci import hessian as hessian_module

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Optional numba helper
# ─────────────────────────────────────────────────────────────────────────────

def _add_numba_decorator(program: str, function_to_evolve: str) -> str:
    """Add @numba.jit(nopython=True) to the target function if possible."""
    try:
        import ast as _ast

        tree = _ast.parse(program)
        imported = any(
            isinstance(n, _ast.Import) and any(a.name == "numba" for a in n.names)
            for n in tree.body
        )
        if not imported:
            tree.body.insert(0, _ast.Import(names=[_ast.alias(name="numba", asname=None)]))

        for node in _ast.walk(tree):
            if isinstance(node, _ast.FunctionDef) and node.name == function_to_evolve:
                dec = _ast.Call(
                    func=_ast.Attribute(
                        value=_ast.Name(id="numba", ctx=_ast.Load()),
                        attr="jit",
                        ctx=_ast.Load(),
                    ),
                    args=[],
                    keywords=[_ast.keyword(arg="nopython", value=_ast.NameConstant(value=True))],
                )
                node.decorator_list.append(dec)
        return _ast.unparse(tree)
    except Exception:
        return program


# ─────────────────────────────────────────────────────────────────────────────
# Program helpers
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
        return ""
    code = f"def _header():\n{generated_code}"
    tree = None
    while tree is None:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            if e.lineno is None:
                return ""
            code = "\n".join(code.splitlines()[: e.lineno - 1])
    v = _FunctionEndVisitor("_header")
    v.visit(tree)
    return "\n".join(code.splitlines()[1 : v.end_line]) + "\n\n"


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
            source_name=f"{function_to_evolve}_v{version_generated}",
            target_name=function_to_evolve,
        )
    program = copy.deepcopy(template)
    fn = program.get_function(function_to_evolve)
    fn.body = body
    return fn, str(program)


def _calls_ancestor(program: str, function_to_evolve: str) -> bool:
    return any(
        name.startswith(f"{function_to_evolve}_v")
        for name in code_manipulation.get_functions_called(program)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sandbox
# ─────────────────────────────────────────────────────────────────────────────

class Sandbox(ABC):
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
    """Subprocess-isolated execution sandbox."""

    def __init__(self, verbose: bool = False, numba_accelerate: bool = False):
        self._verbose = verbose
        self._numba_accelerate = numba_accelerate

    def run(
        self,
        program,
        function_to_run,
        function_to_evolve,
        inputs,
        test_input,
        timeout_seconds,
        **kwargs,
    ) -> tuple[Any, bool]:
        dataset = inputs[test_input]
        q = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=self._run_subprocess,
            args=(program, function_to_run, function_to_evolve, dataset, self._numba_accelerate, q),
        )
        p.start()
        p.join(timeout=timeout_seconds)
        if p.is_alive():
            p.terminate()
            p.join()
            result = (None, False)
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
        return (None, False)

    def _print(self, program: str, result: tuple[Any, bool], **kwargs) -> None:
        print("=" * 53)
        fn = code_manipulation.text_to_program(program).get_function(
            kwargs.get("func_to_evolve", "equation")
        )
        print(str(fn).strip())
        print("-" * 53)
        print(f"Score: {result}")
        print("=" * 53 + "\n")

    @staticmethod
    def _run_subprocess(program, function_to_run, function_to_evolve, dataset, numba_accelerate, q):
        try:
            if numba_accelerate:
                program = _add_numba_decorator(program, function_to_evolve)
            ns: dict[str, Any] = {}
            exec(program, ns)  # noqa: S102
            result = ns[function_to_run](dataset)
            if not isinstance(result, (int, float)):
                q.put((None, False))
                return
            q.put((result, True))
        except Exception as e:
            print(f"Execution Error: {e}")
            q.put((None, False))


# ─────────────────────────────────────────────────────────────────────────────
# Parameter-aware fitting helpers
# ─────────────────────────────────────────────────────────────────────────────

class _ParamIndexVisitor(ast.NodeVisitor):
    """Collect max params[i] index used in equation body."""

    def __init__(self) -> None:
        self.max_idx = -1

    def visit_Subscript(self, node: ast.Subscript) -> None:
        try:
            if isinstance(node.value, ast.Name) and node.value.id == "params":
                idx = None
                if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
                    idx = node.slice.value
                elif hasattr(ast, "Index") and isinstance(node.slice, ast.Index):
                    val = node.slice.value
                    if isinstance(val, ast.Constant) and isinstance(val.value, int):
                        idx = val.value
                if idx is not None:
                    self.max_idx = max(self.max_idx, idx)
        except Exception:
            pass
        self.generic_visit(node)


def _infer_num_params(program_str: str, function_to_evolve: str) -> int:
    try:
        tree = ast.parse(program_str)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == function_to_evolve:
                vis = _ParamIndexVisitor()
                vis.visit(node)
                return vis.max_idx + 1 if vis.max_idx >= 0 else 0
    except Exception:
        pass
    return 0


def _dataset_to_matrix(dataset) -> Optional[np.ndarray]:
    """Convert a dataset of rows [y, x1, x2, ...] into a numeric matrix."""
    try:
        rows = list(dataset)
        if not rows:
            return None
        arr = np.asarray(rows, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] < 2:
            return None
        return arr
    except Exception:
        return None


def _build_direct_loss_fn(
    program_str: str,
    function_to_evolve: str,
    inputs: Sequence[Any],
):
    """Build a parameter-aware MSE loss over all available splits.

    Expected row format:
        [y, x1, x2, ...]
    Equation signature:
        equation(x1, x2, ..., params)
    """
    try:
        ns: dict[str, Any] = {}
        exec(program_str, ns)  # noqa: S102
        eq_fn = ns.get(function_to_evolve)
        if eq_fn is None:
            return None

        matrices: list[np.ndarray] = []
        if isinstance(inputs, dict):
            for _, ds in inputs.items():
                arr = _dataset_to_matrix(ds)
                if arr is not None:
                    matrices.append(arr)
        else:
            for ds in inputs:
                arr = _dataset_to_matrix(ds)
                if arr is not None:
                    matrices.append(arr)

        if not matrices:
            return None

        def loss_fn(params: np.ndarray) -> float:
            try:
                total = 0.0
                count = 0
                for arr in matrices:
                    y_true = arr[:, 0]
                    x_cols = [arr[:, i] for i in range(1, arr.shape[1])]
                    pred = eq_fn(*x_cols, params)
                    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
                    if pred.shape[0] != y_true.shape[0]:
                        return 1e12
                    if not np.all(np.isfinite(pred)):
                        return 1e12
                    err = pred - y_true
                    total += float(np.mean(err * err))
                    count += 1
                return total / max(count, 1)
            except Exception:
                return 1e12

        return loss_fn
    except Exception as exc:
        logger.debug("_build_direct_loss_fn failed: %s", exc)
        return None


def _fit_theta_star(
    loss_fn,
    n_params: int,
    n_restarts: int = 5,
    maxiter: int = 200,
) -> tuple[Optional[np.ndarray], float]:
    if n_params <= 0:
        return None, float("inf")

    best_theta: Optional[np.ndarray] = None
    best_loss = float("inf")
    rng = np.random.default_rng(0)

    init_list = [np.zeros(n_params, dtype=np.float64)]
    while len(init_list) < n_restarts:
        scale = [0.01, 0.1, 1.0, 3.0][min(len(init_list) - 1, 3)]
        init_list.append(rng.normal(0.0, scale, size=n_params).astype(np.float64))

    for theta0 in init_list[:n_restarts]:
        try:
            res = scipy_minimize(
                loss_fn,
                theta0,
                method="L-BFGS-B",
                options={"maxiter": maxiter, "ftol": 1e-12},
            )
            loss = float(res.fun)
            theta = np.asarray(res.x, dtype=np.float64)
            if np.isfinite(loss) and loss < best_loss:
                best_loss = loss
                best_theta = theta
        except Exception as exc:
            logger.debug("_fit_theta_star restart failed: %s", exc)

    return best_theta, best_loss


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class Evaluator:
    """Evaluates LLM-generated skeletons with optional Hessian-guided refit."""

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
        self._database = database
        self._template = template
        self._function_to_evolve = function_to_evolve
        self._function_to_run = function_to_run
        self._inputs = inputs
        self._timeout_seconds = timeout_seconds
        self._sandbox = sandbox_class()

    def analyse(
        self,
        sample: str,
        island_id: Optional[int],
        version_generated: Optional[int],
        **kwargs,
    ) -> None:
        hessian_cfg = kwargs.get("hessian_cfg", None)
        adaptive_scheduler = kwargs.get("adaptive_scheduler", None)
        ablation_mode = kwargs.get("ablation_mode", "full")

        hessian_enabled = (
            hessian_cfg is not None
            and hessian_cfg.enabled
            and ablation_mode in ("full", "hessian_only")
        )

        new_function, program = _sample_to_program(
            sample,
            version_generated,
            self._template,
            self._function_to_evolve,
        )

        scores_per_test: Dict[str, float] = {}
        t0 = time.time()

        for inp in self._inputs:
            output, ok = self._sandbox.run(
                program,
                self._function_to_run,
                self._function_to_evolve,
                self._inputs,
                inp,
                self._timeout_seconds,
            )
            if ok and not _calls_ancestor(program, self._function_to_evolve) and output is not None:
                if not isinstance(output, (int, float)):
                    raise ValueError("@evaluate.run must return int or float.")
                scores_per_test[inp] = output

        hessian_info: dict[str, Any] = {}

        if hessian_enabled and scores_per_test:
            n_params = _infer_num_params(program, self._function_to_evolve)
            loss_fn = _build_direct_loss_fn(program, self._function_to_evolve, self._inputs)

            if loss_fn is not None and n_params > 0:
                theta_star, best_loss = _fit_theta_star(
                    loss_fn,
                    n_params=n_params,
                    n_restarts=4,
                    maxiter=200,
                )

                if theta_star is not None and np.all(np.isfinite(theta_star)):
                    diag = hessian_module.compute_hessian_diagnostics(
                        loss_fn,
                        theta_star,
                        hessian_cfg,
                    )
                    decision = hessian_module.adaptive_optimization_policy(
                        diag,
                        max(scores_per_test.values()),
                        hessian_cfg,
                    )

                    hessian_info = diag.to_dict() if diag is not None else {}
                    hessian_info["policy_action"] = decision.action
                    hessian_info["policy_reason"] = decision.reason
                    hessian_info["theta_star"] = theta_star.tolist()
                    hessian_info["fit_loss"] = float(best_loss)

                    if decision.action == hessian_module.OptimizationDecision.EXTRA_RESTARTS:
                        theta_star, best_loss = self._extra_restarts(
                            loss_fn=loss_fn,
                            theta_star=theta_star,
                            current_best_loss=best_loss,
                            n_restarts=decision.extra_restarts,
                        )
                        hessian_info["theta_star_after_restarts"] = theta_star.tolist()
                        hessian_info["fit_loss_after_restarts"] = float(best_loss)
                    elif decision.action == hessian_module.OptimizationDecision.REDIRECT_EXPLORE:
                        # Keep the candidate, but tell the scheduler to lean exploratory.
                        hessian_info["redirect_explore_only"] = True
                        if adaptive_scheduler is not None:
                            adaptive_scheduler.record_policy_signal("redirect_explore")

        evaluate_time = time.time() - t0

        if scores_per_test:
            new_function.hessian_info = hessian_info  # type: ignore[attr-defined]
            self._database.register_program(
                new_function,
                island_id,
                scores_per_test,
                **kwargs,
                evaluate_time=evaluate_time,
            )
            if adaptive_scheduler is not None:
                adaptive_scheduler.record_score(
                    max(scores_per_test.values()),
                    code=str(new_function),
                )
        else:
            profiler = kwargs.get("profiler", None)
            if profiler:
                new_function.global_sample_nums = kwargs.get("global_sample_nums", None)
                new_function.score = None
                new_function.sample_time = kwargs.get("sample_time", None)
                new_function.evaluate_time = evaluate_time
                new_function.hessian_info = hessian_info  # type: ignore[attr-defined]
                profiler.register_function(new_function)

    def _extra_restarts(
        self,
        loss_fn,
        theta_star: np.ndarray,
        current_best_loss: float,
        n_restarts: int,
    ) -> tuple[np.ndarray, float]:
        best_theta = theta_star.copy()
        best_loss = float(current_best_loss)
        p = len(theta_star)
        rng = np.random.default_rng(123)

        for _ in range(max(0, n_restarts)):
            scale = max(1e-3, float(np.mean(np.abs(best_theta))) + 1e-3)
            theta_init = best_theta + rng.normal(0.0, 0.3 * scale, size=p)
            try:
                res = scipy_minimize(
                    loss_fn,
                    theta_init,
                    method="L-BFGS-B",
                    options={"maxiter": 200, "ftol": 1e-12},
                )
                loss = float(res.fun)
                if np.isfinite(loss) and loss < best_loss:
                    best_loss = loss
                    best_theta = np.asarray(res.x, dtype=np.float64)
            except Exception as exc:
                logger.debug("Restart failed: %s", exc)

        return best_theta, best_loss
