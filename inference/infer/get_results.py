"""
get_results.py – Equation parsing and performance evaluation.

AdaptSci changes vs. original SR-Scientist:
  • `get_performance_with_hessian()` – multi-restart BFGS + Hessian diagnostics
    for every evaluated equation; returns HessianDiag per equation.
  • `get_performance()` / `get_performance_on_test()` – unchanged API, now
    accept an optional `n_restarts` argument for κ-adaptive restarts.
  • `get_final_results()` – unchanged (best-by-train-MSE selection).
"""

import json
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

# Try importing Hessian module; fall back gracefully if not available
try:
    from .hessian import evaluate_with_hessian, HessianDiag
    _HESSIAN_AVAILABLE = True
except ImportError:
    _HESSIAN_AVAILABLE = False

exec_globals = {
    "np": np,
    "minimize": minimize,
}

# ---------------------------------------------------------------------------
# Code-snippet templates (unchanged from SR-Scientist)
# ---------------------------------------------------------------------------

code_snippet_single_dataset = """

{equation}

def evaluate(equation, data, thresholds):
    MAX_NPARAMS = 10
    outputs = np.array([row[0] for row in data])
    inputs = np.array([row[1:] for row in data])

    def loss(params):
        try:
            y_pred = equation(*inputs.T, params)
            return np.mean((y_pred - outputs) ** 2)
        except (FloatingPointError, OverflowError):
            return np.inf

    result = minimize(loss, [1.0] * MAX_NPARAMS, method='BFGS')
    optimized_params = result.x
    mse = result.fun

    if np.isnan(mse) or np.isinf(mse):
        return None, None, None, [None] * len(thresholds), optimized_params

    var_outputs = np.var(outputs)
    nmse = mse / var_outputs if not np.isclose(var_outputs, 0) else (0.0 if np.isclose(mse, 0) else np.inf)

    y_pred = equation(*inputs.T, optimized_params)
    zero_mask = np.isclose(outputs, 0)
    non_zero_mask = ~zero_mask

    mape = np.nan
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((outputs[non_zero_mask] - y_pred[non_zero_mask]) / outputs[non_zero_mask]))
    elif np.allclose(y_pred, 0):
        mape = 0.0
    else:
        mape = np.inf

    accuracies = []
    if len(outputs) == 0:
        accuracies = [0.0] * len(thresholds)
    else:
        for threshold in thresholds:
            correct_predictions = np.zeros_like(outputs, dtype=bool)
            if np.any(non_zero_mask):
                relative_error = np.abs((y_pred[non_zero_mask] - outputs[non_zero_mask]) / outputs[non_zero_mask])
                correct_predictions[non_zero_mask] = relative_error < threshold
            correct_predictions[zero_mask] = np.isclose(y_pred[zero_mask], 0, atol=1e-9)
            accuracies.append(1.0 if np.mean(correct_predictions) >= 0.95 else 0.0)

    return float(mse), float(nmse), float(mape), accuracies, optimized_params
"""

code_snippet_train_test = """

{equation}

def evaluate(equation, train_data, test_data, thresholds):
    MAX_NPARAMS = 10
    train_outputs = np.array([row[0] for row in train_data])
    train_inputs = np.array([row[1:] for row in train_data])

    def loss(params):
        try:
            y_pred_train = equation(*train_inputs.T, params)
            return np.mean((y_pred_train - train_outputs) ** 2)
        except (FloatingPointError, OverflowError):
            return np.inf

    result = minimize(loss, [1.0] * MAX_NPARAMS, method='BFGS')
    optimized_params = result.x

    test_outputs = np.array([row[0] for row in test_data])
    test_inputs = np.array([row[1:] for row in test_data])

    try:
        y_pred_test = equation(*test_inputs.T, optimized_params)
    except (FloatingPointError, OverflowError):
        return None, None, None, [None] * len(thresholds), optimized_params

    mse = np.mean((y_pred_test - test_outputs) ** 2)
    if np.isnan(mse) or np.isinf(mse):
        return None, None, None, [None] * len(thresholds), optimized_params

    var_test_outputs = np.var(test_outputs)
    nmse = mse / var_test_outputs if not np.isclose(var_test_outputs, 0) else (0.0 if np.isclose(mse, 0) else np.inf)

    zero_mask = np.isclose(test_outputs, 0)
    non_zero_mask = ~zero_mask

    mape = np.nan
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((test_outputs[non_zero_mask] - y_pred_test[non_zero_mask]) / test_outputs[non_zero_mask]))
    elif np.allclose(y_pred_test, 0):
        mape = 0.0
    else:
        mape = np.inf

    accuracies = []
    if len(test_outputs) == 0:
        accuracies = [0.0] * len(thresholds)
    else:
        for threshold in thresholds:
            correct_predictions = np.zeros_like(test_outputs, dtype=bool)
            if np.any(non_zero_mask):
                relative_error = np.abs((y_pred_test[non_zero_mask] - test_outputs[non_zero_mask]) / test_outputs[non_zero_mask])
                correct_predictions[non_zero_mask] = relative_error < threshold
            correct_predictions[zero_mask] = np.isclose(y_pred_test[zero_mask], 0, atol=1e-9)
            accuracies.append(1.0 if np.mean(correct_predictions) > 0.95 else 0.0)

    return float(mse), float(nmse), float(mape), accuracies, optimized_params
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_code_block(raw_code_input: str) -> str:
    raw_code = raw_code_input.strip()
    if "```python" in raw_code:
        return raw_code.split("```python")[-1].split("```")[0].strip()
    if "```" in raw_code:
        parts = raw_code.split("```")
        if len(parts) > 1:
            potential_code = parts[1]
            if "\n" in potential_code:
                first_line, rest_of_code = potential_code.split("\n", 1)
                return rest_of_code.strip() if first_line.strip().isalpha() else potential_code.strip()
            else:
                return potential_code.strip()
    return raw_code


def parse_equation(messages: list) -> List[str]:
    """Extract equation source strings from the assistant message history."""
    extracted_equations = []
    raw_code_strings = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            for tc in (msg.get("tool_calls") or []):
                if not isinstance(tc, dict):
                    continue
                fn_info = tc.get("function", {})
                if fn_info.get("name") == "equation_evaluator" and "arguments" in fn_info:
                    args = fn_info["arguments"]
                    if isinstance(args, str):
                        try:
                            parsed = json.loads(args)
                            if "equation" in parsed:
                                raw_code_strings.append(parsed["equation"])
                        except json.JSONDecodeError:
                            continue

    for code_str in raw_code_strings:
        processed = _clean_code_block(code_str)
        if processed:
            extracted_equations.append(processed)

    return extracted_equations


def _load_equation_fn(equation_string: str):
    """exec() the equation string and return the callable `equation`."""
    local_scope = {}
    exec(equation_string, {"np": np}, local_scope)
    return local_scope["equation"]


# ---------------------------------------------------------------------------
# Core evaluation helpers
# ---------------------------------------------------------------------------

def _execute_performance_evaluation(
    equations: List[str],
    code_snippet: str,
    eval_args: tuple,
    mape_threshold: float,
    n_restarts: int = 1,
) -> dict:
    """
    Execute evaluation for a list of equations.

    `n_restarts` is used only when the Hessian module is available; otherwise
    falls back to the original single-BFGS evaluation.
    """
    MSE, NMSE, MAPE = [], [], []
    ACC_0_1, ACC_0_01, ACC_0_001 = [], [], []
    Optimized_Params, SUCCESS = [], []
    HESSIAN_DIAGS: List[Optional[dict]] = []
    thresholds = [0.1, 0.01, 0.001]

    for equation_string in equations:
        success = 0
        hessian_info: Optional[dict] = None
        try:
            full_code = code_snippet.format(equation=equation_string)
            local_scope: dict = {}
            exec(full_code, exec_globals, local_scope)

            evaluate_func = local_scope["evaluate"]
            equation_func = local_scope["equation"]

            # --- Multi-restart BFGS if n_restarts > 1 and Hessian available ---
            if n_restarts > 1 and _HESSIAN_AVAILABLE:
                # Use the first element of eval_args as the dataset for Hessian
                train_data = eval_args[0]
                nmse_h, mape_h, mse_h, hdiag, theta_star = evaluate_with_hessian(
                    equation_func, train_data, n_restarts=n_restarts
                )
                hessian_info = hdiag.to_dict()
                # Still run standard evaluation for consistent metric output
                mse, nmse, mape, accuracies, optimized_params = evaluate_func(
                    equation_func, *eval_args, thresholds
                )
                # Prefer multi-restart MSE if it is better
                if mse is not None and np.isfinite(mse_h) and mse_h < mse:
                    mse = mse_h
                    nmse = nmse_h
                    mape = mape_h
                    optimized_params = theta_star
            else:
                mse, nmse, mape, accuracies, optimized_params = evaluate_func(
                    equation_func, *eval_args, thresholds
                )
                if _HESSIAN_AVAILABLE:
                    train_data = eval_args[0]
                    hdiag = evaluate_with_hessian.__wrapped__(equation_func, train_data) if hasattr(evaluate_with_hessian, "__wrapped__") else None
                    # Lightweight single-run hessian
                    try:
                        from .hessian import compute_hessian_diag
                        _hd = compute_hessian_diag(equation_func, train_data)
                        hessian_info = _hd.to_dict()
                    except Exception:
                        pass

            acc_0_1, acc_0_01, acc_0_001 = (accuracies if accuracies else [None, None, None])
            if mape is not None and mape < mape_threshold:
                success = 1

        except Exception as e:
            mse, nmse, mape = None, None, None
            acc_0_1, acc_0_01, acc_0_001 = None, None, None
            optimized_params = [None] * 10
            success = 0

        MSE.append(mse)
        NMSE.append(nmse)
        MAPE.append(mape)
        ACC_0_1.append(acc_0_1)
        ACC_0_01.append(acc_0_01)
        ACC_0_001.append(acc_0_001)
        Optimized_Params.append(
            optimized_params.tolist() if hasattr(optimized_params, "tolist") else optimized_params
        )
        SUCCESS.append(success)
        HESSIAN_DIAGS.append(hessian_info)

    return {
        "equations": equations,
        "mse": MSE,
        "nmse": NMSE,
        "mape": MAPE,
        "acc_0.1": ACC_0_1,
        "acc_0.01": ACC_0_01,
        "acc_0.001": ACC_0_001,
        "optimized_params": Optimized_Params,
        "success": SUCCESS,
        "hessian_diags": HESSIAN_DIAGS,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_performance(
    equations: List[str],
    dataset: list,
    mape_threshold: float = 0.001,
    n_restarts: int = 1,
) -> dict:
    """Evaluate equations on a single dataset (e.g. training set)."""
    return _execute_performance_evaluation(
        equations, code_snippet_single_dataset, (dataset,), mape_threshold,
        n_restarts=n_restarts,
    )


def get_performance_on_test(
    equations: List[str],
    train_dataset: list,
    test_dataset: list,
    mape_threshold: float = 0.001,
    n_restarts: int = 1,
) -> dict:
    """Fit on train, score on test."""
    return _execute_performance_evaluation(
        equations, code_snippet_train_test, (train_dataset, test_dataset),
        mape_threshold, n_restarts=n_restarts,
    )


def get_final_results(trainset_results: dict, testset_results: dict) -> dict:
    """
    Select the best equation by training-set MSE; return its test-set metrics.
    Unchanged from SR-Scientist.
    """
    train_mses = trainset_results.get("mse", [])
    valid_mses = [(mse, i) for i, mse in enumerate(train_mses) if mse is not None]

    if not valid_mses:
        return {
            "equation": None, "mse": None, "nmse": None, "mape": None,
            "acc_0.1": None, "acc_0.01": None, "acc_0.001": None,
            "optimized_params": None, "success": None,
            "hessian_diag": None,
        }

    _, best_index = min(valid_mses, key=lambda item: item[0])

    return {
        "equation":         testset_results["equations"][best_index],
        "mse":              testset_results["mse"][best_index],
        "nmse":             testset_results["nmse"][best_index],
        "mape":             testset_results["mape"][best_index],
        "acc_0.1":          testset_results["acc_0.1"][best_index],
        "acc_0.01":         testset_results["acc_0.01"][best_index],
        "acc_0.001":        testset_results["acc_0.001"][best_index],
        "optimized_params": testset_results["optimized_params"][best_index],
        "success":          trainset_results["success"][best_index],
        "hessian_diag":     testset_results.get("hessian_diags", [None])[best_index],
    }