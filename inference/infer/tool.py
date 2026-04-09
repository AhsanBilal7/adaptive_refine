"""
tool.py – Sandbox-backed tool execution for AdaptSci / SR-Scientist.

AdaptSci changes vs. original SR-Scientist tool.py:
  • `equation_evaluator` now accepts an optional `n_restarts` argument.
    When n_restarts > 1, multiple BFGS initialisations are run inside the
    sandbox code template; the best result is kept and the Hessian-derived
    signals (κ, n_eff) are printed alongside the standard MAPE output.
  • The Hessian computation uses purely numerical finite-difference code that
    executes fully inside the sandbox – no additional imports needed.
  • All original behaviour (single-restart, original output format) is
    preserved as the default (n_restarts=1).
"""

import asyncio
import httpx
import json
import logging
import os
import threading
import time
import traceback
import uuid
import numpy as np
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DEFAULT_TIMEOUT = 10
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1
API_TIMEOUT = 2700


def convert_numpy_types(obj):
    if isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    else:
        return obj


# ---------------------------------------------------------------------------
# Tool schemas (unchanged)
# ---------------------------------------------------------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "equation_evaluator",
            "description": (
                "Accepts a mathematical equation as a Python function string, "
                "optimizes its parameters to fit a dataset using BFGS (with optional "
                "multi-restart for ill-conditioned surfaces), and returns performance "
                "metrics (MSE, NMSE, MAPE) plus Hessian-derived diagnostics "
                "(condition number κ, effective parameter count n_eff)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "equation": {
                        "type": "string",
                        "description": "The equation to evaluate, as a complete Python function string.",
                    }
                },
                "required": ["equation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "data_analyzer",
            "description": (
                "Executes Python code for data analysis and exploration on the dataset. "
                "Supports statistical analysis, correlation, residual inspection. "
                "Does NOT support plotting libraries such as Matplotlib."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code snippet for data analysis.",
                    }
                },
                "required": ["code"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Sandbox API helpers (unchanged)
# ---------------------------------------------------------------------------

def correct_stderr_line_numbers(stderr_text: str, offset: int) -> str:
    if not stderr_text:
        return ""
    traceback_pattern = re.compile(r"(File \".*?\", line )(\d+)")
    warning_pattern = re.compile(r"([^\"\s]+?\.py:)(\d+)")
    corrected_lines = []
    for line in stderr_text.splitlines():
        if traceback_pattern.search(line):
            corrected_line = re.sub(
                traceback_pattern,
                lambda m: f"{m.group(1)}{int(m.group(2)) - offset}",
                line,
            )
        elif warning_pattern.search(line):
            corrected_line = re.sub(
                warning_pattern,
                lambda m: f"{m.group(1)}{int(m.group(2)) - offset}",
                line,
            )
        else:
            corrected_line = line
        corrected_lines.append(corrected_line)
    return "\n".join(corrected_lines)


async def call_sandbox_api(
    sandbox_fusion_url: str,
    code: str,
    stdin: str,
    compile_timeout: int,
    run_timeout: int,
    memory_limit_mb: int,
    language: str = "python",
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Async sandbox API call with retry logic (unchanged from SR-Scientist)."""
    request_id = str(uuid.uuid4())
    log_prefix = f"[Request ID: {request_id}] "
    payload = json.dumps({
        "compile_timeout": compile_timeout,
        "run_timeout": run_timeout,
        "code": code,
        "stdin": stdin,
        "memory_limit_MB": memory_limit_mb,
        "language": language,
        "files": {},
        "fetch_files": [],
    })
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    request_timeout = compile_timeout + run_timeout + API_TIMEOUT
    last_error = None

    async with httpx.AsyncClient() as client:
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.post(
                    sandbox_fusion_url,
                    headers=headers,
                    data=payload,
                    timeout=request_timeout,
                )
                if response.status_code == 504:
                    last_error = f"{log_prefix}Gateway Timeout (504) attempt {attempt+1}"
                    logger.warning(last_error)
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(INITIAL_RETRY_DELAY * (2**attempt))
                    continue
                response.raise_for_status()
                return response.json(), None
            except httpx.RequestError as e:
                last_error = f"{log_prefix}Request error: {e}"
                break
            except json.JSONDecodeError as e:
                raw = response.text if "response" in locals() else "N/A"
                last_error = f"{log_prefix}JSON decode error: {e}. Raw: {raw}"
                break
            except Exception as e:
                last_error = f"{log_prefix}Unexpected: {e}"
                break

    return None, (last_error or "API call failed after retries")


async def _execute_code(
    code: str,
    stdin_data: Any,
    sandbox_fusion_url: str,
    timeout: int,
    memory_limit_mb: int,
    language: str,
    concurrent_semaphore: Optional[threading.Semaphore] = None,
    tool_name: str = "data_analyzer",
) -> Tuple[int, Dict[str, Any]]:
    """Execute code inside sandbox; return (result_status, metadata)."""
    raw_code = code.strip()
    processed_code = raw_code

    if "```python" in raw_code:
        processed_code = raw_code.split("```python")[-1].split("```")[0].strip()
    elif "```" in raw_code:
        parts = raw_code.split("```")
        if len(parts) > 1:
            potential_code = parts[1]
            if "\n" in potential_code:
                first_line, rest = potential_code.split("\n", 1)
                processed_code = rest.strip() if first_line.strip().isalpha() else potential_code.strip()
            else:
                processed_code = potential_code.strip()

    header = (
        "import os\n"
        "os.environ['OPENBLAS_NUM_THREADS'] = '1'\n"
        "os.environ['OMP_NUM_THREADS'] = '1'\n"
        "os.environ['MKL_NUM_THREADS'] = '1'\n"
    )
    code = header + processed_code
    stdin_data = json.dumps(convert_numpy_types(stdin_data))

    metadata: Dict[str, Any] = {
        "tool_name": tool_name,
        "stdout": None,
        "stderr": None,
        "status": "unknown",
        "run_status": None,
    }

    api_response, error_msg = await call_sandbox_api(
        sandbox_fusion_url=sandbox_fusion_url,
        code=code,
        stdin=str(stdin_data),
        compile_timeout=timeout,
        run_timeout=timeout,
        memory_limit_mb=memory_limit_mb,
        language=language,
    )

    result_status: int = -1

    if error_msg:
        metadata["status"] = "api_error"
        metadata["api_error"] = error_msg
    elif api_response:
        metadata["api_response"] = api_response
        metadata["api_status"] = api_response.get("status")
        compile_result = api_response.get("compile_result")
        run_result = api_response.get("run_result")

        if compile_result:
            metadata["compile_status"] = compile_result.get("status")
            metadata["compile_duration"] = compile_result.get("execution_time")
            metadata["compile_stderr"] = compile_result.get("stderr")
        if run_result:
            metadata["run_status"] = run_result.get("status")
            metadata["stdout"] = run_result.get("stdout")
            metadata["stderr"] = run_result.get("stderr")
            metadata["exit_code"] = run_result.get("return_code")
            metadata["duration"] = run_result.get("execution_time")

        api_status = metadata["api_status"]
        if api_status == "SandboxError":
            metadata["status"] = "sandbox_error"
        elif api_status == "Failed":
            is_compile_error = compile_result and (
                metadata.get("compile_status") in ["Error", "TimeLimitExceeded"]
                or (
                    metadata.get("compile_status") == "Finished"
                    and compile_result.get("return_code") != 0
                )
            )
            if is_compile_error:
                metadata["status"] = (
                    "compile_timeout"
                    if metadata.get("compile_status") == "TimeLimitExceeded"
                    else "compile_error"
                )
                result_status = -4
            elif run_result:
                is_runtime_error = (
                    metadata["run_status"] == "TimeLimitExceeded"
                    or metadata["run_status"] == "Error"
                    or (
                        metadata["run_status"] == "Finished"
                        and run_result.get("return_code") != 0
                    )
                )
                if is_runtime_error:
                    metadata["status"] = (
                        "timeout"
                        if metadata["run_status"] == "TimeLimitExceeded"
                        else "runtime_error"
                    )
                    result_status = -3 if metadata["status"] == "timeout" else -2
                else:
                    metadata["status"] = "unknown_failure"
            else:
                metadata["status"] = "unknown_failure_state"
        elif api_status == "Success":
            if run_result and metadata["run_status"] == "Finished":
                result_status = 1
                metadata["status"] = "success"
            else:
                metadata["status"] = "unexpected_success_state"
        else:
            metadata["status"] = f"unknown_api_status_{api_status}"
    else:
        metadata["status"] = "unknown_api_state"

    return result_status, metadata


# ---------------------------------------------------------------------------
# Multi-restart code template (used when n_restarts > 1)
# ---------------------------------------------------------------------------

_EQUATION_EVALUATOR_MULTI_RESTART_TEMPLATE = """
import numpy as np
import sys
import json
from scipy.optimize import minimize

MAX_NPARAMS = 10
params = [1.0] * MAX_NPARAMS

{equation}

def _detect_active_params(theta_star, threshold=1e-3):
    x0 = np.ones_like(theta_star)
    displacement = np.abs(theta_star - x0)
    active = [i for i, d in enumerate(displacement) if d > threshold]
    if len(active) < 1:
        active = [int(np.argmax(displacement))]
    return sorted(active)

def _finite_diff_hessian_subspace(loss_fn, theta, active_indices, eps=1e-4):
    n = len(active_indices)
    p = len(theta)
    H = np.zeros((n, n))
    for ii, gi in enumerate(active_indices):
        ei = np.zeros(p); ei[gi] = eps
        for jj, gj in enumerate(active_indices[ii:], start=ii):
            ej = np.zeros(p); ej[gj] = eps
            try:
                with np.errstate(all='ignore'):
                    fpp = loss_fn(theta + ei + ej)
                    fpm = loss_fn(theta + ei - ej)
                    fmp = loss_fn(theta - ei + ej)
                    fmm = loss_fn(theta - ei - ej)
                    h_ij = (fpp - fpm - fmp + fmm) / (4.0 * eps ** 2)
                if not np.isfinite(h_ij): h_ij = 0.0
            except Exception:
                h_ij = 0.0
            H[ii, jj] = h_ij; H[jj, ii] = h_ij
    return H

def compute_hessian_signals(loss_fn, theta):
    active = _detect_active_params(theta)
    n = len(active)
    H = _finite_diff_hessian_subspace(loss_fn, theta, active)
    H = 0.5 * (H + H.T) + 1e-8 * np.eye(n)
    eigenvalues = np.clip(np.linalg.eigvalsh(H), 0.0, None)
    lmax = float(eigenvalues[-1])
    lmin = float(eigenvalues[0])
    kappa = lmax / lmin if lmin > 1e-12 else float('inf')
    tr_H = float(np.trace(H))
    tr_H2 = float(np.trace(H @ H))
    n_eff = (tr_H ** 2) / tr_H2 if tr_H2 > 1e-20 else float(n)
    return kappa, n_eff, n

def evaluate(data):
    outputs = np.array([row[0] for row in data])
    inputs = np.array([row[1:] for row in data])
    X = inputs

    def loss(p):
        try:
            with np.errstate(all='ignore'):
                y_pred = equation(*X.T, p)
                val = np.mean((y_pred - outputs) ** 2)
            return float(val) if np.isfinite(val) else 1e30
        except Exception:
            return 1e30

    best_mse = float('inf')
    best_params = np.ones(MAX_NPARAMS)
    rng = np.random.default_rng(42)
    n_restarts = {n_restarts}

    for r in range(n_restarts):
        x0 = np.ones(MAX_NPARAMS) if r == 0 else rng.normal(0, 3.0, MAX_NPARAMS)
        try:
            res = minimize(loss, x0, method='BFGS')
            if np.isfinite(res.fun) and res.fun < best_mse:
                best_mse = res.fun
                best_params = res.x
        except Exception:
            continue

    if np.isnan(best_mse) or np.isinf(best_mse):
        return None, None, None, None, None, None

    var_outputs = np.var(outputs)
    nmse = best_mse / var_outputs if not np.isclose(var_outputs, 0) else (0.0 if np.isclose(best_mse, 0) else np.inf)

    with np.errstate(all='ignore'):
        y_pred = equation(*X.T, best_params)
    zero_mask = np.isclose(outputs, 0)
    non_zero_mask = ~zero_mask
    mape = 0.0
    if np.any(non_zero_mask):
        relative_errors = np.abs((y_pred[non_zero_mask] - outputs[non_zero_mask]) / outputs[non_zero_mask])
        mape = np.mean(relative_errors)

    try:
        kappa, n_eff, n_active = compute_hessian_signals(loss, best_params)
    except Exception:
        kappa, n_eff, n_active = float('inf'), 1.0, 1

    return float(best_mse), float(nmse), float(mape), float(kappa), float(n_eff), int(n_active)


if __name__ == '__main__':
    input_data_str = sys.stdin.read()
    data_list = json.loads(input_data_str)
    mse, nmse, mape, kappa, n_eff, n_active = evaluate(data_list)

    if mse is not None:
        print(f"MSE loss: {{mse:.6e}}; NMSE loss: {{nmse:.6e}}; Mean absolute percentage error: {{mape:.4%}}")
        print(f"Hessian (active {{n_active}} params): kappa={{kappa:.2e}}, n_eff={{n_eff:.2f}}")
        if mape < {mape_threshold}:
            print("Success: The mean absolute percentage error is smaller than {mape_threshold_percentage}.")
        else:
            print("Failure: The mean absolute percentage error is larger than {mape_threshold_percentage}.")
"""

# Original single-restart template (unchanged)
_EQUATION_EVALUATOR_SINGLE_RESTART_TEMPLATE = """
import numpy as np
import sys
import json

MAX_NPARAMS = 10
params = [1.0] * MAX_NPARAMS

{equation}

def evaluate(data: list) -> float:
    outputs = np.array([row[0] for row in data])
    inputs = np.array([row[1:] for row in data])
    X = inputs
    from scipy.optimize import minimize
    def loss(params):
        try:
            y_pred = equation(*X.T, params)
            return np.mean((y_pred - outputs) ** 2)
        except (FloatingPointError, OverflowError):
            return np.inf

    result = minimize(loss, [1.0] * MAX_NPARAMS, method='BFGS')
    optimized_params = result.x
    mse = result.fun

    if np.isnan(mse) or np.isinf(mse):
        return None, None, None

    var_outputs = np.var(outputs)
    if np.isclose(var_outputs, 0):
        nmse = 0.0 if np.isclose(mse, 0) else np.inf
    else:
        nmse = mse / var_outputs

    y_pred = equation(*X.T, optimized_params)
    zero_mask = np.isclose(outputs, 0)
    non_zero_mask = ~zero_mask
    mape = 0.0
    if np.any(non_zero_mask):
        relative_errors = np.abs((y_pred[non_zero_mask] - outputs[non_zero_mask]) / outputs[non_zero_mask])
        mape = np.mean(relative_errors)

    return float(mse), float(nmse), float(mape)


if __name__ == '__main__':
    input_data_str = sys.stdin.read()
    data_list = json.loads(input_data_str)
    mse, nmse, mape = evaluate(data_list)

    if mse is not None:
        print(f"MSE loss: {{mse:.6e}}; NMSE loss: {{nmse:.6e}}; Mean absolute percentage error: {{mape:.4%}}")
        if mape < {mape_threshold}:
            print("Success: The mean absolute percentage error is smaller than {mape_threshold_percentage}.")
        else:
            print("Failure: The mean absolute percentage error is larger than {mape_threshold_percentage}.")
"""


# ---------------------------------------------------------------------------
# Public tool functions
# ---------------------------------------------------------------------------

async def execute_code(
    code: str,
    stdin_data: Any,
    sandbox_fusion_url: str = "http://0.0.0.0:8080/run_code",
    memory_limit_mb: int = 1024,
    timeout: int = 30,
    language: str = "python",
) -> Tuple[str, Dict]:
    """data_analyzer tool (unchanged)."""
    result_status, metadata = await _execute_code(
        code, stdin_data, sandbox_fusion_url=sandbox_fusion_url,
        timeout=timeout, memory_limit_mb=memory_limit_mb, language=language,
    )
    stdout = metadata.get("stdout", "") or ""
    stderr = metadata.get("stderr", "") or ""
    stderr = correct_stderr_line_numbers(stderr, offset=4)

    if metadata.get("run_status") == "Finished":
        actual_output = f"STDOUT:\n{stdout}" if not stderr else f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
        return actual_output, metadata
    else:
        failed_output = (
            f"Execution Failed: {metadata.get('run_status', 'Unknown')}\n\n"
            f"---STDOUT---\n{stdout}\n\n---STDERR---\n{stderr}"
        )
        return failed_output, metadata


async def equation_evaluator(
    equation: str,
    stdin_data: Any,
    sandbox_fusion_url: str = "http://0.0.0.0:8080/run_code",
    memory_limit_mb: int = 1024,
    timeout: int = 60,
    language: str = "python",
    mape_threshold: float = 0.001,
    n_restarts: int = 1,
) -> Tuple[str, Dict]:
    """
    equation_evaluator tool.

    AdaptSci extension: when n_restarts > 1 the multi-restart template is used
    so that the sandbox itself runs multiple BFGS initialisations and reports
    Hessian signals (κ, n_eff) in the output the LLM sees.
    """
    mape_threshold_percentage = f"{mape_threshold:.4%}"

    # Clean code fences
    raw_code = equation.strip()
    cleaned = raw_code
    if "```python" in raw_code:
        cleaned = raw_code.split("```python")[-1].split("```")[0].strip()
    elif "```" in raw_code:
        parts = raw_code.split("```")
        if len(parts) > 1:
            potential = parts[1]
            if "\n" in potential:
                first_line, rest = potential.split("\n", 1)
                cleaned = rest.strip() if first_line.strip().isalpha() else potential.strip()
            else:
                cleaned = potential.strip()

    # Choose template
    if n_restarts > 1:
        template = _EQUATION_EVALUATOR_MULTI_RESTART_TEMPLATE
        code = template.format(
            equation=cleaned,
            n_restarts=n_restarts,
            mape_threshold=mape_threshold,
            mape_threshold_percentage=mape_threshold_percentage,
        )
    else:
        template = _EQUATION_EVALUATOR_SINGLE_RESTART_TEMPLATE
        code = template.format(
            equation=cleaned,
            mape_threshold=mape_threshold,
            mape_threshold_percentage=mape_threshold_percentage,
        )

    result_status, metadata = await _execute_code(
        code, stdin_data,
        sandbox_fusion_url=sandbox_fusion_url,
        timeout=timeout,
        memory_limit_mb=memory_limit_mb,
        language=language,
        tool_name="equation_evaluator",
    )
    stdout = metadata.get("stdout", "") or ""
    stderr = metadata.get("stderr", "") or ""
    stderr = correct_stderr_line_numbers(stderr, offset=4)

    if metadata.get("run_status") == "Finished":
        actual_output = f"STDOUT:\n{stdout}" if not stderr else f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
        return actual_output, metadata
    else:
        failed_output = (
            f"Execution Failed: {metadata.get('run_status', 'Unknown')}\n\n"
            f"---STDOUT---\n{stdout}\n\n---STDERR---\n{stderr}"
        )
        return failed_output, metadata


def get_function_by_name(name: str):
    if name == "data_analyzer":
        return execute_code
    elif name == "equation_evaluator":
        return equation_evaluator
    raise ValueError(f"Unknown tool: {name}")