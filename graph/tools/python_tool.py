"""
Restricted Python execution sandbox for the Data Analysis Agent.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats

logger = logging.getLogger(__name__)
PANDAS_EXEC_TIMEOUT_SECONDS = float(os.getenv("PANDAS_EXEC_TIMEOUT_SECONDS", "10"))

# Whitelist of modules that can be imported inside the sandbox.
_ALLOWED_IMPORTS = frozenset({
    "pandas", "numpy", "scipy", "scipy.stats",
})


def _restricted_import(name: str, *args, **kwargs):
    """Only allow imports of whitelisted modules (pandas, numpy, scipy)."""
    top_level = name.split(".")[0]
    if name in _ALLOWED_IMPORTS or top_level in _ALLOWED_IMPORTS:
        return __import__(name, *args, **kwargs)
    raise ImportError(f"Module '{name}' is not available in the safe sandbox.")


# Restricted execution namespace — whitelist only.
PYTHON_SAFE_NAMESPACE: dict = {
    "__builtins__": {
        "__import__": _restricted_import,
        "abs": abs,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "isinstance": isinstance,
        "KeyError": KeyError,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "print": print,
        "range": range,
        "round": round,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "zip": zip,
    },
    "pd": pd,
    "np": np,
    "stats": stats,
}


def _sandbox_worker(
    code: str,
    local_vars: dict[str, Any],
    result_picker: Callable[[dict[str, Any]], Any],
    result_queue: mp.Queue,
) -> None:
    try:
        global_vars = dict(PYTHON_SAFE_NAMESPACE)
        global_vars.update(local_vars)
        compiled = compile(code, "<python_sandbox>", "exec")
        exec(compiled, global_vars, local_vars)
        result_queue.put(("ok", result_picker(local_vars)))
    except Exception as exc:
        result_queue.put(("error", str(exc)))


def _run_python_safe(
    code: str,
    local_vars: dict[str, Any],
    result_picker: Callable[[dict[str, Any]], Any],
    timeout_seconds: float = PANDAS_EXEC_TIMEOUT_SECONDS,
) -> Any:
    ctx = mp.get_context("fork")
    result_queue: mp.Queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_sandbox_worker, args=(code, local_vars, result_picker, result_queue))
    proc.start()

    try:
        status, payload = result_queue.get(timeout=timeout_seconds)
    except queue.Empty as exc:
        proc.terminate()
        proc.join()
        raise RuntimeError(f"Python execution timed out after {timeout_seconds:g}s") from exc

    proc.join(timeout=1)
    if proc.is_alive():
        proc.terminate()
        proc.join()

    if status == "error":
        raise RuntimeError(f"Python execution failed: {payload}")
    return payload


def _pick_dataframe_result(local_vars: dict[str, Any], original_df: pd.DataFrame) -> pd.DataFrame:
    for key, val in reversed(list(local_vars.items())):
        if isinstance(val, pd.DataFrame) and key != "df":
            return val

    result_df = local_vars.get("df")
    if isinstance(result_df, pd.DataFrame):
        if result_df.shape != original_df.shape or not result_df.equals(original_df):
            return result_df

    raise RuntimeError(
        "Code did not produce a DataFrame result. "
        "Ensure the last expression evaluates to a DataFrame."
    )


def _pick_dict_result(local_vars: dict[str, Any]) -> dict:
    result = local_vars.get("result")
    if isinstance(result, dict):
        return result


    for val in reversed(list(local_vars.values())):
        if isinstance(val, dict):
            return val


    raise RuntimeError("Code did not produce a dict result.")


def run_pandas_safe(
    code: str,
    df: pd.DataFrame,
    timeout_seconds: float = PANDAS_EXEC_TIMEOUT_SECONDS,
) -> pd.DataFrame:
    """Execute user-supplied Python code in a restricted sandbox.

    Args:
        code: Python code string. The last expression must evaluate to a DataFrame.
        df: Input DataFrame available as the variable `df` inside the sandbox.

    Returns:
        The resulting DataFrame from the last expression.

    Raises:
        RuntimeError: on any execution failure.
    """
    local_vars: dict[str, Any] = {"df": df.copy()}
    return _run_python_safe(
        code,
        local_vars,
        lambda vars_: _pick_dataframe_result(vars_, df),
        timeout_seconds,
    )


def run_dict_safe(
    code: str,
    local_vars: dict[str, Any],
    timeout_seconds: float = PANDAS_EXEC_TIMEOUT_SECONDS,
) -> dict:
    """Execute sandboxed Python code and return a dict named/assigned as result."""
    return _run_python_safe(code, local_vars, _pick_dict_result, timeout_seconds)
