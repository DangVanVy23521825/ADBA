"""
Python Agent — generate and execute pandas analysis code in safe sandbox.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from graph.state import MultiAgentState
from graph.tools.python_tool import run_pandas_safe
from graph.utils import append_trace, df_from_state, df_to_state, with_routing
from model.model_client import ModelClient

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "data_analysis.txt"

PYTHON_SYSTEM_PROMPT: str = _PROMPT_PATH.read_text(encoding="utf-8")

MAX_RETRIES = 2


def _extract_code(text: str) -> str:
    """Strip markdown fences and extract Python code."""
    text = re.sub(r"```(?:python)?\s*|\s*```", "", text).strip()
    return text


def _find_python_step(state: MultiAgentState) -> dict | None:
    """Find the python step in the execution plan."""
    for step in state.get("execution_plan", []):
        if step.get("agent") == "python":
            return step
    return None


def _build_prompt(sys_template: str, task: str, df_sample: str, columns: list[str]) -> str:
    """Render the system prompt with column info and sample data."""
    return (sys_template
            .replace("{columns}", str(columns))
            .replace("{sample}", df_sample)
            .replace("{task}", task))


def python_agent_node(state: MultiAgentState) -> MultiAgentState:
    """Generate Python analysis code → execute in safe sandbox → update state.

    Reads shared_dataframe from state. On failure retries once with error context.
    """
    step = _find_python_step(state)
    if step is None:
        logger.error("Python Agent: no python step in execution plan")
        return {
            **state,
            "status": "failed",
            "last_error": {"agent": "python", "error_type": "missing_step"},
        }

    task: str = step.get("task", "")

    # ── Input DataFrame check ───────────────────────────────────
    df_state = state.get("shared_dataframe")
    if not df_state:
        logger.error("Python Agent: no shared_dataframe in state")
        return {
            **state,
            "status": "failed",
            "last_error": {
                "agent": "python",
                "error_type": "missing_data",
                "traceback": "shared_dataframe is empty — sql agent must run first",
            },
        }

    df_in = df_from_state(df_state)
    columns = df_in.columns.tolist()
    sample_rows = json.dumps(df_in.head(3).to_dict("records"), ensure_ascii=False)

    system_prompt = _build_prompt(PYTHON_SYSTEM_PROMPT, task, sample_rows, columns)

    trace = append_trace(state, "python", "generate_code",
                         f"Task: {task} | cols: {columns}", "started")

    client = ModelClient(agent_type="python")
    error_context: str = ""

    for attempt in range(1, MAX_RETRIES + 1):
        user_prompt = f"Task: {task}"

        reflector_diag = state.get("shared_metadata", {}).get("reflector_diagnosis", {})
        reflector_hint = reflector_diag.get("corrected_context", "")
        if reflector_hint:
            user_prompt += f"\n\n[HINT from error diagnosis] {reflector_hint}"

        if error_context:
            user_prompt += (f"\n\nPrevious attempt failed with error:\n{error_context}\n\n"
                            "Fix the error and rewrite the code.")

        try:
            raw = client.invoke(system_prompt=system_prompt, user_prompt=user_prompt)
        except Exception as exc:
            error_context = f"ModelClient error: {exc}"
            logger.warning("Python Agent attempt %d: %s", attempt, error_context)
            continue

        code = _extract_code(raw)
        logger.info("Python Agent attempt %d code preview:\n%s", attempt, code[:300])

        try:
            df_out = run_pandas_safe(code, df_in)
        except Exception as exc:
            error_context = str(exc)
            logger.warning("Python Agent execution error (attempt %d): %s", attempt, error_context)
            trace = append_trace(state, "python", "execute_code",
                                 f"Attempt {attempt} error: {error_context}", "error")
            continue

        # ── Success ───────────────────────────────────────────────
        stats = {
            "input_rows": len(df_in),
            "output_rows": len(df_out),
            "new_columns": list(set(df_out.columns) - set(df_in.columns)),
            "output_columns": df_out.columns.tolist(),
        }
        trace = append_trace(state, "python", "execute_code",
                             f"OK — {len(df_out)} rows, new cols: {stats['new_columns']}", "ok")
        logger.info("Python Agent: %d rows output, %d new columns",
                    len(df_out), len(stats["new_columns"]))

        return with_routing({
            **state,
            "current_agent": "python",
            "completed_agents": state.get("completed_agents", []) + ["python"],
            "agent_outputs": {
                **state.get("agent_outputs", {}),
                "python": {"status": "ok", "code": code, "stats": stats},
            },
            "shared_dataframe": df_to_state(df_out),
            "python_result": {"code": code, "stats": stats},
            "shared_metadata": {
                **state.get("shared_metadata", {}),
                "python_row_count": len(df_out),
                "python_stats": stats,
                "python_result": {"code": code, "stats": stats},
            },
            "action_trace": trace,
            "status": "running",
        })

    # ── Exhausted retries ─────────────────────────────────────
    error_summary = f"Python Agent failed after {MAX_RETRIES} attempts. Last error: {error_context}"
    trace = append_trace(state, "python", "execute_code", error_summary, "error")

    return {
        **state,
        "current_agent": "python",
        "agent_outputs": {
            **state.get("agent_outputs", {}),
            "python": {"status": "error", "error": error_summary},
        },
        "error_count": state.get("error_count", 0) + 1,
        "agent_error_counts": {
            **state.get("agent_error_counts", {}),
            "python": state.get("agent_error_counts", {}).get("python", 0) + 1,
        },
        "last_error": {
            "agent": "python",
            "error_type": "python_runtime",
            "traceback": error_summary,
        },
        "action_trace": trace,
        "status": "running",
    }
