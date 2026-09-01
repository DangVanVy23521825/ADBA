"""
Viz Agent — generate matplotlib chart code, execute in safe sandbox, return base64 PNG.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from graph.state import MultiAgentState
from graph.tools.python_tool import run_chart_safe
from graph.utils import append_trace, df_from_state, with_routing
from model.model_client import ModelClient

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "viz_generation.txt"

VIZ_SYSTEM_PROMPT: str = _PROMPT_PATH.read_text(encoding="utf-8")

MAX_RETRIES = 2


def _extract_code(text: str) -> str:
    """Strip markdown fences and pre-supplied imports from Python code."""
    text = re.sub(r"```(?:python)?\s*|\s*```", "", text).strip()
    lines = text.split("\n")
    clean = [l for l in lines if not (
        l.strip().startswith("import matplotlib") or
        l.strip().startswith("from matplotlib") or
        l.strip().startswith("import io") or
        l.strip().startswith("from io") or
        l.strip().startswith("import base64") or
        l.strip().startswith("from base64")
    )]
    return "\n".join(clean)


def _find_viz_step(state: MultiAgentState) -> dict | None:
    for step in state.get("execution_plan", []):
        if step.get("agent") == "viz":
            return step
    return None


def viz_agent_node(state: MultiAgentState) -> MultiAgentState:
    """Generate chart code via LLM → execute → return chart_b64 in state."""
    step = _find_viz_step(state)
    if step is None:
        logger.error("Viz Agent: no viz step in execution plan")
        return {**state, "status": "failed",
                "last_error": {"agent": "viz", "error_type": "missing_step"}}

    task: str = step.get("task", "")

    df_state = state.get("shared_dataframe")
    if not df_state:
        logger.error("Viz Agent: no shared_dataframe in state")
        return {**state, "status": "failed",
                "last_error": {"agent": "viz", "error_type": "missing_data"}}

    df = df_from_state(df_state)
    import json
    columns = df.columns.tolist()
    sample = json.dumps(df.head(3).to_dict("records"), ensure_ascii=False)

    system_prompt = (VIZ_SYSTEM_PROMPT
                     .replace("{columns}", str(columns))
                     .replace("{sample}", sample)
                     .replace("{task}", task))

    trace = append_trace(state, "viz", "generate_chart",
                         f"Task: {task}", "started")

    client = ModelClient(agent_type="viz")
    error_context = ""

    for attempt in range(1, MAX_RETRIES + 1):
        user_prompt = f"Task: {task}"

        reflector_diag = state.get("shared_metadata", {}).get("reflector_diagnosis", {})
        reflector_hint = reflector_diag.get("corrected_context", "")
        if reflector_hint:
            user_prompt += f"\n\n[HINT from error diagnosis] {reflector_hint}"

        if error_context:
            user_prompt += f"\n\nPrevious error:\n{error_context}\n\nFix and rewrite."

        try:
            raw = client.invoke(system_prompt=system_prompt, user_prompt=user_prompt)
        except Exception as exc:
            error_context = f"ModelClient error: {exc}"
            continue

        code = _extract_code(raw)

        # Try executing the LLM-generated code
        try:
            # Wrap bare dict literal at end so exec() captures it
            lines = code.strip().split("\n")
            if lines and lines[-1].strip().startswith("{"):
                lines[-1] = "result = " + lines[-1]
                code = "\n".join(lines)

            result = run_chart_safe(code, df)
            if "chart_b64" not in result:
                raise ValueError("Code did not produce a chart result dict")

        except Exception as exc:
            error_context = str(exc)
            logger.warning("Viz Agent error (attempt %d): %s", attempt, error_context)
            trace = append_trace(state, "viz", "execute_chart",
                                 f"Attempt {attempt}: {error_context}", "error")
            continue

        # ── Success ───────────────────────────────────
        chart_b64 = result.get("chart_b64", "")
        chart_type = result.get("chart_type", "unknown")
        trace = append_trace(state, "viz", "execute_chart",
                             f"OK — {chart_type} chart, "
                             f"{len(chart_b64)//1024} KB base64", "ok")

        return with_routing({
            **state,
            "current_agent": "viz",
            "completed_agents": state.get("completed_agents", []) + ["viz"],
            "agent_outputs": {
                **state.get("agent_outputs", {}),
                "viz": {"status": "ok", "chart_type": chart_type},
            },
            "chart_b64": chart_b64,
            "chart_metadata": {
                "chart_type": chart_type,
                "x_col": result.get("x_col", ""),
                "y_col": result.get("y_col", ""),
                "title": result.get("title", ""),
            },
            "shared_metadata": {
                **state.get("shared_metadata", {}),
                "chart_type": chart_type,
                "chart_metadata": {
                    "chart_type": chart_type,
                    "x_col": result.get("x_col", ""),
                    "y_col": result.get("y_col", ""),
                    "title": result.get("title", ""),
                },
            },
            "action_trace": trace,
            "status": "running",
        })

    # ── Exhausted retries ──────────────────────────
    error_summary = f"Viz Agent failed after {MAX_RETRIES} attempts: {error_context}"
    trace = append_trace(state, "viz", "execute_chart", error_summary, "error")

    return {
        **state,
        "current_agent": "viz",
        "agent_outputs": {
            **state.get("agent_outputs", {}),
            "viz": {"status": "error", "error": error_summary},
        },
        "error_count": state.get("error_count", 0) + 1,
        "agent_error_counts": {
            **state.get("agent_error_counts", {}),
            "viz": state.get("agent_error_counts", {}).get("viz", 0) + 1,
        },
        "last_error": {"agent": "viz", "error_type": "chart_error",
                       "traceback": error_summary},
        "action_trace": trace,
        "status": "running",
    }
