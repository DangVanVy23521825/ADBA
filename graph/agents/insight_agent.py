"""
Insight Agent — generate structured business insight from all agent outputs.
"""

from __future__ import annotations

import logging
from pathlib import Path

from graph.state import MultiAgentState
from graph.utils import append_trace
from model.model_client import ModelClient
from schemas.insight_schema import InsightOutput

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "insight_generation.txt"

INSIGHT_SYSTEM_PROMPT: str = _PROMPT_PATH.read_text(encoding="utf-8")


def _find_insight_step(state: MultiAgentState) -> dict | None:
    for step in state.get("execution_plan", []):
        if step.get("agent") == "insight":
            return step
    return None


def insight_agent_node(state: MultiAgentState) -> MultiAgentState:
    """Gather all agent outputs → call ModelClient.invoke_json() → validate InsightOutput."""
    step = _find_insight_step(state)
    if step is None:
        logger.error("Insight Agent: no insight step in execution plan")
        return {**state, "status": "failed",
                "last_error": {"agent": "insight", "error_type": "missing_step"}}

    query: str = state["query"]

    # Gather context from previous agents
    sql_info = state.get("agent_outputs", {}).get("sql", {})
    sql_text = sql_info.get("sql", state.get("shared_metadata", {}).get("sql_query", "N/A"))

    python_info = state.get("agent_outputs", {}).get("python", {})
    stats = python_info.get("stats", state.get("shared_metadata", {}).get("python_stats", {}))

    anomalies_info = "None detected"
    df_state = state.get("shared_dataframe")
    if df_state:
        from graph.tools.insight_tools import detect_anomaly
        from graph.utils import df_from_state
        try:
            df = df_from_state(df_state)
            num_cols = df.select_dtypes(include="number").columns.tolist()
            anomaly_results = []
            for col in num_cols:
                result = detect_anomaly(df, col)
                if result["anomaly_count"] > 0:
                    anomaly_results.append({"column": col, **result})
            if anomaly_results:
                anomalies_info = str(anomaly_results)
        except Exception:
            pass

    system_prompt = (INSIGHT_SYSTEM_PROMPT
                     .replace("{query}", query)
                     .replace("{sql}", str(sql_text))
                     .replace("{stats}", str(stats))
                     .replace("{anomalies}", str(anomalies_info)))

    trace = append_trace(state, "insight", "generate_insight",
                         f"Query: {query}", "started")

    try:
        client = ModelClient(agent_type="insight")
        raw = client.invoke_json(system_prompt=system_prompt, user_prompt=query)

        # Pydantic validation
        validated = InsightOutput.model_validate(raw)

        trace = append_trace(state, "insight", "generate_insight",
                             f"OK — confidence: {validated.confidence}", "ok")

        return {
            **state,
            "current_agent": "insight",
            "completed_agents": state.get("completed_agents", []) + ["insight"],
            "agent_outputs": {
                **state.get("agent_outputs", {}),
                "insight": {"status": "ok", "output": validated.model_dump()},
            },
            "insight": validated.model_dump(),
            "action_trace": trace,
            "status": "success",
        }

    except Exception as exc:
        error_msg = f"Insight Agent failed: {exc}"
        logger.error(error_msg)
        trace = append_trace(state, "insight", "generate_insight", error_msg, "error")

        return {
            **state,
            "current_agent": "insight",
            "agent_outputs": {
                **state.get("agent_outputs", {}),
                "insight": {"status": "error", "error": error_msg},
            },
            "error_count": state.get("error_count", 0) + 1,
            "agent_error_counts": {
                **state.get("agent_error_counts", {}),
                "insight": state.get("agent_error_counts", {}).get("insight", 0) + 1,
            },
            "last_error": {
                "agent": "insight",
                "error_type": "insight_generation_failure",
                "traceback": error_msg,
            },
            "action_trace": trace,
            "status": "failed",
        }
