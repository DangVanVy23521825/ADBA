"""
Insight Agent — generate structured business insight from all agent outputs.
"""

from __future__ import annotations

import logging
import json
from pathlib import Path

from graph.budget import node_may_run
from graph.errors import BUDGET_EXCEEDED, INSIGHT_GENERATION, MISSING_STEP
from graph.state import MultiAgentState
from graph.utils import append_trace
from model.model_client import BudgetExceededError, ModelClient
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
    may_run, reason = node_may_run(state, "insight")
    if not may_run:
        logger.warning("Insight Agent: %s", reason)
        return {
            **state,
            "current_agent": "insight",
            "completed_agents": state.get("completed_agents", []) + ["insight"],
            "degradation_reason": state.get("degradation_reason", []) + [reason],
            "action_trace": append_trace(state, "insight", "budget_check", reason, "error"),
            "status": "running",
        }

    step = _find_insight_step(state)
    if step is None:
        logger.error("Insight Agent: no insight step in execution plan")
        return {**state, "status": "failed",
                "last_error": {"agent": "insight", "error_type": MISSING_STEP}}

    query: str = state["query"]

    # Gather JSON-serializable context from previous agents.
    agent_outputs = state.get("agent_outputs", {})
    shared_metadata = state.get("shared_metadata", {})
    sql_info = agent_outputs.get("sql", {})
    sql_result = state.get("sql_result") or shared_metadata.get("sql_result") or sql_info
    sql_text = sql_info.get("sql", shared_metadata.get("sql_query", "N/A"))

    python_info = agent_outputs.get("python", {})
    python_result = state.get("python_result") or shared_metadata.get("python_result") or python_info
    stats = python_info.get("stats", shared_metadata.get("python_stats", {}))
    chart_metadata = state.get("chart_metadata") or shared_metadata.get("chart_metadata") or {}
    chart_description = json.dumps(chart_metadata, ensure_ascii=False)

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
                result = detect_anomaly(df, col, method="sigma", sigma_threshold=2.0)
                if result["anomaly_count"] > 0:
                    anomaly_results.append({"column": col, **result})
            if anomaly_results:
                anomalies_info = json.dumps(anomaly_results, ensure_ascii=False)
        except Exception:
            pass

    insight_context = {
        "query": query,
        "sql_result": sql_result,
        "python_result": python_result,
        "stats": stats,
        "anomalies": anomalies_info,
        "chart_metadata": chart_metadata,
        "dataframe_shape": (state.get("shared_dataframe") or {}).get("shape"),
        "dataframe_columns": (state.get("shared_dataframe") or {}).get("columns"),
    }

    system_prompt = (INSIGHT_SYSTEM_PROMPT
                      .replace("{query}", query)
                      .replace("{sql}", str(sql_text))
                      .replace("{stats}", str(stats))
                      .replace("{anomalies}", str(anomalies_info))
                      .replace("{chart_description}", chart_description))

    trace = append_trace(state, "insight", "generate_insight",
                         f"Query: {query}", "started")

    try:
        client = ModelClient(agent_type="insight", deadline_ts=state.get("deadline_ts"))
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
                "insight": {"status": "ok", "output": validated.model_dump(), "context": insight_context},
            },
            "insight": validated.model_dump(),
            "shared_metadata": {
                **shared_metadata,
                "insight_context": insight_context,
            },
            "action_trace": trace,
            "status": "success",
            "llm_calls_used": state.get("llm_calls_used", 0) + 1,
        }

    except Exception as exc:
        # Phân biệt "hết giờ" với "sinh insight hỏng". BudgetExceededError
        # kế thừa RuntimeError nên nhánh chung này bắt được nó; nếu không
        # tách nhãn ra, mọi lượt bị cắt vì ngân sách đều hiện lên log và
        # trace dưới tên `insight_generation` — tức là đổ lỗi cho bước sinh
        # insight về một lời gọi model chưa từng khởi động. Kiểm bằng
        # isinstance thay vì thêm một `except` riêng vì thân khối này dài
        # và hai nhánh chỉ khác đúng một chuỗi nhãn.
        error_label = (
            BUDGET_EXCEEDED if isinstance(exc, BudgetExceededError) else INSIGHT_GENERATION
        )
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
                "error_type": error_label,
                "traceback": error_msg,
            },
            "action_trace": trace,
            "status": "failed",
            "llm_calls_used": state.get("llm_calls_used", 0) + 1,
        }
