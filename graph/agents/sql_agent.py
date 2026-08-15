"""
SQL Agent — generate and execute PostgreSQL queries with self-repair retry loop.
"""

from __future__ import annotations
import json


import logging
import re
from pathlib import Path

from graph.state import MultiAgentState
from graph.tools.sql_tool import execute_sql, explain_query_plan
from graph.utils import append_trace, df_to_state
from model.model_client import ModelClient

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "text_to_sql.txt"

SQL_SYSTEM_PROMPT: str = _PROMPT_PATH.read_text(encoding="utf-8")

MAX_RETRIES = 3


_SQL_KEYWORDS = ("SELECT", "WITH", "select", "with")


def _extract_sql(text: str) -> str:
    """Strip markdown fences and extract the SQL query from model output."""
    text = re.sub(r"```(?:sql)?\s*|\s*```", "", text).strip()
    # Scan for the earliest SQL keyword (WITH may start before SELECT inside a CTE)
    earliest = len(text)
    for kw in _SQL_KEYWORDS:
        idx = text.find(kw)
        if 0 <= idx < earliest:
            earliest = idx
    if earliest < len(text):
        return text[earliest:].strip()
    return text


def _find_sql_step(state: MultiAgentState) -> dict | None:
    """Find the sql step in the execution plan."""
    for step in state.get("execution_plan", []):
        if step.get("agent") == "sql":
            return step
    return None


def sql_agent_node(state: MultiAgentState) -> MultiAgentState:
    """Generate SQL, execute, retry on failure (max 3).

    Returns state with sql_result, shared_dataframe, and updated trace.
    On fatal failure sets status='failed' and populates last_error.
    """
    step = _find_sql_step(state)
    if step is None:
        logger.error("SQL Agent: no sql step in execution plan")
        return {
            **state,
            "status": "failed",
            "last_error": {"agent": "sql", "error_type": "missing_step"},
        }

    task: str = step.get("task", "")
    info_box: dict = state.get("info_box", {})
    system_prompt = SQL_SYSTEM_PROMPT.replace("{info_box}", json.dumps(info_box, ensure_ascii=False))

    trace = append_trace(state, "sql", "generate_sql",
                         f"Task: {task}", "started")

    client = ModelClient(agent_type="sql")
    error_context: str = ""

    for attempt in range(1, MAX_RETRIES + 1):
        user_prompt = f"Task: {task}"

        # Check for reflector-provided corrected context
        reflector_diag = state.get("shared_metadata", {}).get("reflector_diagnosis", {})
        reflector_hint = reflector_diag.get("corrected_context", "")
        if reflector_hint:
            user_prompt += f"\n\n[HINT from error diagnosis] {reflector_hint}"

        if error_context:
            user_prompt += f"\n\nPrevious attempt failed with error:\n{error_context}\n\nFix the error and rewrite the SQL."

        try:
            raw = client.invoke(system_prompt=system_prompt, user_prompt=user_prompt)
        except Exception as exc:
            error_context = f"ModelClient error: {exc}"
            logger.warning("SQL Agent attempt %d: %s", attempt, error_context)
            continue

        sql = _extract_sql(raw)
        logger.info("SQL Agent attempt %d:\n%s", attempt, sql)

        # ── Execution ────────────────────────────────────────────
        try:
            meta = state.get("shared_metadata", {})
            df = execute_sql(sql, profile=meta["profile"], user=meta["user"])
        except Exception as exc:
            error_context = str(exc)
            logger.warning("SQL Agent execution error (attempt %d): %s", attempt, error_context)
            trace = append_trace(state, "sql", "execute_sql",
                                 f"Attempt {attempt} error: {error_context}", "error")
            continue

        # ── Success ───────────────────────────────────────────────
        try:
            plan = explain_query_plan(sql)
            cost = plan.get("total_cost_estimate", "?")
        except Exception:
            plan = {}
            cost = "?"

        trace = append_trace(state, "sql", "execute_sql",
                             f"OK — {len(df)} rows, cost ~{cost}", "ok")
        logger.info("SQL Agent: %d rows returned", len(df))

        return {
            **state,
            "current_agent": "sql",
            "completed_agents": state.get("completed_agents", []) + ["sql"],
            "agent_outputs": {
                **state.get("agent_outputs", {}),
                "sql": {"status": "ok", "sql": sql, "row_count": len(df)},
            },
            "shared_dataframe": df_to_state(df),
            "sql_result": {"sql": sql, "row_count": len(df)},
            "shared_metadata": {
                **state.get("shared_metadata", {}),
                "sql_row_count": len(df),
                "sql_query": sql,
                "sql_result": {"sql": sql, "row_count": len(df)},
            },
            "action_trace": trace,
            "status": "running",
        }

    # ── Exhausted retries ─────────────────────────────────────
    error_summary = f"SQL Agent failed after {MAX_RETRIES} attempts. Last error: {error_context}"
    trace = append_trace(state, "sql", "execute_sql", error_summary, "error")

    return {
        **state,
        "current_agent": "sql",
        "agent_outputs": {
            **state.get("agent_outputs", {}),
            "sql": {"status": "error", "error": error_summary},
        },
        "error_count": state.get("error_count", 0) + 1,
        "agent_error_counts": {
            **state.get("agent_error_counts", {}),
            "sql": state.get("agent_error_counts", {}).get("sql", 0) + 1,
        },
        "last_error": {
            "agent": "sql",
            "error_type": "sql_execution",
            "traceback": error_summary,
        },
        "action_trace": trace,
        "status": "running",
    }
