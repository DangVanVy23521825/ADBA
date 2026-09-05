"""
MultiAgentState — shared TypedDict for the LangGraph multi-agent pipeline.
"""

from __future__ import annotations

import time
from typing import Any, NotRequired, TypedDict

from graph.budget import DEFAULT_BUDGET_S, Clock, make_deadline, new_query_id
from perception.schema_context import SchemaContext


class MultiAgentState(TypedDict):

    # ── Input ──────────────────────────────────────────────
    query: str
    schema_context: SchemaContext

    # ── Ngân sách ──────────────────────────────────────────
    query_id: str
    deadline_ts: float
    llm_calls_used: int
    degradation_reason: list[str]

    # ── Supervisor ─────────────────────────────────────────
    execution_plan: list[dict[str, Any]]
    dependency_graph: NotRequired[dict[str, list[str]]]
    ready_agents: NotRequired[list[str]]
    current_agent: str
    completed_agents: list[str]

    # ── Inter-agent Communication ──────────────────────────
    agent_outputs: dict[str, dict[str, Any]]
    shared_dataframe: NotRequired[dict[str, Any] | None]
    shared_metadata: dict[str, Any]

    # ── Error Handling ─────────────────────────────────────
    error_count: int
    agent_error_counts: dict[str, int]
    last_error: NotRequired[dict[str, Any] | None]

    # ── Final Outputs ──────────────────────────────────────
    sql_result: NotRequired[dict[str, Any] | None]
    python_result: NotRequired[dict[str, Any] | None]
    chart_b64: NotRequired[str | None]
    chart_metadata: NotRequired[dict[str, Any] | None]
    insight: NotRequired[dict[str, Any] | None]

    # ── Trace ──────────────────────────────────────────────
    action_trace: list[dict[str, Any]]
    status: str


def make_initial_state(
    query: str,
    schema_context: SchemaContext,
    budget_s: float = DEFAULT_BUDGET_S,
    clock: Clock = time.time,
) -> MultiAgentState:
    """Return a fresh state dict for a new query.

    `budget_s` và `clock` đều có mặc định, nên mọi lời gọi hai tham số
    hiện có vẫn chạy nguyên. `clock` tồn tại để test deadline không phải
    sleep.
    """
    return MultiAgentState(
        query=query,
        schema_context=schema_context,
        query_id=new_query_id(),
        deadline_ts=make_deadline(budget_s, clock),
        llm_calls_used=0,
        degradation_reason=[],
        execution_plan=[],
        dependency_graph={},
        ready_agents=[],
        current_agent="supervisor",
        completed_agents=[],
        agent_outputs={},
        shared_dataframe=None,
        shared_metadata={},
        error_count=0,
        agent_error_counts={},
        last_error=None,
        sql_result=None,
        python_result=None,
        chart_b64=None,
        chart_metadata=None,
        insight=None,
        action_trace=[],
        status="planning",
    )
