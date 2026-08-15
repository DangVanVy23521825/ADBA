"""
MultiAgentState — shared TypedDict for the LangGraph multi-agent pipeline.
"""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict

from perception.schema_context import SchemaContext


class MultiAgentState(TypedDict):

    # ── Input ──────────────────────────────────────────────
    query: str
    schema_context: SchemaContext

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


def make_initial_state(query: str, schema_context: SchemaContext) -> MultiAgentState:
    """Return a fresh state dict for a new query."""
    return MultiAgentState(
        query=query,
        schema_context=schema_context,
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
