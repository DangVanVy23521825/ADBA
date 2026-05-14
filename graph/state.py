"""
MultiAgentState — shared TypedDict for the LangGraph multi-agent pipeline.
"""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class MultiAgentState(TypedDict):

    # ── Input ──────────────────────────────────────────────
    query: str
    info_box: dict[str, Any]

    # ── Supervisor ─────────────────────────────────────────
    execution_plan: list[dict[str, Any]]
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


def make_initial_state(query: str, info_box: dict[str, Any]) -> MultiAgentState:
    """Return a fresh state dict for a new query."""
    return MultiAgentState(
        query=query,
        info_box=info_box,
        execution_plan=[],
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
