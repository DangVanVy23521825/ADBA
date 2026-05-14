"""
LangGraph multi-agent graph skeleton.

Flow:
    supervisor → route_next_agent → sql | python | viz | reflector → (loop) → insight → END
"""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from graph.agents.supervisor import route_next_agent, route_reflector_return, supervisor_node
from graph.agents.sql_agent import sql_agent_node
from graph.agents.python_agent import python_agent_node
from graph.agents.viz_agent import viz_agent_node
from graph.agents.insight_agent import insight_agent_node
from graph.agents.reflector_agent import reflector_agent_node
from graph.state import MultiAgentState, make_initial_state
from graph.utils import append_trace

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────
# Graph assembly
# ────────────────────────────────────────────────

def build_multi_agent_graph() -> CompiledStateGraph:
    """Build and return the compiled LangGraph multi-agent pipeline."""
    builder = StateGraph(MultiAgentState)

    # ── Add nodes ────────────────────────────────
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("sql", sql_agent_node)
    builder.add_node("python", python_agent_node)
    builder.add_node("viz", viz_agent_node)
    builder.add_node("insight", insight_agent_node)
    builder.add_node("reflector", reflector_agent_node)

    # ── Edges ────────────────────────────────────
    builder.set_entry_point("supervisor")

    # After supervisor, route to the next agent in the plan.
    builder.add_conditional_edges(
        "supervisor",
        route_next_agent,
        {
            "sql": "sql",
            "python": "python",
            "viz": "viz",
            "insight": "insight",
            "reflector": "reflector",
            END: END,
        },
    )

    # After each specialist agent, return to routing (check next step).
    for agent in ("sql", "python", "viz"):
        builder.add_conditional_edges(
            agent,
            route_next_agent,
            {
                "sql": "sql",
                "python": "python",
                "viz": "viz",
                "insight": "insight",
                "reflector": "reflector",
                END: END,
            },
        )

    # Insight agent always ends the pipeline.
    builder.add_edge("insight", END)

    # Reflector returns to the failed agent (or dies if no valid target).
    builder.add_conditional_edges(
        "reflector",
        route_reflector_return,
        {
            "sql": "sql",
            "python": "python",
            "viz": "viz",
            "insight": "insight",
            END: END,
        },
    )

    return builder.compile()


def run_graph(query: str, info_box: dict) -> MultiAgentState:
    """Convenience: build the graph and invoke it in one shot."""
    graph = build_multi_agent_graph()
    initial = make_initial_state(query, info_box)
    return graph.invoke(initial)


# ── Quick smoke test — run when this file is executed directly ──
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    test_box = {
        "schemas": {
            "orders": {"columns": ["id", "region", "amount", "order_date"]},
        }
    }
    print("=== Running graph skeleton smoke test ===")
    result = run_graph("So sánh doanh thu theo region năm 2024", test_box)
    print(f"Status: {result['status']}")
    print(f"Completed agents: {result['completed_agents']}")
    print(f"Trace entries: {len(result['action_trace'])}")
    print("=== OK ===")
