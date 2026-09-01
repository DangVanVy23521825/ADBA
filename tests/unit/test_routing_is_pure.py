"""Conditional edge phải là hàm thuần (spec 5.5).

LangGraph gọi conditional edge để lấy TÊN ROUTE. Gán vào state bên trong
nó không đi qua reducer, nên thay đổi có thể mất — mà mất im lặng, vì
đọc code thì thấy như đã cập nhật.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from graph.agents.supervisor import route_next_agent, routing_snapshot
from graph.state import make_initial_state
from graph.utils import with_routing
from perception.schema_context import SchemaContext

PLAN = [
    {"step": 1, "agent": "sql", "task": "t1", "depends_on": [], "skill_type": "text-to-sql"},
    {"step": 2, "agent": "insight", "task": "t2", "depends_on": ["sql"],
     "skill_type": "insight-generation"},
]


def _state(**over):
    s = make_initial_state("q", SchemaContext())
    s["execution_plan"] = PLAN
    s["completed_agents"] = ["supervisor"]
    s["status"] = "running"
    s.update(over)
    return s


def test_route_does_not_mutate_the_state_it_is_given():
    s = _state()
    before = copy.deepcopy(s)
    route_next_agent(s)
    assert s == before, "conditional edge đã ghi vào state — xem spec 5.5"


def test_route_still_returns_the_right_agent():
    assert route_next_agent(_state()) == "sql"


def test_route_returns_insight_after_sql_completes():
    assert route_next_agent(_state(completed_agents=["supervisor", "sql"])) == "insight"


def test_snapshot_reports_ready_agents_without_writing():
    s = _state()
    before = copy.deepcopy(s)
    snap = routing_snapshot(s)
    assert snap["ready_agents"] == ["sql"]
    assert s == before


def test_snapshot_reports_the_dependency_graph():
    assert routing_snapshot(_state())["dependency_graph"] == {"sql": [], "insight": ["sql"]}


def test_snapshot_reports_blocked_agents():
    s = _state(
        agent_outputs={"sql": {"status": "error"}, "reflector": {"status": "ok"}},
        agent_error_counts={"sql": 5},
        shared_metadata={
            "reflect_passes_per_agent": {"sql": 9},
            "reflect_error_snapshot": {"sql": 5},
        },
    )
    assert "sql" in routing_snapshot(s)["blocked_agents"]


def test_with_routing_writes_the_snapshot_into_the_returned_state():
    """Node ghi — không phải conditional edge. Đây là chỗ ghi hợp lệ."""
    out = with_routing(_state())
    assert out["ready_agents"] == ["sql"]
    assert out["dependency_graph"] == {"sql": [], "insight": ["sql"]}
    assert out["shared_metadata"]["parallel_ready"] is False


def test_with_routing_does_not_mutate_its_input():
    s = _state()
    before = copy.deepcopy(s)
    with_routing(s)
    assert s == before
