"""
Unit tests — Supervisor Agent routing logic.

10 routing cases for route_next_agent() covering:
  - simple plan → [sql, insight]
  - complex plan → [sql, python, viz, insight]
  - dependency ordering
  - error-limit → reflector
  - terminal states (failed / success)
  - empty plan
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Project root on sys.path (same pattern used by all scripts in this repo)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langgraph.graph import END

from graph.agents.supervisor import route_next_agent, supervisor_node
from graph.state import MultiAgentState, make_initial_state


# ── Fixtures ──────────────────────────────────────────────────────────────────

INFO_BOX = {
    "schemas": {
        "orders": {"columns": ["id", "region", "amount", "order_date"]},
    }
}

PLAN_SIMPLE = [
    {"step": 1, "agent": "sql",
     "task": "Lấy tổng doanh thu theo region năm 2024",
     "depends_on": [], "skill_type": "text-to-sql"},
    {"step": 2, "agent": "insight",
     "task": "Nhận xét region dẫn đầu",
     "depends_on": ["sql"], "skill_type": "insight-generation"},
]

PLAN_COMPLEX = [
    {"step": 1, "agent": "sql",
     "task": "Lấy doanh thu Q4 2024 và Q4 2023 theo region",
     "depends_on": [], "skill_type": "text-to-sql"},
    {"step": 2, "agent": "python",
     "task": "Tính YoY %, phát hiện outlier IQR",
     "depends_on": ["sql"], "skill_type": "data-analysis"},
    {"step": 3, "agent": "viz",
     "task": "Grouped bar chart, highlight anomaly",
     "depends_on": ["python"], "skill_type": "visualization"},
    {"step": 4, "agent": "insight",
     "task": "Đề xuất hành động cho Q1 2025",
     "depends_on": ["sql", "python", "viz"], "skill_type": "insight-generation"},
]

PLAN_NO_VIZ = [
    {"step": 1, "agent": "sql",
     "task": "Lấy data",
     "depends_on": [], "skill_type": "text-to-sql"},
    {"step": 2, "agent": "python",
     "task": "Phân tích",
     "depends_on": ["sql"], "skill_type": "data-analysis"},
    {"step": 3, "agent": "insight",
     "task": "Nhận xét",
     "depends_on": ["sql", "python"], "skill_type": "insight-generation"},
]


def _state(plan=None, completed=None, status="running", error_counts=None,
           last_error=None, agent_outputs=None, shared_metadata=None) -> MultiAgentState:
    """Build a minimal state dict for routing tests."""
    s = make_initial_state("test query", INFO_BOX)
    s["execution_plan"] = plan or []
    s["completed_agents"] = completed or []
    s["status"] = status
    s["agent_error_counts"] = error_counts or {}
    if last_error is not None:
        s["last_error"] = last_error
    if agent_outputs is not None:
        s["agent_outputs"] = agent_outputs
    if shared_metadata is not None:
        s["shared_metadata"] = shared_metadata
    return s


# ── Routing tests ─────────────────────────────────────────────────────────────

class TestRouteNextAgent:

    # 1. Simple plan, nothing completed → first agent is sql
    def test_simple_plan_starts_with_sql(self):
        s = _state(PLAN_SIMPLE)
        assert route_next_agent(s) == "sql"

    # 2. Simple plan, sql done → next is insight
    def test_simple_plan_sql_done_routes_to_insight(self):
        s = _state(PLAN_SIMPLE, completed=["sql"])
        assert route_next_agent(s) == "insight"

    # 3. Complex plan, nothing completed → sql
    def test_complex_plan_starts_with_sql(self):
        s = _state(PLAN_COMPLEX)
        assert route_next_agent(s) == "sql"

    # 4. Complex plan, sql done → python (python depends on sql)
    def test_complex_plan_sql_done_routes_to_python(self):
        s = _state(PLAN_COMPLEX, completed=["sql"])
        assert route_next_agent(s) == "python"

    # 5. Complex plan, sql+python done → viz (viz depends on python)
    def test_complex_plan_sql_python_done_routes_to_viz(self):
        s = _state(PLAN_COMPLEX, completed=["sql", "python"])
        assert route_next_agent(s) == "viz"

    # 6. Complex plan, sql+python+viz done → insight (depends on all three)
    def test_complex_plan_all_done_routes_to_insight(self):
        s = _state(PLAN_COMPLEX, completed=["sql", "python", "viz"])
        assert route_next_agent(s) == "insight"

    # 7. Plan with all agents completed → END
    def test_all_completed_routes_to_end(self):
        s = _state(PLAN_SIMPLE, completed=["sql", "insight"])
        assert route_next_agent(s) == END

    # 8. Failed status → END regardless of plan
    def test_failed_status_routes_to_end(self):
        s = _state(PLAN_COMPLEX, status="failed")
        assert route_next_agent(s) == END

    # 9. Agent error count >= 3 → reflector
    def test_error_limit_triggers_reflector(self):
        s = _state(PLAN_COMPLEX, completed=[], error_counts={"sql": 3})
        assert route_next_agent(s) == "reflector"

    # 10. Empty plan → END
    def test_empty_plan_routes_to_end(self):
        s = _state(plan=[], completed=[])
        assert route_next_agent(s) == END

    # 11. New failure wave after reflector → reflector again (not stuck skip)
    def test_sql_second_failure_wave_routes_back_to_reflector(self):
        s = _state(
            PLAN_SIMPLE,
            agent_outputs={
                "reflector": {"status": "ok"},
                "sql": {"status": "error", "error": "boom"},
            },
            error_counts={"sql": 2},
            shared_metadata={
                "reflect_error_snapshot": {"sql": 1},
                "reflect_passes_per_agent": {"sql": 1},
            },
        )
        assert route_next_agent(s) == "reflector"

    # 12. Fresh failure wave but reflector budget exhausted → END (stuck specialist)
    def test_reflector_cap_exceeded_ends_when_sql_stuck(self):
        from graph.agents.supervisor import MAX_REFLECTOR_PASSES_PER_AGENT

        s = _state(
            PLAN_SIMPLE,
            agent_outputs={
                "reflector": {"status": "ok"},
                "sql": {"status": "error", "error": "still failing"},
            },
            error_counts={"sql": 5},  # new wave after diagnosis at snapshot 4
            shared_metadata={
                "reflect_error_snapshot": {"sql": 4},
                "reflect_passes_per_agent": {"sql": MAX_REFLECTOR_PASSES_PER_AGENT},
            },
        )
        assert route_next_agent(s) == END


# ── Supervisor node tests ─────────────────────────────────────────────────────
class TestSupervisorNode:

    VALID_PLAN_DICT = {
        "plan_summary": "Tổng doanh thu theo region",
        "steps": [
            {"step": 1, "agent": "sql",
             "task": "Lấy tổng doanh thu theo region năm 2024",
             "depends_on": [], "skill_type": "text-to-sql"},
            {"step": 2, "agent": "insight",
             "task": "Nhận xét region dẫn đầu",
             "depends_on": ["sql"], "skill_type": "insight-generation"},
        ],
    }

    @patch("graph.agents.supervisor.ModelClient")
    def test_supervisor_success_populates_plan(self, MockClient):
        """ModelClient.invoke_json returns valid plan → state.execution_plan populated."""
        mock_instance = MagicMock()
        mock_instance.invoke_json.return_value = self.VALID_PLAN_DICT
        MockClient.return_value = mock_instance

        s = make_initial_state("Tổng doanh thu theo region 2024", INFO_BOX)
        result = supervisor_node(s)

        assert result["status"] == "running"
        assert len(result["execution_plan"]) == 2
        assert result["execution_plan"][0]["agent"] == "sql"
        assert result["execution_plan"][1]["agent"] == "insight"
        assert "supervisor" in result["completed_agents"]
        assert result["agent_outputs"]["supervisor"]["status"] == "ok"

    @patch("graph.agents.supervisor.ModelClient")
    def test_supervisor_model_error_sets_failed(self, MockClient):
        """ModelClient raises → state.status='failed', last_error populated."""
        mock_instance = MagicMock()
        mock_instance.invoke_json.side_effect = RuntimeError("Ollama down")
        MockClient.return_value = mock_instance

        s = make_initial_state("test", INFO_BOX)
        result = supervisor_node(s)

        assert result["status"] == "failed"
        assert result["error_count"] == 1
        assert result["last_error"]["agent"] == "supervisor"
        assert "Ollama down" in result["last_error"]["traceback"]

    @patch("graph.agents.supervisor.ModelClient")
    def test_supervisor_invalid_json_sets_failed(self, MockClient):
        """ModelClient returns garbage JSON → Pydantic validation fails → status='failed'."""
        mock_instance = MagicMock()
        mock_instance.invoke_json.return_value = {"bad": "structure"}
        MockClient.return_value = mock_instance

        s = make_initial_state("test", INFO_BOX)
        result = supervisor_node(s)

        assert result["status"] == "failed"
        assert result["last_error"]["agent"] == "supervisor"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
