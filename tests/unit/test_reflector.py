"""
Unit tests — Reflector Agent.

Tests:
  - reflector_agent_node: success diagnosis, fallback on ModelClient error
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from graph.agents.reflector_agent import reflector_agent_node
from graph.state import make_initial_state
from perception.schema_context import SchemaContext

SCHEMA_CONTEXT = SchemaContext()

PLAN_SQL = [
    {"step": 1, "agent": "sql",
     "task": "Lấy tổng doanh thu theo region",
     "depends_on": [], "skill_type": "text-to-sql"},
]

DIAGNOSIS = {
    "root_cause": "Column 'total_revenue' does not exist in orders table",
    "error_category": "schema_mismatch",
    "fix_strategy": "Use SUM(amount) AS total_revenue instead",
    "corrected_context": "Query total revenue by region using SUM(amount) grouped by region",
}


def _state():
    s = make_initial_state("test", SCHEMA_CONTEXT)
    s["execution_plan"] = PLAN_SQL
    s["last_error"] = {
        "agent": "sql",
        "error_type": "sql_execution",
        "traceback": "psycopg2.UndefinedColumn: column 'total_revenue' does not exist",
    }
    s["agent_error_counts"] = {"sql": 2}
    return s


class TestReflectorAgentNode:
    @patch("graph.agents.reflector_agent.ModelClient")
    def test_success_diagnosis(self, MockClient):
        """Reflector produces correct diagnosis → state updated."""
        mock_instance = MagicMock()
        mock_instance.invoke_json.return_value = DIAGNOSIS
        MockClient.return_value = mock_instance

        s = _state()
        result = reflector_agent_node(s)

        assert result["status"] == "running"
        assert result["agent_outputs"]["reflector"]["status"] == "ok"
        assert result["agent_outputs"]["reflector"]["output"]["error_category"] == "schema_mismatch"
        assert "reflector_diagnosis" in result["shared_metadata"]

    @patch("graph.agents.reflector_agent.ModelClient")
    def test_fallback_on_client_error(self, MockClient):
        """ModelClient fails → reflector produces fallback diagnosis."""
        mock_instance = MagicMock()
        mock_instance.invoke_json.side_effect = RuntimeError("Ollama down")
        MockClient.return_value = mock_instance

        s = _state()
        result = reflector_agent_node(s)

        assert result["status"] == "running"
        assert result["agent_outputs"]["reflector"]["status"] == "ok"
        diag = result["agent_outputs"]["reflector"]["output"]
        assert diag["error_category"] == "unknown"
        assert "corrected_context" in diag

    @patch("graph.agents.reflector_agent.ModelClient")
    def test_diagnosis_with_python_error(self, MockClient):
        """Reflector diagnoses Python runtime error."""
        mock_instance = MagicMock()
        mock_instance.invoke_json.return_value = {
            "root_cause": "KeyError on column 'bad_col'",
            "error_category": "python_runtime",
            "fix_strategy": "Check column exists before accessing",
            "corrected_context": "Filter df to columns: ['region', 'rev_2024', 'rev_2023']",
        }
        MockClient.return_value = mock_instance

        s = _state()
        s["last_error"] = {
            "agent": "python",
            "error_type": "python_runtime",
            "traceback": "KeyError: 'bad_col'",
        }
        result = reflector_agent_node(s)

        assert result["agent_outputs"]["reflector"]["output"]["error_category"] == "python_runtime"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
