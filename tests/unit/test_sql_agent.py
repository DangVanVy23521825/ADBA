"""
Unit tests — SQL Agent.

Tests:
  - sql_agent_node with missing step
  - sql_agent_node success path (mock ModelClient + mock execute_sql)
  - sql_agent_node retry on SQL error, success on 2nd attempt
  - sql_agent_node exhaust retries
  - _extract_sql with fences and prose
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from graph.agents.sql_agent import _extract_sql, sql_agent_node
from graph.state import make_initial_state

INFO_BOX = {"schemas": {"orders": {"columns": ["id", "region", "amount"]}}}

PLAN_SQL_INSIGHT = [
    {"step": 1, "agent": "sql",
     "task": "Lấy tổng doanh thu theo region năm 2024",
     "depends_on": [], "skill_type": "text-to-sql"},
    {"step": 2, "agent": "insight",
     "task": "Nhận xét",
     "depends_on": ["sql"], "skill_type": "insight-generation"},
]

PLAN_NO_SQL = [
    {"step": 1, "agent": "insight",
     "task": "Nhận xét", "depends_on": [], "skill_type": "insight-generation"},
]


def _state(plan=None):
    s = make_initial_state("test", INFO_BOX)
    s["execution_plan"] = plan or []
    return s


class TestExtractSql:
    def test_plain_sql(self):
        assert _extract_sql("SELECT * FROM orders") == "SELECT * FROM orders"

    def test_sql_with_fences(self):
        assert _extract_sql("```sql\nSELECT * FROM orders\n```") == "SELECT * FROM orders"

    def test_sql_with_prose_before(self):
        raw = "Here is the query:\n\nSELECT region, SUM(amount) FROM orders GROUP BY region"
        assert _extract_sql(raw) == "SELECT region, SUM(amount) FROM orders GROUP BY region"

    def test_sql_with_cte(self):
        raw = "```sql\nWITH cte AS (SELECT * FROM orders) SELECT * FROM cte\n```"
        assert _extract_sql(raw) == "WITH cte AS (SELECT * FROM orders) SELECT * FROM cte"

    def test_no_sql_keyword_returns_original(self):
        assert _extract_sql("Just some text") == "Just some text"


class TestSqlAgentNode:
    @patch("graph.agents.sql_agent.explain_query_plan")
    @patch("graph.agents.sql_agent.execute_sql")
    @patch("graph.agents.sql_agent.ModelClient")
    def test_success_populates_state(self, MockClient, mock_execute, mock_explain):
        """Successful SQL generation + execution populates sql_result and shared_dataframe."""
        mock_explain.return_value = {"total_cost_estimate": 15.5}

        mock_instance = MagicMock()
        mock_instance.invoke.return_value = "SELECT region, SUM(amount) AS total FROM orders " \
                                             "WHERE year=2024 AND status IN ('completed','processing') GROUP BY region"
        MockClient.return_value = mock_instance

        df = pd.DataFrame({"region": ["North", "South"], "total": [150000, 120000]})
        mock_execute.return_value = df

        s = _state(PLAN_SQL_INSIGHT)
        result = sql_agent_node(s)

        assert result["status"] == "running"
        assert "sql" in result["completed_agents"]
        assert result["agent_outputs"]["sql"]["status"] == "ok"
        assert result["sql_result"]["row_count"] == 2
        assert result["shared_dataframe"]["shape"] == [2, 2]
        assert result["shared_metadata"]["sql_row_count"] == 2

    @patch("graph.agents.sql_agent.explain_query_plan")
    @patch("graph.agents.sql_agent.execute_sql")
    @patch("graph.agents.sql_agent.ModelClient")
    def test_retry_succeeds_on_second_attempt(self, MockClient, mock_execute, mock_explain):
        """SQL execution fails on attempt 1, LLM is called again, succeeds on attempt 2."""
        mock_explain.return_value = {}
        mock_instance = MagicMock()
        mock_instance.invoke.side_effect = [
            "SELECT * FROM non_existent_table",
            "SELECT region, SUM(amount) FROM orders WHERE year=2024 GROUP BY region",
        ]
        MockClient.return_value = mock_instance

        mock_execute.side_effect = [
            Exception("relation 'non_existent_table' does not exist"),
            pd.DataFrame({"region": ["North"], "total": [100000]}),
        ]

        s = _state(PLAN_SQL_INSIGHT)
        result = sql_agent_node(s)

        assert result["status"] == "running"
        assert "sql" in result["completed_agents"]
        assert mock_instance.invoke.call_count == 2
        assert result["sql_result"]["row_count"] == 1

    @patch("graph.agents.sql_agent.execute_sql")
    @patch("graph.agents.sql_agent.ModelClient")
    def test_exhaust_retries_returns_error_state(self, MockClient, mock_execute):
        """All 3 attempts fail → error state with last_error."""
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = "SELECT bad FROM nowhere"
        MockClient.return_value = mock_instance
        mock_execute.side_effect = Exception("syntax error")

        s = _state(PLAN_SQL_INSIGHT)
        result = sql_agent_node(s)

        assert result["status"] == "running"
        assert result["error_count"] == 1
        assert result["agent_error_counts"]["sql"] == 1
        assert result["last_error"]["agent"] == "sql"
        assert "syntax error" in result["last_error"]["traceback"]

    def test_missing_step_returns_failed(self):
        """No sql step in plan → status='failed'."""
        s = _state(PLAN_NO_SQL)
        result = sql_agent_node(s)
        assert result["status"] == "failed"
        assert result["last_error"]["agent"] == "sql"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
