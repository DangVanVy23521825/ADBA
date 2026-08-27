"""
Unit tests — Insight Agent + insight_tools.

Tests:
  - detect_anomaly: IQR, no anomaly, single row
  - compare_periods: YoY change + anomaly flag
  - insight_agent_node: success, Pydantic validation fail, missing step
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from graph.agents.insight_agent import insight_agent_node
from graph.state import make_initial_state
from graph.tools.insight_tools import compare_periods, detect_anomaly
from graph.utils import df_to_state
from perception.schema_context import SchemaContext

SCHEMA_CONTEXT = SchemaContext()

VALID_INSIGHT = {
    "finding": "Miền Bắc tăng trưởng +88% YoY trong Q4 2024, gấp 3.8 lần trung bình cả nước.",
    "evidence": [
        "Doanh thu Q4 2024 Miền Bắc: 142,000,000 VND vs Q4 2023: 75,400,000 VND",
        "Tăng trưởng trung bình toàn quốc Q4 2024: +23% YoY",
        "Miền Bắc vượt 3.6 sigma so với baseline các region khác",
    ],
    "anomaly": {"type": "positive_outlier", "detail": "Tăng trưởng bất thường do campaign B2B."},
    "action": "Tăng tồn kho khu vực Miền Bắc ít nhất 50% trước Q1 2025.",
    "confidence": "high",
}

PLAN_WITH_INSIGHT = [
    {"step": 1, "agent": "sql",
     "task": "Lấy data", "depends_on": [], "skill_type": "text-to-sql"},
    {"step": 2, "agent": "insight",
     "task": "Nhận xét", "depends_on": ["sql"],
     "skill_type": "insight-generation"},
]

PLAN_NO_INSIGHT = [
    {"step": 1, "agent": "sql",
     "task": "Lấy data", "depends_on": [], "skill_type": "text-to-sql"},
]

ANALYSIS_DF = pd.DataFrame({
    "region": ["North", "South", "East"],
    "rev_2024": [150000, 120000, 95000],
    "rev_2023": [80000, 110000, 90000],
})


def _state(plan=None):
    s = make_initial_state("Test query", SCHEMA_CONTEXT)
    s["execution_plan"] = plan or []
    s["agent_outputs"] = {
        "sql": {"status": "ok", "sql": "SELECT * FROM orders WHERE year=2024"},
        "python": {"status": "ok", "stats": {"output_rows": 3, "new_columns": ["yoy_pct"]}},
    }
    s["shared_dataframe"] = df_to_state(ANALYSIS_DF)
    s["shared_metadata"] = {"sql_query": "SELECT ...", "sql_row_count": 3}
    return s


class TestInsightTools:
    def test_detect_anomaly_finds_outlier(self):
        df = pd.DataFrame({
            "region": ["A", "B", "C", "D", "E", "Outlier"],
            "value": [10, 12, 11, 9, 13, 50],
        })
        result = detect_anomaly(df, "value")
        assert result["anomaly_count"] == 1
        assert result["anomalies"][0]["direction"] == "positive_outlier"
        assert result["thresholds"]["upper"] > 0

    def test_detect_anomaly_no_outliers(self):
        df = pd.DataFrame({"region": ["A", "B", "C"], "value": [10, 12, 11]})
        result = detect_anomaly(df, "value")
        assert result["anomaly_count"] == 0

    def test_detect_anomaly_single_row(self):
        df = pd.DataFrame({"region": ["A"], "value": [10]})
        result = detect_anomaly(df, "value")
        assert result["anomaly_count"] == 0

    def test_compare_periods(self):
        df = pd.DataFrame({
            "region": ["North", "South"],
            "rev_2024": [150000, 120000],
            "rev_2023": [80000, 110000],
        })
        result = compare_periods(df, "rev_2024", "rev_2023")
        assert "pct_change" in result.columns
        assert result["pct_change"].iloc[0] == 87.5
        assert result["pct_change"].iloc[1] == pytest.approx(9.09, abs=0.1)
        assert "is_anomaly" in result.columns


class TestInsightAgentNode:
    @patch("graph.agents.insight_agent.ModelClient")
    def test_success_populates_insight(self, MockClient):
        """Valid insight JSON → Pydantic validation → state updated."""
        mock_instance = MagicMock()
        mock_instance.invoke_json.return_value = VALID_INSIGHT
        MockClient.return_value = mock_instance

        s = _state(PLAN_WITH_INSIGHT)
        result = insight_agent_node(s)

        assert result["status"] == "success"
        assert "insight" in result["completed_agents"]
        assert result["insight"]["finding"] == VALID_INSIGHT["finding"]
        assert result["insight"]["confidence"] == "high"

    @patch("graph.agents.insight_agent.ModelClient")
    def test_invalid_json_sets_failed(self, MockClient):
        """Invalid output → Pydantic validation fails → status='failed'."""
        mock_instance = MagicMock()
        mock_instance.invoke_json.return_value = {"bad": "json"}
        MockClient.return_value = mock_instance

        s = _state(PLAN_WITH_INSIGHT)
        result = insight_agent_node(s)

        assert result["status"] == "failed"
        assert result["last_error"]["agent"] == "insight"

    def test_missing_step_returns_failed(self):
        s = _state(PLAN_NO_INSIGHT)
        result = insight_agent_node(s)
        assert result["status"] == "failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
