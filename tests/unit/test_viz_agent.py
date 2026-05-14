"""
Unit tests — Viz Agent + viz_tool.

Tests:
  - generate_chart: bar, line, pie, anomaly colors, auto chart type
  - viz_agent_node: success, missing step, missing data, retry
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from graph.agents.viz_agent import viz_agent_node, _extract_code
from graph.state import make_initial_state
from graph.tools.viz_tool import generate_chart, _auto_chart_type, _draw_bar
from graph.utils import df_to_state

INFO_BOX = {"schemas": {}}

PLAN_WITH_VIZ = [
    {"step": 1, "agent": "sql",
     "task": "Lấy data", "depends_on": [], "skill_type": "text-to-sql"},
    {"step": 2, "agent": "python",
     "task": "Analyze", "depends_on": ["sql"], "skill_type": "data-analysis"},
    {"step": 3, "agent": "viz",
     "task": "Grouped bar chart so sánh rev_2024 vs rev_2023",
     "depends_on": ["python"], "skill_type": "visualization"},
    {"step": 4, "agent": "insight",
     "task": "Nhận xét", "depends_on": ["sql", "python", "viz"],
     "skill_type": "insight-generation"},
]

PLAN_NO_VIZ = [
    {"step": 1, "agent": "insight",
     "task": "Nhận xét", "depends_on": [], "skill_type": "insight-generation"},
]

ANOMALY_DF = pd.DataFrame({
    "region": ["North", "South", "East"],
    "rev_2024": [150000, 120000, 200000],
    "rev_2023": [80000, 110000, 180000],
    "yoy_pct": [87.5, 9.09, 11.11],
    "is_anomaly": [True, False, False],
})


def _state(plan=None, has_df=True):
    s = make_initial_state("test", INFO_BOX)
    s["execution_plan"] = plan or []
    if has_df:
        s["shared_dataframe"] = df_to_state(ANOMALY_DF)
    return s


class TestVizTool:
    def test_generate_bar_chart(self):
        df = pd.DataFrame({"region": ["North", "South"], "revenue": [150, 120]})
        result = generate_chart(df, "bar", "Q4 Revenue", "region", "revenue")
        assert "chart_b64" in result
        assert len(result["chart_b64"]) > 100
        assert result["chart_type"] == "bar"
        assert result["chart_metadata"]["title"] == "Q4 Revenue"

    def test_generate_bar_with_anomaly(self):
        df = ANOMALY_DF.copy()
        result = generate_chart(df, "bar", "Q4 Revenue", "region", "rev_2024")
        assert len(result["chart_b64"]) > 100

    def test_generate_line_chart(self):
        df = pd.DataFrame({"month": ["Jan", "Feb", "Mar"], "sales": [100, 150, 130]})
        result = generate_chart(df, "line", "Monthly Sales", "month", "sales")
        assert result["chart_type"] == "line"

    def test_generate_pie_chart(self):
        df = pd.DataFrame({"region": ["North", "South", "East"], "share": [40, 35, 25]})
        result = generate_chart(df, "pie", "Revenue Share", "region", "share")
        assert result["chart_type"] == "pie"

    def test_auto_chart_pie(self):
        df = pd.DataFrame({"cat": ["A", "B", "C"], "val": [10, 20, 30]})
        assert _auto_chart_type(df, "cat", "val") == "pie"

    def test_auto_chart_bar(self):
        df = pd.DataFrame({"cat": ["A", "B", "C", "D", "E", "F", "G"], "val": range(7)})
        assert _auto_chart_type(df, "cat", "val") == "bar"


class TestVizAgentNode:
    @patch("graph.agents.viz_agent.ModelClient")
    def test_success_populates_chart_b64(self, MockClient):
        """Successful chart generation produces chart_b64."""
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = (
            "import matplotlib; matplotlib.use('Agg')\n"
            "import matplotlib.pyplot as plt\n"
            "import io, base64\n"
            "plt.style.use('seaborn-v0_8-whitegrid')\n"
            "fig, ax = plt.subplots(figsize=(10, 6))\n"
            "colors = ['#E24B4A' if a else '#4A72B0' for a in df['is_anomaly']]\n"
            "ax.bar(df['region'], df['rev_2024'], color=colors)\n"
            "ax.set_title('Revenue by Region')\n"
            "ax.set_xlabel('Region')\n"
            "ax.set_ylabel('Revenue')\n"
            "fig.tight_layout()\n"
            "buf = io.BytesIO()\n"
            "plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')\n"
            "buf.seek(0)\n"
            "b64 = base64.b64encode(buf.read()).decode('utf-8')\n"
            "plt.close()\n"
            "{'chart_b64': b64, 'chart_type': 'bar', 'title': 'Revenue by Region',"
            " 'x_col': 'region', 'y_col': 'rev_2024'}"
        )
        MockClient.return_value = mock_instance

        s = _state(PLAN_WITH_VIZ)
        result = viz_agent_node(s)

        assert result["status"] == "running"
        assert "viz" in result["completed_agents"]
        assert result["chart_b64"] is not None
        assert len(result["chart_b64"]) > 100
        assert result["chart_metadata"]["chart_type"] == "bar"

    @patch("graph.agents.viz_agent.ModelClient")
    def test_retry_succeeds(self, MockClient):
        """First attempt fails, second succeeds."""
        mock_instance = MagicMock()
        mock_instance.invoke.side_effect = [
            "bad code that fails",
            (
                "import matplotlib; matplotlib.use('Agg')\n"
                "import matplotlib.pyplot as plt\n"
                "import io, base64\n"
                "plt.style.use('seaborn-v0_8-whitegrid')\n"
                "fig, ax = plt.subplots(figsize=(10, 6))\n"
                "ax.bar(df['region'], df['rev_2024'])\n"
                "fig.tight_layout()\n"
                "buf = io.BytesIO()\n"
                "plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')\n"
                "buf.seek(0)\n"
                "b64 = base64.b64encode(buf.read()).decode('utf-8')\n"
                "plt.close()\n"
                "{'chart_b64': b64, 'chart_type': 'bar'}"
            ),
        ]
        MockClient.return_value = mock_instance

        s = _state(PLAN_WITH_VIZ)
        result = viz_agent_node(s)

        assert "viz" in result["completed_agents"]
        assert mock_instance.invoke.call_count == 2

    def test_missing_step_returns_failed(self):
        s = _state(PLAN_NO_VIZ)
        result = viz_agent_node(s)
        assert result["status"] == "failed"

    def test_missing_data_returns_failed(self):
        s = _state(PLAN_WITH_VIZ, has_df=False)
        result = viz_agent_node(s)
        assert result["status"] == "failed"
        assert result["last_error"]["error_type"] == "missing_data"

    def test_extract_code_strips_fences(self):
        raw = "```python\nimport pandas as pd\n```"
        assert _extract_code(raw) == "import pandas as pd"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
