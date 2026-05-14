"""
Unit tests — Python Agent + python_tool.

Tests:
  - python_agent_node with missing step
  - python_agent_node with missing shared_dataframe
  - python_agent_node success path (mock ModelClient + mock run_pandas_safe)
  - python_agent_node retry on error, success on 2nd attempt
  - _extract_code with fences
  - run_pandas_safe with valid and invalid code
  - run_pandas_safe sandbox security checks
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from graph.agents.python_agent import _extract_code, python_agent_node
from graph.state import make_initial_state
from graph.tools.python_tool import PYTHON_SAFE_NAMESPACE, run_pandas_safe
from graph.utils import df_to_state

INFO_BOX = {"schemas": {}}

PLAN_WITH_PYTHON = [
    {"step": 1, "agent": "sql",
     "task": "Lấy data", "depends_on": [], "skill_type": "text-to-sql"},
    {"step": 2, "agent": "python",
     "task": "Tính YoY % thay đổi, phát hiện outlier IQR",
     "depends_on": ["sql"], "skill_type": "data-analysis"},
    {"step": 3, "agent": "insight",
     "task": "Nhận xét", "depends_on": ["python"], "skill_type": "insight-generation"},
]

PLAN_NO_PYTHON = [
    {"step": 1, "agent": "sql",
     "task": "Lấy data", "depends_on": [], "skill_type": "text-to-sql"},
    {"step": 2, "agent": "insight",
     "task": "Nhận xét", "depends_on": ["sql"], "skill_type": "insight-generation"},
]

INPUT_DF = pd.DataFrame({
    "region": ["North", "South"],
    "rev_2024": [150000, 120000],
    "rev_2023": [100000, 110000],
})


def _state(plan=None, has_df=True):
    s = make_initial_state("test", INFO_BOX)
    s["execution_plan"] = plan or []
    if has_df:
        s["shared_dataframe"] = df_to_state(INPUT_DF)
    return s


class TestExtractCode:
    def test_plain_code(self):
        assert _extract_code("df.head()") == "df.head()"

    def test_code_with_fences(self):
        raw = "```python\nimport pandas as pd\ndf.head()\n```"
        assert _extract_code(raw) == "import pandas as pd\ndf.head()"

    def test_code_no_lang_fence(self):
        raw = "```\ndf.head()\n```"
        assert _extract_code(raw) == "df.head()"


class TestRunPandasSafe:
    def test_simple_transform(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        code = "import pandas as pd\ndf = df.copy()\ndf['b'] = df['a'] * 2\ndf"
        result = run_pandas_safe(code, df)
        assert list(result["b"]) == [2, 4, 6]

    def test_yoy_iqr_detection(self):
        code = (
            "import pandas as pd\nimport numpy as np\n"
            "df = df.copy()\n"
            "df['yoy_pct'] = ((df['rev_2024'] - df['rev_2023']) / df['rev_2023'] * 100).round(2)\n"
            "q1 = df['yoy_pct'].quantile(0.25)\n"
            "q3 = df['yoy_pct'].quantile(0.75)\n"
            "iqr = q3 - q1\n"
            "df['is_anomaly'] = ((df['yoy_pct'] < q1 - 1.5*iqr) | (df['yoy_pct'] > q3 + 1.5*iqr))\n"
            "df"
        )
        result = run_pandas_safe(code, INPUT_DF)
        assert "yoy_pct" in result.columns
        assert "is_anomaly" in result.columns
        assert result["yoy_pct"].iloc[0] == 50.0

    def test_no_dataframe_result_raises(self):
        code = "x = 1 + 1"
        with pytest.raises(RuntimeError, match="DataFrame"):
            run_pandas_safe(code, INPUT_DF)

    def test_sandbox_no_file_io(self):
        code = "open('/etc/passwd')"
        with pytest.raises(RuntimeError):
            run_pandas_safe(code, INPUT_DF)

    def test_sandbox_no_import_os(self):
        code = "import os\nos.system('ls')"
        with pytest.raises(RuntimeError):
            run_pandas_safe(code, INPUT_DF)

    def test_sandbox_no_eval(self):
        code = "eval('1+1')"
        with pytest.raises(RuntimeError):
            run_pandas_safe(code, INPUT_DF)


class TestPythonAgentNode:
    @patch("graph.agents.python_agent.run_pandas_safe")
    @patch("graph.agents.python_agent.ModelClient")
    def test_success_populates_state(self, MockClient, mock_run):
        """Successful code generation + execution populates python_result."""
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = (
            "import pandas as pd\nimport numpy as np\n"
            "df = df.copy()\n"
            "df['yoy_pct'] = ((df['rev_2024'] - df['rev_2023']) / df['rev_2023'] * 100).round(2)\n"
            "df"
        )
        MockClient.return_value = mock_instance

        out_df = INPUT_DF.copy()
        out_df["yoy_pct"] = [50.0, 9.09]
        mock_run.return_value = out_df

        s = _state(PLAN_WITH_PYTHON)
        result = python_agent_node(s)

        assert result["status"] == "running"
        assert "python" in result["completed_agents"]
        assert result["agent_outputs"]["python"]["status"] == "ok"
        assert result["python_result"]["stats"]["output_rows"] == 2
        assert "yoy_pct" in result["python_result"]["stats"]["new_columns"]

    @patch("graph.agents.python_agent.run_pandas_safe")
    @patch("graph.agents.python_agent.ModelClient")
    def test_retry_succeeds_on_second_attempt(self, MockClient, mock_run):
        """Code execution fails on attempt 1 → LLM called again → succeeds on attempt 2."""
        mock_instance = MagicMock()
        mock_instance.invoke.side_effect = [
            "df['bad_column']",
            "import pandas as pd\ndf = df.copy()\ndf['test'] = 1\ndf",
        ]
        MockClient.return_value = mock_instance

        mock_run.side_effect = [
            RuntimeError("KeyError: 'bad_column'"),
            INPUT_DF.assign(test=1),
        ]

        s = _state(PLAN_WITH_PYTHON)
        result = python_agent_node(s)

        assert result["status"] == "running"
        assert "python" in result["completed_agents"]
        assert mock_instance.invoke.call_count == 2

    @patch("graph.agents.python_agent.run_pandas_safe")
    @patch("graph.agents.python_agent.ModelClient")
    def test_exhaust_retries_returns_error_state(self, MockClient, mock_run):
        """All retries fail → error state."""
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = "df['bad']"
        MockClient.return_value = mock_instance
        mock_run.side_effect = RuntimeError("KeyError")

        s = _state(PLAN_WITH_PYTHON)
        result = python_agent_node(s)

        assert result["error_count"] == 1
        assert result["agent_error_counts"]["python"] == 1
        assert result["last_error"]["agent"] == "python"
        assert "KeyError" in result["last_error"]["traceback"]

    def test_missing_step_returns_failed(self):
        """No python step in plan → status='failed'."""
        s = _state(PLAN_NO_PYTHON)
        result = python_agent_node(s)
        assert result["status"] == "failed"

    def test_missing_dataframe_returns_failed(self):
        """No shared_dataframe → status='failed'."""
        s = _state(PLAN_WITH_PYTHON, has_df=False)
        result = python_agent_node(s)
        assert result["status"] == "failed"
        assert result["last_error"]["error_type"] == "missing_data"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
