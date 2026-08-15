"""Integration test helpers — shared mocks and fixtures for full-graph tests."""

from __future__ import annotations

import base64
import io
from unittest.mock import MagicMock

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from perception.connection_profile import ALL_TABLES, build_profile
from perception.schema_context import resolve_schema_context
from tests.fixtures.mini_schema import MINI_TABLES

matplotlib.use("Agg")


_SCHEMA_PROFILE = build_profile(
    dsn="postgresql://u:p@h:5432/d",
    tables=MINI_TABLES,
    grants={"test_user": frozenset({ALL_TABLES})},
)

# Shared SchemaContext for full-graph integration tests — content doesn't
# matter here since ModelClient is mocked at every agent, but the state
# needs a real SchemaContext to build prompts against.
SCHEMA_CONTEXT = resolve_schema_context(
    _SCHEMA_PROFILE, "test", permitted=frozenset(t.name for t in MINI_TABLES),
)

# ── Sample DataFrames for SQL agent outputs ──────────────────────────────────

DF_REVENUE_BY_REGION = pd.DataFrame({
    "region": ["North", "South", "East", "West"],
    "total_revenue": [150000000, 120000000, 95000000, 80000000],
})

DF_TOP_PRODUCTS = pd.DataFrame({
    "name": ["Product A", "Product B", "Product C", "Product D", "Product E"],
    "total_revenue": [50000000, 45000000, 40000000, 35000000, 30000000],
})

DF_EMPLOYEE_COUNT = pd.DataFrame({
    "department": ["Engineering", "Sales", "Marketing", "HR"],
    "count": [45, 30, 20, 12],
})

DF_MONTHLY_SALES = pd.DataFrame({
    "month": [1, 2, 3, 4, 5, 6],
    "total_revenue": [100000, 120000, 150000, 130000, 160000, 180000],
})

DF_Q4_COMPARISON = pd.DataFrame({
    "region": ["North", "South", "East", "West"],
    "rev_2024": [150000, 120000, 95000, 80000],
    "rev_2023": [80000, 110000, 90000, 75000],
})

DF_PYTHON_OUTPUT = pd.DataFrame({
    "region": ["North", "South", "East", "West"],
    "rev_2024": [150000, 120000, 95000, 80000],
    "rev_2023": [80000, 110000, 90000, 75000],
    "yoy_pct": [87.5, 9.09, 5.56, 6.67],
    "is_anomaly": [True, False, False, False],
})

DF_SALARY_DIST = pd.DataFrame({
    "level": ["Junior", "Mid", "Senior", "Lead"],
    "avg_salary": [15000000, 25000000, 40000000, 60000000],
    "count": [80, 50, 25, 10],
})

DF_INVENTORY_THRESHOLD = pd.DataFrame({
    "product_name": ["Widget X", "Gadget Y", "Tool Z"],
    "current_qty": [5, 8, 3],
    "min_threshold": [20, 15, 25],
    "warehouse": ["HCMC", "Hanoi", "Danang"],
})

# ── Valid chart generation code (used by viz_agent) ───────────────────────────

CHART_CODE_BAR = (
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
    "result = {'chart_b64': b64, 'chart_type': 'bar',"
    " 'title': 'Revenue by Region', 'x_col': 'region', 'y_col': 'rev_2024'}"
)

CHART_CODE_GENERIC = (
    "import matplotlib; matplotlib.use('Agg')\n"
    "import matplotlib.pyplot as plt\n"
    "import io, base64\n"
    "plt.style.use('seaborn-v0_8-whitegrid')\n"
    "fig, ax = plt.subplots(figsize=(10, 6))\n"
    "cols = df.columns.tolist()\n"
    "ax.bar(df[cols[0]].astype(str), df[cols[1]])\n"
    "ax.set_title('Chart')\n"
    "ax.set_xlabel(cols[0])\n"
    "ax.set_ylabel(cols[1])\n"
    "fig.tight_layout()\n"
    "buf = io.BytesIO()\n"
    "plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')\n"
    "buf.seek(0)\n"
    "b64 = base64.b64encode(buf.read()).decode('utf-8')\n"
    "plt.close()\n"
    "result = {'chart_b64': b64, 'chart_type': 'bar'}"
)

# ── Valid insight JSON ───────────────────────────────────────────────────────

VALID_INSIGHT = {
    "finding": "Miền Bắc tăng trưởng +87.5% YoY trong Q4 2024, là outlier duy nhất.",
    "evidence": [
        "Doanh thu Q4 2024 Miền Bắc: 150,000 VND vs Q4 2023: 80,000 VND",
        "Tăng trưởng trung bình toàn quốc Q4: +27.2% YoY",
    ],
    "anomaly": {
        "type": "positive_outlier",
        "detail": "North vượt 2.5 sigma so với các region còn lại.",
    },
    "action": "Tăng tồn kho khu vực Miền Bắc 60% trước Q1 2025.",
    "confidence": "high",
}


# ── Mock helper — configure all agent ModelClient mocks at once ────────────────

def setup_simple_mocks(sql_df: pd.DataFrame):
    """Mock ModelClient for sql + insight agents (simple query)."""
    mocks = {}

    sql_mock = MagicMock()
    sql_mock.invoke.return_value = "SELECT region, SUM(amount) AS total_revenue FROM orders GROUP BY region"
    mocks["sql"] = sql_mock

    insight_mock = MagicMock()
    insight_mock.invoke_json.return_value = VALID_INSIGHT
    mocks["insight"] = insight_mock

    exec_mock = MagicMock()
    exec_mock.return_value = sql_df
    mocks["execute_sql"] = exec_mock

    explain_mock = MagicMock()
    explain_mock.return_value = {"total_cost_estimate": 15.0}
    mocks["explain_query_plan"] = explain_mock

    return mocks


def setup_complex_mocks():
    """Mock ModelClient for sql + python + viz + insight (complex query)."""
    mocks = {}

    sql_mock = MagicMock()
    sql_mock.invoke.return_value = (
        "WITH q4_24 AS ("
        "SELECT region, SUM(amount) AS rev_2024 FROM orders "
        "WHERE year=2024 AND quarter=4 GROUP BY region), "
        "q4_23 AS ("
        "SELECT region, SUM(amount) AS rev_2023 FROM orders "
        "WHERE year=2023 AND quarter=4 GROUP BY region) "
        "SELECT a.region, a.rev_2024, b.rev_2023 "
        "FROM q4_24 a LEFT JOIN q4_23 b USING (region)"
    )
    mocks["sql"] = sql_mock

    python_mock = MagicMock()
    python_mock.invoke.return_value = (
        "import pandas as pd\nimport numpy as np\n"
        "df = df.copy()\n"
        "df['yoy_pct'] = ((df['rev_2024'] - df['rev_2023']) / df['rev_2023'] * 100).round(2)\n"
        "q1 = df['yoy_pct'].quantile(0.25)\nq3 = df['yoy_pct'].quantile(0.75)\n"
        "iqr = q3 - q1\n"
        "df['is_anomaly'] = ((df['yoy_pct'] < q1 - 1.5*iqr) | (df['yoy_pct'] > q3 + 1.5*iqr))\n"
        "df"
    )
    mocks["python"] = python_mock

    viz_mock = MagicMock()
    viz_mock.invoke.return_value = CHART_CODE_BAR
    mocks["viz"] = viz_mock

    insight_mock = MagicMock()
    insight_mock.invoke_json.return_value = VALID_INSIGHT
    mocks["insight"] = insight_mock

    # Tool mocks
    exec_mock = MagicMock()
    exec_mock.return_value = DF_Q4_COMPARISON
    mocks["execute_sql"] = exec_mock

    explain_mock = MagicMock()
    explain_mock.return_value = {"total_cost_estimate": 25.0}
    mocks["explain_query_plan"] = explain_mock

    pandas_mock = MagicMock()
    pandas_mock.return_value = DF_PYTHON_OUTPUT
    mocks["run_pandas_safe"] = pandas_mock

    return mocks
