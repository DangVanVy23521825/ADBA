"""
Integration tests — 10 complex multi-agent queries (sql + python + viz + insight).

All tests mock ModelClient for ALL agents and tool functions so no real
Ollama or PostgreSQL calls are made.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from graph.state import make_initial_state
from graph.multi_agent import build_multi_agent_graph
from perception.connection_profile import ALL_TABLES, build_profile
from tests.fixtures.mini_schema import MINI_TABLES
from tests.integration import (
    SCHEMA_CONTEXT,
    CHART_CODE_BAR,
    CHART_CODE_GENERIC,
    DF_Q4_COMPARISON,
    DF_PYTHON_OUTPUT,
    DF_SALARY_DIST,
    DF_REVENUE_BY_REGION,
    VALID_INSIGHT,
)

TEST_PROFILE = build_profile(
    dsn="postgresql://u:p@h:5432/d",
    tables=MINI_TABLES,
    grants={"test_user": frozenset({ALL_TABLES})},
)

PYTHON_CODE_YOY = (
    "import pandas as pd\nimport numpy as np\ndf = df.copy()\n"
    "df['yoy_pct'] = ((df['rev_2024']-df['rev_2023'])/df['rev_2023']*100).round(2)\n"
    "q1=df['yoy_pct'].quantile(0.25)\nq3=df['yoy_pct'].quantile(0.75)\n"
    "iqr=q3-q1\ndf['is_anomaly']=((df['yoy_pct']<q1-1.5*iqr)|(df['yoy_pct']>q3+1.5*iqr))\ndf"
)


def _supervisor_mock(plan):
    m = MagicMock()
    m.invoke_json.return_value = {"plan_summary": "test", "steps": plan}
    return m


def _complex_plan(sql_t, py_t, viz_t, ins_t):
    return [
        {"step": 1, "agent": "sql", "task": sql_t,
         "depends_on": [], "skill_type": "text-to-sql"},
        {"step": 2, "agent": "python", "task": py_t,
         "depends_on": ["sql"], "skill_type": "data-analysis"},
        {"step": 3, "agent": "viz", "task": viz_t,
         "depends_on": ["python"], "skill_type": "visualization"},
        {"step": 4, "agent": "insight", "task": ins_t,
         "depends_on": ["sql", "python", "viz"],
         "skill_type": "insight-generation"},
    ]


def _state(plan, query="Complex test query"):
    s = make_initial_state(query, SCHEMA_CONTEXT)
    s["execution_plan"] = plan
    s["shared_metadata"] = {"profile": TEST_PROFILE, "user": "test_user"}
    return s


# ── Patch context manager helper ─────────────────────────────────────────

def _all_patches(plan, sql_txt, exec_df, py_code, py_df, viz_code, insight_json=None):
    """Return 8 patches needed to fully mock a complex query pipeline."""
    insight_json = insight_json or VALID_INSIGHT
    return (
        patch("graph.agents.supervisor.ModelClient", return_value=_supervisor_mock(plan)),
        patch("graph.agents.insight_agent.ModelClient",
              return_value=MagicMock(invoke_json=MagicMock(return_value=insight_json))),
        patch("graph.agents.viz_agent.ModelClient",
              return_value=MagicMock(invoke=MagicMock(return_value=viz_code))),
        patch("graph.agents.python_agent.run_pandas_safe", return_value=py_df),
        patch("graph.agents.python_agent.ModelClient",
              return_value=MagicMock(invoke=MagicMock(return_value=py_code))),
        patch("graph.agents.sql_agent.explain_query_plan", return_value={}),
        patch("graph.agents.sql_agent.execute_sql", return_value=exec_df),
        patch("graph.agents.sql_agent.ModelClient",
              return_value=MagicMock(invoke=MagicMock(return_value=sql_txt))),
    )


class TestComplexQueries:

    def test_01_yoy_revenue_comparison_with_anomaly(self):
        """YoY revenue comparison → anomaly → chart → insight."""
        plan = _complex_plan(
            "Lấy doanh thu Q4 2024 vs Q4 2023 theo region",
            "Tính YoY %, outlier IQR, thêm is_anomaly",
            "Grouped bar chart, highlight anomaly red",
            "Phân tích region bất thường, đề xuất hành động",
        )
        sql = "WITH q4_24 AS (...) SELECT a.region, a.rev_2024, b.rev_2023 FROM q4_24 a LEFT JOIN q4_23 b USING (region)"

        with patch("graph.agents.supervisor.ModelClient", return_value=_supervisor_mock(plan)), \
             patch("graph.agents.insight_agent.ModelClient",
                   return_value=MagicMock(invoke_json=MagicMock(return_value=VALID_INSIGHT))), \
              patch("graph.agents.viz_agent.ModelClient",
                    return_value=MagicMock(invoke=MagicMock(return_value=CHART_CODE_BAR))), \
             patch("graph.agents.python_agent.run_pandas_safe", return_value=DF_PYTHON_OUTPUT), \
             patch("graph.agents.python_agent.ModelClient",
                   return_value=MagicMock(invoke=MagicMock(return_value=PYTHON_CODE_YOY))), \
             patch("graph.agents.sql_agent.explain_query_plan", return_value={}), \
             patch("graph.agents.sql_agent.execute_sql", return_value=DF_Q4_COMPARISON), \
             patch("graph.agents.sql_agent.ModelClient",
                   return_value=MagicMock(invoke=MagicMock(return_value=sql))):
            graph = build_multi_agent_graph()
            result = graph.invoke(_state(plan))

        assert result["status"] == "success"
        assert result["completed_agents"] == ["supervisor", "sql", "python", "viz", "insight"]
        assert result["chart_b64"] is not None
        assert len(result["chart_b64"]) > 100
        assert result["chart_metadata"]["chart_type"] == "bar"
        assert result["insight"]["confidence"] == "high"
        assert len(result["action_trace"]) >= 5

    def test_02_revenue_by_region_with_viz(self):
        """Revenue by region → compute share → chart → insight."""
        plan = _complex_plan(
            "Lấy tổng doanh thu theo region năm 2024",
            "Tính tỷ trọng % từng region",
            "Bar chart doanh thu theo region",
            "Xác định region dẫn đầu",
        )
        chart_code = (
            "import matplotlib; matplotlib.use('Agg')\n"
            "import matplotlib.pyplot as plt\nimport io, base64\n"
            "plt.style.use('seaborn-v0_8-whitegrid')\n"
            "fig, ax = plt.subplots(figsize=(10, 6))\n"
            "ax.bar(df['region'], df['total_revenue'])\n"
            "ax.set_title('Revenue by Region')\n"
            "ax.set_xlabel('Region')\nax.set_ylabel('Revenue')\n"
            "fig.tight_layout()\nbuf = io.BytesIO()\n"
            "plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')\n"
            "buf.seek(0)\nb64 = base64.b64encode(buf.read()).decode('utf-8')\n"
            "plt.close()\n"
            "result = {'chart_b64': b64, 'chart_type': 'bar',"
            " 'title': 'Revenue by Region', 'x_col': 'region', 'y_col': 'total_revenue'}"
        )
        sql = "SELECT region, SUM(amount) AS total_revenue FROM orders WHERE year=2024 GROUP BY region"
        py = "import pandas as pd\ndf=df.copy()\ndf['ratio']=(df['total_revenue']/df['total_revenue'].sum()*100).round(1)\ndf"
        py_out = DF_REVENUE_BY_REGION.assign(ratio=[33.7, 27.0, 21.3, 18.0])

        with patch("graph.agents.supervisor.ModelClient", return_value=_supervisor_mock(plan)), \
             patch("graph.agents.insight_agent.ModelClient",
                   return_value=MagicMock(invoke_json=MagicMock(return_value=VALID_INSIGHT))), \
             patch("graph.agents.viz_agent.ModelClient",
                   return_value=MagicMock(invoke=MagicMock(return_value=chart_code))), \
             patch("graph.agents.python_agent.run_pandas_safe", return_value=py_out), \
             patch("graph.agents.python_agent.ModelClient",
                   return_value=MagicMock(invoke=MagicMock(return_value=py))), \
             patch("graph.agents.sql_agent.explain_query_plan", return_value={}), \
             patch("graph.agents.sql_agent.execute_sql", return_value=DF_REVENUE_BY_REGION), \
             patch("graph.agents.sql_agent.ModelClient",
                   return_value=MagicMock(invoke=MagicMock(return_value=sql))):
            graph = build_multi_agent_graph()
            result = graph.invoke(_state(plan))

        assert result["status"] == "success"
        assert result["completed_agents"] == ["supervisor", "sql", "python", "viz", "insight"]
        assert result["chart_b64"] is not None

    def test_03_top_products_with_margin(self):
        """Top products → margin → chart → insight."""
        plan = _complex_plan(
            "Lấy top sản phẩm, kèm margin",
            "Tính margin % từng sản phẩm",
            "Bar chart doanh thu + margin",
            "Xác định sản phẩm margin cao nhất",
        )
        sql = "SELECT p.name, SUM(o.amount) AS revenue, SUM((p.unit_price-p.cost)*o.quantity) AS margin FROM orders o JOIN products p ON o.product_id=p.id GROUP BY p.name LIMIT 10"
        py = "import pandas as pd\ndf=df.copy()\ndf['margin_pct']=(df['margin']/df['revenue']*100).round(1)\ndf"
        df = pd.DataFrame({
            "name": ["A", "B", "C"], "revenue": [50, 45, 40], "margin": [20, 15, 18]})
        py_out = df.assign(margin_pct=[40.0, 33.3, 45.0])

        with patch("graph.agents.supervisor.ModelClient", return_value=_supervisor_mock(plan)), \
             patch("graph.agents.insight_agent.ModelClient",
                   return_value=MagicMock(invoke_json=MagicMock(return_value=VALID_INSIGHT))), \
             patch("graph.agents.viz_agent.ModelClient",
                   return_value=MagicMock(invoke=MagicMock(return_value=CHART_CODE_GENERIC))), \
             patch("graph.agents.python_agent.run_pandas_safe", return_value=py_out), \
             patch("graph.agents.python_agent.ModelClient",
                   return_value=MagicMock(invoke=MagicMock(return_value=py))), \
             patch("graph.agents.sql_agent.explain_query_plan", return_value={}), \
             patch("graph.agents.sql_agent.execute_sql", return_value=df), \
             patch("graph.agents.sql_agent.ModelClient",
                   return_value=MagicMock(invoke=MagicMock(return_value=sql))):
            graph = build_multi_agent_graph()
            result = graph.invoke(_state(plan))

        assert result["status"] == "success"
        assert result["completed_agents"] == ["supervisor", "sql", "python", "viz", "insight"]

    def test_04_employee_salary_distribution(self):
        """Salary by level → stats → chart → insight."""
        plan = _complex_plan(
            "Lấy lương trung bình theo level", "Tính tỷ lệ % nhân viên theo level",
            "Bar chart phân bố lương", "Nhận xét phân bố nhân sự",
        )
        sql = "SELECT e.level, ROUND(AVG(e.salary)) AS avg_salary, COUNT(*) AS count FROM employees e WHERE e.status='active' GROUP BY e.level"
        py = "import pandas as pd\ndf=df.copy()\nt=df['count'].sum()\ndf['pct']=(df['count']/t*100).round(1)\ndf"
        py_out = DF_SALARY_DIST.assign(pct=[48.5, 30.3, 15.2, 6.1])

        with patch("graph.agents.supervisor.ModelClient", return_value=_supervisor_mock(plan)), \
             patch("graph.agents.insight_agent.ModelClient",
                   return_value=MagicMock(invoke_json=MagicMock(return_value=VALID_INSIGHT))), \
             patch("graph.agents.viz_agent.ModelClient",
                   return_value=MagicMock(invoke=MagicMock(return_value=CHART_CODE_GENERIC))), \
             patch("graph.agents.python_agent.run_pandas_safe", return_value=py_out), \
             patch("graph.agents.python_agent.ModelClient",
                   return_value=MagicMock(invoke=MagicMock(return_value=py))), \
             patch("graph.agents.sql_agent.explain_query_plan", return_value={}), \
             patch("graph.agents.sql_agent.execute_sql", return_value=DF_SALARY_DIST), \
             patch("graph.agents.sql_agent.ModelClient",
                   return_value=MagicMock(invoke=MagicMock(return_value=sql))):
            graph = build_multi_agent_graph()
            result = graph.invoke(_state(plan))

        assert result["status"] == "success"

    def test_05_inventory_turnover_analysis(self):
        """Inventory movement → turnover → chart → insight."""
        plan = _complex_plan(
            "Lấy movement theo warehouse", "Tính avg qty per move",
            "Bar chart turnover", "Đề xuất tối ưu phân bổ hàng",
        )
        sql = "SELECT w.name AS warehouse, COUNT(sm.id) AS movements, SUM(sm.quantity) AS total_qty FROM stock_movements sm JOIN warehouses w ON sm.to_warehouse_id=w.id GROUP BY w.name"
        py = "import pandas as pd\ndf=df.copy()\ndf['avg']=(df['total_qty']/df['movements']).round(1)\ndf"
        df = pd.DataFrame({"warehouse": ["HCMC", "Hanoi", "Danang"],
                           "movements": [150, 120, 80], "total_qty": [5000, 4000, 2500]})
        py_out = df.assign(avg=[33.3, 33.3, 31.3])

        with patch("graph.agents.supervisor.ModelClient", return_value=_supervisor_mock(plan)), \
             patch("graph.agents.insight_agent.ModelClient",
                   return_value=MagicMock(invoke_json=MagicMock(return_value=VALID_INSIGHT))), \
             patch("graph.agents.viz_agent.ModelClient",
                   return_value=MagicMock(invoke=MagicMock(return_value=CHART_CODE_GENERIC))), \
             patch("graph.agents.python_agent.run_pandas_safe", return_value=py_out), \
             patch("graph.agents.python_agent.ModelClient",
                   return_value=MagicMock(invoke=MagicMock(return_value=py))), \
             patch("graph.agents.sql_agent.explain_query_plan", return_value={}), \
             patch("graph.agents.sql_agent.execute_sql", return_value=df), \
             patch("graph.agents.sql_agent.ModelClient",
                   return_value=MagicMock(invoke=MagicMock(return_value=sql))):
            graph = build_multi_agent_graph()
            result = graph.invoke(_state(plan))

        assert result["status"] == "success"

    def test_06_customer_segment_profitability(self):
        """Segment revenue → avg per customer → chart → insight."""
        plan = _complex_plan(
            "Lấy doanh thu theo segment", "Tính avg per customer",
            "Bar chart doanh thu theo segment", "Xác định segment sinh lời nhất",
        )
        sql = "SELECT c.segment, SUM(o.amount) AS rev, COUNT(DISTINCT c.id) AS cust FROM orders o JOIN customers c ON o.customer_id=c.id GROUP BY c.segment"
        py = "import pandas as pd\ndf=df.copy()\ndf['avg']=(df['rev']/df['cust']).round(0)\ndf"
        df = pd.DataFrame({"segment": ["Enterprise", "SMB", "Startup"],
                           "rev": [500, 350, 150], "cust": [120, 250, 80]})
        py_out = df.assign(avg=[4.2, 1.4, 1.9])

        with patch("graph.agents.supervisor.ModelClient", return_value=_supervisor_mock(plan)), \
             patch("graph.agents.insight_agent.ModelClient",
                   return_value=MagicMock(invoke_json=MagicMock(return_value=VALID_INSIGHT))), \
             patch("graph.agents.viz_agent.ModelClient",
                   return_value=MagicMock(invoke=MagicMock(return_value=CHART_CODE_GENERIC))), \
             patch("graph.agents.python_agent.run_pandas_safe", return_value=py_out), \
             patch("graph.agents.python_agent.ModelClient",
                   return_value=MagicMock(invoke=MagicMock(return_value=py))), \
             patch("graph.agents.sql_agent.explain_query_plan", return_value={}), \
             patch("graph.agents.sql_agent.execute_sql", return_value=df), \
             patch("graph.agents.sql_agent.ModelClient",
                   return_value=MagicMock(invoke=MagicMock(return_value=sql))):
            graph = build_multi_agent_graph()
            result = graph.invoke(_state(plan))

        assert result["status"] == "success"
        assert result["insight"] is not None

    def test_07_qoq_growth_analysis(self):
        """QoQ revenue → growth → chart → insight."""
        plan = _complex_plan(
            "Lấy doanh thu theo quý 2024", "Tính % QoQ growth",
            "Line/bar chart theo quý", "Nhận xét xu hướng, dự báo",
        )
        chart = (
            "import matplotlib; matplotlib.use('Agg')\n"
            "import matplotlib.pyplot as plt\nimport io, base64\n"
            "plt.style.use('seaborn-v0_8-whitegrid')\n"
            "fig, ax = plt.subplots(figsize=(10, 6))\n"
            "ax.bar(df['quarter'].astype(str), df['revenue'])\n"
            "ax.set_title('Revenue by Quarter')\n"
            "fig.tight_layout()\nbuf = io.BytesIO()\n"
            "plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')\n"
            "buf.seek(0)\nb64 = base64.b64encode(buf.read()).decode('utf-8')\n"
            "plt.close()\n"
            "result = {'chart_b64': b64, 'chart_type': 'bar',"
            " 'title': 'Revenue by Quarter'}"
        )
        sql = "SELECT quarter, SUM(amount) AS revenue FROM orders WHERE year=2024 GROUP BY quarter ORDER BY quarter"
        py = "import pandas as pd\ndf=df.copy()\ndf['qoq']=(df['revenue'].pct_change()*100).round(1)\ndf"
        qdf = pd.DataFrame({"quarter": [1, 2, 3, 4], "revenue": [280, 310, 350, 420]})
        py_out = qdf.assign(qoq=[float("nan"), 10.7, 12.9, 20.0])

        with patch("graph.agents.supervisor.ModelClient", return_value=_supervisor_mock(plan)), \
             patch("graph.agents.insight_agent.ModelClient",
                   return_value=MagicMock(invoke_json=MagicMock(return_value=VALID_INSIGHT))), \
             patch("graph.agents.viz_agent.ModelClient",
                    return_value=MagicMock(invoke=MagicMock(return_value=chart))), \
             patch("graph.agents.python_agent.run_pandas_safe", return_value=py_out), \
             patch("graph.agents.python_agent.ModelClient",
                    return_value=MagicMock(invoke=MagicMock(return_value=py))), \
             patch("graph.agents.sql_agent.explain_query_plan", return_value={}), \
             patch("graph.agents.sql_agent.execute_sql", return_value=qdf), \
             patch("graph.agents.sql_agent.ModelClient",
                   return_value=MagicMock(invoke=MagicMock(return_value=sql))):
            graph = build_multi_agent_graph()
            result = graph.invoke(_state(plan))

        assert result["status"] == "success"
        assert result["chart_metadata"] is not None

    def test_08_payroll_analysis_by_department(self):
        """Payroll by dept → avg → chart → insight."""
        plan = _complex_plan(
            "Lấy payroll theo phòng ban", "Tính avg/người",
            "Bar chart chi phí lương", "Đề xuất tối ưu chi phí",
        )
        sql = "SELECT d.name AS dept, SUM(p.base_salary+p.bonus-p.deduction) AS net, COUNT(DISTINCT p.employee_id) AS emp FROM payroll p JOIN employees e ON p.employee_id=e.id JOIN departments d ON e.department_id=d.id WHERE p.year=2024 GROUP BY d.name"
        py = "import pandas as pd\ndf=df.copy()\ndf['avg']=(df['net']/df['emp']).round(0)\ndf"
        df = pd.DataFrame({"dept": ["Eng", "Sales", "Mktg"],
                           "net": [1500, 900, 600], "emp": [45, 30, 20]})
        py_out = df.assign(avg=[33.3, 30.0, 30.0])

        with patch("graph.agents.supervisor.ModelClient", return_value=_supervisor_mock(plan)), \
             patch("graph.agents.insight_agent.ModelClient",
                   return_value=MagicMock(invoke_json=MagicMock(return_value=VALID_INSIGHT))), \
             patch("graph.agents.viz_agent.ModelClient",
                   return_value=MagicMock(invoke=MagicMock(return_value=CHART_CODE_GENERIC))), \
             patch("graph.agents.python_agent.run_pandas_safe", return_value=py_out), \
             patch("graph.agents.python_agent.ModelClient",
                   return_value=MagicMock(invoke=MagicMock(return_value=py))), \
             patch("graph.agents.sql_agent.explain_query_plan", return_value={}), \
             patch("graph.agents.sql_agent.execute_sql", return_value=df), \
             patch("graph.agents.sql_agent.ModelClient",
                   return_value=MagicMock(invoke=MagicMock(return_value=sql))):
            graph = build_multi_agent_graph()
            result = graph.invoke(_state(plan))

        assert result["status"] == "success"
        assert result["insight"] is not None

    def test_09_stock_movement_anomaly(self):
        """Stock movement → detect anomaly → chart → insight."""
        plan = _complex_plan(
            "Lấy stock movements theo sản phẩm", "Phát hiện outlier IQR",
            "Bar chart movement, highlight anomaly", "Điều tra movement bất thường",
        )
        sql = "SELECT p.name AS product, sm.movement_type, SUM(sm.quantity) AS total_qty FROM stock_movements sm JOIN products p ON sm.product_id=p.id GROUP BY p.name, sm.movement_type"
        py = "import pandas as pd\nimport numpy as np\ndf=df.copy()\nq1=df['total_qty'].quantile(0.25)\nq3=df['total_qty'].quantile(0.75)\niqr=q3-q1\ndf['is_anomaly']=((df['total_qty']<q1-1.5*iqr)|(df['total_qty']>q3+1.5*iqr))\ndf"
        df = pd.DataFrame({
            "product": ["W", "G", "T", "W"], "movement_type": ["in", "in", "out", "out"],
            "total_qty": [500, 300, 100, 450]})
        py_out = df.assign(is_anomaly=[False, False, True, False])

        with patch("graph.agents.supervisor.ModelClient", return_value=_supervisor_mock(plan)), \
             patch("graph.agents.insight_agent.ModelClient",
                   return_value=MagicMock(invoke_json=MagicMock(return_value=VALID_INSIGHT))), \
             patch("graph.agents.viz_agent.ModelClient",
                   return_value=MagicMock(invoke=MagicMock(return_value=CHART_CODE_GENERIC))), \
             patch("graph.agents.python_agent.run_pandas_safe", return_value=py_out), \
             patch("graph.agents.python_agent.ModelClient",
                   return_value=MagicMock(invoke=MagicMock(return_value=py))), \
             patch("graph.agents.sql_agent.explain_query_plan", return_value={}), \
             patch("graph.agents.sql_agent.execute_sql", return_value=df), \
             patch("graph.agents.sql_agent.ModelClient",
                   return_value=MagicMock(invoke=MagicMock(return_value=sql))):
            graph = build_multi_agent_graph()
            result = graph.invoke(_state(plan))

        assert result["status"] == "success"
        assert result["insight"] is not None

    def test_10_cross_domain_sales_inventory(self):
        """Cross-domain: sales + inventory → ratio → chart → insight."""
        plan = _complex_plan(
            "Lấy doanh thu + tồn kho (cross-domain)", "Tính stock_to_sales ratio",
            "Bar chart doanh thu vs tồn kho", "Đề xuất giảm overstock",
        )
        chart = (
            "import matplotlib; matplotlib.use('Agg')\n"
            "import matplotlib.pyplot as plt\nimport io, base64\n"
            "plt.style.use('seaborn-v0_8-whitegrid')\n"
            "fig, ax = plt.subplots(figsize=(10, 6))\n"
            "ax.bar(df['product'], df['sales'])\n"
            "ax.set_title('Sales by Product')\n"
            "fig.tight_layout()\nbuf = io.BytesIO()\n"
            "plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')\n"
            "buf.seek(0)\nb64 = base64.b64encode(buf.read()).decode('utf-8')\n"
            "plt.close()\n"
            "result = {'chart_b64': b64, 'chart_type': 'bar',"
            " 'title': 'Sales by Product'}"
        )
        sql = "SELECT p.name AS product, COALESCE(SUM(o.amount),0) AS sales, COALESCE(s.quantity,0) AS stock FROM products p LEFT JOIN orders o ON o.product_id=p.id LEFT JOIN stock s ON s.product_id=p.id GROUP BY p.name, s.quantity"
        py = "import pandas as pd\ndf=df.copy()\ndf['ratio']=(df['stock']/df['sales']*1000).round(2)\ndf"
        df = pd.DataFrame({"product": ["A", "B", "C"],
                           "sales": [500, 400, 300], "stock": [120, 80, 350]})
        py_out = df.assign(ratio=[240.0, 200.0, 1166.67])

        with patch("graph.agents.supervisor.ModelClient", return_value=_supervisor_mock(plan)), \
             patch("graph.agents.insight_agent.ModelClient",
                   return_value=MagicMock(invoke_json=MagicMock(return_value=VALID_INSIGHT))), \
             patch("graph.agents.viz_agent.ModelClient",
                   return_value=MagicMock(invoke=MagicMock(return_value=CHART_CODE_GENERIC))), \
             patch("graph.agents.python_agent.run_pandas_safe", return_value=py_out), \
             patch("graph.agents.python_agent.ModelClient",
                   return_value=MagicMock(invoke=MagicMock(return_value=py))), \
             patch("graph.agents.sql_agent.explain_query_plan", return_value={}), \
             patch("graph.agents.sql_agent.execute_sql", return_value=df), \
             patch("graph.agents.sql_agent.ModelClient",
                   return_value=MagicMock(invoke=MagicMock(return_value=sql))):
            graph = build_multi_agent_graph()
            result = graph.invoke(_state(plan))

        assert result["status"] == "success"
        assert result["insight"]["confidence"] == "high"
        assert len(result["completed_agents"]) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
