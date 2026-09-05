"""
Integration tests — 10 simple SQL-only queries (sql + insight).

All tests mock ModelClient for supervisor, sql, and insight agents so no
real Ollama or PostgreSQL calls are made.
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
    DF_REVENUE_BY_REGION,
    DF_TOP_PRODUCTS,
    DF_EMPLOYEE_COUNT,
    DF_MONTHLY_SALES,
    DF_INVENTORY_THRESHOLD,
    DF_Q4_COMPARISON,
    VALID_INSIGHT,
)

TEST_PROFILE = build_profile(
    dsn="postgresql://u:p@h:5432/d",
    tables=MINI_TABLES,
    grants={"test_user": frozenset({ALL_TABLES})},
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _supervisor_mock(plan: list[dict]) -> MagicMock:
    m = MagicMock()
    m.invoke_json.return_value = {
        "plan_summary": "test",
        "steps": plan,
    }
    m.calls_made = 1
    return m


def _sql_mock(sql_text: str) -> MagicMock:
    m = MagicMock()
    m.invoke.return_value = sql_text
    m.calls_made = 1
    return m


def _insight_mock() -> MagicMock:
    m = MagicMock()
    m.invoke_json.return_value = VALID_INSIGHT
    m.calls_made = 1
    return m


def _simple_plan(sql_task: str) -> list[dict]:
    return [
        {"step": 1, "agent": "sql", "task": sql_task,
         "depends_on": [], "skill_type": "text-to-sql"},
        {"step": 2, "agent": "insight",
         "task": "Tóm tắt kết quả và đưa ra nhận xét",
         "depends_on": ["sql"], "skill_type": "insight-generation"},
    ]


def _state(plan: list[dict], query: str = "Test query") -> dict:
    s = make_initial_state(query, SCHEMA_CONTEXT)
    s["execution_plan"] = plan
    s["shared_metadata"] = {"profile": TEST_PROFILE, "user": "test_user"}
    return s


def _all_patches(plan, sql_text, df, test_fn):
    """Apply all 5 patches needed for a simple query integration test."""
    p1 = patch("graph.agents.supervisor.ModelClient", return_value=_supervisor_mock(plan))
    p2 = patch("graph.agents.insight_agent.ModelClient", return_value=_insight_mock())
    p3 = patch("graph.agents.sql_agent.explain_query_plan", return_value={})
    p4 = patch("graph.agents.sql_agent.execute_sql", return_value=df)
    p5 = patch("graph.agents.sql_agent.ModelClient", return_value=_sql_mock(sql_text))
    return p1, p2, p3, p4, p5


# ── Tests ────────────────────────────────────────────────────────────────────

class TestSimpleQueries:

    def test_01_revenue_by_region(self):
        """Query: Tổng doanh thu theo region năm 2024."""
        plan = _simple_plan("Lấy tổng doanh thu theo region năm 2024")
        sql = "SELECT region, SUM(amount) AS total_revenue FROM orders WHERE year=2024 GROUP BY region"

        with patch("graph.agents.supervisor.ModelClient", return_value=_supervisor_mock(plan)), \
             patch("graph.agents.insight_agent.ModelClient", return_value=_insight_mock()), \
             patch("graph.agents.sql_agent.explain_query_plan", return_value={}), \
             patch("graph.agents.sql_agent.execute_sql", return_value=DF_REVENUE_BY_REGION), \
             patch("graph.agents.sql_agent.ModelClient", return_value=_sql_mock(sql)):
            graph = build_multi_agent_graph()
            result = graph.invoke(_state(plan))

        assert result["status"] == "success"
        assert result["completed_agents"] == ["supervisor", "sql", "insight"]
        assert result["insight"]["confidence"] == "high"
        assert result["shared_dataframe"]["shape"] == [4, 2]

    def test_02_top_5_products(self):
        """Query: Top 5 sản phẩm bán chạy nhất theo doanh thu."""
        plan = _simple_plan("Lấy top 5 sản phẩm bán chạy nhất theo doanh thu")
        sql = "SELECT p.name, SUM(o.amount) AS total_revenue FROM orders o JOIN products p ON o.product_id = p.id GROUP BY p.name ORDER BY total_revenue DESC LIMIT 5"

        with patch("graph.agents.supervisor.ModelClient", return_value=_supervisor_mock(plan)), \
             patch("graph.agents.insight_agent.ModelClient", return_value=_insight_mock()), \
             patch("graph.agents.sql_agent.explain_query_plan", return_value={}), \
             patch("graph.agents.sql_agent.execute_sql", return_value=DF_TOP_PRODUCTS), \
             patch("graph.agents.sql_agent.ModelClient", return_value=_sql_mock(sql)):
            graph = build_multi_agent_graph()
            result = graph.invoke(_state(plan))

        assert result["status"] == "success"
        assert result["completed_agents"] == ["supervisor", "sql", "insight"]
        assert len(result["action_trace"]) >= 3

    def test_03_employee_count_by_department(self):
        """Query: Số lượng nhân viên theo từng phòng ban."""
        plan = _simple_plan("Đếm số nhân viên theo phòng ban")
        sql = "SELECT d.name AS department, COUNT(e.id) AS count FROM departments d JOIN employees e ON d.id = e.department_id WHERE e.status = 'active' GROUP BY d.name ORDER BY count DESC"

        with patch("graph.agents.supervisor.ModelClient", return_value=_supervisor_mock(plan)), \
             patch("graph.agents.insight_agent.ModelClient", return_value=_insight_mock()), \
             patch("graph.agents.sql_agent.explain_query_plan", return_value={}), \
             patch("graph.agents.sql_agent.execute_sql", return_value=DF_EMPLOYEE_COUNT), \
             patch("graph.agents.sql_agent.ModelClient", return_value=_sql_mock(sql)):
            graph = build_multi_agent_graph()
            result = graph.invoke(_state(plan))

        assert result["status"] == "success"
        assert result["completed_agents"] == ["supervisor", "sql", "insight"]

    def test_04_monthly_sales_trend(self):
        """Query: Doanh thu theo tháng trong năm 2024."""
        plan = _simple_plan("Doanh thu theo từng tháng năm 2024")
        sql = "SELECT EXTRACT(MONTH FROM order_date) AS month, SUM(amount) AS total_revenue FROM orders WHERE year=2024 GROUP BY EXTRACT(MONTH FROM order_date) ORDER BY month"

        with patch("graph.agents.supervisor.ModelClient", return_value=_supervisor_mock(plan)), \
             patch("graph.agents.insight_agent.ModelClient", return_value=_insight_mock()), \
             patch("graph.agents.sql_agent.explain_query_plan", return_value={}), \
             patch("graph.agents.sql_agent.execute_sql", return_value=DF_MONTHLY_SALES), \
             patch("graph.agents.sql_agent.ModelClient", return_value=_sql_mock(sql)):
            graph = build_multi_agent_graph()
            result = graph.invoke(_state(plan))

        assert result["status"] == "success"
        assert result["shared_dataframe"]["shape"] == [6, 2]

    def test_05_average_order_value_by_region(self):
        """Query: Giá trị đơn hàng trung bình theo region."""
        plan = _simple_plan("Tính giá trị đơn hàng trung bình theo region")
        sql = "SELECT region, ROUND(AVG(amount)) AS avg_order_value FROM orders WHERE status IN ('completed','processing') GROUP BY region"
        df = DF_REVENUE_BY_REGION.rename(columns={"total_revenue": "avg_order_value"})

        with patch("graph.agents.supervisor.ModelClient", return_value=_supervisor_mock(plan)), \
             patch("graph.agents.insight_agent.ModelClient", return_value=_insight_mock()), \
             patch("graph.agents.sql_agent.explain_query_plan", return_value={}), \
             patch("graph.agents.sql_agent.execute_sql", return_value=df), \
             patch("graph.agents.sql_agent.ModelClient", return_value=_sql_mock(sql)):
            graph = build_multi_agent_graph()
            result = graph.invoke(_state(plan))

        assert result["status"] == "success"
        assert "insight" in result["completed_agents"]

    def test_06_customer_count_by_segment(self):
        """Query: Số lượng khách hàng theo phân khúc."""
        plan = _simple_plan("Đếm số lượng khách hàng theo segment")
        sql = "SELECT segment, COUNT(id) AS customer_count FROM customers GROUP BY segment ORDER BY customer_count DESC"

        with patch("graph.agents.supervisor.ModelClient", return_value=_supervisor_mock(plan)), \
             patch("graph.agents.insight_agent.ModelClient", return_value=_insight_mock()), \
             patch("graph.agents.sql_agent.explain_query_plan", return_value={}), \
             patch("graph.agents.sql_agent.execute_sql", return_value=pd.DataFrame({
                 "segment": ["Enterprise", "SMB", "Startup"], "customer_count": [120, 250, 80]})), \
             patch("graph.agents.sql_agent.ModelClient", return_value=_sql_mock(sql)):
            graph = build_multi_agent_graph()
            result = graph.invoke(_state(plan))

        assert result["status"] == "success"

    def test_07_inventory_below_threshold(self):
        """Query: Sản phẩm có tồn kho dưới ngưỡng tối thiểu."""
        plan = _simple_plan("Tìm sản phẩm có tồn kho dưới ngưỡng tối thiểu")
        sql = "SELECT p.name, s.quantity, s.min_threshold, w.name AS warehouse FROM stock s JOIN products p ON s.product_id = p.id JOIN warehouses w ON s.warehouse_id = w.id WHERE s.quantity < s.min_threshold"

        with patch("graph.agents.supervisor.ModelClient", return_value=_supervisor_mock(plan)), \
             patch("graph.agents.insight_agent.ModelClient", return_value=_insight_mock()), \
             patch("graph.agents.sql_agent.explain_query_plan", return_value={}), \
             patch("graph.agents.sql_agent.execute_sql", return_value=DF_INVENTORY_THRESHOLD), \
             patch("graph.agents.sql_agent.ModelClient", return_value=_sql_mock(sql)):
            graph = build_multi_agent_graph()
            result = graph.invoke(_state(plan))

        assert result["status"] == "success"
        assert result["shared_dataframe"]["shape"] == [3, 4]

    def test_08_revenue_by_quarter(self):
        """Query: Tổng doanh thu theo quý năm 2024."""
        plan = _simple_plan("Tổng doanh thu theo quý năm 2024")
        sql = "SELECT quarter, SUM(amount) AS total_revenue FROM orders WHERE year=2024 GROUP BY quarter ORDER BY quarter"

        with patch("graph.agents.supervisor.ModelClient", return_value=_supervisor_mock(plan)), \
             patch("graph.agents.insight_agent.ModelClient", return_value=_insight_mock()), \
             patch("graph.agents.sql_agent.explain_query_plan", return_value={}), \
             patch("graph.agents.sql_agent.execute_sql", return_value=pd.DataFrame({
                 "quarter": [1, 2, 3, 4], "total_revenue": [280, 310, 350, 420]})), \
             patch("graph.agents.sql_agent.ModelClient", return_value=_sql_mock(sql)):
            graph = build_multi_agent_graph()
            result = graph.invoke(_state(plan))

        assert result["status"] == "success"

    def test_09_top_products_by_quantity(self):
        """Query: Top sản phẩm theo số lượng bán ra."""
        plan = _simple_plan("Top sản phẩm bán chạy nhất theo số lượng")
        sql = "SELECT p.name, SUM(o.quantity) AS total_quantity FROM orders o JOIN products p ON o.product_id = p.id GROUP BY p.name ORDER BY total_quantity DESC LIMIT 10"

        with patch("graph.agents.supervisor.ModelClient", return_value=_supervisor_mock(plan)), \
             patch("graph.agents.insight_agent.ModelClient", return_value=_insight_mock()), \
             patch("graph.agents.sql_agent.explain_query_plan", return_value={}), \
             patch("graph.agents.sql_agent.execute_sql", return_value=pd.DataFrame({
                 "name": ["Widget A", "Gadget B"], "total_quantity": [1500, 1200]})), \
             patch("graph.agents.sql_agent.ModelClient", return_value=_sql_mock(sql)):
            graph = build_multi_agent_graph()
            result = graph.invoke(_state(plan))

        assert result["status"] == "success"

    def test_10_revenue_q4_comparison(self):
        """Query: So sánh doanh thu Q4 2024 vs Q4 2023 theo region."""
        plan = _simple_plan("So sánh doanh thu Q4 2024 vs Q4 2023 theo region")
        sql = "WITH q4_24 AS (...) SELECT a.region, a.rev_2024, b.rev_2023 FROM q4_24 a LEFT JOIN q4_23 b USING (region)"

        with patch("graph.agents.supervisor.ModelClient", return_value=_supervisor_mock(plan)), \
             patch("graph.agents.insight_agent.ModelClient", return_value=_insight_mock()), \
             patch("graph.agents.sql_agent.explain_query_plan", return_value={}), \
             patch("graph.agents.sql_agent.execute_sql", return_value=DF_Q4_COMPARISON), \
             patch("graph.agents.sql_agent.ModelClient", return_value=_sql_mock(sql)):
            graph = build_multi_agent_graph()
            result = graph.invoke(_state(plan))

        assert result["status"] == "success"
        assert result["sql_result"]["row_count"] == 4
        assert len(result["action_trace"]) >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
