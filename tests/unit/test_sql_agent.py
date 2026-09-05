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

import re
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from graph.agents.sql_agent import _extract_sql, build_system_prompt, sql_agent_node
from graph.state import make_initial_state
from graph.tools.sql_tool import TableNotPermittedError, assert_tables_permitted
from perception.connection_profile import build_profile
from perception.schema_context import SchemaContext
from tests.fixtures.mini_schema import MINI_TABLES

SCHEMA_CONTEXT = SchemaContext()

TEST_PROFILE = build_profile(
    dsn="postgresql://u:p@h:5432/d",
    tables=MINI_TABLES,
    grants={"test_user": frozenset({"orders", "customers", "products"})},
)

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
    s = make_initial_state("test", SCHEMA_CONTEXT)
    s["execution_plan"] = plan or []
    s["shared_metadata"] = {"profile": TEST_PROFILE, "user": "test_user"}
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

    def test_no_sql_keyword_raises_fail_closed(self):
        """Superseded by fail-closed extraction — see TestExtractSqlFailsClosed.
        Kept here (renamed) so the class's coverage of `_extract_sql`'s normal
        cases stays next to its one abnormal case."""
        with pytest.raises(ValueError, match="Không trích được SQL"):
            _extract_sql("Just some text")

    def test_the_english_word_with_in_prose_is_not_mistaken_for_a_cte(self):
        """Ca quan sát được thật trên BIRD: 8 trong 18 ca SQL-sinh-lỗi
        (44%) là do đúng lỗi này, không phải do model viết SQL sai.

        Bản cũ dùng `str.find("with")` nên cắt ngay từ chữ "with" giữa câu
        văn và trả về một mớ không phải SQL."""
        raw = (
            "To find the school with the highest average score in Reading, "
            "you can use the following query:\n"
            "SELECT name FROM schools ORDER BY score DESC LIMIT 1"
        )
        assert _extract_sql(raw) == "SELECT name FROM schools ORDER BY score DESC LIMIT 1"

    def test_a_real_cte_after_prose_is_still_found(self):
        """Sửa ca trên không được làm mất CTE thật: `WITH ... SELECT` phải
        giữ nguyên cả mệnh đề WITH, không được cắt vào SELECT bên trong."""
        raw = (
            "Here is a query with a CTE:\n"
            "WITH cte AS (SELECT * FROM orders) SELECT * FROM cte"
        )
        assert _extract_sql(raw) == "WITH cte AS (SELECT * FROM orders) SELECT * FROM cte"

    def test_prose_wrapped_around_a_fenced_block_is_dropped(self):
        raw = (
            "Sure! To answer that with the data you have:\n"
            "```sql\nSELECT 1 FROM t\n```\n"
            "Let me know if you want it filtered."
        )
        assert _extract_sql(raw) == "SELECT 1 FROM t"

    def test_prose_after_a_semicolon_is_dropped(self):
        """7 trong 8 ca lỗi cú pháp trên BIRD có hình dạng này: model viết
        xong SQL, chấm phẩy, rồi giải thích tiếp."""
        raw = (
            "SELECT a FROM t ORDER BY b DESC LIMIT 1;\n\n"
            "This query performs the following steps:\n"
            "1. Joins the tables on the key."
        )
        assert _extract_sql(raw) == "SELECT a FROM t ORDER BY b DESC LIMIT 1;"

    def test_prose_after_a_blank_line_is_dropped_when_there_is_no_semicolon(self):
        """Ca thứ 8: không có dấu chấm phẩy nào để cắt."""
        raw = (
            "SELECT a\nFROM t\nWHERE x = (SELECT MAX(y) FROM u)\n\n"
            "This query joins the tables based on their respective keys."
        )
        assert _extract_sql(raw) == "SELECT a\nFROM t\nWHERE x = (SELECT MAX(y) FROM u)"

    def test_a_blank_line_inside_the_query_does_not_truncate_it(self):
        """Điều kiện bảo vệ: cắt vô điều kiện ở dòng trống sẽ làm cụt một
        truy vấn được định dạng thoáng — hỏng nặng hơn hẳn thứ đang sửa."""
        raw = "SELECT a\nFROM t\n\nORDER BY a DESC"
        assert _extract_sql(raw) == raw

    def test_the_word_with_inside_an_identifier_is_not_a_start(self):
        """`\b` ở cả hai đầu: thiếu nó thì `WITH` khớp trong `WITHIN`."""
        raw = "SELECT x FROM t GROUP BY y WITHIN GROUP (ORDER BY z)"
        assert _extract_sql(raw) == raw


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
        assert result["sql_result"]["truncated"] is False
        assert result["degradation_reason"] == []

    @patch("graph.agents.sql_agent.explain_query_plan")
    @patch("graph.agents.sql_agent.execute_sql")
    @patch("graph.agents.sql_agent.ModelClient")
    def test_a_truncated_result_is_named_in_degradation_reason(
        self, MockClient, mock_execute, mock_explain
    ):
        """`execute_sql` cắt kết quả ở `SQL_MAX_ROWS` qua `df.attrs["truncated"]`
        (graph/tools/sql_tool.py) — trước bản này, cờ đó chỉ tới được
        `action_trace` (một dòng chữ chôn trong danh sách trace), không
        phải `degradation_reason`. `finalize_node` phân loại "success" khi
        không có lý do suy giảm nào — nên một câu SELECT trả về nhiều hơn
        trần dòng vẫn được báo "success", và người dùng nhận một bảng
        THIẾU dữ liệu mà không có banner nào cảnh báo."""
        mock_explain.return_value = {"total_cost_estimate": 99.0}

        mock_instance = MagicMock()
        mock_instance.invoke.return_value = "SELECT * FROM orders"
        MockClient.return_value = mock_instance

        df = pd.DataFrame({"id": range(50000)})
        df.attrs["truncated"] = True
        df.attrs["row_cap"] = 50000
        mock_execute.return_value = df

        s = _state(PLAN_SQL_INSIGHT)
        result = sql_agent_node(s)

        assert result["status"] == "running"
        assert result["sql_result"]["truncated"] is True
        assert result["sql_result"]["row_cap"] == 50000
        assert result["agent_outputs"]["sql"]["truncated"] is True
        assert result["degradation_reason"], "kết quả bị cắt phải được ghi thành lý do suy giảm"
        assert any("50000" in r for r in result["degradation_reason"])

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

    @patch("graph.agents.sql_agent.explain_query_plan")
    @patch("graph.agents.sql_agent.execute_sql")
    @patch("graph.agents.sql_agent.ModelClient")
    def test_every_attempt_reaches_the_trace_not_just_the_last_one(
        self, MockClient, mock_execute, mock_explain
    ):
        """`append_trace` dựng danh sách mới từ `state["action_trace"]`.

        Vòng lặp retry nào cũng gọi nó với CÙNG một `state` gốc thì mỗi
        lần lại dựng lại từ danh sách ban đầu, và chỉ mục cuối sống sót.
        Chẩn đoán "Attempt 1: ..." — đúng thứ mà lớp fail-closed của
        `_extract_sql` sinh ra để người ta đọc được — không bao giờ tới
        được UI hay log JSONL.

        Ba mục phải có mặt: "started" trước vòng lặp, lỗi của lần 1, và OK
        của lần 2."""
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
        before = len(s["action_trace"])
        trace = sql_agent_node(s)["action_trace"]

        added = trace[before:]
        assert len(added) == 3, f"mất mục trung gian: {[e['observation'] for e in added]}"
        assert added[0]["status"] == "started"
        assert "Attempt 1" in added[1]["observation"]
        assert added[1]["status"] == "error"
        assert added[2]["status"] == "ok"

    def test_missing_step_returns_failed(self):
        """No sql step in plan → status='failed'."""
        s = _state(PLAN_NO_SQL)
        result = sql_agent_node(s)
        assert result["status"] == "failed"
        assert result["last_error"]["agent"] == "sql"

    @patch("graph.agents.sql_agent.execute_sql")
    @patch("graph.agents.sql_agent.ModelClient")
    def test_table_not_permitted_error_never_leaks_the_internal_segment(
        self, MockClient, mock_execute,
    ):
        """Permanent regression test for the leak-path finding (Task 8/10 review).

        assert_tables_permitted() (graph/tools/sql_tool.py) raises
        TableNotPermittedError with `.public` (safe to surface) and
        `.forbidden` (exact forbidden table names — must never reach state).
        sql_agent_node sanitizes via _public_error_message() before anything
        enters state, because that state (last_error, action_trace,
        agent_outputs) can reach the UI via app.py or travel through
        reflector_agent_node into its own LLM prompt and trace entry.

        The exception here is PROVOKED from the real guard
        (assert_tables_permitted against TEST_PROFILE, which grants
        test_user only orders/customers/products — not payroll) rather than
        hand-built with a "(nội bộ: [...])" string. A hand-built message
        would stay green even if the marker/format assert_tables_permitted
        actually produces changed — this construction fails loudly instead
        if the two drift apart, since it exercises the real code path that
        must stay in sync with the sanitizer.
        """
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = "SELECT * FROM payroll"
        MockClient.return_value = mock_instance
        try:
            assert_tables_permitted("SELECT * FROM payroll", TEST_PROFILE, "test_user")
            pytest.fail("expected TableNotPermittedError — test fixture drifted")
        except TableNotPermittedError as real_exc:
            mock_execute.side_effect = real_exc

        s = _state(PLAN_SQL_INSIGHT)
        result = sql_agent_node(s)

        assert result["status"] == "running"  # retries exhausted, not a crash
        haystacks = [
            str(result["last_error"]),
            str(result["action_trace"]),
            str(result["agent_outputs"]),
        ]
        for text in haystacks:
            assert "payroll" not in text, f"forbidden table name leaked: {text!r}"
            assert "nội bộ" not in text, f"internal-segment marker leaked: {text!r}"


class TestBudgetCanRunOutBetweenModelAndTool:
    """Regression: `node_may_run` chỉ gác cửa TRƯỚC lời gọi model — nó
    không thấy phần chi tiêu SAU đó. Trước bản sửa này, một lượt qua được
    cửa gác (còn đủ ~15s cho model) vẫn có thể chạy `execute_sql` với
    `SQL_TIMEOUT_MS` (10s) độc lập với deadline ngay sau, cộng lại vượt
    ngân sách toàn cục dù cửa gác nói "còn giờ".

    `_clock` ở đây trả về hai mốc: mốc đầu (dùng cho `node_may_run` ở đầu
    hàm) còn nhiều thời gian; mốc sau (dùng cho kiểm tra SAU khi model trả
    lời) đã hết. Không `time.sleep` — cùng kỹ thuật tiêm đồng hồ với
    `tests/unit/test_budget.py`.
    """

    @staticmethod
    def _state_with_budget_dying_after_the_model_call():
        s = _state(PLAN_SQL_INSIGHT)
        seq = [1000.0, 1050.001]

        def clock() -> float:
            return seq.pop(0) if len(seq) > 1 else seq[0]

        s["_clock"] = clock
        s["deadline_ts"] = 1050.0
        return s

    @patch("graph.agents.sql_agent.execute_sql")
    @patch("graph.agents.sql_agent.ModelClient")
    def test_execute_sql_is_never_called_once_the_budget_is_gone(self, MockClient, mock_execute):
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = "SELECT region, SUM(amount) FROM orders GROUP BY region"
        MockClient.return_value = mock_instance

        s = self._state_with_budget_dying_after_the_model_call()
        result = sql_agent_node(s)

        mock_execute.assert_not_called()
        assert result["last_error"]["error_type"] == "budget_exceeded"

    @patch("graph.agents.sql_agent.execute_sql")
    @patch("graph.agents.sql_agent.ModelClient")
    def test_the_reason_says_budget_not_a_generic_sql_failure(self, MockClient, mock_execute):
        """Nhãn sai (`sql_execution`) khiến reflector chẩn đoán sai lớp lỗi —
        chính lý do nhãn `budget_exceeded` tồn tại (xem graph/errors.py)."""
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = "SELECT region, SUM(amount) FROM orders GROUP BY region"
        MockClient.return_value = mock_instance

        s = self._state_with_budget_dying_after_the_model_call()
        result = sql_agent_node(s)

        assert "ngân sách" in result["last_error"]["traceback"]


class TestBuildSystemPromptHasNoSurvivingPlaceholders:
    def test_no_placeholder_survives_rendering(self):
        """IMPORTANT 1 (final review): assert on the RENDERED OUTPUT, not on an
        exact-string .replace() call matching the template byte-for-byte.

        build_system_prompt drops "Task: {task}\\n\\n" by exact string match.
        tests/unit/test_prompts_are_schema_agnostic.py separately REQUIRES the
        template to contain the literal "{task}" placeholder — so the template
        is contractually required to hold the exact substring the code must
        remove verbatim. If a future edit changes "Task: {task}" to
        "Task:{task}", adds a trailing space, or the file picks up CRLF line
        endings, the .replace() call silently stops matching and the model is
        told its task is the literal string "{task}" — while every test that
        checks the template file directly stays green. Only a test on the
        function's OUTPUT catches that.
        """
        ctx = SchemaContext(
            retrieved_tables=("orders",),
            rendered_text="CREATE TABLE orders (id INT PRIMARY KEY);",
            few_shots=({"question": "Doanh thu tháng này?", "sql": "SELECT 1"},),
        )
        rendered = build_system_prompt(ctx)
        assert re.search(r"\{[a-z_]+\}", rendered) is None, rendered


class TestExtractSqlFailsClosed:
    def test_prose_without_sql_raises(self):
        from graph.agents.sql_agent import _extract_sql

        with pytest.raises(ValueError, match="Không trích được SQL"):
            _extract_sql("Xin lỗi, tôi không đủ thông tin để viết câu truy vấn này.")

    def test_empty_output_raises(self):
        from graph.agents.sql_agent import _extract_sql

        with pytest.raises(ValueError, match="Không trích được SQL"):
            _extract_sql("   ")

    def test_a_real_select_still_comes_through(self):
        from graph.agents.sql_agent import _extract_sql

        assert _extract_sql("```sql\nSELECT 1\n```") == "SELECT 1"

    def test_a_with_cte_still_comes_through(self):
        from graph.agents.sql_agent import _extract_sql

        out = _extract_sql("Đây là câu trả lời:\nWITH t AS (SELECT 1) SELECT * FROM t")
        assert out.startswith("WITH")


class TestSqlTimeoutCeiling:
    def test_default_statement_timeout_is_ten_seconds(self):
        """Spec 4.1 lớp 4: 30s → 10s, nằm trong ngân sách wall-clock mục 5."""
        import importlib

        import graph.tools.sql_tool as sql_tool

        importlib.reload(sql_tool)
        assert sql_tool.SQL_TIMEOUT_MS == 10000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
