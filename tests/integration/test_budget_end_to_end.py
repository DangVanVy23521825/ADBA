"""Ngân sách nhìn từ ngoài graph.

PHẠM VI — ĐÃ CẬP NHẬT. Bản đầu của file này (viết ở Task 9) ghi rằng
`budget_s=0` không chặn được gì, vì `route_next_agent()` khi đó mới chỉ
biết `calls_exhausted` (trần SỐ LƯỢT gọi model, Task 6) và chưa hề đọc
`deadline_ts`/`time_left`. Ghi chú đó GIỜ ĐÃ SAI: Task 10 nối
`graph.budget.node_may_run` vào cả năm specialist node, nên hết ngân
sách THỜI GIAN là một cơ chế chặn có thật ở cấp graph. Hai test dựa trên
`budget_s` bên dưới kiểm đúng điều đó.

Hai lớp chặn được kiểm riêng, vì chúng hỏng theo hai cách khác nhau:

  * trần lời gọi (`llm_calls_used = MAX_LLM_CALLS_PER_QUERY`) — chặn ở
    router, trước khi node chạy;
  * deadline (`node_may_run`) — chặn Ở TRONG node, và node bị chặn TỰ
    THÊM mình vào `completed_agents` để router không chọn lại nó. Chính
    chi tiết đó từng khiến `finalize` gọi một lượt bị cắt sạch là
    "success" và xoá luôn `degradation_reason` (lỗi C1 của bản review
    toàn nhánh). `test_sql_succeeds_then_the_budget_runs_out_before_insight`
    là bản kiểm ở CẤP GRAPH cho lỗi đó; bản kiểm ở cấp node nằm trong
    `tests/unit/test_finalize.py`.

Mọi lời gọi model đều mock — không chạm Ollama, không chạm Postgres.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from graph.budget import DEFAULT_BUDGET_S, MAX_LLM_CALLS_PER_QUERY
from graph.multi_agent import build_multi_agent_graph
from graph.state import make_initial_state
from perception.connection_profile import ALL_TABLES, build_profile
from tests.fixtures.mini_schema import MINI_TABLES
from tests.integration import DF_REVENUE_BY_REGION, SCHEMA_CONTEXT, setup_simple_mocks

PROFILE = build_profile(
    dsn="postgresql://u:p@h:5432/d",
    tables=MINI_TABLES,
    grants={"test_user": frozenset({ALL_TABLES})},
)

PLAN_JSON = {
    "plan_summary": "s",
    "steps": [
        {"step": 1, "agent": "sql", "task": "t1", "depends_on": [], "skill_type": "text-to-sql"},
        {"step": 2, "agent": "insight", "task": "t2", "depends_on": ["sql"],
         "skill_type": "insight-generation"},
    ],
}


def _initial(budget_s: float = DEFAULT_BUDGET_S, llm_calls_used: int = 0):
    s = make_initial_state("q", SCHEMA_CONTEXT, budget_s=budget_s)
    s["shared_metadata"] = {"profile": PROFILE, "user": "test_user"}
    s["llm_calls_used"] = llm_calls_used
    return s


def _supervisor_mock() -> MagicMock:
    m = MagicMock()
    m.invoke_json.return_value = PLAN_JSON
    m.calls_made = 1
    return m


def test_every_run_ends_at_finalize():
    """Không còn đường nào ra khỏi graph mà bỏ qua finalize — kể cả khi
    kế hoạch chạy trót lọt (trước Task 9, insight đi thẳng ra END)."""
    mocks = setup_simple_mocks(DF_REVENUE_BY_REGION)

    with patch("graph.agents.supervisor.ModelClient", return_value=_supervisor_mock()), \
         patch("graph.agents.sql_agent.ModelClient", return_value=mocks["sql"]), \
         patch("graph.agents.sql_agent.execute_sql", mocks["execute_sql"]), \
         patch("graph.agents.sql_agent.explain_query_plan", mocks["explain_query_plan"]), \
         patch("graph.agents.insight_agent.ModelClient", return_value=mocks["insight"]):
        out = build_multi_agent_graph().invoke(_initial())

    assert out["current_agent"] == "finalize"
    assert out["status"] == "success"


def test_a_call_ceiling_state_never_reaches_the_sql_node():
    """Trần số lời gọi model (Task 6) là cơ chế chặn có thật hôm nay: đã
    chạm trần trước khi supervisor định tuyến → sql không được phép chạy."""
    with patch("graph.agents.supervisor.ModelClient", return_value=_supervisor_mock()), \
         patch("graph.agents.sql_agent.ModelClient") as sql:
        out = build_multi_agent_graph().invoke(
            _initial(llm_calls_used=MAX_LLM_CALLS_PER_QUERY)
        )
        sql.return_value.invoke.assert_not_called()

    assert out["current_agent"] == "finalize"


def test_a_call_ceiling_state_yields_a_reason_not_a_silent_failure():
    """Bị chặn ở trần lời gọi thì finalize phải nói ra vì sao, không phải
    im lặng."""
    with patch("graph.agents.supervisor.ModelClient", return_value=_supervisor_mock()), \
         patch("graph.agents.sql_agent.ModelClient"):
        out = build_multi_agent_graph().invoke(
            _initial(llm_calls_used=MAX_LLM_CALLS_PER_QUERY)
        )

    assert out["degradation_reason"], "phải nói ra vì sao bị cắt"
    assert out["status"] in {"partial", "failed"}


# ── deadline thời gian, ở cấp graph ─────────────────────────────────────────

def test_a_zero_budget_never_reaches_the_sql_node():
    """`budget_s=0` — hết ngân sách THỜI GIAN, không phải trần lời gọi.

    Đây là điều docstring cũ của file này nói là "chưa tồn tại". Task 10
    làm nó tồn tại: `node_may_run` chặn ngay trong node `sql`, trước khi
    một lời gọi model nào được khởi động."""
    with patch("graph.agents.supervisor.ModelClient", return_value=_supervisor_mock()), \
         patch("graph.agents.sql_agent.ModelClient") as sql, \
         patch("graph.agents.insight_agent.ModelClient") as insight:
        out = build_multi_agent_graph().invoke(_initial(budget_s=0))
        sql.return_value.invoke.assert_not_called()
        insight.return_value.invoke_json.assert_not_called()

    assert out["current_agent"] == "finalize"
    assert out["status"] in {"partial", "failed"}
    assert out["status"] != "success"
    assert out["degradation_reason"], "hết ngân sách mà không nói gì là nuốt lỗi"


def test_sql_succeeds_then_the_budget_runs_out_before_insight():
    """Hồi quy CẤP GRAPH cho lỗi C1.

    Kịch bản: SQL chạy xong và trả dữ liệu thật; ngân sách cạn TRONG lúc
    nó chạy; `insight` bị `node_may_run` cắt. Node bị cắt tự thêm mình vào
    `completed_agents`, nên `_skipped_agents()` của finalize trả về [] —
    và bản trước vì thế gọi lượt chạy này là "success" rồi xoá sạch
    `degradation_reason`. Người dùng nhận một bảng trần: không insight,
    không banner, không một chữ nào nói có thứ gì đã bị cắt.

    Thời gian trôi thật ở đây, không giả lập: `node_may_run` đọc
    `time.time()` bên trong node, và LangGraph chỉ mang các khoá đã khai
    trong `MultiAgentState` nên không tiêm được đồng hồ giả qua state.
    `MODEL_CALL_ESTIMATE_S` bị vá xuống 2s để cái giá phải trả là ~1,5s
    ngủ, không phải ~15s."""
    mocks = setup_simple_mocks(DF_REVENUE_BY_REGION)

    def slow_execute_sql(*args, **kwargs):
        # Ngân sách cạn TRONG lúc SQL chạy — không phải trước khi nó bắt đầu.
        time.sleep(1.5)
        return DF_REVENUE_BY_REGION

    with patch("graph.budget.MODEL_CALL_ESTIMATE_S", 2.0), \
         patch("graph.agents.supervisor.ModelClient", return_value=_supervisor_mock()), \
         patch("graph.agents.sql_agent.ModelClient", return_value=mocks["sql"]), \
         patch("graph.agents.sql_agent.execute_sql", slow_execute_sql), \
         patch("graph.agents.sql_agent.explain_query_plan", mocks["explain_query_plan"]), \
         patch("graph.agents.insight_agent.ModelClient", return_value=mocks["insight"]) as insight:
        out = build_multi_agent_graph().invoke(_initial(budget_s=3.0))
        insight.return_value.invoke_json.assert_not_called()

    assert out["sql_result"], "SQL phải chạy xong — đây không phải ca chặn từ đầu"
    assert out["insight"] is None, "insight phải bị cắt"
    assert out["status"] != "success", "một lượt bị cắt không được gọi là success"
    assert out["status"] == "partial"
    assert out["degradation_reason"], "lý do bị cắt không được xoá đi"
    assert any("insight" in r for r in out["degradation_reason"])
