"""Ngân sách nhìn từ ngoài graph.

Mọi lời gọi model đều mock — không chạm Ollama, không chạm Postgres.

LƯU Ý PHẠM VI: bản đầu của file này định dùng `budget_s=0` (hết ngân
sách THỜI GIAN) làm đòn bẩy để chặn graph trước khi tới node `sql`. Đọc
lại `route_next_agent()` (graph/agents/supervisor.py) cho thấy điều đó
CHƯA đúng ở Task 9 — hàm đó mới chỉ chặn theo `calls_exhausted` (trần SỐ
LƯỢT gọi model, có từ Task 6), chưa hề đọc `deadline_ts`/`time_left`. Nối
`deadline_ts` vào từng specialist node (`graph.budget.node_may_run` +
`ModelClient(..., deadline_ts=...)`) là phạm vi Task 10
(`graph/agents/{sql,python,viz,insight,reflector}_agent.py`,
`tests/unit/test_budget_gating.py`) — chưa tồn tại khi Task 9 chạy, nên
`budget_s=0` một mình không chặn được gì. Vì vậy các test "bị chặn" dưới
đây dùng trạng thái đã chạm trần lời gọi model
(`llm_calls_used = MAX_LLM_CALLS_PER_QUERY`) — cơ chế chặn DUY NHẤT có
thật ngay bây giờ — thay vì `budget_s=0`, để không khẳng định một hành vi
mà code chưa làm được. `test_every_run_ends_at_finalize` vẫn dùng một
lượt chạy THÀNH CÔNG bình thường, vì đó chính là điều Task 9 nối thêm:
trước đây `insight` đi thẳng ra `END`, giờ phải qua `finalize`.
"""

from __future__ import annotations

import sys
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
