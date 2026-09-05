"""Node finalize — thang suy giảm có kiểm soát (spec 5.3)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from graph.agents.finalize import finalize_node
from graph.state import make_initial_state
from perception.schema_context import SchemaContext

PLAN = [
    {"step": 1, "agent": "sql", "task": "t1", "depends_on": [], "skill_type": "text-to-sql"},
    {"step": 2, "agent": "viz", "task": "t2", "depends_on": ["sql"],
     "skill_type": "visualization"},
    {"step": 3, "agent": "insight", "task": "t3", "depends_on": ["sql"],
     "skill_type": "insight-generation"},
]


def _state(**over):
    """State mặc định: kế hoạch 3 bước, deadline CÒN HẠN.

    Đồng hồ thật, không phải `lambda: 1000.0`. `finalize_node` gọi
    `is_expired(deadline_ts)` với `time.time()` — một deadline giả ở mốc
    1045.0 (năm 1970) luôn là ĐÃ HẾT HẠN với đồng hồ thật, nên mọi state ở
    đây từng mang sẵn một lý do suy giảm mà không test nào định dựng lên.
    Điều đó không lộ ra trước bản sửa C1 chỉ vì nhánh "success" xoá sạch
    `reasons` vô điều kiện — chính là lỗi C1. Test nào cần một deadline đã
    hết hạn thì đặt `deadline_ts` tường minh (xem
    `test_expired_deadline_is_named_in_the_reason`).
    """
    s = make_initial_state("q", SchemaContext(), budget_s=45)
    s["execution_plan"] = PLAN
    s["completed_agents"] = ["supervisor"]
    s["status"] = "running"
    s.update(over)
    return s


# ── success ─────────────────────────────────────────────────────────────────

def test_full_plan_completed_is_success():
    s = _state(
        completed_agents=["supervisor", "sql", "viz", "insight"],
        sql_result={"sql": "SELECT 1", "row_count": 3},
        insight={"summary": "x"},
    )
    assert finalize_node(s)["status"] == "success"


def test_success_has_no_degradation_reason():
    s = _state(
        completed_agents=["supervisor", "sql", "viz", "insight"],
        sql_result={"sql": "SELECT 1", "row_count": 3},
        insight={"summary": "x"},
    )
    assert finalize_node(s)["degradation_reason"] == []


# ── partial ─────────────────────────────────────────────────────────────────

def test_sql_without_insight_is_partial_not_failed():
    """Người dùng nhận bảng dữ liệu. Đó có giá trị hơn một thông báo lỗi."""
    s = _state(completed_agents=["supervisor", "sql"], sql_result={"sql": "SELECT 1", "row_count": 3})
    assert finalize_node(s)["status"] == "partial"


def test_partial_names_every_step_that_was_skipped():
    s = _state(completed_agents=["supervisor", "sql"], sql_result={"sql": "SELECT 1", "row_count": 3})
    reasons = " ".join(finalize_node(s)["degradation_reason"])
    assert "viz" in reasons
    assert "insight" in reasons


def test_expired_deadline_is_named_in_the_reason():
    s = _state(
        completed_agents=["supervisor", "sql"],
        sql_result={"sql": "SELECT 1", "row_count": 3},
        deadline_ts=900.0,
    )
    out = finalize_node(s)
    assert out["status"] == "partial"
    assert any("ngân sách" in r for r in out["degradation_reason"])


def test_call_ceiling_is_named_in_the_reason():
    s = _state(
        completed_agents=["supervisor", "sql"],
        sql_result={"sql": "SELECT 1", "row_count": 3},
        llm_calls_used=12,
    )
    assert any("lời gọi" in r for r in finalize_node(s)["degradation_reason"])


def test_a_budget_cut_step_is_partial_even_though_it_marked_itself_completed():
    """Node bị `node_may_run` gác ra vì hết ngân sách TỰ THÊM mình vào
    `completed_agents` (để router không chọn lại nó mỗi vòng). Vì thế
    `_skipped_agents()` trả về [] cho một lượt chạy bị cắt sạch — và bản
    trước phân loại nó là "success" rồi xoá luôn `degradation_reason`.

    Người dùng khi đó nhận đúng một bảng SQL trần: không insight, không
    biểu đồ, không banner cảnh báo (UI chỉ hiện banner cho partial/failed),
    và không một chữ nào nói rằng có thứ gì đã bị cắt."""
    s = _state(
        completed_agents=["supervisor", "sql", "viz", "insight"],
        sql_result={"sql": "SELECT 1", "row_count": 3},
        degradation_reason=[
            "Bỏ qua 'viz': còn 8.0s, không đủ ngân sách cho một lời gọi model.",
            "Bỏ qua 'insight': còn 3.0s, không đủ ngân sách cho một lời gọi model.",
        ],
    )
    out = finalize_node(s)
    assert out["status"] == "partial", "một lượt bị cắt không được gọi là success"
    assert out["degradation_reason"], "lý do bị cắt không được xoá đi"
    assert any("viz" in r for r in out["degradation_reason"])
    assert any("insight" in r for r in out["degradation_reason"])


def test_a_truncated_sql_result_alone_blocks_the_success_label():
    """`sql_agent_node` ghi lý do "bị cắt ở N dòng" vào `degradation_reason`
    khi `execute_sql` cắt kết quả ở `SQL_MAX_ROWS` (graph/agents/
    sql_agent.py, xem test tương ứng ở tests/unit/test_sql_agent.py).
    finalize_node phải thấy được lý do đó — mọi bước khác chạy xong bình
    thường không được che đi việc kết quả SQL thiếu dữ liệu."""
    s = _state(
        completed_agents=["supervisor", "sql", "viz", "insight"],
        sql_result={"sql": "SELECT * FROM orders", "row_count": 50000,
                    "truncated": True, "row_cap": 50000},
        degradation_reason=[
            "Kết quả SQL bị cắt ở 50000 dòng — câu truy vấn trả về nhiều hơn thế.",
        ],
    )
    out = finalize_node(s)
    assert out["status"] == "partial", "kết quả bị cắt không được gọi là success"
    assert out["degradation_reason"]
    assert any("cắt" in r for r in out["degradation_reason"])


def test_an_expired_deadline_alone_blocks_the_success_label():
    """Kể cả khi mọi bước đều nằm trong completed_agents: deadline đã hết
    là một lý do suy giảm có thật, nên không thể là "success"."""
    s = _state(
        completed_agents=["supervisor", "sql", "viz", "insight"],
        sql_result={"sql": "SELECT 1", "row_count": 3},
        deadline_ts=900.0,
    )
    out = finalize_node(s)
    assert out["status"] == "partial"
    assert any("ngân sách" in r for r in out["degradation_reason"])


# ── failed ──────────────────────────────────────────────────────────────────

def test_no_sql_result_is_failed():
    """Không có gì dùng được thì gọi đúng tên nó là thất bại."""
    assert finalize_node(_state())["status"] == "failed"


def test_failed_keeps_the_last_error_for_the_trace():
    s = _state(last_error={"agent": "sql", "error_type": "sql_execution", "traceback": "boom"})
    out = finalize_node(s)
    assert out["status"] == "failed"
    assert out["last_error"]["traceback"] == "boom"


# ── bất biến ────────────────────────────────────────────────────────────────

def test_finalize_always_appends_a_trace_entry():
    before = len(_state()["action_trace"])
    assert len(finalize_node(_state())["action_trace"]) == before + 1


def test_finalize_never_raises_on_a_bare_empty_state():
    """finalize LUÔN chạy — kể cả khi state chỉ là {} (ghi dở, crash giữa
    chừng trước khi make_initial_state từng chạy). Nó là node cuối duy
    nhất, nên nó ném lỗi là người dùng không nhận được gì."""
    assert finalize_node({})["status"] in {"success", "partial", "failed"}


def test_finalize_never_raises_when_execution_plan_is_none():
    """Khoá có mặt nhưng giá trị None — khác với khoá vắng mặt, và
    `.get(key, default)` không đỡ được trường hợp này."""
    s = _state(execution_plan=None)
    assert finalize_node(s)["status"] in {"success", "partial", "failed"}


def test_finalize_never_raises_when_completed_agents_is_none():
    s = _state(completed_agents=None)
    assert finalize_node(s)["status"] in {"success", "partial", "failed"}


def test_finalize_never_raises_when_degradation_reason_is_none():
    s = _state(degradation_reason=None)
    assert finalize_node(s)["status"] in {"success", "partial", "failed"}


def test_finalize_never_raises_when_last_error_is_not_a_dict():
    s = _state(last_error="boom")
    assert finalize_node(s)["status"] in {"success", "partial", "failed"}


def test_finalize_handles_a_malformed_plan_step_instead_of_crashing():
    """Một bước hỏng (không phải dict) trong execution_plan — ghi dở hoặc
    dữ liệu bị hỏng — không được làm sập finalize."""
    s = _state(
        completed_agents=["supervisor", "sql"],
        sql_result={"sql": "SELECT 1", "row_count": 3},
        execution_plan=[{"step": 1, "agent": "sql", "task": "t1", "depends_on": [],
                          "skill_type": "text-to-sql"}, "not_a_dict", 42],
    )
    out = finalize_node(s)
    assert out["status"] in {"success", "partial", "failed"}


def test_malformed_plan_step_is_not_silently_dropped_from_classification():
    """Một bước hỏng vẫn CÓ MẶT trong kế hoạch — nó không được biến mất
    lặng lẽ khỏi phân loại. Nếu mọi agent thật đều xong nhưng một bước
    hỏng chưa từng được "hoàn thành" (nó không phải agent thật, nên không
    thể nằm trong completed_agents), kết quả phải là "partial", không
    phải "success" giả."""
    s = _state(
        completed_agents=["supervisor", "sql", "viz"],
        sql_result={"sql": "SELECT 1", "row_count": 3},
        execution_plan=[
            {"step": 1, "agent": "sql", "task": "t1", "depends_on": [],
             "skill_type": "text-to-sql"},
            {"step": 2, "agent": "viz", "task": "t2", "depends_on": ["sql"],
             "skill_type": "visualization"},
            "corrupted-step",
        ],
    )
    out = finalize_node(s)
    assert out["status"] == "partial"


def test_finalize_never_raises_when_degradation_reason_has_non_string_elements():
    """degradation_reason là list hợp lệ về kiểu container, nhưng phần tử
    không phải str (ghi dở, dữ liệu hỏng) — nhánh "partial" nối chúng bằng
    " ".join(reasons), việc đó ném lỗi nếu không ép kiểu trước."""
    s = _state(
        completed_agents=["supervisor", "sql"],
        sql_result={"sql": "SELECT 1", "row_count": 3},
        degradation_reason=[1, 2, 3],
    )
    out = finalize_node(s)
    assert out["status"] in {"success", "partial", "failed"}


def test_finalize_never_raises_when_action_trace_is_none():
    """action_trace có mặt nhưng là None — append_trace() (graph/utils.py)
    có cùng lỗ hổng present-but-None như các trường khác; finalize gọi
    append_trace() vô điều kiện nên không được phép sập vì lỗ hổng đó."""
    s = _state(action_trace=None)
    assert finalize_node(s)["status"] in {"success", "partial", "failed"}


def test_finalize_does_not_mutate_its_input():
    import copy

    s = _state()
    before = copy.deepcopy(s)
    finalize_node(s)
    assert s == before
