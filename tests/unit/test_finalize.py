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
    s = make_initial_state("q", SchemaContext(), budget_s=45, clock=lambda: 1000.0)
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


def test_finalize_skips_a_malformed_plan_step_instead_of_crashing():
    """Một bước hỏng (không phải dict) trong execution_plan — ghi dở hoặc
    dữ liệu bị hỏng — bị bỏ qua thay vì làm sập finalize."""
    s = _state(
        completed_agents=["supervisor", "sql"],
        sql_result={"sql": "SELECT 1", "row_count": 3},
        execution_plan=[{"step": 1, "agent": "sql", "task": "t1", "depends_on": [],
                          "skill_type": "text-to-sql"}, "not_a_dict", 42],
    )
    out = finalize_node(s)
    assert out["status"] in {"success", "partial", "failed"}


def test_finalize_does_not_mutate_its_input():
    import copy

    s = _state()
    before = copy.deepcopy(s)
    finalize_node(s)
    assert s == before
