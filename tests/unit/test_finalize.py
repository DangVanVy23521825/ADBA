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


def test_finalize_never_raises_on_an_empty_state():
    """finalize LUÔN chạy — quá hạn, xong kế hoạch, hay lỗi chí mạng. Nó là
    node cuối duy nhất, nên nó ném lỗi là người dùng không nhận được gì."""
    s = make_initial_state("q", SchemaContext())
    assert finalize_node(s)["status"] in {"success", "partial", "failed"}


def test_finalize_does_not_mutate_its_input():
    import copy

    s = _state()
    before = copy.deepcopy(s)
    finalize_node(s)
    assert s == before
