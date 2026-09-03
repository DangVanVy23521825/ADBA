"""Cấp phát theo dự trữ (spec 5.2).

Chia đều sẽ bỏ đói bước cuối, mà bước cuối lại là thứ người dùng đọc.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from graph.budget import node_may_run
from graph.state import make_initial_state
from perception.schema_context import SchemaContext


def _state(seconds_left: float, **over):
    s = make_initial_state("q", SchemaContext(), budget_s=seconds_left, clock=lambda: 1000.0)
    s["_clock"] = lambda: 1000.0
    s.update(over)
    return s


def test_sql_may_run_with_plenty_of_time():
    ok, _ = node_may_run(_state(45), "sql")
    assert ok is True


def test_sql_may_run_even_inside_the_insight_reserve():
    """sql là bắt buộc: không có nó thì không có gì để nhận xét, nên nó
    được phép lấn vào phần dự trữ."""
    ok, _ = node_may_run(_state(16), "sql")
    assert ok is True


def test_viz_is_cut_before_the_reserve_is_touched():
    """Còn 20s, cần ~15s, dự trữ 12s → 20 - 12 = 8 < 15. Cắt viz."""
    ok, reason = node_may_run(_state(20), "viz")
    assert ok is False
    assert "viz" in reason


def test_python_is_cut_before_the_reserve_is_touched():
    ok, _ = node_may_run(_state(20), "python")
    assert ok is False


def test_viz_runs_when_there_is_room_beyond_the_reserve():
    """Còn 40s, dự trữ 12s → 28s dùng được, đủ cho một lời gọi ~15s."""
    ok, _ = node_may_run(_state(40), "viz")
    assert ok is True


def test_insight_may_spend_the_reserve_it_is_reserved_for():
    """Dự trữ dành riêng cho insight, nên insight không tự trừ nó của mình."""
    ok, _ = node_may_run(_state(16), "insight")
    assert ok is True


def test_insight_is_cut_when_even_the_reserve_is_gone():
    ok, reason = node_may_run(_state(5), "insight")
    assert ok is False
    assert "ngân sách" in reason


def test_nothing_may_run_past_the_deadline():
    for agent in ("sql", "python", "viz", "insight", "reflector"):
        ok, _ = node_may_run(_state(0), agent)
        assert ok is False, f"{agent} chạy sau deadline"


def test_reflector_is_cut_before_specialists_are():
    """Reflector là chẩn đoán, không phải kết quả. Nó đi trước trên đoạn ván."""
    ok, _ = node_may_run(_state(20), "reflector")
    assert ok is False


def test_the_call_ceiling_blocks_a_node_that_still_has_time():
    ok, reason = node_may_run(_state(45, llm_calls_used=12), "sql")
    assert ok is False
    assert "lời gọi" in reason
