"""Cấp phát theo dự trữ (spec 5.2).

Chia đều sẽ bỏ đói bước cuối, mà bước cuối lại là thứ người dùng đọc.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from graph.budget import INSIGHT_RESERVE_S, MODEL_CALL_ESTIMATE_S, node_may_run
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
    """Còn 20s, cần ~15s, dự trữ 15s → 20 - 15 = 5 < 15. Cắt viz.

    Số dự trữ là 15 chứ không phải 12 kể từ khi `INSIGHT_RESERVE_S` được
    dẫn ra từ `MODEL_CALL_ESTIMATE_S` — một dự trữ nhỏ hơn lời gọi mà nó
    phải chừa chỗ thì không bảo vệ được gì. Phân biệt hành vi mà test này
    kiểm KHÔNG đổi: 20s vẫn là "bị cắt", chỉ là biên giờ ở 30s thay vì
    27s."""
    ok, reason = node_may_run(_state(20), "viz")
    assert ok is False
    assert "viz" in reason


def test_python_is_cut_before_the_reserve_is_touched():
    ok, _ = node_may_run(_state(20), "python")
    assert ok is False


def test_viz_runs_when_there_is_room_beyond_the_reserve():
    """Còn 40s, dự trữ 15s → 25s dùng được, đủ cho một lời gọi ~15s."""
    ok, _ = node_may_run(_state(40), "viz")
    assert ok is True


def test_insight_may_spend_the_reserve_it_is_reserved_for():
    """Dự trữ dành riêng cho insight, nên insight không tự trừ nó của mình.

    16s: quá ít cho viz (16 - 15 = 1 < 15) nhưng đủ cho insight, vì
    insight tính với reserve = 0. Đó chính là điều phần dự trữ mua được."""
    ok, _ = node_may_run(_state(16), "insight")
    assert ok is True
    assert node_may_run(_state(16), "viz")[0] is False


def test_the_reserve_leaves_exactly_enough_room_for_the_insight_call():
    """Bất biến của I4, kiểm ở đúng chỗ nó có hiệu lực.

    Ngay tại biên mà viz vừa bị cắt, phần thời gian được chừa lại phải đủ
    cho một lời gọi model của insight. Bản trước dự trữ 12s cho một lời
    gọi ước lượng 15s, nên "bảo vệ insight" chỉ là cái tên."""
    at_the_edge = INSIGHT_RESERVE_S
    assert node_may_run(_state(at_the_edge), "insight")[0] is True, (
        "insight phải chạy được bằng đúng phần dự trữ của nó"
    )
    assert INSIGHT_RESERVE_S >= MODEL_CALL_ESTIMATE_S


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
    # ...trong khi sql, thứ được lấn vào dự trữ, vẫn chạy với cùng 20s đó.
    assert node_may_run(_state(20), "sql")[0] is True


def test_the_call_ceiling_blocks_a_node_that_still_has_time():
    ok, reason = node_may_run(_state(45, llm_calls_used=12), "sql")
    assert ok is False
    assert "lời gọi" in reason
