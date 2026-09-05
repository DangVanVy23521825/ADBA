"""Ngân sách thời gian — hàm thuần, clock tiêm vào.

Không test nào ở đây được dùng time.sleep. Nếu phải sleep để test một
deadline thì deadline đó đang đọc đồng hồ ở chỗ không tiêm được, và đó
mới là lỗi cần sửa.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from graph.budget import (
    INSIGHT_RESERVE_S,
    MAX_LLM_CALLS_PER_QUERY,
    MODEL_CALL_ESTIMATE_S,
    calls_exhausted,
    calls_remaining,
    clamped_tool_timeout_s,
    has_room_for,
    is_expired,
    make_deadline,
    new_query_id,
    time_left,
)
from graph.state import make_initial_state
from perception.schema_context import SchemaContext


def fake_clock(*ticks: float):
    """Đồng hồ trả lần lượt các mốc cho trước, giữ nguyên mốc cuối."""
    seq = list(ticks)

    def clock() -> float:
        return seq.pop(0) if len(seq) > 1 else seq[0]

    return clock


# ── deadline ────────────────────────────────────────────────────────────────

def test_deadline_is_now_plus_budget():
    assert make_deadline(45, clock=fake_clock(1000.0)) == 1045.0


def test_time_left_shrinks_as_the_clock_advances():
    assert time_left(1045.0, clock=fake_clock(1030.0)) == 15.0


def test_time_left_goes_negative_past_the_deadline():
    assert time_left(1045.0, clock=fake_clock(1060.0)) == -15.0


def test_not_expired_before_the_deadline():
    assert is_expired(1045.0, clock=fake_clock(1044.9)) is False


def test_expired_exactly_at_the_deadline():
    """Đúng mốc là hết. Biên nghiêng về phía an toàn."""
    assert is_expired(1045.0, clock=fake_clock(1045.0)) is True


# ── dự trữ ──────────────────────────────────────────────────────────────────

def test_room_when_plenty_of_time_left():
    assert has_room_for(1100.0, needed_s=15, clock=fake_clock(1000.0)) is True


def test_no_room_when_the_work_would_overrun():
    assert has_room_for(1010.0, needed_s=15, clock=fake_clock(1000.0)) is False


def test_reserve_is_subtracted_before_the_decision():
    """40s còn lại, cần 15s, dự trữ 30s → không đủ chỗ.

    Đây là quy tắc bảo vệ bước insight: node python/viz bị cắt trước, để
    người dùng luôn nhận được nhận xét nếu SQL đã chạy được (spec 5.2).
    """
    assert has_room_for(1040.0, needed_s=15, reserve_s=30, clock=fake_clock(1000.0)) is False


def test_without_a_reserve_the_same_call_has_room():
    assert has_room_for(1040.0, needed_s=15, clock=fake_clock(1000.0)) is True


def test_the_insight_reserve_is_never_smaller_than_what_it_protects():
    """Đây là BẤT BIẾN, không phải một con số.

    Phần dự trữ tồn tại để bước insight còn chỗ chạy một lời gọi model.
    Một dự trữ nhỏ hơn `MODEL_CALL_ESTIMATE_S` không bảo vệ được gì: nó
    vẫn cắt python/viz đúng lúc, nhưng chỗ nó chừa lại không đủ cho chính
    insight, nên lời hứa "SQL xong thì luôn có nhận xét" sụp về mặt cấu
    trúc. Bản trước viết cứng 12 trong khi ước lượng là 15 — đúng cái ca
    đó. Khẳng định quan hệ, đừng khẳng định con số, để việc chỉnh ước
    lượng qua env không lặng lẽ mở lại lỗ hổng."""
    assert INSIGHT_RESERVE_S >= MODEL_CALL_ESTIMATE_S


def test_the_insight_reserve_does_not_shrink_below_its_historical_floor():
    """12s là sàn cũ. Dẫn ra từ ước lượng chỉ được phép làm nó TO ra."""
    assert INSIGHT_RESERVE_S >= 12


# ── timeout tool sau model ──────────────────────────────────────────────────
#
# `node_may_run` gác cửa TRƯỚC lời gọi model. Nó không thấy phần chi tiêu
# SAU đó — SQL/sandbox vẫn dùng timeout cấu hình sẵn, độc lập với deadline,
# nên một lượt qua được cửa gác vẫn có thể vượt ngân sách toàn cục ở tool.
# `clamped_tool_timeout_s` là điểm kiểm giữa, đóng đúng lỗ đó.

def test_no_deadline_means_no_clamp():
    """Không deadline (test cũ, lời gọi ngoài graph) → trả nguyên cấu hình."""
    assert clamped_tool_timeout_s(None, 25.0, clock=fake_clock(1000.0)) == 25.0


def test_clamps_down_to_remaining_time_when_it_is_the_smaller_number():
    """Còn 8s, cấu hình 25s → chỉ được dùng 8s, không phải cấu hình."""
    assert clamped_tool_timeout_s(1008.0, 25.0, clock=fake_clock(1000.0)) == 8.0


def test_configured_ceiling_wins_when_plenty_of_time_remains():
    """Còn 400s, cấu hình 25s → vẫn chỉ 25s — không giãn quá cấu hình."""
    assert clamped_tool_timeout_s(1400.0, 25.0, clock=fake_clock(1000.0)) == 25.0


def test_floors_at_zero_past_the_deadline_never_goes_negative():
    """Deadline đã qua 5s — bên gọi cần một mốc so sánh được (`<= 0`), không
    phải một số âm mà mỗi caller phải tự kẹp lại."""
    assert clamped_tool_timeout_s(995.0, 25.0, clock=fake_clock(1000.0)) == 0.0


def test_zero_remaining_exactly_at_the_deadline_floors_at_zero():
    assert clamped_tool_timeout_s(1000.0, 25.0, clock=fake_clock(1000.0)) == 0.0


# ── trần cứng số lời gọi ────────────────────────────────────────────────────

def test_call_ceiling_is_twelve():
    assert MAX_LLM_CALLS_PER_QUERY == 12


def test_calls_not_exhausted_below_the_ceiling():
    assert calls_exhausted(11) is False


def test_calls_exhausted_at_the_ceiling():
    assert calls_exhausted(12) is True


def test_calls_remaining_never_goes_negative():
    assert calls_remaining(11) == 1
    assert calls_remaining(12) == 0
    assert calls_remaining(99) == 0


# ── query_id ────────────────────────────────────────────────────────────────

def test_query_ids_are_unique():
    assert len({new_query_id() for _ in range(100)}) == 100


# ── state ───────────────────────────────────────────────────────────────────

def test_initial_state_carries_a_deadline():
    s = make_initial_state("q", SchemaContext(), budget_s=45, clock=fake_clock(1000.0))
    assert s["deadline_ts"] == 1045.0


def test_initial_state_carries_a_query_id():
    s = make_initial_state("q", SchemaContext())
    assert s["query_id"]


def test_initial_state_starts_with_zero_llm_calls():
    assert make_initial_state("q", SchemaContext())["llm_calls_used"] == 0


def test_initial_state_starts_with_no_degradation():
    assert make_initial_state("q", SchemaContext())["degradation_reason"] == []


def test_old_two_argument_call_still_works():
    """Hàng chục test hiện có gọi make_initial_state(query, ctx). Đừng làm hỏng."""
    s = make_initial_state("q", SchemaContext())
    assert s["query"] == "q"
    assert s["deadline_ts"] > 0


# ── trần cứng thay đếm retry ─────────────────────────────────────────────────

def test_reflector_cap_is_one():
    """Spec 5.4: reflector sinh corrected_context. Một chẩn đoán + một lần
    thử lại không sửa được thì bảy lần nữa cũng vậy — model kẹt ở cùng lớp
    lỗi, và reflector_json_rate = 100% cho thấy nó không hỏng ở khâu sinh
    output."""
    from graph.agents.supervisor import MAX_REFLECTOR_PASSES_PER_AGENT

    assert MAX_REFLECTOR_PASSES_PER_AGENT == 1


def test_sql_node_retry_is_two():
    """Một lần sinh + một lần sửa theo error context."""
    from graph.agents.sql_agent import MAX_RETRIES

    assert MAX_RETRIES == 2


def test_router_goes_to_finalize_when_the_call_ceiling_is_hit():
    from graph.agents.supervisor import route_next_agent

    s = _state_for_ceiling()
    s["llm_calls_used"] = 12
    assert route_next_agent(s) == "finalize"


def test_router_still_routes_below_the_call_ceiling():
    from graph.agents.supervisor import route_next_agent

    s = _state_for_ceiling()
    s["llm_calls_used"] = 11
    assert route_next_agent(s) == "sql"


def _state_for_ceiling():
    from graph.state import make_initial_state
    from perception.schema_context import SchemaContext

    s = make_initial_state("q", SchemaContext())
    s["execution_plan"] = [
        {"step": 1, "agent": "sql", "task": "t", "depends_on": [], "skill_type": "text-to-sql"},
    ]
    s["completed_agents"] = ["supervisor"]
    s["status"] = "running"
    return s
