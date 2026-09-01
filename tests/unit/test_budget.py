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
    calls_exhausted,
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


def test_insight_reserve_defaults_to_twelve_seconds():
    assert INSIGHT_RESERVE_S == 12


# ── trần cứng số lời gọi ────────────────────────────────────────────────────

def test_call_ceiling_is_twelve():
    assert MAX_LLM_CALLS_PER_QUERY == 12


def test_calls_not_exhausted_below_the_ceiling():
    assert calls_exhausted(11) is False


def test_calls_exhausted_at_the_ceiling():
    assert calls_exhausted(12) is True


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
