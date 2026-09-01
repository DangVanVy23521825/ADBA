"""Node cuối duy nhất của graph.

Trước bản này, hết retry → status='failed' → người dùng nhận KHÔNG GÌ CẢ,
sau khi đã chờ. Một bảng dữ liệu kèm dòng "phần insight bị bỏ do hết thời
gian" có giá trị hơn hẳn một thông báo lỗi — và chính điều đó khiến việc
cắt cứng theo deadline trở thành chấp nhận được thay vì tàn nhẫn (spec 5.3).

`finalize` LUÔN chạy: quá hạn, xong kế hoạch, hay lỗi chí mạng. Vì thế nó
không được phép ném lỗi — nó là chỗ cuối cùng còn cơ hội trả về thứ gì đó.
"""

from __future__ import annotations

import logging

from graph.budget import calls_exhausted, is_expired
from graph.state import MultiAgentState
from graph.utils import append_trace

logger = logging.getLogger(__name__)


def _planned_agents(state: MultiAgentState) -> list[str]:
    return [str(step.get("agent", "")) for step in state.get("execution_plan", [])]


def _skipped_agents(state: MultiAgentState) -> list[str]:
    completed = set(state.get("completed_agents", []))
    return [agent for agent in _planned_agents(state) if agent not in completed]


def finalize_node(state: MultiAgentState) -> MultiAgentState:
    """Gom mọi thứ đang có và phân loại kết quả.

    | status    | Điều kiện                              |
    |-----------|----------------------------------------|
    | success   | Kế hoạch hoàn tất                      |
    | partial   | Có kết quả SQL, một số bước bị bỏ      |
    | failed    | Không có output nào dùng được          |

    `degradation_reason` ghi ra để UI hiển thị được — nuốt lỗi im lặng ở
    đây là cách chắc chắn nhất khiến không ai biết hệ thống đang cắt gì.
    """
    skipped = _skipped_agents(state)
    has_sql = bool(state.get("sql_result"))

    reasons: list[str] = list(state.get("degradation_reason", []))
    if is_expired(state.get("deadline_ts", 0.0)):
        reasons.append("Hết ngân sách thời gian trước khi kế hoạch chạy xong.")
    if calls_exhausted(state.get("llm_calls_used", 0)):
        reasons.append(
            f"Chạm trần {state.get('llm_calls_used', 0)} lời gọi model cho một câu hỏi."
        )
    if skipped:
        reasons.append("Bước bị bỏ: " + ", ".join(skipped) + ".")

    if not skipped and has_sql:
        status = "success"
        reasons = []
        observation = "Kế hoạch hoàn tất."
    elif has_sql:
        status = "partial"
        observation = "Kết quả một phần — " + " ".join(reasons)
    else:
        status = "failed"
        last_error = state.get("last_error") or {}
        observation = (
            "Không có output dùng được. "
            f"Lỗi cuối: {last_error.get('error_type', 'không rõ')}."
        )
        if not reasons:
            reasons = [observation]

    logger.info("Finalize: status=%s, skipped=%s", status, skipped)
    trace = append_trace(state, "finalize", "finalize", observation, status)

    return {
        **state,
        "current_agent": "finalize",
        "status": status,
        "degradation_reason": reasons,
        "action_trace": trace,
    }
