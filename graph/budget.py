"""Ngân sách thời gian cho một lượt truy vấn.

Toàn bộ module là hàm thuần với đồng hồ tiêm vào. Lý do không phải là
thẩm mỹ: deadline chỉ kiểm được bằng test nhanh khi đồng hồ tiêm được,
và một bộ test phải `sleep` để kiểm ngân sách thì sẽ bị người ta tắt đi.

Module này KHÔNG import gì từ `graph/` — nó nằm dưới đáy cây phụ thuộc
để mọi node và cả `model/` dùng được mà không tạo vòng import.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Callable

Clock = Callable[[], float]

# 45s là SLO trong spec mục 1, đo ở cấu hình vLLM/PEFT (12,4s mỗi lời gọi
# model). Trên Ollama q5 — 75,8s mỗi lời gọi — con số này không đạt được,
# và đó là lý do nó có env override thay vì bị viết cứng: dev đặt
# ADBA_QUERY_BUDGET_S=400, production giữ 45.
DEFAULT_BUDGET_S = float(os.getenv("ADBA_QUERY_BUDGET_S", "45"))

# Trần cứng, độc lập với thời gian. Deadline ép SLO; trần này chặn vòng
# lặp vô hạn khi đồng hồ vì lý do nào đó không cứu được (spec 5.4).
MAX_LLM_CALLS_PER_QUERY = int(os.getenv("ADBA_MAX_LLM_CALLS", "12"))

# Ước lượng thời gian một lời gọi model, dùng để quyết định CÓ khởi động
# lời gọi hay không. Cố ý bi quan: bỏ qua một node còn hơn khởi động một
# lời gọi chắc chắn không kịp về, vì lời gọi đó vẫn tiêu hết phần thời
# gian còn lại rồi mới thất bại.
MODEL_CALL_ESTIMATE_S = float(os.getenv("ADBA_MODEL_CALL_ESTIMATE_S", "15"))

# Phần cuối ngân sách giữ riêng cho bước insight. Chia đều sẽ bỏ đói bước
# cuối, mà bước cuối lại là thứ người dùng thực sự đọc (spec 5.2).
#
# DẪN RA TỪ `MODEL_CALL_ESTIMATE_S`, không phải một số độc lập. Bản trước
# viết cứng 12 trong khi ước lượng một lời gọi model là 15 — tức phần "dự
# trữ cho insight" nhỏ hơn chính thứ nó phải chừa chỗ. Hệ quả: node
# python/viz bị cắt đúng lúc còn 12..15s, nhưng insight — thứ vừa được
# "bảo vệ" — cũng không chạy nổi trong 12s đó, nên lời hứa "SQL xong thì
# luôn có nhận xét" không đứng vững về mặt cấu trúc. `max` giữ cho nó
# không bao giờ nhỏ hơn thứ nó bảo vệ, kể cả khi ai đó chỉnh ước lượng
# lên qua biến môi trường.
INSIGHT_RESERVE_S = max(
    float(os.getenv("ADBA_INSIGHT_RESERVE_S", "12")),
    MODEL_CALL_ESTIMATE_S,
)


def new_query_id() -> str:
    """Khoá vòng đời của một lượt truy vấn — dùng cho trace và log."""
    return uuid.uuid4().hex[:12]


def make_deadline(budget_s: float = DEFAULT_BUDGET_S, clock: Clock = time.time) -> float:
    return clock() + budget_s


def time_left(deadline_ts: float, clock: Clock = time.time) -> float:
    return deadline_ts - clock()


def is_expired(deadline_ts: float, clock: Clock = time.time) -> bool:
    """Đúng mốc deadline đã tính là hết — biên nghiêng về phía an toàn."""
    return time_left(deadline_ts, clock) <= 0


def has_room_for(
    deadline_ts: float,
    needed_s: float,
    *,
    reserve_s: float = 0.0,
    clock: Clock = time.time,
) -> bool:
    """Còn đủ chỗ cho một việc tốn `needed_s` mà không lấn vào phần dự trữ?"""
    return time_left(deadline_ts, clock) - reserve_s >= needed_s


def calls_exhausted(llm_calls_used: int) -> bool:
    return llm_calls_used >= MAX_LLM_CALLS_PER_QUERY


def calls_remaining(llm_calls_used: int) -> int:
    """Số lời gọi model còn được khởi động trong lượt truy vấn này."""
    return max(0, MAX_LLM_CALLS_PER_QUERY - llm_calls_used)


def clamped_tool_timeout_s(
    deadline_ts: float | None,
    configured_s: float,
    *,
    clock: Clock = time.time,
) -> float:
    """Timeout thật cho một tool (SQL/sandbox) SAU KHI lời gọi model đã trả về.

    `node_may_run` gác CỬA TRƯỚC lời gọi model — nó không thấy phần chi
    tiêu SAU đó. Một node được cho qua cửa vì còn đủ chỗ cho MỘT lời gọi
    model (~15s ước lượng) vẫn có thể tốn 25s ở sandbox hay 10s ở
    statement_timeout ngay sau đó, và cả hai con số đó độc lập với deadline
    — cộng lại, một lượt chạy có thể vượt ngân sách toàn cục dù mọi cửa
    gác từng nói "còn giờ".

    Hàm này là điểm kiểm giữa — sau model, trước tool — trả về nhỏ hơn
    giữa cấu hình sẵn có và thời gian thật sự còn lại. `0.0` (không âm)
    khi hết giờ, để bên gọi coi đó là "không còn chỗ" mà không phải tự
    kiểm dấu âm. Không deadline (test cũ, lời gọi ngoài graph) → trả
    nguyên cấu hình, không clamp gì — cùng quy ước với
    `ModelClient.effective_timeout_s()`.
    """
    if deadline_ts is None:
        return configured_s
    return max(0.0, min(configured_s, time_left(deadline_ts, clock)))


# Node nào được lấn vào phần dự trữ dành cho insight.
#
#   sql       — bắt buộc. Không có kết quả SQL thì không có gì để nhận xét,
#               nên nó được lấn.
#   insight   — chính là chủ của phần dự trữ, nên không tự trừ nó của mình.
#   python/viz/reflector — bị cắt trước tiên. Người dùng mất biểu đồ nhưng
#               vẫn nhận được bảng và nhận xét (spec 5.2).
_MAY_USE_RESERVE = frozenset({"sql", "insight"})


def node_may_run(state, agent: str) -> tuple[bool, str]:
    """Node `agent` còn được phép chạy không, và nếu không thì vì sao.

    `state` là MultiAgentState nhưng không annotate ở đây: module này nằm
    dưới đáy cây phụ thuộc và import graph.state sẽ tạo vòng.

    Đọc `state["_clock"]` nếu có — chỉ để test tiêm đồng hồ; production
    không đặt khoá đó và rơi về time.time.
    """
    clock: Clock = state.get("_clock", time.time)
    deadline_ts = state.get("deadline_ts", 0.0)

    if calls_exhausted(state.get("llm_calls_used", 0)):
        return False, (
            f"Bỏ qua '{agent}': đã chạm trần "
            f"{state.get('llm_calls_used', 0)} lời gọi model cho câu hỏi này."
        )

    reserve = 0.0 if agent in _MAY_USE_RESERVE else INSIGHT_RESERVE_S
    if not has_room_for(deadline_ts, MODEL_CALL_ESTIMATE_S, reserve_s=reserve, clock=clock):
        return False, (
            f"Bỏ qua '{agent}': còn {time_left(deadline_ts, clock):.1f}s, "
            f"không đủ ngân sách cho một lời gọi model"
            + (f" mà không lấn vào {reserve:.0f}s dự trữ cho insight." if reserve else ".")
        )
    return True, ""
