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

# Phần cuối ngân sách giữ riêng cho bước insight. Chia đều sẽ bỏ đói bước
# cuối, mà bước cuối lại là thứ người dùng thực sự đọc (spec 5.2).
INSIGHT_RESERVE_S = float(os.getenv("ADBA_INSIGHT_RESERVE_S", "12"))

# Trần cứng, độc lập với thời gian. Deadline ép SLO; trần này chặn vòng
# lặp vô hạn khi đồng hồ vì lý do nào đó không cứu được (spec 5.4).
MAX_LLM_CALLS_PER_QUERY = int(os.getenv("ADBA_MAX_LLM_CALLS", "12"))

# Ước lượng thời gian một lời gọi model, dùng để quyết định CÓ khởi động
# lời gọi hay không. Cố ý bi quan: bỏ qua một node còn hơn khởi động một
# lời gọi chắc chắn không kịp về, vì lời gọi đó vẫn tiêu hết phần thời
# gian còn lại rồi mới thất bại.
MODEL_CALL_ESTIMATE_S = float(os.getenv("ADBA_MODEL_CALL_ESTIMATE_S", "15"))


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
