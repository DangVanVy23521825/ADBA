"""
Model configuration constants for ADBA.
"""

from __future__ import annotations

import os

# Local Ollama models
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "qwen2.5-coder:7b-instruct-q5_K_M")
BACKUP_MODEL = os.getenv("BACKUP_MODEL", "llama3.1:8b-instruct-q4_K_M")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))

# Default retries / timeout for model invocations
AGENT_TIMEOUT_S = {
    "supervisor": 300,   # complex JSON plan
    "sql":        200,
    "python":     200,
    "viz":        150,
    "insight":    200,
    "reflector":  120,
    "annotate":   200,   # perception/annotate.py -- see AGENT_MAX_TOKENS below
}

MODEL_MAX_RETRIES = int(os.getenv("MODEL_MAX_RETRIES", "3"))

# ── Ranh giới egress ────────────────────────────────────────────────────────

DEPLOYMENT_ENV_VAR = "ADBA_DEPLOYMENT"
ONPREM = "onprem"
HYBRID = "hybrid"
_DEPLOYMENT_MODES = {ONPREM, HYBRID}


def deployment_mode() -> str:
    """Chế độ triển khai, đọc lại mỗi lần gọi (không cache lúc import).

    Mặc định `onprem`: KHÔNG có gì rời khỏi mạng khách. Đây là mặc định vì
    lời hứa on-prem là thứ khách chọn sản phẩm này để mua, và một mặc định
    an toàn nghĩa là quên cấu hình thì hỏng về phía giữ dữ liệu lại, không
    phải phía gửi nó đi.

    `hybrid` cho phép fallback ra model ngoài. Đó là một quyết định thương
    mại, ghi trong hợp đồng với khách, nên nó phải được BẬT TƯỜNG MINH —
    không bao giờ là thứ mà một biến môi trường bị quên lại vô tình bật.

    Giá trị lạ → `onprem`. Cùng nguyên tắc với `read_profile` xử lý một
    `schema_mode` hỏng: chuỗi không đọc được phải rơi về phía an toàn, chứ
    không phải phía im lặng gửi dữ liệu đi.
    """
    raw = os.getenv(DEPLOYMENT_ENV_VAR, ONPREM).strip().lower()
    return raw if raw in _DEPLOYMENT_MODES else ONPREM


def egress_allowed() -> bool:
    """Tiến trình này có được phép gọi model bên ngoài mạng khách không?"""
    return deployment_mode() == HYBRID

AGENT_TEMPERATURES = {
    "supervisor": 0.1,
    "sql": 0.0,
    "python": 0.1,
    "viz": 0.2,
    "insight": 0.2,
    "reflector": 0.1,
    "annotate": 0.2,
}

AGENT_MAX_TOKENS = {
    "supervisor": 800,
    "sql":       1024,
    "python":    1024,
    "viz":       1024,
    "insight":    512,
    "reflector":  512,
    # Own budget for perception/annotate.py, deliberately NOT borrowed from
    # "insight". Borrowing was exactly how wide tables (Match: 115 cols,
    # cards: 74, Laboratory: 44, frpm: 29) silently lost their annotation:
    # the model's reply gets cut off mid-JSON before it reaches the closing
    # brace, and `_parse` then extracts some smaller-but-complete object
    # buried in the fragment -- which either lacks a "table" key or has one
    # with empty text, so the table is counted as a failure.
    #
    # 512 is kept at the same size as "insight" on purpose: the real fix for
    # wide tables is chunking their columns across several calls (see
    # perception.annotate._column_batches), not raising this ceiling --
    # raising it to ~3000 to fit one 44-column table in a single call was
    # tried and killed after 5 minutes on a 7B local model. Keeping the
    # per-call budget small keeps every individual call fast; chunking is
    # what makes wide tables fit.
    "annotate":   512,
}
