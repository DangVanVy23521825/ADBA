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
