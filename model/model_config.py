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
}

MODEL_MAX_RETRIES = int(os.getenv("MODEL_MAX_RETRIES", "3"))

AGENT_TEMPERATURES = {
    "supervisor": 0.1,
    "sql": 0.0,
    "python": 0.1,
    "viz": 0.2,
    "insight": 0.2,
    "reflector": 0.1,
}

AGENT_MAX_TOKENS = {
    "supervisor": 800,  
    "sql":       1024,
    "python":    1024,
    "viz":       1024,
    "insight":    512,
    "reflector":  512,
}
