"""
Model package exports.
"""

from model.model_client import ModelClient, safe_parse_json

__all__ = ["ModelClient", "safe_parse_json"]
# import httpx
# import ollama
# from model.model_config import OLLAMA_BASE_URL

# ollama_client = ollama.Client(
#     host=OLLAMA_BASE_URL,
#     timeout=httpx.Timeout( timeout_s=10.0, connect=10.0)
# )

# def get_ollama_client_with_default_timeout():
#     return ollama.Client(
#         host=OLLAMA_BASE_URL,
#         timeout=httpx.Timeout( timeout_s=10.0, connect=10.0)
#     )  

# def get_ollama_client_with_timeout(timeout_s: float = 10.0):
#     return ollama.Client(host=OLLAMA_BASE_URL, timeout=httpx.Timeout(timeout_s, connect=timeout_s))