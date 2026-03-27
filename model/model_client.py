"""
ModelClient wrapper with local-first Ollama and OpenAI fallback.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import ollama

from model.model_config import (
    AGENT_MAX_TOKENS,
    AGENT_TEMPERATURES,
    AGENT_TIMEOUT_S,
    MODEL_MAX_RETRIES,
    OLLAMA_BASE_URL,
    OLLAMA_NUM_CTX,
    PRIMARY_MODEL,
)


def _strip_markdown_fences(text: str) -> str:
    return re.sub(r"```(?:json|python|sql)?\s*|\s*```", "", text).strip()


def safe_parse_json(text: str) -> dict[str, Any]:
    """
    Strip markdown fences and parse JSON object.
    Raises ValueError when no valid JSON object is found.
    """
    clean = _strip_markdown_fences(text)

    try:
        data = json.loads(clean)
        if not isinstance(data, dict):
            raise ValueError("JSON parsed but root is not an object.")
        return data
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", clean)
    if not match:
        raise ValueError("No JSON object found in model output.")

    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("Extracted JSON is not an object.")
    return data


class ModelClient:
    """
    Local-first model wrapper:
      - primary path: Ollama
      - fallback path: OpenAI (optional, per agent)
    """

    def __init__(
        self,
        agent_type: str,
        model_name: str | None = None,
        timeout_s: float | None = None,
        max_retries: int = MODEL_MAX_RETRIES,
        enable_openai_fallback: bool | None = None,
        openai_model: str | None = None,
    ) -> None:
        self.agent_type = agent_type
        self.model_name = model_name or PRIMARY_MODEL
        self.timeout_s = (
            float(AGENT_TIMEOUT_S.get(agent_type, AGENT_TIMEOUT_S["sql"]))
            if timeout_s is None
            else timeout_s
        )
        self.max_retries = max_retries
        self.temperature = AGENT_TEMPERATURES.get(agent_type, 0.1)
        self.max_tokens = AGENT_MAX_TOKENS.get(agent_type, 1024)

        self.ollama_client = ollama.Client(host=OLLAMA_BASE_URL)

        # Default: only fallback for supervisor + sql in hybrid mode.
        if enable_openai_fallback is None:
            env_flag = os.getenv("ENABLE_OPENAI_FALLBACK", "1").strip().lower()
            self.enable_openai_fallback = env_flag not in {"0", "false", "no"}
        else:
            self.enable_openai_fallback = enable_openai_fallback

        self.openai_model = openai_model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.fallback_agents = {"supervisor", "sql"}

    def _chat_once_ollama(self, messages: list[dict[str, str]]) -> str:
        response = self.ollama_client.chat(
            model=self.model_name,
            messages=messages,
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "num_ctx": OLLAMA_NUM_CTX,
            },
        )
        return response["message"]["content"].strip()

    def _invoke_ollama(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            start = time.time()
            try:
                out = self._chat_once_ollama(messages)
                elapsed = time.time() - start
                if elapsed > self.timeout_s:
                    raise TimeoutError(
                        f"Ollama response exceeded timeout: {elapsed:.1f}s > {self.timeout_s:.1f}s"
                    )
                if not out:
                    raise ValueError("Ollama returned empty content.")
                return _strip_markdown_fences(out)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(2 ** (attempt - 1))

        raise RuntimeError(
            f"Ollama invocation failed after {self.max_retries} attempts for agent "
            f"'{self.agent_type}'. Last error: {last_error}"
        )

    def _invoke_openai(self, system_prompt: str, user_prompt: str) -> str:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing; cannot use OpenAI fallback.")

        try:
            from openai import OpenAI
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "OpenAI package is not available. Install with: pip install openai"
            ) from exc

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        content = response.choices[0].message.content or ""
        if not content.strip():
            raise RuntimeError("OpenAI returned empty content.")
        return _strip_markdown_fences(content)

    def invoke(self, system_prompt: str, user_prompt: str) -> str:
        """
        Invoke model text with local-first strategy.
        """
        try:
            return self._invoke_ollama(system_prompt=system_prompt, user_prompt=user_prompt)
        except Exception as ollama_error:  # noqa: BLE001
            can_fallback = (
                self.enable_openai_fallback and self.agent_type in self.fallback_agents
            )
            if not can_fallback:
                raise RuntimeError(
                    f"Ollama failed for agent '{self.agent_type}' and fallback is disabled. "
                    f"Error: {ollama_error}"
                ) from ollama_error

            try:
                return self._invoke_openai(system_prompt=system_prompt, user_prompt=user_prompt)
            except Exception as openai_error:  # noqa: BLE001
                raise RuntimeError(
                    f"Both Ollama and OpenAI fallback failed for agent '{self.agent_type}'. "
                    f"Ollama error: {ollama_error}. OpenAI error: {openai_error}"
                ) from openai_error

    def invoke_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """
        Invoke model and parse JSON object safely.
        """
        raw = self.invoke(system_prompt=system_prompt, user_prompt=user_prompt)
        try:
            return safe_parse_json(raw)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f"Failed to parse JSON from model output: {exc}. Raw output: {raw[:300]}"
            ) from exc

