"""ModelClient tôn trọng deadline (spec 5.1, điểm kiểm 2)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model.model_client import BudgetExceededError, ModelClient


def clock_at(t: float):
    return lambda: t


def test_no_deadline_means_no_budget_check():
    """Mặc định giữ nguyên hành vi cũ — hàng chục test hiện có dựa vào đó."""
    with patch("model.model_client.ollama.Client") as fake:
        fake.return_value.chat.return_value = {"message": {"content": "ok"}}
        assert ModelClient(agent_type="sql").invoke("s", "u") == "ok"


def test_call_is_refused_when_not_enough_time_remains():
    with patch("model.model_client.ollama.Client") as fake:
        client = ModelClient(agent_type="sql", deadline_ts=1005.0, clock=clock_at(1000.0))
        with pytest.raises(BudgetExceededError, match="ngân sách"):
            client.invoke("s", "u")
        fake.return_value.chat.assert_not_called()


def test_call_proceeds_when_there_is_room():
    with patch("model.model_client.ollama.Client") as fake:
        fake.return_value.chat.return_value = {"message": {"content": "ok"}}
        client = ModelClient(agent_type="sql", deadline_ts=1100.0, clock=clock_at(1000.0))
        assert client.invoke("s", "u") == "ok"


def test_call_is_refused_after_the_deadline_has_passed():
    with patch("model.model_client.ollama.Client") as fake:
        client = ModelClient(agent_type="sql", deadline_ts=900.0, clock=clock_at(1000.0))
        with pytest.raises(BudgetExceededError):
            client.invoke("s", "u")
        fake.return_value.chat.assert_not_called()


def test_timeout_is_clamped_to_the_time_actually_left():
    """AGENT_TIMEOUT_S['sql'] là 200s. Còn 30s thì timeout phải là 30, không phải 200."""
    with patch("model.model_client.ollama.Client"):
        client = ModelClient(agent_type="sql", deadline_ts=1030.0, clock=clock_at(1000.0))
        assert client.effective_timeout_s() == pytest.approx(30.0)


def test_timeout_is_unclamped_without_a_deadline():
    with patch("model.model_client.ollama.Client"):
        assert ModelClient(agent_type="sql").effective_timeout_s() == 200


def test_budget_error_does_not_trigger_the_openai_fallback():
    """Hết ngân sách không phải lỗi của Ollama. Gọi sang OpenAI chỉ tiêu thêm
    thời gian mình không có, và làm schema rời khỏi mạng khách."""
    with patch("model.model_client.ollama.Client"), \
         patch("model.model_client.ModelClient._invoke_openai") as fallback:
        client = ModelClient(
            agent_type="sql", deadline_ts=900.0, clock=clock_at(1000.0),
            enable_openai_fallback=True,
        )
        with pytest.raises(BudgetExceededError):
            client.invoke("s", "u")
        fallback.assert_not_called()
