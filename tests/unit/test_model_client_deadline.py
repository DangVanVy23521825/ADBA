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


def test_the_http_client_is_given_a_real_timeout_not_just_measured_after():
    """Deadline phải NGẮT được lời gọi, không chỉ đo nó sau khi xong.

    Bản trước dựng `ollama.Client(host=...)` không timeout, và chỉ so
    elapsed với ngân sách SAU khi `.chat()` trả về. Một lời gọi treo vì
    thế chạy tới bao lâu tuỳ nó, rồi mới bị tuyên là quá hạn — deadline
    trở thành cái nhiệt kế, không phải cái phanh."""
    with patch("model.model_client.ollama.Client") as fake:
        fake.return_value.chat.return_value = {"message": {"content": "ok"}}
        client = ModelClient(agent_type="sql", deadline_ts=1030.0, clock=clock_at(1000.0))
        client.invoke("s", "u")

    timeouts = [c.kwargs.get("timeout") for c in fake.call_args_list]
    assert any(t is not None for t in timeouts), (
        "ollama.Client được dựng mà không có timeout — không gì ngắt được "
        "một lời gọi đang treo"
    )
    assert timeouts[-1] == pytest.approx(30.0), (
        "timeout phải bằng thời gian CÒN LẠI, không phải AGENT_TIMEOUT_S=200"
    )


def test_the_enforced_timeout_never_drops_to_zero():
    """`effective_timeout_s()` về 0 khi deadline sát nút. `timeout=0` với
    httpx là hỏng-ngay, biến một lời gọi còn kịp thành lỗi giả."""
    with patch("model.model_client.ollama.Client") as fake:
        fake.return_value.chat.return_value = {"message": {"content": "ok"}}
        client = ModelClient(agent_type="sql")
        client.timeout_s = 0.0
        client._chat_once_ollama([{"role": "user", "content": "x"}], 0.0)

    assert fake.call_args_list[-1].kwargs["timeout"] >= 1.0


def test_the_retry_backoff_never_sleeps_past_the_deadline():
    """Ngủ 2s khi chỉ còn 0,5s là đem nốt phần ngân sách cuối đổi lấy một
    lỗi: lần thử sau chắc chắn bị `_assert_budget` chặn."""
    with patch("model.model_client.ollama.Client"), \
         patch("model.model_client.time.sleep") as sleep:
        client = ModelClient(agent_type="sql", deadline_ts=1000.5, clock=clock_at(1000.0))
        client._sleep_before_retry(attempt=3)  # backoff trần trụi sẽ là 4s

    sleep.assert_called_once()
    assert sleep.call_args.args[0] == pytest.approx(0.5)


def test_no_sleep_at_all_once_the_deadline_has_passed():
    with patch("model.model_client.ollama.Client"), \
         patch("model.model_client.time.sleep") as sleep:
        client = ModelClient(agent_type="sql", deadline_ts=900.0, clock=clock_at(1000.0))
        client._sleep_before_retry(attempt=1)

    sleep.assert_not_called()


def test_backoff_is_unchanged_without_a_deadline():
    """Không deadline thì giữ nguyên hành vi cũ — nhiều test hiện có dựa
    vào đó."""
    with patch("model.model_client.ollama.Client"), \
         patch("model.model_client.time.sleep") as sleep:
        ModelClient(agent_type="sql")._sleep_before_retry(attempt=2)

    sleep.assert_called_once_with(2.0)


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
