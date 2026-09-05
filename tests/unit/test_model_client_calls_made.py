"""`ModelClient.calls_made` đếm đúng số lời gọi mạng THẬT.

Trước bản này, mỗi node tự ước lượng số lời gọi model bằng biến vòng lặp
NGOÀI của chính nó (`attempt`, hay `MAX_RETRIES` khi hết lượt). Con số đó
KHÔNG PHẢI số lời gọi HTTP thật, vì hai lý do độc lập:

  1. `_invoke_ollama` có vòng lặp retry RIÊNG (`MODEL_MAX_RETRIES`), ẩn
     hoàn toàn với caller — một `client.invoke()` có thể tốn tới 3 lời
     gọi Ollama thật. Supervisor với `MAX_SUPERVISOR_RETRIES=3` vòng
     ngoài × 3 vòng trong = tới 9 lời gọi thật, báo cáo thành 3.
  2. Fallback OpenAI (`_invoke_openai`) là MỘT lời gọi mạng thật nữa, tới
     một backend khác — không nơi nào đếm nó.

`MAX_LLM_CALLS_PER_QUERY` là tuyến phòng thủ CUỐI khi đồng hồ vì lý do
nào đó không cứu được (spec 5.4) — một trần đếm sai số thật thì không
còn là trần nữa. `calls_made` đếm Ở NƠI lời gọi mạng THẬT sự xảy ra, nên
các test dưới đây gọi thẳng `ModelClient` thật (không mock cả class),
chỉ mock `ollama.Client`/`openai.OpenAI` ở biên mạng.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model.model_client import BudgetExceededError, ModelClient


def test_one_successful_call_counts_as_one():
    with patch("model.model_client.ollama.Client") as fake:
        fake.return_value.chat.return_value = {"message": {"content": "ok"}}
        client = ModelClient(agent_type="sql")
        client.invoke("s", "u")
        assert client.calls_made == 1


def test_internal_retry_is_counted_not_hidden():
    """Đây chính là lỗi 9-báo-thành-3: `_invoke_ollama` tự retry
    (MODEL_MAX_RETRIES=3) khi Ollama trả rỗng hay ném lỗi — vòng lặp đó
    ẩn hoàn toàn với node gọi `invoke()`. Hai lần đầu hỏng, lần ba mới
    xong: `calls_made` phải là 3, không phải 1 (như thể chỉ có một lời
    gọi diễn ra) hay bất kỳ số nào suy luận từ phía caller."""
    with patch("model.model_client.ollama.Client") as fake, \
         patch("model.model_client.time.sleep"):
        fake.return_value.chat.side_effect = [
            {"message": {"content": ""}},  # rỗng → tự retry lần 2
            RuntimeError("Ollama transient"),  # lỗi → tự retry lần 3
            {"message": {"content": "ok"}},  # xong
        ]
        client = ModelClient(agent_type="sql")
        result = client.invoke("s", "u")

    assert result == "ok"
    assert client.calls_made == 3, (
        "ba lời gọi Ollama thật đã xảy ra bên trong MỘT client.invoke() — "
        "che giấu điều đó là chính lỗi mà calls_made tồn tại để sửa"
    )


def test_calls_made_accumulates_across_multiple_invoke_calls_on_the_same_instance():
    """Mọi agent dựng MỘT client trước vòng lặp ngoài của nó rồi gọi
    invoke() nhiều lần trên CÙNG instance (đã kiểm cả 6 file agent) — node
    đọc `client.calls_made` một lần Ở CUỐI, nên nó phải cộng dồn qua các
    lần invoke() riêng biệt, không phải reset mỗi lần."""
    with patch("model.model_client.ollama.Client") as fake:
        fake.return_value.chat.return_value = {"message": {"content": "ok"}}
        client = ModelClient(agent_type="sql")
        client.invoke("s", "u1")
        client.invoke("s", "u2")

    assert client.calls_made == 2


def test_a_call_that_exhausts_every_internal_retry_still_counts_every_attempt():
    """Cả ba lần thử nội bộ đều hỏng — `invoke()` cuối cùng ném lỗi, nhưng
    ba lời gọi mạng thật đã thật sự xảy ra trước khi nó bỏ cuộc."""
    with patch("model.model_client.ollama.Client") as fake, \
         patch("model.model_client.time.sleep"):
        fake.return_value.chat.side_effect = RuntimeError("Ollama down")
        client = ModelClient(agent_type="viz")  # không nằm trong fallback_agents
        with pytest.raises(RuntimeError):
            client.invoke("s", "u")

    assert client.calls_made == 3


def test_openai_fallback_is_counted_as_a_real_call():
    """Fallback là lời gọi mạng THẬT tới MỘT BACKEND KHÁC (OpenAI, không
    phải Ollama) — trước bản này không nơi nào đếm nó vào `llm_calls_used`,
    nên trần 12 lời gọi/câu hỏi không tính tới nó dù nó tốn tiền thật và
    thời gian thật giống hệt một lời gọi Ollama."""
    fake_openai_response = MagicMock()
    fake_openai_response.choices = [MagicMock(message=MagicMock(content="fallback ok"))]

    with patch("model.model_client.ollama.Client") as fake_ollama, \
         patch("openai.OpenAI") as fake_openai_cls, \
         patch.dict(
             "os.environ",
             {"OPENAI_API_KEY": "test-key", "ADBA_DEPLOYMENT": "hybrid"},
         ):
        fake_ollama.return_value.chat.side_effect = RuntimeError("Ollama down")
        fake_openai_cls.return_value.chat.completions.create.return_value = fake_openai_response

        client = ModelClient(
            agent_type="sql",  # phải nằm trong fallback_agents
            enable_openai_fallback=True,
        )
        result = client.invoke("s", "u")

    assert result == "fallback ok"
    # MODEL_MAX_RETRIES lần Ollama thật (đều hỏng) + 1 lần OpenAI thật.
    from model.model_config import MODEL_MAX_RETRIES

    assert client.calls_made == MODEL_MAX_RETRIES + 1


def test_a_budget_exceeded_refusal_makes_no_real_call():
    """`_assert_budget()` chặn TRƯỚC khi bất kỳ lời gọi mạng nào xảy ra —
    calls_made phải là 0, không phải 1, vì không lời gọi thật nào đã xảy
    ra để đếm."""
    with patch("model.model_client.ollama.Client"):
        client = ModelClient(agent_type="sql", deadline_ts=900.0, clock=lambda: 1000.0)
        with pytest.raises(BudgetExceededError):
            client.invoke("s", "u")

    assert client.calls_made == 0


def test_call_budget_blocks_internal_retry_before_next_network_call():
    """Regression cho ca state đang ở 11/12.

    Trước bản vá, router/node entry cho node chạy vì 11 < 12, nhưng một
    `client.invoke()` vẫn có thể tự retry Ollama 3 lần bên trong và biến
    tổng thật thành 14/12 trước khi state được cộng `calls_made`.
    """
    with patch("model.model_client.ollama.Client") as fake, \
         patch("model.model_client.time.sleep"):
        fake.return_value.chat.side_effect = RuntimeError("Ollama transient")
        client = ModelClient(agent_type="viz", call_budget_remaining=1)

        with pytest.raises(BudgetExceededError, match="quota"):
            client.invoke("s", "u")

    assert client.calls_made == 1
    assert fake.return_value.chat.call_count == 1


def test_call_budget_zero_refuses_before_any_network_call():
    with patch("model.model_client.ollama.Client") as fake:
        client = ModelClient(agent_type="sql", call_budget_remaining=0)

        with pytest.raises(BudgetExceededError, match="quota"):
            client.invoke("s", "u")

    assert client.calls_made == 0
    fake.return_value.chat.assert_not_called()
