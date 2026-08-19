"""Ranh giới egress: dữ liệu khách có được rời khỏi mạng của họ không.

Prompt của agent `sql` mang schema khách, mô tả nghiệp vụ do nhân viên họ
viết, và câu hỏi thật đang được hỏi. On-prem là lý do khách chọn sản phẩm
này, nên "không có gì rời khỏi đây" phải là hành vi MẶC ĐỊNH, và phải kiểm
được — không phải một sự trùng hợp giữa hai biến môi trường.
"""

from __future__ import annotations

import pytest

from model.model_client import ModelClient
from model.model_config import DEPLOYMENT_ENV_VAR, deployment_mode, egress_allowed


# ── chế độ triển khai ───────────────────────────────────────────────────────


def test_deployment_defaults_to_onprem_when_unset(monkeypatch):
    """Quên cấu hình phải hỏng về phía GIỮ dữ liệu lại."""
    monkeypatch.delenv(DEPLOYMENT_ENV_VAR, raising=False)
    assert deployment_mode() == "onprem"
    assert egress_allowed() is False


@pytest.mark.parametrize(
    "raw", ["", "  ", "cloud", "Onprem ", "HYBRID_", "prod", "0", "true"]
)
def test_an_unrecognised_mode_falls_back_to_onprem(monkeypatch, raw):
    """Chuỗi không đọc được rơi về phía an toàn.

    Cùng nguyên tắc đã áp cho `schema_mode` hỏng trong `read_profile`: một
    giá trị lạ không bao giờ được chọn nhánh im lặng gửi dữ liệu đi.
    """
    monkeypatch.setenv(DEPLOYMENT_ENV_VAR, raw)
    assert deployment_mode() == "onprem"
    assert egress_allowed() is False


@pytest.mark.parametrize("raw", ["hybrid", "HYBRID", " Hybrid "])
def test_hybrid_is_recognised_case_insensitively(monkeypatch, raw):
    monkeypatch.setenv(DEPLOYMENT_ENV_VAR, raw)
    assert deployment_mode() == "hybrid"
    assert egress_allowed() is True


def test_the_mode_is_read_per_call_not_frozen_at_import(monkeypatch):
    """Đọc lại mỗi lần gọi, để test và tiến trình dài đổi được chế độ."""
    monkeypatch.setenv(DEPLOYMENT_ENV_VAR, "hybrid")
    assert egress_allowed() is True
    monkeypatch.setenv(DEPLOYMENT_ENV_VAR, "onprem")
    assert egress_allowed() is False


# ── ModelClient: chế độ triển khai là lớp NGOÀI ────────────────────────────


def test_onprem_blocks_fallback_even_with_the_env_flag_on(monkeypatch):
    """Đây là lỗi thật đã tìm thấy: `sql` nằm trong fallback_agents VÀ
    ENABLE_OPENAI_FALLBACK mặc định bật, nên khi Ollama hỏng lúc chạy thật,
    schema khách đi sang OpenAI."""
    monkeypatch.setenv(DEPLOYMENT_ENV_VAR, "onprem")
    monkeypatch.setenv("ENABLE_OPENAI_FALLBACK", "1")

    with pytest.warns(RuntimeWarning, match=DEPLOYMENT_ENV_VAR):
        client = ModelClient(agent_type="sql")

    assert client.enable_openai_fallback is False


def test_onprem_is_the_default_so_an_unconfigured_process_cannot_egress(
    monkeypatch, recwarn
):
    monkeypatch.delenv(DEPLOYMENT_ENV_VAR, raising=False)
    monkeypatch.delenv("ENABLE_OPENAI_FALLBACK", raising=False)

    client = ModelClient(agent_type="sql")

    assert client.enable_openai_fallback is False, (
        "một tiến trình chưa cấu hình gì không được phép gửi dữ liệu ra ngoài"
    )
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)], (
        "bản cài on-prem bình thường phải im lặng — cảnh báo chỉ dành cho "
        "cấu hình mâu thuẫn, kêu ở mọi lần dựng client sẽ dạy người ta bỏ qua nó"
    )


def test_hybrid_allows_fallback_when_the_env_flag_is_on(monkeypatch):
    monkeypatch.setenv(DEPLOYMENT_ENV_VAR, "hybrid")
    monkeypatch.setenv("ENABLE_OPENAI_FALLBACK", "1")

    client = ModelClient(agent_type="sql")

    assert client.enable_openai_fallback is True


def test_hybrid_still_honours_the_env_flag_being_off(monkeypatch):
    """Chọn hybrid không có nghĩa là bắt buộc bật; hai lớp vẫn độc lập."""
    monkeypatch.setenv(DEPLOYMENT_ENV_VAR, "hybrid")
    monkeypatch.setenv("ENABLE_OPENAI_FALLBACK", "0")

    assert ModelClient(agent_type="sql").enable_openai_fallback is False


def test_an_explicit_false_wins_in_hybrid_too(monkeypatch):
    """`onboard._local_invoke` dựa vào điều này.

    Prompt chú giải mang DÒNG DỮ LIỆU THẬT của khách, nên nơi gọi đó tự tắt
    fallback bất kể chế độ triển khai. Chế độ không được bật lại hộ nó.
    """
    monkeypatch.setenv(DEPLOYMENT_ENV_VAR, "hybrid")
    monkeypatch.setenv("ENABLE_OPENAI_FALLBACK", "1")

    client = ModelClient(agent_type="sql", enable_openai_fallback=False)

    assert client.enable_openai_fallback is False


def test_an_explicit_true_does_not_override_onprem(monkeypatch):
    """Chế độ triển khai là ranh giới ngoài cùng.

    Một nơi gọi trong code không được tự cấp cho mình quyền gửi dữ liệu ra
    khỏi mạng khách — nếu không thì lời hứa on-prem chỉ đúng tới khi ai đó
    thêm một tham số.
    """
    monkeypatch.setenv(DEPLOYMENT_ENV_VAR, "onprem")

    with pytest.warns(RuntimeWarning):
        client = ModelClient(agent_type="sql", enable_openai_fallback=True)

    assert client.enable_openai_fallback is False


def test_no_warning_when_nothing_was_requested(monkeypatch, recwarn):
    """Cảnh báo chỉ dành cho cấu hình MÂU THUẪN.

    Một bản cài on-prem bình thường không bật gì cả; kêu ở đó sẽ dạy người
    vận hành bỏ qua cảnh báo.
    """
    monkeypatch.setenv(DEPLOYMENT_ENV_VAR, "onprem")
    monkeypatch.setenv("ENABLE_OPENAI_FALLBACK", "0")

    ModelClient(agent_type="sql")

    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


# ── thông báo lỗi phải nói đúng nguyên nhân ────────────────────────────────


def test_onprem_failure_message_points_at_ollama_not_at_the_egress_switch(monkeypatch):
    """Không có fallback ở onprem là hành vi ĐÚNG, không phải cấu hình sai.

    Một câu "fallback is disabled" trần trụi khiến người vận hành đi tìm
    công tắc để bật — tức là đi phá đúng thứ đang bảo vệ khách hàng của họ.
    """
    monkeypatch.setenv(DEPLOYMENT_ENV_VAR, "onprem")
    monkeypatch.delenv("ENABLE_OPENAI_FALLBACK", raising=False)
    client = ModelClient(agent_type="sql")

    def _boom(**_kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(client, "_invoke_ollama", _boom)

    with pytest.raises(RuntimeError) as excinfo:
        client.invoke(system_prompt="s", user_prompt="u")

    message = str(excinfo.value)
    assert DEPLOYMENT_ENV_VAR in message
    assert "Ollama" in message
    assert "đừng mở egress" in message


def test_a_non_fallback_agent_gets_the_ordinary_message(monkeypatch):
    """`insight` vốn không nằm trong fallback_agents — không liên quan egress."""
    monkeypatch.setenv(DEPLOYMENT_ENV_VAR, "onprem")
    monkeypatch.delenv("ENABLE_OPENAI_FALLBACK", raising=False)
    client = ModelClient(agent_type="insight")

    def _boom(**_kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(client, "_invoke_ollama", _boom)

    with pytest.raises(RuntimeError) as excinfo:
        client.invoke(system_prompt="s", user_prompt="u")

    assert DEPLOYMENT_ENV_VAR not in str(excinfo.value)
