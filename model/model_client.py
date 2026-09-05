"""
ModelClient wrapper with local-first Ollama and OpenAI fallback.
"""

from __future__ import annotations

import json
import os
import re
import time
import warnings
from typing import TYPE_CHECKING, Any

import ollama

from model.model_config import (
    AGENT_MAX_TOKENS,
    AGENT_TEMPERATURES,
    AGENT_TIMEOUT_S,
    DEPLOYMENT_ENV_VAR,
    HYBRID,
    MODEL_MAX_RETRIES,
    OLLAMA_BASE_URL,
    OLLAMA_NUM_CTX,
    ONPREM,
    PRIMARY_MODEL,
    deployment_mode,
    egress_allowed,
)

if TYPE_CHECKING:
    # Chỉ dùng cho type hint (`clock: Clock = time.time`). Import thật ở
    # cấp module sẽ tạo vòng: `graph/__init__.py` import tới
    # `graph.agents.supervisor`, mà module đó lại `import model.model_client`
    # — nếu import graph.budget ở đây trước khi ModelClient được định nghĩa
    # xong, Python sẽ gặp module `model.model_client` đang dở dang. Các hàm
    # dùng `graph.budget` thật sự (`effective_timeout_s`, `_assert_budget`)
    # import nó cục bộ, lúc đó toàn bộ cây import đã nạp xong.
    from graph.budget import Clock


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


class BudgetExceededError(RuntimeError):
    """Không còn đủ thời gian để khởi động một lời gọi model.

    Tách khỏi RuntimeError thường vì nó KHÔNG phải lỗi của Ollama, nên
    không được kích hoạt đường fallback sang OpenAI: gọi sang API ngoài
    lúc sắp hết giờ chỉ tiêu thêm thời gian mình không có, và làm schema
    của khách rời khỏi mạng của họ.
    """


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
        deadline_ts: float | None = None,
        call_budget_remaining: int | None = None,
        clock: Clock = time.time,
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

        # Giữ lại cho tương thích ngược (nơi gọi bên ngoài và test hiện có
        # chạm tới thuộc tính này). Đường gọi THẬT không dùng nó nữa —
        # `_chat_once_ollama` dựng client riêng mỗi lần với timeout đúng
        # bằng thời gian còn lại; client này không có timeout nên không
        # ngắt được gì.
        self.ollama_client = ollama.Client(host=OLLAMA_BASE_URL)

        # Fallback ra OpenAI được quyết bởi HAI lớp, và lớp ngoài thắng.
        #
        # Lớp ngoài là chế độ triển khai. Ở `onprem` — mặc định — không có
        # gì rời khỏi mạng khách, bất kể tham số hay biến môi trường nói gì.
        # Prompt của agent `sql` mang schema khách, mô tả nghiệp vụ do nhân
        # viên họ viết, và câu hỏi thật; ở khách ngân hàng hay y tế thì đó
        # không phải chuyện cấu hình mà là chuyện hợp đồng.
        #
        # Lớp trong là `ENABLE_OPENAI_FALLBACK` (hoặc tham số truyền vào),
        # chỉ có tác dụng khi đã ở `hybrid`.
        #
        # Một tham số `False` TƯỜNG MINH luôn thắng ở cả hai chế độ: nơi gọi
        # nào đã tự tắt thì không được bật lại bởi chế độ triển khai. Đó là
        # phòng thủ nhiều lớp, và `onboard._local_invoke` dựa vào nó.
        raw_env = os.getenv("ENABLE_OPENAI_FALLBACK")
        if enable_openai_fallback is None:
            requested = (raw_env or "1").strip().lower() not in {"0", "false", "no"}
            # Chỉ tính là "người ta CHỦ ĐỘNG xin" khi biến được đặt thật.
            # Mặc định "1" là di sản của thiết kế hybrid-first, không phải
            # một yêu cầu ai đó phát ra.
            explicitly_requested = raw_env is not None and requested
        else:
            requested = enable_openai_fallback
            explicitly_requested = enable_openai_fallback

        self.deployment_mode = deployment_mode()
        self.enable_openai_fallback = requested and egress_allowed()

        # Cấu hình mâu thuẫn phải nhìn thấy được. Người vận hành đặt
        # ENABLE_OPENAI_FALLBACK=1 rồi thấy nó không chạy sẽ đi tìm lỗi ở
        # OpenAI key, ở mạng, ở mọi nơi trừ chỗ thật sự chặn nó.
        #
        # Nhưng CHỈ cảnh báo khi có mâu thuẫn thật. Một bản cài on-prem bình
        # thường không đặt gì cả; kêu ở đó sẽ dạy người vận hành bỏ qua
        # cảnh báo, và cảnh báo bị bỏ qua thì vô dụng đúng lúc cần nhất.
        if explicitly_requested and not self.enable_openai_fallback:
            warnings.warn(
                f"ENABLE_OPENAI_FALLBACK được bật nhưng {DEPLOYMENT_ENV_VAR}"
                f"={self.deployment_mode!r} nên fallback bị CHẶN: không có "
                "prompt nào rời khỏi mạng này. Đặt "
                f"{DEPLOYMENT_ENV_VAR}={HYBRID} nếu khách hàng đã đồng ý cho "
                "gọi model ngoài.",
                RuntimeWarning,
                stacklevel=2,
            )

        self.openai_model = openai_model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.fallback_agents = {"supervisor", "sql"}

        self.deadline_ts = deadline_ts
        self.call_budget_remaining = call_budget_remaining
        self.clock = clock

        # Đếm SỐ LỜI GỌI MẠNG THẬT — không phải số vòng lặp. Cộng dồn cho
        # TRỌN VÒNG ĐỜI của instance này, KHÔNG reset mỗi lần invoke(): mọi
        # node dựng client MỘT LẦN trước vòng lặp ngoài của nó rồi gọi
        # invoke() nhiều lần trên CÙNG instance (đã kiểm cả 6 file agent),
        # nên đọc `calls_made` một lần ở cuối hàm cho ra đúng tổng thật của
        # cả node, bất kể vòng lặp ngoài chạy mấy lượt.
        #
        # Trước bản này, mỗi node tự ước lượng số lời gọi bằng biến vòng
        # lặp NGOÀI của chính nó (`attempt`, hay `MAX_RETRIES` khi hết
        # lượt) — con số đó KHÔNG PHẢI số lời gọi HTTP thật, vì hai lý do
        # độc lập:
        #   1. `_invoke_ollama` có vòng lặp retry RIÊNG, ẩn hoàn toàn với
        #      caller — một `client.invoke()` có thể tốn tới
        #      `MODEL_MAX_RETRIES` (3) lời gọi Ollama thật. Supervisor với
        #      `MAX_SUPERVISOR_RETRIES=3` vòng ngoài × 3 vòng trong = tới 9
        #      lời gọi thật, báo cáo thành 3.
        #   2. Fallback OpenAI (`_invoke_openai`) là MỘT lời gọi mạng thật
        #      nữa, tới một backend khác — không nơi nào đếm nó.
        # `MAX_LLM_CALLS_PER_QUERY` là tuyến phòng thủ CUỐI khi đồng hồ vì
        # lý do nào đó không cứu được (spec 5.4) — một trần đếm sai số thật
        # thì không còn là trần nữa. Sửa tận gốc: đếm Ở NƠI lời gọi mạng
        # THẬT sự xảy ra (`_chat_once_ollama`, `_invoke_openai`'s request),
        # không phải suy luận từ biến vòng lặp ở một lớp khác.
        #
        # Mọi node phải cộng `client.calls_made` vào `llm_calls_used` khi
        # trả về — KHÔNG PHẢI `attempt`/`MAX_RETRIES`/hằng số `1` nữa.
        self.calls_made: int = 0

    # Sàn cho timeout đẩy xuống HTTP client. `effective_timeout_s()` có thể
    # trả về 0 khi deadline đã cận, và `timeout=0` với httpx nghĩa là hỏng
    # ngay lập tức — biến một lời gọi lẽ ra vẫn kịp thành một lỗi giả.
    _MIN_HTTP_TIMEOUT_S = 1.0

    def _assert_call_budget(self) -> None:
        """Refuse before the next real network call would exceed the query cap.

        `llm_calls_used` lives in LangGraph state and only updates when a node
        returns. Internal retries happen inside this object, so the hard cap
        must also be checked here, immediately before each Ollama/OpenAI
        request.
        """
        if self.call_budget_remaining is None:
            return
        if self.calls_made >= self.call_budget_remaining:
            from graph.budget import MAX_LLM_CALLS_PER_QUERY

            raise BudgetExceededError(
                f"Không khởi động lời gọi model cho agent '{self.agent_type}': "
                f"đã dùng hết quota còn lại ({self.calls_made}/"
                f"{self.call_budget_remaining}) trong trần "
                f"{MAX_LLM_CALLS_PER_QUERY} lời gọi/câu hỏi."
            )

    def _chat_once_ollama(self, messages: list[dict[str, str]], timeout_s: float) -> str:
        # Client dựng THEO TỪNG LỜI GỌI, với timeout thật đẩy xuống httpx.
        #
        # `self.ollama_client` dựng ở __init__ không có timeout nào cả, nên
        # trước bản này KHÔNG gì ngắt được một lời gọi đang treo: deadline
        # chỉ được đối chiếu SAU khi `.chat()` trả về, tức là nó đo chứ
        # không ngăn. Một lời gọi treo vì thế vượt ngân sách của cả lượt
        # tới trọn AGENT_TIMEOUT_S (200s với agent `sql`) rồi mới bị phát
        # hiện là đã quá hạn.
        #
        # Timeout phải nằm ở đây chứ không ở __init__: thời gian còn lại
        # co dần trong suốt một lượt, nên con số đúng chỉ biết được ngay
        # trước lời gọi. `ollama.Client(host=..., **kwargs)` chuyển kwargs
        # thẳng xuống `httpx.Client`, nên `timeout` được httpx cưỡng chế
        # thật (đã kiểm với ollama đang cài).
        client = ollama.Client(
            host=OLLAMA_BASE_URL,
            timeout=max(timeout_s, self._MIN_HTTP_TIMEOUT_S),
        )
        response = client.chat(
            model=self.model_name,
            messages=messages,
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "num_ctx": OLLAMA_NUM_CTX,
            },
        )
        return response["message"]["content"].strip()

    def _sleep_before_retry(self, attempt: int) -> None:
        """Backoff mũ, nhưng không bao giờ ngủ xuyên qua deadline.

        `time.sleep(2 ** (attempt - 1))` trần trụi tiêu thời gian mà không
        làm được việc gì — và nếu nó ngủ quá mốc deadline thì lần thử tiếp
        theo chắc chắn bị `_assert_budget` chặn, nên giấc ngủ đó đổi phần
        ngân sách cuối cùng lấy đúng một lỗi. Cắt nó theo phần thời gian
        thật sự còn lại.
        """
        backoff = float(2 ** (attempt - 1))
        if self.deadline_ts is None:
            time.sleep(backoff)
            return

        from graph.budget import time_left

        remaining = time_left(self.deadline_ts, self.clock)
        if remaining <= 0:
            return
        time.sleep(min(backoff, remaining))

    def _invoke_ollama(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            # Tính lại mỗi lần thử: thời gian còn lại co đi sau mỗi lần
            # hỏng cộng mỗi lần backoff. Lấy một lần trước vòng lặp sẽ cho
            # lần thử thứ hai một ngân sách đã tiêu mất rồi.
            budget = self.effective_timeout_s()
            start = time.time()
            # Cộng dồn NGAY TRƯỚC lời gọi thật, không phải sau khi biết kết
            # quả — một lần thử timeout/lỗi vẫn là MỘT lời gọi mạng đã xảy
            # ra, dù nó không trả về gì dùng được.
            self._assert_call_budget()
            self.calls_made += 1
            try:
                out = self._chat_once_ollama(messages, budget)
                elapsed = time.time() - start
                if elapsed > budget:
                    raise TimeoutError(
                        f"Ollama response exceeded timeout: {elapsed:.1f}s > {budget:.1f}s"
                    )
                if not out:
                    raise ValueError("Ollama returned empty content.")
                return _strip_markdown_fences(out)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= self.max_retries:
                    break
                self._sleep_before_retry(attempt)

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
        # Cùng nguyên tắc với _invoke_ollama: cộng NGAY TRƯỚC lời gọi mạng
        # thật, không phải sau khi biết kết quả. Đây là lời gọi tới MỘT
        # BACKEND KHÁC (OpenAI, không phải Ollama) — trước bản này không
        # nơi nào đếm nó vào `llm_calls_used`, nên trần 12 lời gọi/câu hỏi
        # (spec 5.4) không tính tới nó dù nó tốn tiền thật và thời gian
        # thật giống hệt một lời gọi Ollama.
        self._assert_call_budget()
        self.calls_made += 1
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

    def effective_timeout_s(self) -> float:
        """Timeout thực dùng: nhỏ hơn giữa timeout của agent và thời gian còn lại.

        AGENT_TIMEOUT_S['sql'] là 200s — lớn hơn cả ngân sách 45s của cả
        lượt. Không siết theo thời gian còn lại thì tham số deadline chỉ là
        trang trí.
        """
        if self.deadline_ts is None:
            return self.timeout_s
        from graph.budget import time_left

        return max(0.0, min(self.timeout_s, time_left(self.deadline_ts, self.clock)))

    def _assert_budget(self) -> None:
        if self.deadline_ts is None:
            return
        from graph.budget import MODEL_CALL_ESTIMATE_S, has_room_for, time_left

        if not has_room_for(self.deadline_ts, MODEL_CALL_ESTIMATE_S, clock=self.clock):
            raise BudgetExceededError(
                f"Không khởi động lời gọi model cho agent '{self.agent_type}': còn "
                f"{time_left(self.deadline_ts, self.clock):.1f}s, cần khoảng "
                f"{MODEL_CALL_ESTIMATE_S:.0f}s — vượt ngân sách."
            )

    def invoke(self, system_prompt: str, user_prompt: str) -> str:
        """
        Invoke model text with local-first strategy.
        """
        self._assert_budget()
        try:
            return self._invoke_ollama(system_prompt=system_prompt, user_prompt=user_prompt)
        except BudgetExceededError:
            raise
        except Exception as ollama_error:  # noqa: BLE001
            can_fallback = (
                self.enable_openai_fallback and self.agent_type in self.fallback_agents
            )
            if not can_fallback:
                # Nói rõ VÌ SAO không có fallback. "fallback is disabled"
                # trần trụi khiến người vận hành đi tìm một cái công tắc,
                # trong khi ở chế độ onprem thì không có fallback là hành vi
                # ĐÚNG và cố ý — thứ cần sửa là Ollama, không phải cấu hình
                # egress.
                if self.deployment_mode == ONPREM and self.agent_type in self.fallback_agents:
                    reason = (
                        f"chế độ triển khai {DEPLOYMENT_ENV_VAR}={ONPREM!r} chặn "
                        "mọi lời gọi model ngoài (đúng thiết kế: prompt mang "
                        "schema và câu hỏi của khách). Hãy sửa Ollama, đừng "
                        "mở egress"
                    )
                else:
                    reason = "fallback đang tắt cho agent này"
                raise RuntimeError(
                    f"Ollama hỏng với agent '{self.agent_type}' và {reason}. "
                    f"Lỗi: {ollama_error}"
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
