"""
Viz Agent — generate matplotlib chart code, execute in safe sandbox, return base64 PNG.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

from graph.budget import calls_remaining, clamped_tool_timeout_s, node_may_run
from graph.errors import BUDGET_EXCEEDED, CHART_ERROR, MISSING_DATA, MISSING_STEP
from graph.state import MultiAgentState
from graph.tools.python_tool import CHART_EXEC_TIMEOUT_SECONDS, run_chart_safe
from graph.utils import append_trace, df_from_state, with_routing
from model.model_client import BudgetExceededError, ModelClient

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "viz_generation.txt"

VIZ_SYSTEM_PROMPT: str = _PROMPT_PATH.read_text(encoding="utf-8")

MAX_RETRIES = 2


def _extract_code(text: str) -> str:
    """Strip markdown fences and pre-supplied imports from Python code."""
    text = re.sub(r"```(?:python)?\s*|\s*```", "", text).strip()
    lines = text.split("\n")
    clean = [l for l in lines if not (
        l.strip().startswith("import matplotlib") or
        l.strip().startswith("from matplotlib") or
        l.strip().startswith("import io") or
        l.strip().startswith("from io") or
        l.strip().startswith("import base64") or
        l.strip().startswith("from base64")
    )]
    return "\n".join(clean)


def _find_viz_step(state: MultiAgentState) -> dict | None:
    for step in state.get("execution_plan", []):
        if step.get("agent") == "viz":
            return step
    return None


def viz_agent_node(state: MultiAgentState) -> MultiAgentState:
    """Generate chart code via LLM → execute → return chart_b64 in state."""
    may_run, reason = node_may_run(state, "viz")
    if not may_run:
        logger.warning("Viz Agent: %s", reason)
        return {
            **state,
            "current_agent": "viz",
            "completed_agents": state.get("completed_agents", []) + ["viz"],
            "degradation_reason": state.get("degradation_reason", []) + [reason],
            "action_trace": append_trace(state, "viz", "budget_check", reason, "error"),
            "status": "running",
        }

    step = _find_viz_step(state)
    if step is None:
        logger.error("Viz Agent: no viz step in execution plan")
        return {**state, "status": "failed",
                "last_error": {"agent": "viz", "error_type": MISSING_STEP}}

    task: str = step.get("task", "")

    df_state = state.get("shared_dataframe")
    if not df_state:
        logger.error("Viz Agent: no shared_dataframe in state")
        return {**state, "status": "failed",
                "last_error": {"agent": "viz", "error_type": MISSING_DATA}}

    df = df_from_state(df_state)
    import json
    columns = df.columns.tolist()
    sample = json.dumps(df.head(3).to_dict("records"), ensure_ascii=False)

    system_prompt = (VIZ_SYSTEM_PROMPT
                     .replace("{columns}", str(columns))
                     .replace("{sample}", sample)
                     .replace("{task}", task))

    trace = append_trace(state, "viz", "generate_chart",
                         f"Task: {task}", "started")
    state = {**state, "action_trace": trace}

    client = ModelClient(
        agent_type="viz",
        deadline_ts=state.get("deadline_ts"),
        call_budget_remaining=calls_remaining(state.get("llm_calls_used", 0)),
    )
    error_context = ""

    # Nhãn cho đường thất bại — xem nhánh BudgetExceededError bên dưới.
    error_label: str = CHART_ERROR

    for attempt in range(1, MAX_RETRIES + 1):
        user_prompt = f"Task: {task}"

        reflector_diag = state.get("shared_metadata", {}).get("reflector_diagnosis", {})
        reflector_hint = reflector_diag.get("corrected_context", "")
        if reflector_hint:
            user_prompt += f"\n\n[HINT from error diagnosis] {reflector_hint}"

        if error_context:
            user_prompt += f"\n\nPrevious error:\n{error_context}\n\nFix and rewrite."

        try:
            raw = client.invoke(system_prompt=system_prompt, user_prompt=user_prompt)
        except BudgetExceededError as exc:
            # Trước `except Exception`: BudgetExceededError là RuntimeError,
            # nên nhánh chung sẽ dán nhãn `chart_error` cho một lượt mà
            # matplotlib còn chưa được gọi tới.
            error_label = BUDGET_EXCEEDED
            error_context = str(exc)
            logger.warning("Viz Agent attempt %d hết ngân sách: %s", attempt, error_context)
            continue
        except Exception as exc:
            error_context = f"ModelClient error: {exc}"
            continue

        code = _extract_code(raw)

        # Ngân sách còn lại SAU model, TRƯỚC khi vào sandbox — `node_may_run`
        # ở đầu hàm chỉ gác cửa cho MỘT lời gọi model, không thấy phần chi
        # tiêu sau đó. Một lượt qua cửa vì còn ~15s cho model vẫn có thể
        # tốn tới `CHART_EXEC_TIMEOUT_SECONDS` (25s) ở sandbox ngay sau —
        # cộng lại có thể vượt ngân sách toàn cục dù cửa gác nói "còn giờ".
        clock = state.get("_clock", time.time)
        deadline_ts = state.get("deadline_ts")
        sandbox_timeout_s = clamped_tool_timeout_s(
            deadline_ts, CHART_EXEC_TIMEOUT_SECONDS, clock=clock
        )
        if sandbox_timeout_s <= 0:
            error_label = BUDGET_EXCEEDED
            error_context = "Hết ngân sách thời gian sau khi model trả lời — không còn chỗ để vẽ biểu đồ."
            logger.warning("Viz Agent attempt %d: %s", attempt, error_context)
            continue

        # Try executing the LLM-generated code
        try:
            # Wrap bare dict literal at end so exec() captures it
            lines = code.strip().split("\n")
            if lines and lines[-1].strip().startswith("{"):
                lines[-1] = "result = " + lines[-1]
                code = "\n".join(lines)

            result = run_chart_safe(code, df, timeout_seconds=sandbox_timeout_s)
            if "chart_b64" not in result:
                raise ValueError("Code did not produce a chart result dict")

        except Exception as exc:
            error_context = str(exc)
            logger.warning("Viz Agent error (attempt %d): %s", attempt, error_context)
            trace = append_trace(state, "viz", "execute_chart",
                                 f"Attempt {attempt}: {error_context}", "error")
            # Ghi lại vào state: `append_trace` dựng danh sách mới từ
            # `state["action_trace"]`, nên vòng lặp sau mà vẫn đọc `state`
            # cũ sẽ dựng lại từ danh sách GỐC và ghi đè mất mục vừa thêm.
            # Không có dòng này, chỉ mục cuối cùng sống sót và mọi chẩn
            # đoán của các lần thử trước biến mất khỏi UI lẫn log JSONL.
            state = {**state, "action_trace": trace}
            continue

        # ── Success ───────────────────────────────────
        chart_b64 = result.get("chart_b64", "")
        chart_type = result.get("chart_type", "unknown")
        trace = append_trace(state, "viz", "execute_chart",
                             f"OK — {chart_type} chart, "
                             f"{len(chart_b64)//1024} KB base64", "ok")

        return with_routing({
            **state,
            "current_agent": "viz",
            "completed_agents": state.get("completed_agents", []) + ["viz"],
            "agent_outputs": {
                **state.get("agent_outputs", {}),
                "viz": {"status": "ok", "chart_type": chart_type},
            },
            "chart_b64": chart_b64,
            "chart_metadata": {
                "chart_type": chart_type,
                "x_col": result.get("x_col", ""),
                "y_col": result.get("y_col", ""),
                "title": result.get("title", ""),
            },
            "shared_metadata": {
                **state.get("shared_metadata", {}),
                "chart_type": chart_type,
                "chart_metadata": {
                    "chart_type": chart_type,
                    "x_col": result.get("x_col", ""),
                    "y_col": result.get("y_col", ""),
                    "title": result.get("title", ""),
                },
            },
            "action_trace": trace,
            "status": "running",
            # `client.calls_made`, không phải `attempt` — xem ghi chú ở
            # model/model_client.py::ModelClient.__init__.
            "llm_calls_used": state.get("llm_calls_used", 0) + client.calls_made,
        })

    # ── Exhausted retries ──────────────────────────
    error_summary = f"Viz Agent failed after {MAX_RETRIES} attempts: {error_context}"
    trace = append_trace(state, "viz", "execute_chart", error_summary, "error")

    return {
        **state,
        "current_agent": "viz",
        "agent_outputs": {
            **state.get("agent_outputs", {}),
            "viz": {"status": "error", "error": error_summary},
        },
        "error_count": state.get("error_count", 0) + 1,
        "agent_error_counts": {
            **state.get("agent_error_counts", {}),
            "viz": state.get("agent_error_counts", {}).get("viz", 0) + 1,
        },
        "last_error": {"agent": "viz", "error_type": error_label,
                       "traceback": error_summary},
        "action_trace": trace,
        "status": "running",
        "llm_calls_used": state.get("llm_calls_used", 0) + client.calls_made,
    }
