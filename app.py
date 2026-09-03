"""
ADBA — Streamlit Chat UI
=========================
Interactive chat interface for the multi-agent BI system.

Features:
  - Chat input + message history
  - DataFrame display (st.dataframe)
  - Chart display (st.image from base64)
  - Insight card (styled card)
  - Agent trace expander (st.expander per agent)
"""

from __future__ import annotations

import base64
import io
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# Project root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from graph.multi_agent import run_graph
from graph.state import MultiAgentState
from graph.tools.sql_tool import TableNotPermittedError
from perception.connection_profile import permitted_tables
from perception.profile_store import read_profile
from perception.retrieval import LexicalRetriever
from perception.schema_context import resolve_schema_context
from schemas.insight_schema import InsightOutput
from schemas.plan_schema import ExecutionPlan


# ────────────────────────────────────────────────
# Page config
# ────────────────────────────────────────────────

st.set_page_config(
    page_title="ADBA — BI Agent",
    page_icon="📊",
    layout="wide",
)

st.title("ADBA — Autonomous Data & BI Agent")
st.caption("Ask questions about your data. The multi-agent pipeline will plan, query, analyze, and visualize.")


# ────────────────────────────────────────────────
# Profile (read once per session, from disk)
# ────────────────────────────────────────────────
# The onboarding pipeline (onboard.py: extract → annotate → build → verify)
# writes profile/ once, at install time. app.py used to rebuild it — via
# build_profile() — on every single question: re-introspecting the database,
# re-hashing, re-deciding schema_mode. That repeated onboarding work on every
# Enter press, though the spec says schema_mode is decided once at install.
# Reading from disk instead means the analyst's annotation edits (written to
# schema.yaml by the "Chú giải schema" review page) only need the profile to
# be re-read to take effect — see the "Nạp lại profile" button in the
# sidebar below, which clears these caches deliberately rather than on an
# automatic staleness check at startup (out of scope for this plan).

PROFILE_DIR = Path(os.environ.get("ADBA_PROFILE_DIR", "profile"))

NO_PROFILE_MSG = (
    f"Chưa có profile trong `{PROFILE_DIR}`. Đặt biến môi trường `ADBA_DSN` "
    "(tránh truyền DSN qua dòng lệnh — nó lộ trong `ps aux` và lịch sử "
    "shell), rồi chạy:\n\n"
    "```\n"
    "export ADBA_DSN=postgresql://...\n"
    f"python onboard.py extract --profile {PROFILE_DIR}\n"
    f"python onboard.py annotate --profile {PROFILE_DIR}\n"
    f"python onboard.py build --profile {PROFILE_DIR} --grant local=*\n"
    "```"
)


@st.cache_resource
def _load_profile():
    """Đọc profile một lần cho cả phiên.

    Trước đây khối xử lý câu hỏi dựng lại profile mỗi lần — introspect,
    hash, dựng index — tức lặp công việc onboarding trên mỗi lần gõ Enter,
    và quyết lại `schema_mode` mỗi lượt dù spec nói quyết một lần lúc cài
    đặt. `@st.cache_resource` giữ kết quả cho cả phiên; nút "Nạp lại
    profile" ở sidebar gọi `.clear()` để nạp lại sau khi chú giải được sửa.
    """
    return read_profile(PROFILE_DIR)


@st.cache_resource
def _load_retriever(_profile):
    """Retriever theo profile hiện tại. `_profile` không được hash (dấu `_`
    đứng đầu báo Streamlit bỏ qua nó khi tính cache key) — nó chỉ cần đổi
    khi `_load_profile` được nạp lại, việc mà `.clear()` đã lo.
    """
    return LexicalRetriever(_profile.tables)


# ────────────────────────────────────────────────
# Session state
# ────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────

def _decode_chart(b64: str) -> bytes | None:
    """Decode base64 chart image to bytes."""
    try:
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        return base64.b64decode(b64)
    except Exception:
        return None


def _dict_to_dataframe(d: dict[str, Any]) -> pd.DataFrame:
    """Convert agent output dict to DataFrame."""
    if "rows" in d and "columns" in d:
        return pd.DataFrame(d["rows"], columns=d["columns"])
    if "data" in d:
        return pd.DataFrame(d["data"])
    return pd.DataFrame([d])


def _render_insight_card(insight: dict) -> None:
    """Render a styled insight card."""
    try:
        parsed = InsightOutput.model_validate(insight)
    except Exception:
        st.json(insight)
        return

    cols = st.columns([3, 1])
    with cols[0]:
        st.markdown("**Finding**")
        st.info(parsed.finding)

        st.markdown("**Evidence**")
        for i, ev in enumerate(parsed.evidence, 1):
            st.markdown(f"{i}. {ev}")

        st.markdown("**Recommended Action**")
        verb_color = "#10b981" if parsed.is_positive() else "#f59e0b" if parsed.has_anomaly() else "#6366f1"
        st.markdown(
            f"<div style='padding: 0.75rem; border-left: 4px solid {verb_color}; "
            f"background: #1f2937; border-radius: 4px;'>{parsed.action}</div>",
            unsafe_allow_html=True,
        )

    with cols[1]:
        st.markdown("**Confidence**")
        conf_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(parsed.confidence, "⚪")
        st.markdown(f"{conf_emoji} **{parsed.confidence.upper()}**")

        st.markdown("**Anomaly**")
        if parsed.has_anomaly():
            anomaly_emoji = {"positive_outlier": "📈", "negative_outlier": "📉"}.get(parsed.anomaly.type, "⚠️")
            st.markdown(f"{anomaly_emoji} **{parsed.anomaly.type}**")
            st.caption(parsed.anomaly.detail)
        else:
            st.markdown("None detected")


def _render_agent_trace(trace: list[dict]) -> None:
    """Render expanders for each agent trace entry."""
    with st.expander(f"Agent Trace ({len(trace)} steps)", expanded=False):
        for i, entry in enumerate(trace, 1):
            agent = entry.get("agent", "unknown")
            status = entry.get("status", "unknown")
            duration = entry.get("duration_s", None)

            status_icon = "✅" if status == "success" else "❌" if status == "error" else "⏳"
            label = f"{status_icon} Step {i}: {agent}"
            if duration is not None:
                label += f" ({duration:.1f}s)"

            with st.expander(label, expanded=False):
                st.json({
                    "agent": agent,
                    "status": status,
                    "task": entry.get("task", ""),
                    "duration_s": duration,
                    "output_summary": entry.get("output_summary", ""),
                })

                if "error" in entry and entry["error"]:
                    st.error(entry["error"])


def _display_result(result: MultiAgentState) -> None:
    """Display all result sections from a completed run."""
    st.divider()

    # ── Trạng thái và lý do bị cắt ──
    status = result.get("status", "unknown")
    reasons = result.get("degradation_reason") or []
    if status == "partial":
        st.warning(
            "**Kết quả một phần.** " + " ".join(reasons)
            + "\n\nPhần hiển thị bên dưới là những gì chạy xong trong ngân sách thời gian."
        )
    elif status == "failed" and reasons:
        st.error("**Không hoàn thành.** " + " ".join(reasons))

    # ── Execution Plan ──
    plan = result.get("execution_plan")
    if plan:
        with st.expander("Execution Plan", expanded=True):
            try:
                parsed_plan = ExecutionPlan(
                    plan_summary=result.get("query", ""),
                    steps=plan,
                )
                st.markdown(f"**Summary:** {parsed_plan.plan_summary}")
                plan_df = pd.DataFrame([
                    {"Step": s.step, "Agent": s.agent, "Task": s.task, "Depends On": ", ".join(s.depends_on) or "—"}
                    for s in parsed_plan.steps
                ])
                st.dataframe(plan_df, use_container_width=True, hide_index=True)
            except Exception:
                st.json(plan)

    # ── Agent Trace ──
    trace = result.get("action_trace", [])
    if trace:
        _render_agent_trace(trace)

    # ── DataFrames ──
    for key, label in [("sql_result", "SQL Query Result"), ("python_result", "Python Analysis Result")]:
        data = result.get(key)
        if data and isinstance(data, dict):
            with st.expander(label, expanded=True):
                try:
                    df = _dict_to_dataframe(data)
                    st.dataframe(df, use_container_width=True)
                except Exception:
                    st.json(data)

    # ── Chart ──
    chart_b64 = result.get("chart_b64")
    if chart_b64:
        with st.expander("Visualization", expanded=True):
            img_bytes = _decode_chart(chart_b64)
            if img_bytes:
                st.image(img_bytes, use_container_width=True)
                chart_meta = result.get("chart_metadata")
                if chart_meta:
                    st.json(chart_meta)
            else:
                st.warning("Could not decode chart image.")

    # ── Insight ──
    insight = result.get("insight")
    if insight:
        with st.expander("Business Insight", expanded=True):
            _render_insight_card(insight)


# ────────────────────────────────────────────────
# Chat UI
# ────────────────────────────────────────────────

# Display message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your data..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run the multi-agent pipeline
    with st.chat_message("assistant"):
        with st.spinner("Planning and executing agents..."):
            try:
                profile = _load_profile()
            except FileNotFoundError:
                st.error(NO_PROFILE_MSG)
                st.stop()

            user = "local"  # TODO(plan-4): thay bằng danh tính từ lớp xác thực
            try:
                # Resolve the per-question schema slice (retrieval or full, per profile.schema_mode).
                ctx = resolve_schema_context(
                    profile, prompt, permitted_tables(profile, user),
                    retriever=_load_retriever(profile),
                )

                result = run_graph(query=prompt, schema_context=ctx, profile=profile, user=user)
                st.session_state.last_result = result

                # Display results
                _display_result(result)

                # Add assistant summary to history
                plan = result.get("execution_plan", [])
                completed = result.get("completed_agents", [])
                summary = f"Completed {len(completed)} agent(s): {', '.join(completed)}"
                if plan:
                    summary = f"Plan: {result.get('status', 'done')}. " + summary
                st.session_state.messages.append({"role": "assistant", "content": summary})

            except TableNotPermittedError as exc:
                # Read .public, not str(exc) — the exception's full message
                # (via .forbidden) carries the exact forbidden table names,
                # which are themselves information the user must not receive.
                # See graph/tools/sql_tool.py.
                error_msg = exc.public
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})


# ────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────

with st.sidebar:
    st.header("Session")
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_result = None
        st.rerun()

    st.divider()
    st.caption(
        "Đã sửa chú giải ở trang 'Chú giải schema'? Profile được cache cho "
        "cả phiên nên bấm nút dưới để nạp lại — không tự kiểm tra lúc khởi "
        "động."
    )
    if st.button("Nạp lại profile", use_container_width=True):
        _load_profile.clear()
        _load_retriever.clear()
        st.session_state.messages = []
        st.session_state.last_result = None
        st.rerun()

    if st.session_state.last_result:
        st.divider()
        st.header("Last Run")
        result = st.session_state.last_result
        st.metric("Status", result.get("status", "unknown"))
        st.metric("Agents Completed", len(result.get("completed_agents", [])))
        st.metric("Errors", result.get("error_count", 0))
        st.metric("Lời gọi model", result.get("llm_calls_used", 0))
