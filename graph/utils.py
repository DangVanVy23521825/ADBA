"""
Helpers for LangGraph state — DataFrame serialization and trace logging.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from graph.state import MultiAgentState

_LOG_DIR = Path("logs")
_LOG_DIR.mkdir(exist_ok=True)
_TRACE_DIR = _LOG_DIR / "traces"


def df_to_state(df: pd.DataFrame) -> dict[str, Any]:
    """Convert DataFrame → JSON-serializable dict for LangGraph state."""
    return {
        "records": df.to_dict("records"),
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "shape": list(df.shape),
    }


def df_from_state(data: dict[str, Any]) -> pd.DataFrame:
    """Restore DataFrame from a dict produced by df_to_state()."""
    df = pd.DataFrame(data.get("records", []))
    for col, dtype in data.get("dtypes", {}).items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except (ValueError, TypeError):
                pass
    return df


def _write_trace_line(entry: dict[str, Any]) -> None:
    """Nối một dòng JSONL vào logs/traces/<query_id>.jsonl.

    Nuốt mọi lỗi có chủ ý: đây là log, và một câu trả lời đúng không được
    mất chỉ vì đĩa đầy hay thư mục không ghi được.
    """
    query_id = entry.get("query_id")
    if not query_id:
        return
    try:
        _TRACE_DIR.mkdir(parents=True, exist_ok=True)
        with (_TRACE_DIR / f"{query_id}.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception:  # noqa: BLE001
        pass


def append_trace(
    state: MultiAgentState,
    agent: str,
    action: str,
    observation: str,
    status: str,
    *,
    duration_ms: int | None = None,
) -> list[dict[str, Any]]:
    """Log an action-trace entry and return the updated trace list.

    `duration_ms` là keyword-only có mặc định, nên hàng chục lời gọi năm
    tham số vị trí đang có vẫn chạy nguyên.
    """
    deadline_ts = state.get("deadline_ts")
    entry: dict[str, Any] = {
        "query_id": state.get("query_id", ""),
        "agent": agent,
        "action": action,
        "observation": observation,
        "status": status,
        "timestamp": time.time(),
        "llm_calls": state.get("llm_calls_used", 0),
        "deadline_remaining_ms": (
            None if deadline_ts is None else int((deadline_ts - time.time()) * 1000)
        ),
        "duration_ms": duration_ms,
    }
    _write_trace_line(entry)
    return (state.get("action_trace") or []) + [entry]


def with_routing(state: MultiAgentState) -> MultiAgentState:
    """Gắn ảnh chụp định tuyến vào state mà node sắp trả về.

    Đây là chỗ ghi HỢP LỆ cho dependency_graph/ready_agents: node trả state
    qua reducer của LangGraph, conditional edge thì không (spec 5.5).
    Import cục bộ để tránh vòng import graph.utils ↔ graph.agents.supervisor.
    """
    from graph.agents.supervisor import routing_snapshot

    snapshot = routing_snapshot(state)
    return {
        **state,
        "dependency_graph": snapshot["dependency_graph"],
        "ready_agents": snapshot["ready_agents"],
        "shared_metadata": {
            **state.get("shared_metadata", {}),
            "dependency_graph": snapshot["dependency_graph"],
            "ready_agents": snapshot["ready_agents"],
            "parallel_ready": len(snapshot["ready_agents"]) > 1,
        },
    }
