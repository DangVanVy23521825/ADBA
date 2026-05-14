"""
Helpers for LangGraph state — DataFrame serialization and trace logging.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from graph.state import MultiAgentState

_LOG_DIR = Path("logs")
_LOG_DIR.mkdir(exist_ok=True)


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


def append_trace(
    state: MultiAgentState,
    agent: str,
    action: str,
    observation: str,
    status: str,
) -> list[dict[str, Any]]:
    """Log an action-trace entry and return the updated trace list."""
    entry: dict[str, Any] = {
        "agent": agent,
        "action": action,
        "observation": observation,
        "status": status,
        "timestamp": time.time(),
    }
    return state.get("action_trace", []) + [entry]
