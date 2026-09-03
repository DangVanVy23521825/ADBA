"""Trace mang ngân sách, và ghi ra đĩa được (spec 5.6)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from graph.errors import ERROR_LABELS
from graph.state import make_initial_state
from graph.utils import append_trace
from perception.schema_context import SchemaContext


def _state():
    s = make_initial_state("q", SchemaContext(), budget_s=45, clock=lambda: 1000.0)
    s["llm_calls_used"] = 3
    return s


def test_trace_entry_carries_the_query_id():
    s = _state()
    entry = append_trace(s, "sql", "execute_sql", "ok", "ok")[-1]
    assert entry["query_id"] == s["query_id"]


def test_trace_entry_carries_the_llm_call_count():
    assert append_trace(_state(), "sql", "a", "o", "ok")[-1]["llm_calls"] == 3


def test_trace_entry_carries_remaining_budget():
    entry = append_trace(_state(), "sql", "a", "o", "ok")[-1]
    assert entry["deadline_remaining_ms"] is not None


def test_trace_entry_carries_duration_when_given():
    entry = append_trace(_state(), "sql", "a", "o", "ok", duration_ms=1234)[-1]
    assert entry["duration_ms"] == 1234


def test_duration_is_none_when_not_measured():
    assert append_trace(_state(), "sql", "a", "o", "ok")[-1]["duration_ms"] is None


def test_old_five_argument_call_still_works():
    """Mọi node đang gọi năm tham số vị trí. Đừng làm hỏng."""
    assert len(append_trace(_state(), "sql", "a", "o", "ok")) == 1


def test_entries_are_written_as_jsonl_keyed_by_query_id(tmp_path, monkeypatch):
    monkeypatch.setattr("graph.utils._TRACE_DIR", tmp_path)
    s = _state()
    append_trace(s, "sql", "execute_sql", "ok", "ok")
    lines = (tmp_path / f"{s['query_id']}.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["agent"] == "sql"


def test_a_write_failure_never_breaks_the_query(monkeypatch):
    """Log hỏng không được làm hỏng câu trả lời của người dùng."""
    monkeypatch.setattr("graph.utils._TRACE_DIR", Path("/khong/ton/tai/duoc"))
    assert len(append_trace(_state(), "sql", "a", "o", "ok")) == 1


def test_the_nine_spec_labels_are_all_present():
    assert {
        "sql_generation", "sql_execution", "sql_timeout", "python_runtime",
        "chart_error", "schema_mismatch", "model_timeout", "budget_exceeded",
        "tool_unavailable",
    } <= ERROR_LABELS


def test_planning_and_insight_labels_are_present_too():
    """Sai lệch có chủ ý so với spec 5.6 — xem docstring graph/errors.py."""
    assert {"planning_failure", "insight_generation"} <= ERROR_LABELS
