import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from onboard import cmd_annotate, cmd_extract, main
from perception.annotations import load_annotations
from perception.profile_store import STRUCTURE_JSON

from tests.fixtures.mini_schema import MINI_TABLES


def test_extract_writes_structure_without_annotations(tmp_path, monkeypatch):
    monkeypatch.setattr("onboard.introspect_schema", lambda dsn, **kw: MINI_TABLES)  # noqa: ARG005
    cmd_extract("postgresql://x", tmp_path)
    raw = json.loads((tmp_path / STRUCTURE_JSON).read_text(encoding="utf-8"))
    assert {t["name"] for t in raw} == {t.name for t in MINI_TABLES}
    assert not (tmp_path / "schema.yaml").exists()


def test_annotate_writes_schema_yaml(tmp_path, monkeypatch):
    monkeypatch.setattr("onboard.introspect_schema", lambda dsn, **kw: MINI_TABLES)  # noqa: ARG005
    monkeypatch.setattr("onboard.sample_rows", lambda dsn, table, n=5: [])  # noqa: ARG005
    cmd_extract("postgresql://x", tmp_path)

    reply = '{"table": {"text": "mô tả", "confidence": "high"}, "columns": {}}'
    cmd_annotate(tmp_path, "postgresql://x", invoke=lambda s, u: reply)  # noqa: ARG005

    ann = load_annotations(tmp_path / "schema.yaml")
    assert ann.tables["orders"].text == "mô tả"


def test_annotate_preserves_human_edits_on_a_second_run(tmp_path, monkeypatch):
    monkeypatch.setattr("onboard.introspect_schema", lambda dsn, **kw: MINI_TABLES)  # noqa: ARG005
    monkeypatch.setattr("onboard.sample_rows", lambda dsn, table, n=5: [])  # noqa: ARG005
    cmd_extract("postgresql://x", tmp_path)

    from perception.annotations import Annotation, SchemaAnnotations, save_annotations
    save_annotations(
        SchemaAnnotations(tables={"orders": Annotation("NGƯỜI VIẾT", reviewed_by="human")}),
        tmp_path / "schema.yaml",
    )

    reply = '{"table": {"text": "LLM ghi đè", "confidence": "high"}, "columns": {}}'
    cmd_annotate(tmp_path, "postgresql://x", invoke=lambda s, u: reply)  # noqa: ARG005

    assert load_annotations(tmp_path / "schema.yaml").tables["orders"].text == "NGƯỜI VIẾT"


# --- Amendment 2: progress + failure reporting -----------------------------


def test_annotate_prints_per_table_progress(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("onboard.introspect_schema", lambda dsn, **kw: MINI_TABLES)  # noqa: ARG005
    monkeypatch.setattr("onboard.sample_rows", lambda dsn, table, n=5: [])  # noqa: ARG005
    cmd_extract("postgresql://x", tmp_path)

    reply = '{"table": {"text": "mô tả", "confidence": "high"}, "columns": {}}'
    capsys.readouterr()
    cmd_annotate(tmp_path, "postgresql://x", invoke=lambda s, u: reply)  # noqa: ARG005

    out = capsys.readouterr().out
    total = len(MINI_TABLES)
    for i, t in enumerate(MINI_TABLES, start=1):
        assert f"{i}/{total}" in out
        assert t.name in out


def test_annotate_reports_failure_count_separately_from_success_line(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr("onboard.introspect_schema", lambda dsn, **kw: MINI_TABLES)  # noqa: ARG005
    monkeypatch.setattr("onboard.sample_rows", lambda dsn, table, n=5: [])  # noqa: ARG005
    cmd_extract("postgresql://x", tmp_path)

    # First table fails (raises), rest succeed.
    calls = {"n": 0}

    def flaky_invoke(system, user):  # noqa: ARG001
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return '{"table": {"text": "mô tả", "confidence": "high"}, "columns": {}}'

    capsys.readouterr()
    cmd_annotate(tmp_path, "postgresql://x", invoke=flaky_invoke)

    out = capsys.readouterr().out
    total = len(MINI_TABLES)
    assert f"1/{total}" in out or "1/" in out  # failure count present in some form
    lines = [ln for ln in out.splitlines() if ln.strip()]
    # There must be a success/summary line and a distinct line mentioning the failure count.
    assert any("bảng" in ln and "duyệt" in ln for ln in lines)
    assert any(str(1) in ln and ("thất bại" in ln.lower() or "lỗi" in ln.lower()) for ln in lines)


def test_annotate_all_failed_warns_local_model_unreachable(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("onboard.introspect_schema", lambda dsn, **kw: MINI_TABLES)  # noqa: ARG005
    monkeypatch.setattr("onboard.sample_rows", lambda dsn, table, n=5: [])  # noqa: ARG005
    cmd_extract("postgresql://x", tmp_path)

    def always_fails(system, user):  # noqa: ARG001
        raise RuntimeError("connection refused")

    capsys.readouterr()
    cmd_annotate(tmp_path, "postgresql://x", invoke=always_fails)

    out = capsys.readouterr().out
    assert "OLLAMA_BASE_URL" in out
    assert "CẢNH BÁO" in out or "cảnh báo" in out.lower()


def test_annotate_partial_failure_does_not_warn(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("onboard.introspect_schema", lambda dsn, **kw: MINI_TABLES)  # noqa: ARG005
    monkeypatch.setattr("onboard.sample_rows", lambda dsn, table, n=5: [])  # noqa: ARG005
    cmd_extract("postgresql://x", tmp_path)

    calls = {"n": 0}

    def flaky_invoke(system, user):  # noqa: ARG001
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return '{"table": {"text": "mô tả", "confidence": "high"}, "columns": {}}'

    capsys.readouterr()
    cmd_annotate(tmp_path, "postgresql://x", invoke=flaky_invoke)

    out = capsys.readouterr().out
    assert "OLLAMA_BASE_URL" not in out


# --- Amendment 3: --dsn optional, ADBA_DSN fallback -------------------------


def test_cli_extract_falls_back_to_adba_dsn_env_var(tmp_path, monkeypatch):
    monkeypatch.setattr("onboard.introspect_schema", lambda dsn, **kw: MINI_TABLES)  # noqa: ARG005
    monkeypatch.setenv("ADBA_DSN", "postgresql://from-env")
    monkeypatch.setattr(
        sys, "argv", ["onboard.py", "extract", "--profile", str(tmp_path)]
    )
    main()
    raw = json.loads((tmp_path / STRUCTURE_JSON).read_text(encoding="utf-8"))
    assert {t["name"] for t in raw} == {t.name for t in MINI_TABLES}


def test_cli_extract_missing_dsn_exits_with_clear_error(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ADBA_DSN", raising=False)
    monkeypatch.setattr(
        sys, "argv", ["onboard.py", "extract", "--profile", str(tmp_path)]
    )
    with pytest.raises(SystemExit):
        main()
    captured = capsys.readouterr()
    assert "ADBA_DSN" in captured.out + captured.err


def test_cli_extract_dsn_flag_takes_precedence_over_env(tmp_path, monkeypatch):
    seen = {}

    def fake_introspect(dsn, **kw):  # noqa: ARG001
        seen["dsn"] = dsn
        return MINI_TABLES

    monkeypatch.setattr("onboard.introspect_schema", fake_introspect)
    monkeypatch.setenv("ADBA_DSN", "postgresql://from-env")
    monkeypatch.setattr(
        sys,
        "argv",
        ["onboard.py", "extract", "--dsn", "postgresql://from-flag", "--profile", str(tmp_path)],
    )
    main()
    assert seen["dsn"] == "postgresql://from-flag"


# --- Amendment 4: _local_invoke disables OpenAI fallback -------------------


def test_local_invoke_disables_openai_fallback():
    from onboard import _local_invoke

    fake_client = MagicMock()
    fake_client.invoke.return_value = "{}"

    with patch("onboard.ModelClient", return_value=fake_client) as mock_cls:
        result = _local_invoke("sys prompt", "user prompt")

    assert result == "{}"
    _args, kwargs = mock_cls.call_args
    assert kwargs.get("enable_openai_fallback") is False
    fake_client.invoke.assert_called_once_with(
        system_prompt="sys prompt", user_prompt="user prompt"
    )
