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


# --- Task 9: `onboard build` ------------------------------------------------


def test_build_writes_a_loadable_profile(tmp_path, monkeypatch):
    monkeypatch.setattr("onboard.introspect_schema", lambda dsn, **kw: MINI_TABLES)  # noqa: ARG005
    cmd_extract("postgresql://x", tmp_path)

    from onboard import cmd_build
    from perception.connection_profile import ALL_TABLES, permitted_tables
    from perception.profile_store import read_profile

    cmd_build(tmp_path, "postgresql://x", grants={"admin": frozenset({ALL_TABLES})})
    profile = read_profile(tmp_path)
    assert permitted_tables(profile, "admin") == {t.name for t in MINI_TABLES}


def test_build_refuses_when_annotations_are_mostly_unreviewed(tmp_path, monkeypatch):
    """Chặn bàn giao sớm: profile không chú giải sẽ cho recall thấp và không ai biết vì sao."""
    monkeypatch.setattr("onboard.introspect_schema", lambda dsn, **kw: MINI_TABLES)  # noqa: ARG005
    cmd_extract("postgresql://x", tmp_path)

    from onboard import UnreviewedAnnotationsError, cmd_build

    with pytest.raises(UnreviewedAnnotationsError):
        cmd_build(tmp_path, "postgresql://x", grants={}, min_reviewed=0.5)


# --- Amendment 1: gate denominator must include un-annotated columns -------


def test_build_refuses_when_only_table_level_entries_are_reviewed(tmp_path, monkeypatch):
    """`review_progress(ann)` (một-đối-số) đếm mọi mục *đã có chú giải* là đã
    duyệt/tổng dựa trên tập đó — 4 bảng, cả 4 đều `human`, cho ratio 1.0 và
    KHÔNG chặn được. Nhưng schema này có 17 cột chưa hề được chú giải; dùng
    dạng hai-đối-số (`review_progress(ann, tables)`) đưa những cột đó vào
    mẫu số, kéo ratio xuống dưới ngưỡng và cổng phải chặn.
    """
    monkeypatch.setattr("onboard.introspect_schema", lambda dsn, **kw: MINI_TABLES)  # noqa: ARG005
    cmd_extract("postgresql://x", tmp_path)

    from perception.annotations import Annotation, SchemaAnnotations, save_annotations

    save_annotations(
        SchemaAnnotations(
            tables={
                t.name: Annotation(f"mô tả {t.name}", reviewed_by="human")
                for t in MINI_TABLES
            }
        ),
        tmp_path / "schema.yaml",
    )

    from onboard import UnreviewedAnnotationsError, cmd_build

    with pytest.raises(UnreviewedAnnotationsError):
        cmd_build(tmp_path, "postgresql://x", grants={}, min_reviewed=0.5)


# --- Amendment 2: a build with no grants must say so ------------------------


def test_build_with_no_grants_warns_nobody_can_see_anything(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("onboard.introspect_schema", lambda dsn, **kw: MINI_TABLES)  # noqa: ARG005
    cmd_extract("postgresql://x", tmp_path)

    from onboard import cmd_build

    capsys.readouterr()
    cmd_build(tmp_path, "postgresql://x", grants={})

    out = capsys.readouterr().out
    assert "--grant user=*" in out


def test_build_with_grants_does_not_print_no_grant_warning(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("onboard.introspect_schema", lambda dsn, **kw: MINI_TABLES)  # noqa: ARG005
    cmd_extract("postgresql://x", tmp_path)

    from onboard import cmd_build
    from perception.connection_profile import ALL_TABLES

    capsys.readouterr()
    cmd_build(tmp_path, "postgresql://x", grants={"admin": frozenset({ALL_TABLES})})

    out = capsys.readouterr().out
    assert "--grant user=*" not in out


# --- Amendment 3: `build` uses the same DSN resolution as extract/annotate --


def test_cli_build_falls_back_to_adba_dsn_env_var(tmp_path, monkeypatch):
    monkeypatch.setattr("onboard.introspect_schema", lambda dsn, **kw: MINI_TABLES)  # noqa: ARG005
    cmd_extract("postgresql://x", tmp_path)

    monkeypatch.setenv("ADBA_DSN", "postgresql://from-env")
    monkeypatch.setattr(
        sys, "argv", ["onboard.py", "build", "--profile", str(tmp_path)]
    )
    main()

    from perception.profile_store import read_profile

    profile = read_profile(tmp_path)
    assert profile.dsn == "postgresql://from-env"


def test_cli_build_missing_dsn_exits_with_clear_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("onboard.introspect_schema", lambda dsn, **kw: MINI_TABLES)  # noqa: ARG005
    cmd_extract("postgresql://x", tmp_path)

    monkeypatch.delenv("ADBA_DSN", raising=False)
    monkeypatch.setattr(
        sys, "argv", ["onboard.py", "build", "--profile", str(tmp_path)]
    )
    with pytest.raises(SystemExit):
        main()
    captured = capsys.readouterr()
    assert "ADBA_DSN" in captured.out + captured.err


def test_cli_build_parses_grant_flags(tmp_path, monkeypatch):
    monkeypatch.setattr("onboard.introspect_schema", lambda dsn, **kw: MINI_TABLES)  # noqa: ARG005
    cmd_extract("postgresql://x", tmp_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "onboard.py",
            "build",
            "--dsn",
            "postgresql://from-flag",
            "--profile",
            str(tmp_path),
            "--grant",
            "admin=*",
            "--grant",
            "analyst=orders,customers",
        ],
    )
    main()

    from perception.connection_profile import permitted_tables
    from perception.profile_store import read_profile

    profile = read_profile(tmp_path)
    assert permitted_tables(profile, "admin") == {t.name for t in MINI_TABLES}
    assert permitted_tables(profile, "analyst") == {"orders", "customers"}


# --- Fix round 1, item 1: a `--grant` with no '=' is a typo, not a phantom -


def test_cli_build_grant_without_equals_exits_with_clear_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("onboard.introspect_schema", lambda dsn, **kw: MINI_TABLES)  # noqa: ARG005
    cmd_extract("postgresql://x", tmp_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "onboard.py",
            "build",
            "--dsn",
            "postgresql://x",
            "--profile",
            str(tmp_path),
            "--grant",
            "admin",
        ],
    )
    with pytest.raises(SystemExit):
        main()

    captured = capsys.readouterr()
    text = captured.out + captured.err
    # Names the offending value...
    assert "admin" in text
    # ...and shows the correct syntax.
    assert "user=bang1,bang2" in text or "user=*" in text

    # No profile.json must have been written — a rejected --grant is a hard
    # stop, not a build that silently drops the bad grant.
    assert not (tmp_path / "profile.json").exists()


# --- Fix round 1, item 1 (decision) + item 3: explicit empty grant ----------


def test_cli_build_explicit_empty_grant_is_accepted_and_suppresses_warning(
    tmp_path, monkeypatch, capsys
):
    """`--grant sales=` (an explicit empty value) is accepted as a deliberate
    "grant nothing to sales for now", distinct from omitting --grant
    entirely. Because the resulting `grants` mapping is non-empty (it has
    the key `sales`), the no-grants warning must NOT fire — that warning is
    about nobody having any grants recorded at all, not about any one
    user's grant being empty.
    """
    monkeypatch.setattr("onboard.introspect_schema", lambda dsn, **kw: MINI_TABLES)  # noqa: ARG005
    cmd_extract("postgresql://x", tmp_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "onboard.py",
            "build",
            "--dsn",
            "postgresql://x",
            "--profile",
            str(tmp_path),
            "--grant",
            "sales=",
        ],
    )
    capsys.readouterr()
    main()

    out = capsys.readouterr().out
    assert "--grant user=*" not in out
    assert "CẢNH BÁO" not in out

    from perception.connection_profile import permitted_tables
    from perception.profile_store import read_profile

    profile = read_profile(tmp_path)
    assert permitted_tables(profile, "sales") == frozenset()


# --- Fix round 1, item 2: gate arithmetic ------------------------------------


def test_build_allows_when_ratio_exactly_equals_threshold(tmp_path, monkeypatch):
    """Strict `<`, not `<=`: a ratio exactly at the threshold must pass."""
    from perception.annotations import Annotation, SchemaAnnotations, save_annotations
    from perception.schema_model import Column, Table

    tables = (
        Table(name="t1", columns=(Column("a", "integer"),), primary_key=("a",), row_count=1),
        Table(name="t2", columns=(Column("b", "integer"),), primary_key=("b",), row_count=1),
    )
    monkeypatch.setattr("onboard.introspect_schema", lambda dsn, **kw: tables)  # noqa: ARG005
    cmd_extract("postgresql://x", tmp_path)

    # 4 reviewable items total (t1, t1.a, t2, t2.b); 2 human-reviewed ->
    # ratio == 0.5 exactly.
    save_annotations(
        SchemaAnnotations(
            tables={"t1": Annotation("mô tả t1", reviewed_by="human")},
            columns={"t2": {"b": Annotation("mô tả b", reviewed_by="human")}},
        ),
        tmp_path / "schema.yaml",
    )

    from onboard import cmd_build

    # Must NOT raise: ratio (0.5) == min_reviewed (0.5).
    cmd_build(tmp_path, "postgresql://x", grants={}, min_reviewed=0.5)


def test_build_gate_on_empty_schema_does_not_divide_by_zero_and_still_gates(
    tmp_path, monkeypatch
):
    """`total == 0` must not raise ZeroDivisionError, and must NOT be treated
    as 100% reviewed: a nonzero --min-reviewed still has to gate it.
    """
    monkeypatch.setattr("onboard.introspect_schema", lambda dsn, **kw: ())  # noqa: ARG005
    cmd_extract("postgresql://x", tmp_path)

    from onboard import UnreviewedAnnotationsError, cmd_build

    with pytest.raises(UnreviewedAnnotationsError):
        cmd_build(tmp_path, "postgresql://x", grants={}, min_reviewed=0.1)

    # min_reviewed=0.0 on an empty schema is the one case that must pass:
    # ratio defaults to 0.0, and 0.0 < 0.0 is false.
    cmd_build(tmp_path, "postgresql://x", grants={}, min_reviewed=0.0)
