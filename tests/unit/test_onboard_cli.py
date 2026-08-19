import json
import sys
from unittest.mock import MagicMock, patch

import psycopg2
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
    # M2: the old assertion here was `f"1/{total}" in out or "1/" in out`,
    # which is trivially true regardless of failure counting -- `_with_progress`
    # unconditionally prints `[1/{total}]` for the FIRST table processed,
    # succeeding or not, so the left side of the `or` (and thus the whole
    # assertion) can never be false. Narrow it to the line this test is
    # actually named for: the dedicated failure-count line `cmd_annotate`
    # prints, `Thất bại: {failures}/{total} bảng.`.
    assert f"Thất bại: 1/{total} bảng." in out
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


# --- I5: annotate's headline number must be review_progress, not ----------
# --- pending_review (a priority queue, not a coverage measure) ------------


def test_annotate_headline_uses_review_progress_not_pending_review_count(
    tmp_path, monkeypatch, capsys
):
    """cmd_build gates on review_progress(ann, tables); `annotate` must
    report the SAME quantity as its headline, or an operator sees "3 mục
    cần người duyệt" from annotate and then a `--min-reviewed` gate later
    reporting hundreds outstanding -- two commands disagreeing about the
    same number in one pipeline."""
    monkeypatch.setattr("onboard.introspect_schema", lambda dsn, **kw: MINI_TABLES)  # noqa: ARG005
    monkeypatch.setattr("onboard.sample_rows", lambda dsn, table, n=5: [])  # noqa: ARG005
    cmd_extract("postgresql://x", tmp_path)

    # High-confidence table-level descriptions only, no column annotations
    # at all -- pending_review() would report 0 (nothing low-confidence),
    # while review_progress(merged, tables) must report a much larger
    # denominator: every column MINI_TABLES has, annotated or not.
    reply = '{"table": {"text": "mô tả", "confidence": "high"}, "columns": {}}'
    capsys.readouterr()
    merged = cmd_annotate(tmp_path, "postgresql://x", invoke=lambda s, u: reply)  # noqa: ARG005

    from perception.review_state import review_progress

    done, total = review_progress(merged, MINI_TABLES)
    assert done == 0  # nothing human-reviewed yet
    assert total > len(MINI_TABLES), "denominator must include every column, not just tables"

    out = capsys.readouterr().out
    assert f"Đã duyệt {done}/{total} mục" in out


def test_annotate_still_prints_pending_review_as_a_separate_priority_line(
    tmp_path, monkeypatch, capsys
):
    """pending_review is still useful as a "look at these first" line -- it
    just must not be presented as the coverage headline (see test above)."""
    monkeypatch.setattr("onboard.introspect_schema", lambda dsn, **kw: MINI_TABLES)  # noqa: ARG005
    monkeypatch.setattr("onboard.sample_rows", lambda dsn, table, n=5: [])  # noqa: ARG005
    cmd_extract("postgresql://x", tmp_path)

    reply = '{"table": {"text": "mô tả", "confidence": "low"}, "columns": {}}'
    capsys.readouterr()
    cmd_annotate(tmp_path, "postgresql://x", invoke=lambda s, u: reply)  # noqa: ARG005

    out = capsys.readouterr().out
    assert f"{len(MINI_TABLES)} mục ưu tiên xem trước" in out


# --- I3(a): one bad table's sampling must not kill a 150-table run ---------


def test_a_sampling_failure_on_one_table_does_not_abort_the_whole_run(
    tmp_path, monkeypatch, capsys
):
    """Trước fix: `sample_rows` lỗi ở BẤT KỲ bảng nào ném ra khỏi một dict
    comprehension chạy xong hẳn trước `annotate_schema`, huỷ toàn bộ lượt
    annotate. Một role thiếu quyền SELECT trên một bảng, hay một lần mất
    kết nối thoáng qua, không được phép giết 149 bảng còn lại."""
    monkeypatch.setattr("onboard.introspect_schema", lambda dsn, **kw: MINI_TABLES)  # noqa: ARG005
    cmd_extract("postgresql://x", tmp_path)

    def flaky_sample_rows(dsn, table, n=5):  # noqa: ARG001
        if table == "payroll":
            raise psycopg2.OperationalError("permission denied for table payroll")
        return []

    monkeypatch.setattr("onboard.sample_rows", flaky_sample_rows)

    reply = '{"table": {"text": "mô tả", "confidence": "high"}, "columns": {}}'
    capsys.readouterr()
    # Must not raise -- the whole point of the fix.
    merged = cmd_annotate(tmp_path, "postgresql://x", invoke=lambda s, u: reply)  # noqa: ARG005

    # Every table, including the one whose sampling failed, still got
    # annotated -- empty samples are a supported `annotate_schema` input.
    assert set(merged.tables) == {t.name for t in MINI_TABLES}
    assert merged.tables["payroll"].text == "mô tả"

    out = capsys.readouterr().out
    assert "Lấy mẫu thất bại: 1/" in out


def test_a_sampling_failure_does_not_leave_that_table_without_annotation_input(
    tmp_path, monkeypatch
):
    """Bảng lấy mẫu lỗi phải nhận mẫu RỖNG (input hợp lệ cho
    annotate_schema), không phải bị loại khỏi dict `samples` hoàn toàn."""
    monkeypatch.setattr("onboard.introspect_schema", lambda dsn, **kw: MINI_TABLES)  # noqa: ARG005

    from onboard import _sample_all_tables

    def flaky_sample_rows(dsn, table, n=5):  # noqa: ARG001
        if table == "payroll":
            raise psycopg2.OperationalError("connection reset")
        return [{"id": 1}]

    monkeypatch.setattr("onboard.sample_rows", flaky_sample_rows)

    samples, failures = _sample_all_tables("postgresql://x", MINI_TABLES)
    assert failures == 1
    assert samples["payroll"] == []
    assert samples["orders"] == [{"id": 1}]


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


# ── chạy sai thứ tự lệnh ────────────────────────────────────────────────────


def test_a_missing_structure_file_says_to_run_extract_first(tmp_path):
    """Chạy sai thứ tự lệnh là việc đầu tiên người vận hành làm sai.

    Trước đây `FileNotFoundError` thoát ra nguyên dạng, nên họ nhận một
    trang traceback Python thay vì câu chỉ đúng việc phải làm — trong khi
    trang Streamlit vốn đã xử lý đúng tình huống này.
    """
    from onboard import OnboardError, _load_structure

    with pytest.raises(OnboardError) as excinfo:
        _load_structure(tmp_path / "chua-extract")

    message = str(excinfo.value)
    assert "structure.json" in message
    assert "extract" in message, "phải nói rõ lệnh cần chạy, không chỉ nói thiếu file"


def test_the_missing_structure_message_never_carries_the_dsn(tmp_path):
    """Thông báo lỗi không được mang mật khẩu — cùng ràng buộc với profile.json."""
    from onboard import OnboardError, _load_structure

    with pytest.raises(OnboardError) as excinfo:
        _load_structure(tmp_path / "chua-extract")

    assert "postgresql://" not in str(excinfo.value)


# ── C1: annotate không được xoá công sức người khi extract hỏng ─────────────
#
# Ruling B5 phán rằng điều kiện "`fresh` phải phủ toàn bộ schema" thuộc về
# NƠI GỌI, và nó được mang vào `cmd_refresh`. Nhưng `cmd_annotate` CŨNG là
# một nơi gọi — nó là bước 2 trong công thức ba lệnh mà chính app.py in ra —
# và nó không có chốt chặn nào. Review toàn nhánh tái lập được: một
# `structure.json` rỗng biến hai tuần công sức của analyst thành `{}`, kèm
# một thông báo mang hình dạng THÀNH CÔNG.


def _profile_with_human_work(tmp_path):
    """Thư mục profile có chú giải người, và một `structure.json` RỖNG."""
    from perception.annotations import Annotation, SchemaAnnotations, save_annotations

    (tmp_path / STRUCTURE_JSON).write_text("[]", encoding="utf-8")
    save_annotations(
        SchemaAnnotations(
            tables={"orders": Annotation("NGƯỜI VIẾT", reviewed_by="human")},
            columns={"orders": {"flg_tt": Annotation("CỘT NGƯỜI", reviewed_by="human")}},
        ),
        tmp_path / "schema.yaml",
    )
    return (tmp_path / "schema.yaml").read_text(encoding="utf-8")


def test_annotate_refuses_when_structure_is_empty_but_annotations_exist(tmp_path):
    from onboard import OnboardError

    raw_before = _profile_with_human_work(tmp_path)

    with pytest.raises(OnboardError) as excinfo:
        cmd_annotate(tmp_path, "postgresql://u:p@h/db", invoke=lambda s, u: "{}")  # noqa: ARG005

    assert "structure.json" in str(excinfo.value)
    assert (tmp_path / "schema.yaml").read_text(encoding="utf-8") == raw_before, (
        "phải chặn TRƯỚC khi ghi, không phải cảnh báo sau khi đã ghi đè"
    )


def test_annotate_force_lets_a_deliberate_operator_through(tmp_path):
    """Khách xoá thật hơn nửa số bảng thì vẫn phải có đường đi tiếp."""
    _profile_with_human_work(tmp_path)

    cmd_annotate(
        tmp_path,
        "postgresql://u:p@h/db",
        invoke=lambda s, u: "{}",  # noqa: ARG005
        force=True,
    )

    assert load_annotations(tmp_path / "schema.yaml").tables == {}


def test_annotate_still_works_normally_when_the_schema_is_intact(tmp_path, monkeypatch):
    """Chốt chặn không được cản đường chạy bình thường."""
    monkeypatch.setattr("onboard.introspect_schema", lambda dsn, **kw: MINI_TABLES)  # noqa: ARG005
    monkeypatch.setattr("onboard.sample_rows", lambda dsn, table, n=5: [])  # noqa: ARG005
    cmd_extract("postgresql://u:p@h/db", tmp_path)

    reply = '{"table": {"text": "mô tả", "confidence": "high"}, "columns": {}}'
    cmd_annotate(tmp_path, "postgresql://u:p@h/db", invoke=lambda s, u: reply)  # noqa: ARG005

    assert load_annotations(tmp_path / "schema.yaml").tables["orders"].text == "mô tả"


# ── C2: DSN gõ sai không được đẩy mật khẩu ra console ───────────────────────
#
# libpq echo NGUYÊN chuỗi kết nối khi DSN không đúng hình dạng URI:
#   invalid dsn: missing "=" after "postgres//u:pw@h/db" in connection info string
# Ngòi nổ là một lỗi gõ tầm thường, và mật khẩu production rơi vào terminal,
# scrollback, log CI, rồi ticket hỗ trợ — đúng kênh mà profile_store đã bịt,
# chỉ khác một tầng.

_SECRET = "hunter2SECRET"


@pytest.mark.parametrize(
    "bad_dsn",
    [
        f"postgres//u:{_SECRET}@h/db",  # thiếu dấu hai chấm
        f"u:{_SECRET}@h/db",  # không có scheme
        f"postgresql://u:{_SECRET}@",  # không có host
    ],
)
def test_a_malformed_dsn_is_rejected_without_echoing_the_password(bad_dsn):
    from onboard import OnboardError, _resolve_dsn

    with pytest.raises(OnboardError) as excinfo:
        _resolve_dsn(bad_dsn)

    message = str(excinfo.value)
    assert _SECRET not in message, "mật khẩu không được xuất hiện trong thông báo lỗi"
    assert bad_dsn not in message, "cả DSN cũng không được in ra"
    assert "ADBA_DSN" in message, "phải nói rõ kiểm biến môi trường nào"


def test_a_well_formed_dsn_passes_through_unchanged():
    from onboard import _resolve_dsn

    good = f"postgresql://u:{_SECRET}@h:5432/db"
    assert _resolve_dsn(good) == good
