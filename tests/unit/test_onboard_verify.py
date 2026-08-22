import json

import pytest

from onboard import OnboardError, cmd_verify
from perception.connection_profile import ALL_TABLES
from perception.profile_store import write_profile
from perception.annotations import SchemaAnnotations
from perception.schema_model import Column, Table
from tests.fixtures.mini_schema import MINI_TABLES

GOLDEN = [
    {"question": "doanh thu theo khách", "sql": "SELECT * FROM orders JOIN customers ON 1=1"},
    {"question": "lương nhân viên", "sql": "SELECT * FROM payroll"},
]


def _write_profile(tmp_path):
    write_profile(tmp_path, dsn="postgresql://x", tables=MINI_TABLES,
                  annotations=SchemaAnnotations(),
                  grants={"admin": frozenset({ALL_TABLES})})


def _setup(tmp_path):
    _write_profile(tmp_path)
    golden = tmp_path / "golden.jsonl"
    golden.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in GOLDEN),
                      encoding="utf-8")
    return golden


def _write_golden(tmp_path, name, lines):
    """lines: list of raw strings, written one per line, in order given."""
    path = tmp_path / name
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_reports_recall_over_the_golden_set(tmp_path):
    report = cmd_verify(tmp_path, _setup(tmp_path), user="admin")
    assert report.total == 2
    assert 0.0 <= report.recall <= 1.0


def test_full_mode_reaches_perfect_recall(tmp_path):
    """4 bảng nằm dưới ngưỡng nên profile ở chế độ full — mọi bảng đều có mặt."""
    report = cmd_verify(tmp_path, _setup(tmp_path), user="admin")
    assert report.recall == 1.0
    assert report.passed is True


def test_a_user_without_grants_fails_every_question(tmp_path):
    golden = _setup(tmp_path)
    report = cmd_verify(tmp_path, golden, user="nguoi_la")
    assert report.recall == 0.0
    assert report.passed is False


def test_misses_name_the_tables_that_were_absent(tmp_path):
    golden = _setup(tmp_path)
    report = cmd_verify(tmp_path, golden, user="nguoi_la")
    assert any("orders" in missing for _, missing in report.misses)


def test_writes_a_markdown_report(tmp_path):
    cmd_verify(tmp_path, _setup(tmp_path), user="admin")
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "recall" in text.lower()


def test_the_report_states_the_handover_threshold(tmp_path):
    cmd_verify(tmp_path, _setup(tmp_path), user="admin")
    assert "95" in (tmp_path / "report.md").read_text(encoding="utf-8")


# --- Amendment 1: report.md must not overstate what it measured -----------


def test_unscoreable_lines_are_disclosed_in_the_report(tmp_path):
    """SQL mẫu không parse ra bảng nào ('SELECT 1') là lỗi dữ liệu, bị
    measure_recall bỏ qua — nhưng report.md phải nói rõ có bao nhiêu dòng
    golden đọc được so với bao nhiêu câu thực sự chấm được, và tại sao.
    """
    _write_profile(tmp_path)
    golden = _write_golden(tmp_path, "golden.jsonl", [
        json.dumps({"question": "câu không parse được bảng", "sql": "SELECT 1"},
                   ensure_ascii=False),
        json.dumps({"question": "doanh thu theo khách",
                    "sql": "SELECT * FROM orders JOIN customers ON 1=1"},
                   ensure_ascii=False),
    ])

    report = cmd_verify(tmp_path, golden, user="admin")
    assert report.total == 1  # only the scoreable question counted

    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "2" in text  # lines read
    assert "1" in text  # questions scored
    # the reason must be spelled out, not just numbers side by side
    assert "parse" in text.lower() or "lỗi dữ liệu" in text.lower()


def test_report_omits_the_disclosure_when_every_line_is_scoreable(tmp_path):
    cmd_verify(tmp_path, _setup(tmp_path), user="admin")
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "lỗi dữ liệu" not in text.lower()


def test_a_golden_line_missing_the_question_field_names_the_line_number(tmp_path):
    _write_profile(tmp_path)
    golden = _write_golden(tmp_path, "golden.jsonl", [
        json.dumps({"question": "câu hợp lệ", "sql": "SELECT * FROM orders"},
                   ensure_ascii=False),
        json.dumps({"sql": "SELECT * FROM payroll"}, ensure_ascii=False),
    ])

    with pytest.raises(OnboardError) as excinfo:
        cmd_verify(tmp_path, golden, user="admin")
    assert "2" in str(excinfo.value)


def test_a_golden_line_missing_the_sql_field_names_the_line_number(tmp_path):
    _write_profile(tmp_path)
    golden = _write_golden(tmp_path, "golden.jsonl", [
        json.dumps({"question": "thiếu sql"}, ensure_ascii=False),
    ])

    with pytest.raises(OnboardError) as excinfo:
        cmd_verify(tmp_path, golden, user="admin")
    assert "1" in str(excinfo.value)


def test_a_missing_golden_file_is_an_onboard_error_not_a_traceback(tmp_path):
    _write_profile(tmp_path)
    with pytest.raises(OnboardError):
        cmd_verify(tmp_path, tmp_path / "does-not-exist.jsonl", user="admin")


# --- Amendment 2: truncated misses must say how many were omitted ---------


def test_misses_beyond_twenty_are_counted_not_silently_dropped(tmp_path):
    _write_profile(tmp_path)
    rows = [{"question": f"câu {i}", "sql": "SELECT * FROM orders"} for i in range(25)]
    golden = _write_golden(
        tmp_path, "golden.jsonl",
        [json.dumps(r, ensure_ascii=False) for r in rows],
    )

    report = cmd_verify(tmp_path, golden, user="nguoi_la")
    assert len(report.misses) == 25

    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    # M1: the old assertion here was `assert "5" in text`, meant to prove
    # the truncation line "... và 5 câu khác" exists. It passed regardless
    # of that line, because the handover threshold line already contains
    # "95%" -- "5" always appears in the report whether or not truncation
    # is reported correctly. Assert the actual sentence onboard.py emits.
    assert "… và 5 câu khác không hiện ở đây." in text


# --- Fix round 1, Important 1: verify before build must not traceback -----


def test_verify_before_build_raises_onboard_error_naming_the_next_step(tmp_path):
    """No profile/ has been built yet — `read_profile` would raise a bare
    `FileNotFoundError`. `verify` is the LAST pipeline step, so running it
    early (or pointing `--profile` at the wrong directory) is one of the
    most likely operator mistakes there is; it must get the same
    `OnboardError` treatment `_load_structure` gives the same mistake for
    `annotate`/`build`, not a Python traceback.
    """
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        json.dumps({"question": "x", "sql": "SELECT * FROM orders"}, ensure_ascii=False),
        encoding="utf-8",
    )
    with pytest.raises(OnboardError) as excinfo:
        cmd_verify(tmp_path, golden, user="admin")
    msg = str(excinfo.value)
    assert "build" in msg
    assert str(tmp_path) in msg


def test_a_golden_path_pointing_at_a_directory_is_an_onboard_error(tmp_path):
    """`--golden` pointing at a directory (not a file) `.exists()`s but
    `.read_text()` on it raises `IsADirectoryError` — the same class of
    bare-traceback bug as the missing-file case, just one guard away.
    """
    _write_profile(tmp_path)
    golden_dir = tmp_path / "golden_as_dir"
    golden_dir.mkdir()
    with pytest.raises(OnboardError):
        cmd_verify(tmp_path, golden_dir, user="admin")


# --- Fix round 1, Minor 3: a bare scalar golden line must not TypeError ---


def test_a_bare_scalar_golden_line_names_the_line_number(tmp_path):
    """A line like `42` is valid JSON but not an object — `"question" not
    in row` on an int raises `TypeError`, not `OnboardError`. Must be
    caught and reported with the line number like every other malformed
    line.
    """
    _write_profile(tmp_path)
    golden = _write_golden(tmp_path, "golden.jsonl", [
        json.dumps({"question": "câu hợp lệ", "sql": "SELECT * FROM orders"},
                   ensure_ascii=False),
        "42",
    ])
    with pytest.raises(OnboardError) as excinfo:
        cmd_verify(tmp_path, golden, user="admin")
    assert "2" in str(excinfo.value)


# --- Fix round 1, Important 2: grants vs. annotation as the miss cause ----


def test_report_names_the_user_and_points_at_grant_when_permitted_is_empty(tmp_path):
    """`nguoi_la` has no entry in `grants` at all — `permitted_tables`
    returns the empty set, so every scored question misses every table.
    The report must say plainly that this is a grants problem, name the
    user, point at `--grant`, and must NOT send the operator to the
    annotation page — the tables are perfectly well annotated.
    """
    golden = _setup(tmp_path)
    cmd_verify(tmp_path, golden, user="nguoi_la")
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "nguoi_la" in text
    assert "--grant" in text
    assert "Chú giải schema" not in text


def test_report_names_the_specific_table_blocked_by_partial_grants(tmp_path):
    """`limited` is granted `customers` and `products` but not `orders`.
    The only golden question needs `orders` — a real table that exists in
    the schema but sits outside this user's grants. Every miss here is
    fully explained by grants, so the report must name `orders` as a
    grants problem and must NOT show the generic annotation guidance.
    """
    write_profile(tmp_path, dsn="postgresql://x", tables=MINI_TABLES,
                  annotations=SchemaAnnotations(),
                  grants={"limited": frozenset({"customers", "products"})})
    golden = _write_golden(tmp_path, "golden.jsonl", [
        json.dumps({"question": "đơn hàng gần đây", "sql": "SELECT * FROM orders"},
                   ensure_ascii=False),
    ])
    report = cmd_verify(tmp_path, golden, user="limited")
    assert report.recall == 0.0
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "orders" in text
    assert "quyền" in text
    assert "Chú giải schema" not in text


def test_report_keeps_annotation_guidance_when_grants_do_not_fully_explain_the_misses(tmp_path):
    """One miss (`orders`) is a grants problem for `limited`; the other
    references a table (`ghost_table`) that doesn't exist in the schema at
    all — that one is NOT explained by grants, so the generic annotation
    guidance is still potentially relevant and must stay, alongside a note
    naming the grants-caused table.
    """
    write_profile(tmp_path, dsn="postgresql://x", tables=MINI_TABLES,
                  annotations=SchemaAnnotations(),
                  grants={"limited": frozenset({"customers", "products"})})
    golden = _write_golden(tmp_path, "golden.jsonl", [
        json.dumps({"question": "đơn hàng gần đây", "sql": "SELECT * FROM orders"},
                   ensure_ascii=False),
        json.dumps({"question": "bảng không tồn tại", "sql": "SELECT * FROM ghost_table"},
                   ensure_ascii=False),
    ])
    cmd_verify(tmp_path, golden, user="limited")
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "orders" in text
    assert "quyền" in text
    assert "Chú giải schema" in text


# --- Identifier-case bug found against a real BIRD Postgres database ------
#
# A schema like BIRD's formula1 has camelCase tables (`lapTimes`). Golden
# SQL against it is written unquoted (`SELECT * FROM lapTimes`), which
# tables_in_sql() now folds to `laptimes` per Postgres's own rule — that
# never matches the catalog's `lapTimes`, so the miss shows up here exactly
# like a genuine gap. Sending the operator to fix annotations for a table
# that is already perfectly annotated wastes their time; the report must
# name identifier case as the cause instead, the same way it already does
# for the empty-grants case.

_CAMEL_TABLES = MINI_TABLES + (
    Table(name="lapTimes", columns=(Column("raceId", "integer"),),
          primary_key=("raceId",), description="Thời gian mỗi vòng đua"),
)


def _write_camel_profile(tmp_path, grants):
    write_profile(tmp_path, dsn="postgresql://x", tables=_CAMEL_TABLES,
                  annotations=SchemaAnnotations(), grants=grants)


def test_report_flags_case_mismatch_instead_of_blaming_annotations(tmp_path):
    """`lapTimes` is granted and perfectly fine — the miss is purely a
    quoting/case artifact of the unquoted golden SQL, not an annotation gap.
    """
    _write_camel_profile(tmp_path, {"admin": frozenset({ALL_TABLES})})
    golden = _write_golden(tmp_path, "golden.jsonl", [
        json.dumps({"question": "vòng đua nhanh nhất", "sql": "SELECT * FROM lapTimes"},
                   ensure_ascii=False),
    ])

    report = cmd_verify(tmp_path, golden, user="admin")
    assert report.recall == 0.0  # confirms the miss actually happened

    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "lapTimes" in text
    assert "hoa" in text.lower() or "case" in text.lower()
    # This is not an annotation problem — must not send the operator there.
    assert "Chú giải schema" not in text


def test_report_still_gives_annotation_guidance_when_the_missing_table_truly_does_not_exist(tmp_path):
    """`ghost_table` has no case-insensitive match anywhere in the profile
    — this is a genuine gap, not a case artifact, so the ordinary
    annotation guidance must still appear and no case-mismatch text must
    be fabricated for it.
    """
    _write_camel_profile(tmp_path, {"admin": frozenset({ALL_TABLES})})
    golden = _write_golden(tmp_path, "golden.jsonl", [
        json.dumps({"question": "bảng không tồn tại", "sql": "SELECT * FROM ghost_table"},
                   ensure_ascii=False),
    ])

    report = cmd_verify(tmp_path, golden, user="admin")
    assert report.recall == 0.0

    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Chú giải schema" in text
    assert "hoa" not in text.lower()
