import json

import pytest

from onboard import OnboardError, cmd_verify
from perception.connection_profile import ALL_TABLES
from perception.profile_store import write_profile
from perception.annotations import SchemaAnnotations
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
    assert "5" in text  # 25 misses - 20 shown = 5 omitted, must be stated somewhere
