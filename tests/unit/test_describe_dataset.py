from pathlib import Path

from eval.datasets import load_normalized
from eval.describe_dataset import describe

FIX = Path("tests/fixtures/mini_dataset")


def _records():
    return load_normalized(FIX / "questions.jsonl", FIX / "schemas.json")


def test_records_without_a_known_db_are_dropped():
    recs = _records()
    assert len(recs) == 3
    assert all(r.db_id in {"db_a", "db_b"} for r in recs)


def test_each_record_carries_its_own_schema():
    by_db = {r.db_id: r for r in _records()}
    assert {t.name for t in by_db["db_a"].tables} == {"t1", "t2"}
    assert {t.name for t in by_db["db_b"].tables} == {"u1"}


def test_describe_counts_databases_and_questions():
    p = describe("mini", _records())
    assert p.n_databases == 2
    assert p.n_questions == 3


def test_describe_reports_table_counts_per_database():
    p = describe("mini", _records())
    assert p.max_tables_per_db == 2
    assert p.avg_tables_per_db == 1.5


def test_describe_of_empty_input_does_not_divide_by_zero():
    p = describe("rỗng", [])
    assert p.n_databases == 0
    assert p.avg_tables_per_db == 0.0


def test_markdown_row_is_pipe_delimited():
    row = describe("mini", _records()).as_markdown_row()
    assert row.startswith("| mini |")
    assert row.count("|") == 7
