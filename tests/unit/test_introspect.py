import os

import pytest

from perception.introspect import introspect_schema, sample_rows

DSN = os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")
pytestmark = pytest.mark.skipif(not DSN, reason="cần DATABASE_URL trỏ Postgres thật")


def test_finds_every_public_table():
    names = {t.name for t in introspect_schema(DSN)}
    assert {"orders", "customers", "products", "payroll"} <= names


def test_columns_carry_name_and_type():
    orders = next(t for t in introspect_schema(DSN) if t.name == "orders")
    by_name = {c.name: c for c in orders.columns}
    assert "id" in by_name
    assert by_name["id"].data_type


def test_primary_key_is_detected():
    orders = next(t for t in introspect_schema(DSN) if t.name == "orders")
    assert orders.primary_key


def test_foreign_keys_render_as_table_paren_column():
    orders = next(t for t in introspect_schema(DSN) if t.name == "orders")
    assert any(ref.startswith("customers(") for ref in orders.foreign_keys.values())


def test_generated_columns_are_flagged():
    orders = next(t for t in introspect_schema(DSN) if t.name == "orders")
    assert any(c.is_generated for c in orders.columns), "orders có cột GENERATED"


def test_descriptions_start_empty():
    """Introspect chỉ ra cấu trúc. Ý nghĩa đến từ bước chú giải.

    Column chưa có field `description` (Task 2 sẽ thêm) nên ở đây chỉ kiểm
    tra Table.description; xem task-1-report.md.
    """
    tables = introspect_schema(DSN)
    assert all(t.description == "" for t in tables)


def test_sample_rows_returns_dicts_capped_at_n():
    rows = sample_rows(DSN, "customers", n=3)
    assert len(rows) <= 3
    assert all(isinstance(r, dict) for r in rows)


def test_sample_rows_rejects_a_table_name_that_is_not_an_identifier():
    with pytest.raises(ValueError):
        sample_rows(DSN, "orders; DROP TABLE customers", n=1)


def test_sample_rows_rejects_a_table_name_with_trailing_newline():
    """re.match + '$' matches just before a trailing '\\n' in Python — a
    known footgun that would let "orders\\n" slip past re.match() even
    though it isn't a bare identifier. fullmatch() closes that gap."""
    with pytest.raises(ValueError):
        sample_rows(DSN, "orders\n", n=1)
