import pytest

from perception.connection_profile import (
    ALL_TABLES,
    build_profile,
    permitted_tables,
    schema_fingerprint,
)
from perception.schema_model import Column, Table
from tests.fixtures.mini_schema import MINI_TABLES

DSN = "postgresql://u:p@localhost:5432/db"


def _profile(grants=None, tables=MINI_TABLES):
    return build_profile(dsn=DSN, tables=tables, grants=grants or {})


def test_unknown_user_gets_nothing():
    """Mặc định đóng. Không có grant nghĩa là không thấy gì."""
    assert permitted_tables(_profile(), "nguoi_la") == frozenset()


def test_wildcard_grant_gives_every_table():
    p = _profile({"admin": frozenset({ALL_TABLES})})
    assert permitted_tables(p, "admin") == {t.name for t in MINI_TABLES}


def test_explicit_grant_gives_exactly_those_tables():
    p = _profile({"sales": frozenset({"orders", "customers"})})
    assert permitted_tables(p, "sales") == frozenset({"orders", "customers"})


def test_grant_naming_a_table_that_does_not_exist_is_dropped():
    p = _profile({"sales": frozenset({"orders", "bang_khong_ton_tai"})})
    assert permitted_tables(p, "sales") == frozenset({"orders"})


def test_payroll_is_not_reachable_without_an_explicit_grant():
    p = _profile({"sales": frozenset({"orders", "customers", "products"})})
    assert "payroll" not in permitted_tables(p, "sales")


def test_fingerprint_is_stable_across_calls():
    assert schema_fingerprint(MINI_TABLES) == schema_fingerprint(MINI_TABLES)


def test_fingerprint_ignores_table_ordering():
    assert schema_fingerprint(MINI_TABLES) == schema_fingerprint(tuple(reversed(MINI_TABLES)))


def test_fingerprint_changes_when_a_column_is_added():
    before = schema_fingerprint(MINI_TABLES)
    grown = list(MINI_TABLES)
    t = grown[0]
    grown[0] = Table(
        name=t.name,
        columns=t.columns + (Column("phone", "character varying"),),
        primary_key=t.primary_key,
        foreign_keys=t.foreign_keys,
        row_count=t.row_count,
        description=t.description,
    )
    assert schema_fingerprint(tuple(grown)) != before


def test_fingerprint_ignores_row_count():
    """row_count đổi mỗi ngày; nó không phải thay đổi schema."""
    before = schema_fingerprint(MINI_TABLES)
    t = MINI_TABLES[0]
    same = (Table(name=t.name, columns=t.columns, primary_key=t.primary_key,
                  foreign_keys=t.foreign_keys, row_count=999999,
                  description=t.description),) + MINI_TABLES[1:]
    assert schema_fingerprint(same) == before


def test_small_schema_gets_full_mode():
    assert _profile().schema_mode == "full"


def test_large_schema_gets_retrieval_mode():
    many = tuple(
        Table(name=f"t{i}", columns=tuple(Column(f"c{j}", "integer") for j in range(12)))
        for i in range(200)
    )
    assert build_profile(dsn=DSN, tables=many, grants={}).schema_mode == "retrieval"


def test_threshold_is_configurable():
    assert build_profile(dsn=DSN, tables=MINI_TABLES, grants={},
                         threshold_tokens=1).schema_mode == "retrieval"
