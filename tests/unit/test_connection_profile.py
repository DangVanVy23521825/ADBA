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


def test_grant_present_but_empty_gives_nothing():
    """Có mục trong grants nhưng tập quyền rỗng vẫn đi vào nhánh mặc định đóng."""
    p = _profile({"sales": frozenset()})
    assert permitted_tables(p, "sales") == frozenset()


def test_grants_mapping_cannot_be_widened_in_place():
    """profile.grants phải bị khóa cứng — không ai chèn thêm user sau khi build.

    dataclass(frozen=True) chỉ chặn `profile.grants = ...`, không chặn
    `profile.grants["x"] = ...` trên dict thường bên trong. Phải dùng
    MappingProxyType để cả thao tác tại chỗ cũng bị chặn — cùng cách
    Table.foreign_keys tự khóa ở schema_model.py.
    """
    p = _profile({"sales": frozenset({"orders"})})
    with pytest.raises(TypeError):
        p.grants["attacker"] = frozenset({ALL_TABLES})


def test_mutating_callers_original_grant_after_build_does_not_change_access():
    """Caller mutate tập quyền gốc sau khi build_profile trả về không được rò vào profile.

    Nếu build_profile chỉ lưu tham chiếu tới đúng object của caller (không
    convert sang frozenset của riêng mình), caller mutate set gốc thì
    permitted_tables đọc thấy ngay — không cần build lại profile, guard bảo
    mật tụt xuống lời hứa.
    """
    original = {"orders"}
    grants = {"sales": original}
    p = build_profile(dsn=DSN, tables=MINI_TABLES, grants=grants)
    assert permitted_tables(p, "sales") == frozenset({"orders"})

    original.add("payroll")

    assert permitted_tables(p, "sales") == frozenset({"orders"})


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


def test_fingerprint_changes_when_a_column_becomes_generated():
    """IMPORTANT 5 (final review): is_generated is not cosmetic.

    A column becoming GENERATED changes what SQL is valid against it (the
    SQL prompt template has a rule about exactly that: GENERATED columns are
    read-only). A DBA making that change must invalidate the profile, same
    as adding a column does — unlike row_count/description, which are
    correctly excluded because they're volatile/cosmetic and not schema.
    """
    before = schema_fingerprint(MINI_TABLES)
    t = MINI_TABLES[0]
    flipped_col = Column(t.columns[0].name, t.columns[0].data_type, is_generated=True)
    changed = (Table(name=t.name, columns=(flipped_col,) + t.columns[1:],
                      primary_key=t.primary_key, foreign_keys=t.foreign_keys,
                      row_count=t.row_count, description=t.description),) + MINI_TABLES[1:]
    assert schema_fingerprint(changed) != before


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
