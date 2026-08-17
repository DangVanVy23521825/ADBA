from dataclasses import replace

from perception.render_schema import estimate_tokens, render_schema
from perception.schema_model import Column, Table
from tests.fixtures.mini_schema import MINI_TABLES


def test_renders_create_table_per_table():
    out = render_schema(MINI_TABLES)
    for name in ("customers", "products", "orders", "payroll"):
        assert f"CREATE TABLE {name} (" in out


def test_primary_key_is_marked():
    out = render_schema([MINI_TABLES[0]])
    assert "id INT PRIMARY KEY" in out


def test_foreign_key_renders_as_references():
    orders = next(t for t in MINI_TABLES if t.name == "orders")
    out = render_schema([orders])
    assert "REFERENCES customers(id)" in out
    assert "REFERENCES products(id)" in out


def test_generated_column_is_marked():
    orders = next(t for t in MINI_TABLES if t.name == "orders")
    assert "GENERATED" in render_schema([orders])


def test_description_renders_as_leading_comment():
    out = render_schema([MINI_TABLES[0]])
    assert "-- Khách hàng doanh nghiệp và cá nhân" in out


def test_verbose_postgres_types_are_shortened():
    out = render_schema(MINI_TABLES)
    assert "character varying" not in out
    assert "VARCHAR" in out


def test_unknown_type_is_passed_through_uppercased():
    t = Table(name="x", columns=(Column("c", "jsonb"),))
    assert "JSONB" in render_schema([t])


def test_stays_under_700_bytes_per_table():
    """Spec criterion 3: DDL must not exceed 700 bytes per table.

    This asserts the hard spec constraint. For drift detection, see
    test_fixture_output_stays_compact().
    """
    out = render_schema(MINI_TABLES)
    assert len(out.encode()) / len(MINI_TABLES) <= 700


def test_fixture_output_stays_compact():
    """Drift guard: detect regressions before they reach spec ceiling.

    Current fixture produces 697 bytes / 4 tables = ~174 B/table.
    Threshold of 250 B/table catches ~1.4x bloat (e.g., stray duplicated
    comments, reverting to verbose type names) while allowing reasonable
    organic growth. Measured baseline: 697 bytes total for MINI_TABLES.
    """
    out = render_schema(MINI_TABLES)
    assert len(out.encode()) <= 750  # ~187 B/table, ~1.1x current


def test_realistic_schema_stays_under_700_bytes_per_table():
    """Realistic case: longer descriptions and more columns.

    Fixture has short Vietnamese descriptions and 3-6 columns per table.
    Real schemas have 8-15 columns and longer descriptions. Verify the
    spec constraint still holds for realistic content.
    """
    # Enhance fixture with longer descriptions and more columns
    realistic = (
        replace(
            MINI_TABLES[0],
            description="Danh sách khách hàng doanh nghiệp và cá nhân tham gia hệ thống bán hàng, được phân loại theo segment và mức độ hoạt động",
            columns=MINI_TABLES[0].columns + (
                Column("email", "character varying"),
                Column("phone", "character varying"),
                Column("address", "text"),
                Column("created_at", "timestamp without time zone"),
                Column("updated_at", "timestamp without time zone"),
            ),
        ),
        replace(
            MINI_TABLES[1],
            description="Danh mục sản phẩm bán ra gồm các mặt hàng khác nhau với giá cả được tính theo quy tắc chiết khấu theo khối lượng đơn hàng",
            columns=MINI_TABLES[1].columns + (
                Column("sku", "character varying"),
                Column("description", "text"),
                Column("stock_quantity", "integer"),
                Column("reorder_level", "integer"),
                Column("supplier_id", "integer"),
            ),
        ),
        MINI_TABLES[2],
        MINI_TABLES[3],
    )
    out = render_schema(realistic)
    assert len(out.encode()) / len(realistic) <= 700


def test_output_is_deterministic():
    assert render_schema(MINI_TABLES) == render_schema(MINI_TABLES)


def test_empty_input_gives_empty_string():
    assert render_schema([]) == ""


def test_estimate_tokens_is_quarter_of_length():
    assert estimate_tokens("a" * 400) == 100


def test_column_description_renders_as_a_trailing_comment():
    t = Table(
        name="orders",
        columns=(
            Column("id", "integer"),
            Column("flg_tt", "boolean", description="Cờ đã thanh toán"),
        ),
    )
    out = render_schema([t])
    assert "flg_tt BOOL  -- Cờ đã thanh toán" in out


def test_column_without_a_description_gets_no_comment():
    t = Table(name="x", columns=(Column("a", "integer"),))
    assert "--" not in render_schema([t])


def test_column_description_does_not_break_the_700_byte_ceiling():
    """Chú giải làm bảng to lên; trần vẫn phải giữ."""
    import dataclasses

    from tests.fixtures.mini_schema import MINI_TABLES

    fat = tuple(
        dataclasses.replace(
            t,
            columns=tuple(
                dataclasses.replace(c, description="Mô tả nghiệp vụ dài vừa phải cho cột này")
                for c in t.columns
            ),
        )
        for t in MINI_TABLES
    )
    out = render_schema(fat)
    assert len(out.encode()) / len(fat) <= 700
