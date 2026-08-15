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
    """Tiêu chí 3 của spec."""
    out = render_schema(MINI_TABLES)
    assert len(out.encode()) / len(MINI_TABLES) <= 700


def test_output_is_deterministic():
    assert render_schema(MINI_TABLES) == render_schema(MINI_TABLES)


def test_empty_input_gives_empty_string():
    assert render_schema([]) == ""


def test_estimate_tokens_is_quarter_of_length():
    assert estimate_tokens("a" * 400) == 100
