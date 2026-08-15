from perception.schema_model import Column, Table, tables_from_info_box


def _info_box() -> dict:
    return {
        "domain": "test",
        "tables": [
            {
                "table_name": "customers",
                "row_count": 500,
                "primary_key": ["id"],
                "columns": [
                    {"name": "id", "data_type": "integer", "is_generated": False},
                    {"name": "name", "data_type": "character varying", "is_generated": False},
                ],
                "foreign_keys": [],
                "sample_rows": [{"id": 1, "name": "bí mật của khách"}],
            },
            {
                "table_name": "orders",
                "row_count": 29830,
                "primary_key": ["id"],
                "columns": [
                    {"name": "id", "data_type": "integer", "is_generated": False},
                    {"name": "customer_id", "data_type": "integer", "is_generated": False},
                    {"name": "year", "data_type": "smallint", "is_generated": True},
                ],
                "foreign_keys": [{"column": "customer_id", "references": "customers.id"}],
                "sample_rows": [],
            },
        ],
        "cross_domain_hints": [
            {"from_table": "orders", "from_column": "id",
             "to_table": "customers", "to_column": "id", "note": "bịa cho test"},
        ],
    }


def test_maps_basic_fields():
    tables = tables_from_info_box(_info_box())
    by_name = {t.name: t for t in tables}
    assert set(by_name) == {"customers", "orders"}
    assert by_name["customers"].row_count == 500
    assert by_name["customers"].primary_key == ("id",)


def test_foreign_keys_render_as_table_paren_column():
    tables = tables_from_info_box(_info_box())
    orders = next(t for t in tables if t.name == "orders")
    assert orders.foreign_keys["customer_id"] == "customers(id)"


def test_cross_domain_hints_become_foreign_keys():
    tables = tables_from_info_box(_info_box())
    orders = next(t for t in tables if t.name == "orders")
    assert orders.foreign_keys["id"] == "customers(id)"


def test_generated_columns_are_flagged():
    tables = tables_from_info_box(_info_box())
    orders = next(t for t in tables if t.name == "orders")
    year = next(c for c in orders.columns if c.name == "year")
    assert year.is_generated is True


def test_sample_rows_are_never_carried_into_the_model():
    """Tiêu chí 9 của spec: dữ liệu thật không được lọt vào profile."""
    tables = tables_from_info_box(_info_box())
    blob = repr(tables)
    assert "sample_rows" not in blob
    assert "bí mật của khách" not in blob


def test_tables_are_immutable():
    t = Table(name="x", columns=(Column("a", "integer"),))
    try:
        t.name = "y"
    except Exception as exc:
        assert isinstance(exc, (AttributeError, TypeError))
    else:
        raise AssertionError("Table phải bất biến")
