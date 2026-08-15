import inspect

import pytest

from graph.tools import sql_tool
from perception.connection_profile import ALL_TABLES, build_profile
from tests.fixtures.mini_schema import MINI_TABLES


def _profile(grants):
    return build_profile(dsn="postgresql://u:p@h:5432/d", tables=MINI_TABLES, grants=grants)


def test_query_touching_a_forbidden_table_is_refused():
    p = _profile({"sales": frozenset({"orders"})})
    with pytest.raises(sql_tool.TableNotPermittedError, match="payroll"):
        sql_tool.assert_tables_permitted("SELECT * FROM payroll", p, "sales")


def test_query_within_permitted_tables_is_accepted():
    p = _profile({"sales": frozenset({"orders", "customers"})})
    sql_tool.assert_tables_permitted(
        "SELECT * FROM orders o JOIN customers c ON o.customer_id = c.id", p, "sales")


def test_permitted_but_not_retrieved_is_allowed():
    """spec 3.4.1: retrieval không phải cơ chế bảo mật.

    Model đoán đúng tên một bảng mà retriever bỏ sót thì câu đó phải chạy,
    vì bảng đó vốn nằm trong quyền của người dùng.
    """
    p = _profile({"admin": frozenset({ALL_TABLES})})
    sql_tool.assert_tables_permitted("SELECT * FROM payroll", p, "admin")


def test_user_without_grants_is_refused_everything():
    p = _profile({})
    with pytest.raises(sql_tool.TableNotPermittedError):
        sql_tool.assert_tables_permitted("SELECT * FROM orders", p, "nguoi_la")


def test_cte_name_is_not_mistaken_for_a_forbidden_table():
    p = _profile({"sales": frozenset({"orders"})})
    sql_tool.assert_tables_permitted(
        "WITH tmp AS (SELECT * FROM orders) SELECT * FROM tmp", p, "sales")


def test_sql_that_parses_to_no_tables_is_refused():
    """Fail closed: không đọc được bảng nào thì từ chối, không cho qua."""
    p = _profile({"admin": frozenset({ALL_TABLES})})
    with pytest.raises(sql_tool.TableNotPermittedError):
        sql_tool.assert_tables_permitted("khong phai sql", p, "admin")


def test_execute_sql_has_no_parameter_that_widens_table_access():
    """Tiêu chí 11 của spec — kiểm bằng chữ ký hàm, không bằng quy ước."""
    params = set(inspect.signature(sql_tool.execute_sql).parameters)
    forbidden = {"allowed_tables", "permitted", "permitted_tables", "tables", "allow"}
    assert not (params & forbidden), f"execute_sql không được nhận: {params & forbidden}"


def test_module_no_longer_exposes_a_static_whitelist():
    """Tiêu chí 1 của spec: không tên bảng hardcode."""
    assert not hasattr(sql_tool, "_ALLOWED_TABLES")
