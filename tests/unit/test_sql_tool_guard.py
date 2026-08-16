import inspect
from unittest.mock import patch

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


def test_explain_query_plan_checks_guard_before_connecting():
    """Same shape as test_execute_sql_checks_guard_before_connecting — mirrors
    it for explain_query_plan, whose guard call (and the switch to
    profile.dsn) had zero coverage: every existing test patches
    explain_query_plan wholesale, so deleting the guard line inside it would
    break nothing. The second assertion (mock_connect not called) is the
    substance — it proves the guard runs BEFORE the connection is opened,
    not merely that an error surfaces somewhere.
    """
    p = _profile({"sales": frozenset({"orders"})})
    with patch.object(sql_tool.psycopg2, "connect") as mock_connect:
        with pytest.raises(sql_tool.TableNotPermittedError):
            sql_tool.explain_query_plan("SELECT * FROM payroll", profile=p, user="sales")
        mock_connect.assert_not_called()


def test_execute_sql_checks_guard_before_connecting():
    """Chứng minh guard thực sự nằm trên đường thực thi, không chỉ tồn tại
    như một hàm độc lập không ai gọi.

    Nếu dòng assert_tables_permitted() bị xoá khỏi execute_sql, test này
    fail vì psycopg2.connect được gọi — không chỉ vì exception biến mất.
    """
    p = _profile({"sales": frozenset({"orders"})})
    with patch.object(sql_tool.psycopg2, "connect") as mock_connect:
        with pytest.raises(sql_tool.TableNotPermittedError):
            sql_tool.execute_sql("SELECT * FROM payroll", profile=p, user="sales")
        mock_connect.assert_not_called()


def test_bare_update_statement_is_refused_even_when_table_is_permitted():
    """execute_sql chỉ đọc: UPDATE bị chặn dù bảng đích nằm trong quyền."""
    p = _profile({"admin": frozenset({ALL_TABLES})})
    with pytest.raises(sql_tool.TableNotPermittedError):
        sql_tool.assert_tables_permitted("UPDATE payroll SET base_salary = 0", p, "admin")


def test_update_with_select_subquery_cannot_hide_the_write_target():
    """UPDATE payroll SET x = (SELECT y FROM orders) chỉ để lộ 'orders' cho
    tables_in_sql — 'payroll', bảng thật sự bị ghi, không nằm trong tập đó.

    Kiểm loại statement (không phải SELECT) phải chặn câu này trước khi tới
    bước so tập bảng, nên user chỉ được quyền trên 'orders' vẫn bị từ chối.
    """
    p = _profile({"sales": frozenset({"orders"})})
    with pytest.raises(sql_tool.TableNotPermittedError):
        sql_tool.assert_tables_permitted(
            "UPDATE payroll SET base_salary = (SELECT amount FROM orders LIMIT 1) "
            "WHERE id = 1",
            p, "sales")


def test_with_select_is_still_permitted_by_the_read_only_check():
    """WITH ... SELECT phải qua được kiểm loại statement — nó là SELECT."""
    p = _profile({"sales": frozenset({"orders"})})
    sql_tool.assert_tables_permitted(
        "WITH recent AS (SELECT * FROM orders) SELECT * FROM recent", p, "sales")


def test_multi_statement_with_any_non_select_is_refused():
    """Một câu SELECT hợp lệ đứng cạnh một UPDATE vẫn phải bị từ chối cả cụm."""
    p = _profile({"admin": frozenset({ALL_TABLES})})
    with pytest.raises(sql_tool.TableNotPermittedError):
        sql_tool.assert_tables_permitted(
            "SELECT * FROM orders; UPDATE payroll SET base_salary = 0", p, "admin")


def test_cross_statement_cte_shadowing_cannot_bypass_the_guard():
    """CRITICAL: CTE khai báo ở statement 1 không được che bảng thật ở statement 2.

    psycopg2.execute() với vars=None gửi cả chuỗi qua PQexec, chạy hết mọi
    statement và trả result set CUỐI CÙNG. Trước fix, tables_in_sql gộp tên
    CTE của MỌI statement thành một tập rồi trừ chung khỏi tập bảng của MỌI
    statement — nên 'payroll' khai CTE ở statement 1 xoá luôn 'payroll' bảng
    thật ở statement 2 khỏi tập bị chạm, và cả hai statement đều là SELECT
    nên qua được kiểm loại. Kết quả: user chỉ có quyền trên 'orders' đọc
    được toàn bộ 'payroll'. Fix chặn bằng cách đòi đúng MỘT statement.
    """
    p = _profile({"sales": frozenset({"orders"})})
    bypass_sql = "WITH payroll AS (SELECT 1 AS x) SELECT * FROM orders; SELECT * FROM payroll"
    with pytest.raises(sql_tool.TableNotPermittedError):
        sql_tool.assert_tables_permitted(bypass_sql, p, "sales")


def test_multiple_statements_are_refused_even_when_all_are_select():
    """len(statements) != 1 chặn trước cả kiểm loại — không chỉ khi có non-SELECT."""
    p = _profile({"admin": frozenset({ALL_TABLES})})
    with pytest.raises(sql_tool.TableNotPermittedError):
        sql_tool.assert_tables_permitted(
            "SELECT * FROM orders; SELECT * FROM customers", p, "admin")
