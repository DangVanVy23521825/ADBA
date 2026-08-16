import pytest

from perception.sql_tables import tables_in_sql


def test_single_table():
    assert tables_in_sql("SELECT id FROM orders") == frozenset({"orders"})


def test_join_with_aliases_returns_real_names_not_aliases():
    sql = """
        SELECT o.id, p.name
        FROM orders o
        JOIN products p ON o.product_id = p.id
    """
    assert tables_in_sql(sql) == frozenset({"orders", "products"})


def test_cte_name_is_not_a_table():
    sql = """
        WITH q4 AS (SELECT product_id, SUM(amount) AS s FROM orders GROUP BY product_id)
        SELECT p.name, q.s FROM q4 q JOIN products p ON q.product_id = p.id
    """
    # q4 là CTE, không phải bảng thật
    assert tables_in_sql(sql) == frozenset({"orders", "products"})


def test_subquery_tables_are_included():
    sql = "SELECT * FROM (SELECT id FROM employees) e JOIN payroll p ON e.id = p.employee_id"
    assert tables_in_sql(sql) == frozenset({"employees", "payroll"})


def test_schema_qualified_name_keeps_only_table():
    assert tables_in_sql("SELECT * FROM public.orders") == frozenset({"orders"})


def test_case_is_normalised_to_lower():
    assert tables_in_sql("SELECT * FROM Orders") == frozenset({"orders"})


def test_empty_or_unparseable_returns_empty_set():
    assert tables_in_sql("") == frozenset()
    assert tables_in_sql("not sql at all") == frozenset()


def test_extract_from_is_not_a_table_source():
    assert tables_in_sql("SELECT EXTRACT(MONTH FROM order_date) FROM orders") == frozenset({"orders"})


def test_trim_from_is_not_a_table_source():
    assert tables_in_sql("SELECT TRIM(BOTH ' ' FROM name) FROM customers") == frozenset({"customers"})


def test_substring_from_is_not_a_table_source():
    assert tables_in_sql("SELECT SUBSTRING(name FROM 1 FOR 3) FROM products") == frozenset({"products"})


def test_subquery_nested_inside_a_function_is_still_found():
    """Guards against 'fixing' the above by never recursing into functions."""
    sql = "SELECT COALESCE((SELECT MAX(id) FROM orders), 0) FROM customers"
    assert tables_in_sql(sql) == frozenset({"orders", "customers"})


def test_multiple_ctes_in_one_with_clause_are_not_tables():
    sql = """
        WITH a AS (SELECT * FROM orders), b AS (SELECT * FROM products)
        SELECT * FROM a JOIN b ON a.product_id = b.id
    """
    assert tables_in_sql(sql) == frozenset({"orders", "products"})


def test_cte_declared_in_one_statement_does_not_shadow_a_real_table_in_another():
    """CTE không sống qua khỏi statement khai báo nó trong PostgreSQL.

    Nếu tên CTE bị gộp toàn cục trước khi trừ (thay vì trừ theo từng
    statement), 'payroll' — bảng thật ở statement 2 — bị hiểu nhầm là CTE
    của statement 1 và biến mất khỏi kết quả. Đây chính là hình dạng SQL
    dùng để bypass permission guard qua PQexec đa-statement.
    """
    sql = "WITH payroll AS (SELECT 1 AS x) SELECT * FROM orders; SELECT * FROM payroll"
    assert tables_in_sql(sql) == frozenset({"orders", "payroll"})
