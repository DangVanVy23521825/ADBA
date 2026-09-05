import inspect
from unittest.mock import patch

import pytest

from graph.tools import sql_tool
from perception.connection_profile import ALL_TABLES, build_profile
from perception.schema_model import Column, Table
from tests.fixtures.mini_schema import MINI_TABLES


def _profile(grants, tables=MINI_TABLES):
    return build_profile(dsn="postgresql://u:p@h:5432/d", tables=tables, grants=grants)


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


# --- Identifier-case bug found against a real BIRD Postgres database ------
#
# permitted_tables() reads case-preserved catalog names (introspect_schema
# never lowercases). Before the fix, tables_in_sql() lowercased every table
# name unconditionally, so a legitimate quoted query against a camelCase
# table (`lapTimes`, `driverStandings`, ... — normal output of Entity
# Framework/Hibernate/Prisma/Rails) was refused even though the table was
# granted. The fix folds like Postgres: quoted identifiers keep case,
# unquoted ones fold to lower.

_CAMEL_TABLES = MINI_TABLES + (
    Table(name="lapTimes", columns=(Column("raceId", "integer"),),
          primary_key=("raceId",)),
)


def test_quoted_camel_case_table_is_accepted_when_granted():
    p = _profile({"sales": frozenset({"lapTimes"})}, tables=_CAMEL_TABLES)
    sql_tool.assert_tables_permitted('SELECT * FROM "lapTimes"', p, "sales")


def test_unquoted_camel_case_table_is_still_refused_because_it_folds_to_a_different_name():
    """`FROM lapTimes` (unquoted) folds to `laptimes` in real Postgres, which
    is not the granted table (`lapTimes`) — this must still be refused,
    exactly like real Postgres would fail to find the table. Fail closed:
    a table the guard cannot confidently resolve is never allowed through.
    """
    p = _profile({"sales": frozenset({"lapTimes"})}, tables=_CAMEL_TABLES)
    with pytest.raises(sql_tool.TableNotPermittedError):
        sql_tool.assert_tables_permitted("SELECT * FROM lapTimes", p, "sales")


def test_quoted_camel_case_table_is_still_refused_when_genuinely_not_granted():
    """The fix must not become a blanket bypass — a quoted table the user
    was never granted is still refused."""
    p = _profile({"sales": frozenset({"orders"})}, tables=_CAMEL_TABLES)
    with pytest.raises(sql_tool.TableNotPermittedError, match="lapTimes"):
        sql_tool.assert_tables_permitted('SELECT * FROM "lapTimes"', p, "sales")


class TestRowCap:
    """Trần dòng (spec 4.1 lớp 4). Không chạm DB thật — psycopg2 bị mock."""

    @staticmethod
    def _profile():
        from perception.connection_profile import ALL_TABLES, build_profile
        from tests.fixtures.mini_schema import MINI_TABLES

        return build_profile(
            dsn="postgresql://u:p@h:5432/d",
            tables=MINI_TABLES,
            grants={"u": frozenset({ALL_TABLES})},
        )

    @staticmethod
    def _patch_connect(rows):
        """Mock psycopg2.connect sao cho fetchmany(n) tôn trọng n."""
        from unittest.mock import MagicMock, patch

        cur = MagicMock()
        cur.description = [("id",)]
        cur.fetchmany.side_effect = lambda n: rows[:n]
        cur.__enter__ = MagicMock(return_value=cur)
        cur.__exit__ = MagicMock(return_value=False)
        conn = MagicMock()
        conn.cursor.return_value = cur
        return patch("graph.tools.sql_tool.psycopg2.connect", return_value=conn)

    def test_all_rows_come_back_when_under_the_cap(self):
        from graph.tools.sql_tool import execute_sql

        rows = [{"id": i} for i in range(5)]
        with self._patch_connect(rows):
            df = execute_sql("SELECT id FROM orders", self._profile(), "u", max_rows=10)
        assert len(df) == 5
        assert df.attrs["truncated"] is False

    def test_rows_are_capped_and_the_flag_is_set(self):
        from graph.tools.sql_tool import execute_sql

        rows = [{"id": i} for i in range(50)]
        with self._patch_connect(rows):
            df = execute_sql("SELECT id FROM orders", self._profile(), "u", max_rows=10)
        assert len(df) == 10
        assert df.attrs["truncated"] is True

    def test_the_cap_defaults_to_fifty_thousand(self):
        import importlib

        import graph.tools.sql_tool as sql_tool

        importlib.reload(sql_tool)
        assert sql_tool.SQL_MAX_ROWS == 50000

    def test_rows_stream_through_a_server_side_cursor(self):
        """Trần dòng một mình KHÔNG chặn được nổ RAM phía client.

        Cursor mặc định của psycopg2 là client-side: `cur.execute(sql)` kéo
        TRỌN result set về RAM rồi mới trả điều khiển, nên `fetchmany()`
        phía sau chỉ cắt một thứ đã nằm sẵn trong bộ nhớ. Một `SELECT *`
        trên bảng chục triệu dòng vẫn làm sập tiến trình dù trần là 50k.

        Test này kiểm THIẾT KẾ chứ không kiểm bộ nhớ: nó khẳng định
        execute_sql xin một cursor CÓ TÊN (server-side) cho câu SELECT, và
        đặt `itersize`. Hành vi streaming thật chỉ đo được với Postgres
        thật — mock không mô phỏng được nó, và một test giả vờ đo RAM sẽ
        tệ hơn là không có test.
        """
        from unittest.mock import MagicMock, patch

        from graph.tools.sql_tool import SQL_STREAM_ITERSIZE, execute_sql

        cur = MagicMock()
        cur.description = [("id",)]
        cur.fetchmany.side_effect = lambda n: [{"id": 1}][:n]
        cur.__enter__ = MagicMock(return_value=cur)
        cur.__exit__ = MagicMock(return_value=False)
        conn = MagicMock()
        conn.cursor.return_value = cur

        with patch("graph.tools.sql_tool.psycopg2.connect", return_value=conn):
            execute_sql("SELECT id FROM orders", self._profile(), "u")

        named_calls = [c for c in conn.cursor.call_args_list if c.kwargs.get("name")]
        assert named_calls, "câu SELECT phải chạy trên cursor phía server (name=...)"
        assert cur.itersize == SQL_STREAM_ITERSIZE

    def test_the_statement_timeout_is_set_on_a_plain_cursor(self):
        """`SET LOCAL` không hợp lệ trên cursor có tên — psycopg2 dịch nó
        thành `DECLARE ... CURSOR FOR SET LOCAL ...`. Nên nó phải đi trên
        một cursor thường, cùng connection, cùng transaction (không COMMIT
        ở giữa) để `LOCAL` còn phủ tới câu SELECT."""
        from unittest.mock import MagicMock, call, patch

        from graph.tools.sql_tool import execute_sql

        cur = MagicMock()
        cur.description = [("id",)]
        cur.fetchmany.side_effect = lambda n: [][:n]
        cur.__enter__ = MagicMock(return_value=cur)
        cur.__exit__ = MagicMock(return_value=False)
        conn = MagicMock()
        conn.cursor.return_value = cur

        with patch("graph.tools.sql_tool.psycopg2.connect", return_value=conn):
            execute_sql("SELECT id FROM orders", self._profile(), "u", timeout_ms=1234)

        assert call() in conn.cursor.call_args_list, "cursor thường cho SET LOCAL"
        timeout_calls = [
            c for c in cur.execute.call_args_list
            if "statement_timeout" in str(c.args[0])
        ]
        assert timeout_calls, "statement_timeout vẫn phải được đặt"
        assert timeout_calls[0].args[1] == (1234,)
        assert not conn.commit.called, "COMMIT giữa chừng sẽ giết cursor có tên"
