"""Bước bọc nháy định danh chạy trong SẢN PHẨM.

`tests/unit/test_sqlite_dialect.py` khoá thuật toán qua đường eval. File
này khoá những tính chất mà chỉ đường sản phẩm mới quan tâm: phải là
no-op trên schema chữ thường, và không được phá SQL vốn chạy được.
"""

import pytest

from perception.schema_model import Column, Table
from perception.sql_identifiers import SchemaNames, requote_for_tables

LOWER = (
    Table(name="orders", columns=(
        Column("id", "integer"), Column("region", "text"), Column("amount", "numeric"),
    )),
    Table(name="customers", columns=(Column("id", "integer"), Column("name", "text"))),
)

MIXED = (
    Table(name="schools", columns=(
        Column("CDSCode", "text"), Column("MailStreet", "text"), Column("Zip", "text"),
    )),
    Table(name="member", columns=(Column("member_id", "integer"), Column("zip", "text"))),
)


# --- Tính chất an toàn: schema chữ thường thì không được đụng gì ---


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT region, SUM(amount) FROM orders GROUP BY region",
        "SELECT o.id FROM orders o JOIN customers c ON o.id = c.id",
        "SELECT * FROM orders WHERE region = 'Region'",
        'SELECT "region" FROM orders',
        "WITH t AS (SELECT id FROM orders) SELECT * FROM t",
    ],
)
def test_a_lowercase_schema_is_never_touched(sql):
    """Phần lớn khách hàng có schema chữ thường. Với họ bước này phải là
    no-op TUYỆT ĐỐI — nó không thể phá thứ đang chạy được."""
    assert requote_for_tables(sql, LOWER) == sql


def test_is_noop_reports_a_lowercase_schema():
    assert SchemaNames(LOWER).is_noop() is True
    assert SchemaNames(MIXED).is_noop() is False


def test_an_empty_table_list_returns_the_sql_unchanged():
    """`SchemaContext.tables` mặc định rỗng, và có nơi dựng context thủ
    công. Rỗng thì không biết gì về schema nên không được đoán."""
    sql = "SELECT MailStreet FROM schools"
    assert requote_for_tables(sql, ()) == sql


# --- Trên schema giữ hoa thường thì phải sửa ---


def test_a_bare_mixed_case_column_gets_quoted():
    out = requote_for_tables("SELECT MailStreet FROM schools", MIXED)
    assert '"MailStreet"' in out


def test_a_qualified_name_is_resolved_through_the_alias():
    out = requote_for_tables("SELECT s.CDSCode FROM schools AS s", MIXED)
    assert '"CDSCode"' in out


def test_an_ambiguous_bare_name_is_left_alone():
    """`member.zip` và `schools.Zip` cùng tồn tại. Sửa bừa thành `"Zip"`
    sẽ biến một câu hỏi về member thành câu chạy được nhưng đọc nhầm
    bảng — hỏng âm thầm, tệ hơn hỏng ồn ào."""
    assert '"Zip"' not in requote_for_tables("SELECT zip FROM member", MIXED)


def test_a_string_literal_is_not_mistaken_for_an_identifier():
    out = requote_for_tables("SELECT 1 FROM schools WHERE Zip = 'MailStreet'", MIXED)
    assert "'MailStreet'" in out


def test_applying_it_twice_changes_nothing_further():
    once = requote_for_tables("SELECT s.CDSCode FROM schools AS s", MIXED)
    assert requote_for_tables(once, MIXED) == once


def test_the_sql_still_parses_after_rewriting():
    """Chốt tổng: dù sửa gì thì đầu ra vẫn phải là SQL đọc được."""
    import sqlparse

    out = requote_for_tables(
        "SELECT s.CDSCode, COUNT(*) FROM schools s "
        "JOIN member m ON m.zip = s.Zip GROUP BY s.CDSCode",
        MIXED,
    )
    assert sqlparse.parse(out)
    assert out.count('"CDSCode"') == 2
    assert '"Zip"' in out


# --- Thu hẹp theo các bảng của chính câu truy vấn ---


def test_a_globally_ambiguous_name_resolves_when_only_one_table_is_in_play():
    """`schools.Zip` và `member.zip` nhập nhằng trên toàn schema, nhưng
    trong một câu chỉ có `FROM schools` thì không. Đây đúng là cách
    Postgres phân giải cột không định tính."""
    assert '"Zip"' in requote_for_tables("SELECT zip FROM schools", MIXED)


def test_the_same_bare_name_stays_lowercase_for_the_other_table():
    out = requote_for_tables("SELECT zip FROM member", MIXED)
    assert '"Zip"' not in out


def test_it_stays_conservative_when_both_tables_are_joined():
    """Cả hai bảng cùng có tên đó thì chính Postgres cũng báo nhập nhằng;
    đoán hộ nó sẽ biến một lỗi rõ ràng thành một câu đọc nhầm cột."""
    out = requote_for_tables(
        "SELECT zip FROM schools JOIN member ON member.member_id = schools.CDSCode",
        MIXED,
    )
    assert '"Zip"' not in out
