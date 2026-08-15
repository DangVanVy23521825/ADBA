"""
SQL execution tools — psycopg2 + pandas wrappers.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import pandas as pd
import psycopg2
import psycopg2.extras
import psycopg2.sql
import sqlparse

from perception.connection_profile import ConnectionProfile, permitted_tables
from perception.sql_tables import tables_in_sql

logger = logging.getLogger(__name__)

SQL_TIMEOUT_MS = int(os.getenv("SQL_TIMEOUT_MS", "30000"))

# Guards against injection via f-string table-name interpolation.
_TABLE_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# execute_sql là bộ thực thi chỉ-đọc: bất kỳ câu nào không phải SELECT (hoặc
# WITH ... SELECT) đều bị từ chối, bất kể tập bảng nó chạm là gì. Đây không
# phải giới hạn của parser — UPDATE/INSERT/DELETE không bao giờ được chạy qua
# đường này, kể cả khi mọi bảng liên quan đều nằm trong quyền người dùng.
_READ_ONLY_STATEMENT_TYPE = "SELECT"


class TableNotPermittedError(PermissionError):
    """Câu SQL chạm bảng ngoài quyền của người dùng."""


def assert_tables_permitted(sql: str, profile: ConnectionProfile, user: str) -> frozenset[str]:
    """Chặn trước khi chạy. Ném TableNotPermittedError nếu vi phạm.

    Tập quyền dẫn ra TẠI ĐÂY từ profile + danh tính. Nó không phải tham số
    của hàm, và không được biến thành tham số: khi tool tách qua MCP, một
    tham số như vậy sẽ nằm trong payload lời gọi, tức là bên bị ràng buộc
    tự khai ràng buộc của mình. Xem spec mục 3.4.1.

    Chỉ-đọc: bất kỳ statement nào không phải SELECT (kể cả WITH ... SELECT)
    bị từ chối ngay, không xét tới bảng nó chạm — một UPDATE mang subquery
    SELECT trong biểu thức (vd. `UPDATE payroll SET x = (SELECT y FROM
    orders)`) chỉ để lộ `orders` cho tables_in_sql, không phải `payroll` —
    bảng thật sự bị ghi. Kiểm loại statement chặn đúng lớp lỗi đó mà không
    cần dạy tables_in_sql hiểu đích UPDATE/INSERT/DELETE.

    Fail closed: SQL không parse ra bảng nào, hoặc không parse ra statement
    nào cả, cũng bị từ chối.
    """
    statements = sqlparse.parse(sql)
    if not statements or any(s.get_type() != _READ_ONLY_STATEMENT_TYPE for s in statements):
        raise TableNotPermittedError(
            "Chỉ cho phép câu SELECT (kể cả WITH ... SELECT) — "
            "câu này không parse được hoặc không chỉ đọc."
        )
    touched = tables_in_sql(sql)
    if not touched:
        raise TableNotPermittedError(
            "Không đọc được tên bảng nào từ câu SQL — từ chối để an toàn."
        )
    allowed = permitted_tables(profile, user)
    if forbidden := (touched - allowed):
        raise TableNotPermittedError(
            f"Câu hỏi này chạm dữ liệu bạn không được cấp quyền. "
            f"(nội bộ: {sorted(forbidden)})"
        )
    return touched


def execute_sql(
    sql: str,
    profile: ConnectionProfile,
    user: str,
    params: tuple | None = None,
    timeout_ms: int = SQL_TIMEOUT_MS,
) -> pd.DataFrame:
    """Chạy một câu SELECT, trả DataFrame.

    Kiểm quyền trước khi mở kết nối. Không có tham số nào cho phép bên gọi
    nới rộng tập bảng được chạm (spec tiêu chí 11).
    """
    assert_tables_permitted(sql, profile, user)
    conn = psycopg2.connect(profile.dsn)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SET LOCAL statement_timeout = %s", (timeout_ms,))
            cur.execute(sql, params)
            rows = cur.fetchall()
            if not rows:
                columns = [desc[0] for desc in cur.description] if cur.description else []
                return pd.DataFrame(columns=columns)
            return pd.DataFrame(rows)
    finally:
        conn.close()


def get_table_sample(
    table: str,
    profile: ConnectionProfile,
    user: str,
    limit: int = 5,
) -> pd.DataFrame:
    """Return a sample of rows from `table`, gated by the caller's permissions."""
    if not _TABLE_NAME_RE.match(table):
        raise ValueError(f"Tên bảng không hợp lệ: {table!r}")
    if not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    query = f"SELECT * FROM {table} LIMIT %s"  # noqa: S608 — tên bảng đã qua regex
    return execute_sql(query, profile=profile, user=user, params=(limit,))


def explain_query_plan(sql: str, profile: ConnectionProfile, user: str) -> dict[str, Any]:
    """Run EXPLAIN on a SQL query and return the plan as a dict.

    Same guard as execute_sql, same profile-derived DSN — EXPLAIN reveals
    table/index names via the plan text, so it is not exempt from the
    permission check just because it doesn't return rows.
    """
    assert_tables_permitted(sql, profile, user)
    conn = psycopg2.connect(profile.dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("EXPLAIN " + sql)
            rows = cur.fetchall()
            plan_lines = [row[0] for row in rows]
            return {
                "plan": plan_lines,
                "total_cost_estimate": _extract_cost(plan_lines),
            }
    except Exception as exc:
        return {"error": str(exc), "plan": []}
    finally:
        conn.close()


def _extract_cost(lines: list[str]) -> float | None:
    """Attempt to extract the top-level cost from EXPLAIN output."""
    for line in lines:
        if "cost=" in line:
            cost_part = line.split("cost=")[1].split("..")[1].split(" ")[0]
            try:
                return float(cost_part)
            except ValueError:
                return None
    return None
