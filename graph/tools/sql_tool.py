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

# 10s, không phải 30s: trần này nằm BÊN TRONG ngân sách wall-clock 45s
# (spec mục 5). Một câu truy vấn chạy quá 10 giây trên schema BI cỡ này
# gần như luôn là join sai, và để nó chạy tiếp chỉ ăn hết phần thời gian
# lẽ ra dành cho bước insight.
SQL_TIMEOUT_MS = int(os.getenv("SQL_TIMEOUT_MS", "10000"))

# Trần dòng nạp về client. statement_timeout không cứu được trường hợp này:
# Postgres trả dòng đúng hạn, chỗ chết là fetchall() phía client nạp trọn
# result set vào RAM. Một SELECT * trên bảng chục triệu dòng làm sập tiến
# trình trước khi có ai kịp báo lỗi.
SQL_MAX_ROWS = int(os.getenv("SQL_MAX_ROWS", "50000"))

# Guards against injection via f-string table-name interpolation.
_TABLE_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# execute_sql là bộ thực thi chỉ-đọc: bất kỳ câu nào không phải SELECT (hoặc
# WITH ... SELECT) đều bị từ chối, bất kể tập bảng nó chạm là gì. Đây không
# phải giới hạn của parser — UPDATE/INSERT/DELETE không bao giờ được chạy qua
# đường này, kể cả khi mọi bảng liên quan đều nằm trong quyền người dùng.
_READ_ONLY_STATEMENT_TYPE = "SELECT"


class TableNotPermittedError(PermissionError):
    """Câu SQL chạm bảng ngoài quyền của người dùng.

    Structured thay vì text: `.public` là phần an toàn để lộ ra ngoài
    (state, trace, UI); `.forbidden` mang đúng tên các bảng bị chạm ngoài
    quyền, chỉ để log server-side. Trước bản này, ba module
    (graph/tools/sql_tool.py, graph/agents/sql_agent.py, app.py) đều tự
    parse chuỗi lỗi bằng cách split trên literal "(nội bộ:" — nếu marker đó
    đổi ở nơi sinh ra lỗi, hai nơi tiêu thụ còn lại vẫn xanh nhưng tên bảng
    cấm sẽ rò ra last_error/action_trace/UI. Đọc `.public`, không parse text.
    """

    def __init__(self, public: str, forbidden: frozenset[str] = frozenset()):
        self.public = public
        self.forbidden = forbidden
        message = f"{public} (nội bộ: {sorted(forbidden)})" if forbidden else public
        super().__init__(message)


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

    Đúng một statement — bắt buộc, trước cả kiểm loại. CTE không sống qua
    khỏi statement trong PostgreSQL, nhưng tables_in_sql cũ gom tên CTE của
    MỌI statement vào một tập rồi trừ chung khỏi tập bảng của MỌI statement:
    'WITH payroll AS (...) SELECT * FROM orders; SELECT * FROM payroll' khiến
    'payroll' ở câu 2 bị hiểu nhầm là CTE của câu 1 và biến mất khỏi tập
    chạm — trong khi cả hai statement đều là SELECT nên qua được kiểm loại,
    và psycopg2.execute() với vars=None gửi cả chuỗi qua PQexec, chạy cả hai
    và trả về result set cuối cùng (payroll). Chặn tại đây thay vì chỉ sửa
    tables_in_sql vì execute_sql là bộ thực thi một-câu-một-lần theo hợp
    đồng đã có (kiểm loại statement ở trên) — viết hợp đồng đó thành code.
    """
    statements = [s for s in sqlparse.parse(sql) if s.token_first(skip_cm=True) is not None]
    if len(statements) != 1:
        raise TableNotPermittedError("Chỉ cho phép đúng một câu SELECT.")
    if any(s.get_type() != _READ_ONLY_STATEMENT_TYPE for s in statements):
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
            "Câu hỏi này chạm dữ liệu bạn không được cấp quyền.",
            forbidden=forbidden,
        )
    return touched


def execute_sql(
    sql: str,
    profile: ConnectionProfile,
    user: str,
    params: tuple | None = None,
    timeout_ms: int = SQL_TIMEOUT_MS,
    max_rows: int = SQL_MAX_ROWS,
) -> pd.DataFrame:
    """Chạy một câu SELECT, trả DataFrame.

    Kiểm quyền trước khi mở kết nối. Không có tham số nào cho phép bên gọi
    nới rộng tập bảng được chạm (spec tiêu chí 11).

    Kết quả bị cắt ở `max_rows`. Cờ `truncated` đi qua `df.attrs` chứ không
    qua kiểu trả về: đổi kiểu trả về sẽ buộc sql_agent, app.py và toàn bộ
    test hiện có phải sửa, đổi lấy một bit thông tin.
    """
    assert_tables_permitted(sql, profile, user)
    conn = psycopg2.connect(profile.dsn)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SET LOCAL statement_timeout = %s", (timeout_ms,))
            cur.execute(sql, params)
            # Lấy dư một dòng: đó là cách duy nhất phân biệt "đúng max_rows
            # dòng" với "còn nữa" mà không phải đếm cả bảng.
            rows = cur.fetchmany(max_rows + 1)
            truncated = len(rows) > max_rows
            if truncated:
                logger.warning(
                    "Kết quả bị cắt ở %d dòng — câu truy vấn trả về nhiều hơn thế.", max_rows
                )
                rows = rows[:max_rows]

            if not rows:
                columns = [desc[0] for desc in cur.description] if cur.description else []
                df = pd.DataFrame(columns=columns)
            else:
                df = pd.DataFrame(rows)

            df.attrs["truncated"] = truncated
            df.attrs["row_cap"] = max_rows
            return df
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
