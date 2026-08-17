"""Đọc cấu trúc một database Postgres thành kiểu dữ liệu nội bộ.

Chỉ ra CẤU TRÚC. Ý nghĩa — mô tả bảng và cột — đến từ bước chú giải sau đó;
introspect không đoán, vì đoán sai ở đây sẽ lặng lẽ sai suốt chuỗi.
"""

from __future__ import annotations

import re

import psycopg2
import psycopg2.extras
from psycopg2 import sql

from perception.schema_model import Column, Table

_IDENT = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

_COLUMNS_SQL = """
SELECT table_name, column_name, data_type, is_generated
FROM information_schema.columns
WHERE table_schema = %s
ORDER BY table_name, ordinal_position
"""

_PK_SQL = """
SELECT tc.table_name, kcu.column_name
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu
  ON tc.constraint_name = kcu.constraint_name
 AND tc.table_schema = kcu.table_schema
WHERE tc.constraint_type = 'PRIMARY KEY' AND tc.table_schema = %s
ORDER BY kcu.ordinal_position
"""

_FK_SQL = """
SELECT tc.table_name, kcu.column_name,
       ccu.table_name AS ref_table, ccu.column_name AS ref_column
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu
  ON tc.constraint_name = kcu.constraint_name
 AND tc.table_schema = kcu.table_schema
JOIN information_schema.constraint_column_usage ccu
  ON ccu.constraint_name = tc.constraint_name
WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = %s
"""

_COUNT_SQL = """
SELECT c.reltuples::bigint
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relname = %s AND n.nspname = %s
"""


def introspect_schema(dsn: str, schema: str = "public") -> tuple[Table, ...]:
    """Trả về mọi bảng trong `schema`, kèm cột, khoá chính và khoá ngoại.

    `row_count` lấy ước lượng từ `pg_class` chứ không `COUNT(*)`: trên bảng
    hàng chục triệu dòng, một lần đếm thật có thể mất hàng phút và con số đó
    chỉ dùng để tham khảo trong prompt.
    """
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(_COLUMNS_SQL, (schema,))
            cols: dict[str, list[Column]] = {}
            for table, name, dtype, generated in cur.fetchall():
                cols.setdefault(table, []).append(
                    Column(name=name, data_type=dtype, is_generated=(generated == "ALWAYS"))
                )

            cur.execute(_PK_SQL, (schema,))
            pks: dict[str, list[str]] = {}
            for table, column in cur.fetchall():
                pks.setdefault(table, []).append(column)

            cur.execute(_FK_SQL, (schema,))
            fks: dict[str, dict[str, str]] = {}
            for table, column, ref_table, ref_column in cur.fetchall():
                fks.setdefault(table, {})[column] = f"{ref_table}({ref_column})"

            counts: dict[str, int] = {}
            for table in cols:
                cur.execute(_COUNT_SQL, (table, schema))
                row = cur.fetchone()
                counts[table] = int(row[0]) if row and row[0] is not None else 0
    finally:
        conn.close()

    return tuple(
        Table(
            name=table,
            columns=tuple(columns),
            primary_key=tuple(pks.get(table, ())),
            foreign_keys=dict(fks.get(table, {})),
            row_count=counts.get(table),
            description="",
        )
        for table, columns in sorted(cols.items())
    )


def sample_rows(dsn: str, table: str, n: int = 5) -> list[dict]:
    """Vài dòng mẫu để bước chú giải đoán ý nghĩa cột.

    KHÔNG được ghi vào `profile/` — xem Global Constraints. Tên bảng không
    tham số hoá được trong Postgres nên phải qua hai lớp: regex định danh
    trước (để báo lỗi rõ ràng, đọc được, ngay tại đây) rồi `sql.Identifier`
    khi dựng câu lệnh (để việc quote/escape đúng đắn không phụ thuộc vào
    regex có đúng hay không — chính layer regex này từng lọt một trường hợp,
    xem TestIdentifierGuard trong tests/unit/test_introspect.py). `sql.Identifier`
    cũng quote tên, nên một bảng thật tên `Orders` (phân biệt hoa/thường)
    vẫn địa chỉ được đúng.
    """
    if not _IDENT.fullmatch(table):
        raise ValueError(f"Tên bảng không hợp lệ: {table!r}")

    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            query = sql.SQL("SELECT * FROM {table} LIMIT %s").format(
                table=sql.Identifier(table)
            )
            cur.execute(query, (n,))
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()
