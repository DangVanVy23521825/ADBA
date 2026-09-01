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

# Định danh Unicode: chữ cái (bất kỳ ngôn ngữ nào) hoặc '_' ở đầu, theo sau
# là chữ/số/'_'. `\w` trên pattern kiểu `str` của Python 3 mặc định khớp
# Unicode — không cần cờ `re.UNICODE` tường minh (đã là mặc định), giữ ở
# đây chỉ để nói rõ ý định, phòng ai đó đổi sang pattern kiểu `bytes`.
#
# ADBA là sản phẩm tiếng Việt, đọc schema của khách hàng tiếng Việt:
# `introspect_schema` (ở trên) trả nguyên văn `table_name` từ
# `information_schema.columns` — nó sẽ trả về `đơn_hàng`, `khách_hàng`,
# ... một cách bình thường. Regex CŨ (`[a-zA-Z_][a-zA-Z0-9_]*`) không cho
# những tên đó qua được lớp chặn ở dưới, dù `sql.Identifier` (lớp phòng
# thủ THỨ HAI, xem docstring `sample_rows`) quote chúng đúng đắn — tức là
# lớp phòng thủ ĐẦU chặt hơn chính lớp nó bảo vệ. Rộng ra Unicode không
# làm yếu lớp phòng thủ khỏi injection: mọi ký tự dùng trong tấn công
# (khoảng trắng, `;`, `'`, `"`, `-`, `.`, xuống dòng, ...) vẫn không phải
# chữ/số/`_` nên vẫn bị từ chối — xem TestIdentifierGuard trong
# tests/unit/test_introspect.py, cả phần injection lẫn phần tên Việt hợp lệ.
_IDENT = re.compile(r"^[^\W\d]\w*\Z", re.UNICODE)

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

            counts: dict[str, int | None] = {}
            for table in cols:
                cur.execute(_COUNT_SQL, (table, schema))
                row = cur.fetchone()
                if row is None or row[0] is None:
                    counts[table] = 0
                    continue
                reltuples = int(row[0])
                # Postgres dùng -1 làm sentinel cho "chưa từng ANALYZE" — trên
                # một DB khách vừa restore, phần lớn bảng sẽ ở trạng thái này
                # cho tới khi autovacuum chạy tới. -1 không phải một con số
                # dòng thật; nó không được đi tới render_schema()/prompt như
                # một sự thật. Chuẩn hoá về None, giống trường hợp không có
                # ước lượng nào cả — render_schema đã bỏ qua comment khi
                # row_count is None.
                counts[table] = reltuples if reltuples >= 0 else None
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

    Mở và đóng một kết nối riêng cho mỗi lần gọi có chủ đích — xem
    `onboard._sample_all_tables` (I3a): dùng chung MỘT kết nối cho cả lượt
    150 bảng được cân nhắc, nhưng test double của bộ test hiện có
    (`tests/unit/test_onboard_cli.py`) mock nguyên hàm `onboard.sample_rows`
    chứ không mock `psycopg2.connect`, và việc dùng chung một connection
    còn đòi hỏi tự quản lý trạng thái transaction khi một câu SELECT lỗi
    (nếu không bật `autocommit`, một lỗi ở bảng N để transaction ở trạng
    thái "aborted" và lây sang bảng N+1). Cả hai lý do khiến việc gộp kết
    nối là một refactor rộng hơn phạm vi sửa lỗi này — không ép làm ở đây.
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
