"""Biểu diễn schema dạng DDL để đưa vào prompt.

Vì sao DDL chứ không phải JSON: nhỏ hơn 2–3 lần, và Qwen2.5-Coder đã đọc
hàng triệu file SQL lúc pretrain trong khi JSON mô tả schema thì không.
Xem spec mục 3.4.
"""

from __future__ import annotations

from collections.abc import Sequence

from perception.schema_model import Table

_TYPE_SHORT = {
    "character varying": "VARCHAR",
    "character": "CHAR",
    "timestamp without time zone": "TIMESTAMP",
    "timestamp with time zone": "TIMESTAMPTZ",
    "double precision": "FLOAT",
    "integer": "INT",
    "smallint": "SMALLINT",
    "bigint": "BIGINT",
    "numeric": "NUMERIC",
    "boolean": "BOOL",
    "date": "DATE",
    "text": "TEXT",
}


def _short_type(data_type: str) -> str:
    return _TYPE_SHORT.get(data_type.lower(), data_type.upper())


def _render_one(table: Table) -> str:
    lines: list[str] = []
    if table.description:
        lines.append(f"-- {table.description}")
    lines.append(f"CREATE TABLE {table.name} (")

    pk = set(table.primary_key)
    body: list[str] = []
    for col in table.columns:
        parts = [col.name, _short_type(col.data_type)]
        if col.name in pk:
            parts.append("PRIMARY KEY")
        if ref := table.foreign_keys.get(col.name):
            parts.append(f"REFERENCES {ref}")
        if col.is_generated:
            parts.append("GENERATED")
        line = "  " + " ".join(parts)
        if col.description:
            line += f"  -- {col.description}"
        body.append(line)
    lines.append(",\n".join(body))

    tail = ");"
    if table.row_count is not None:
        tail += f"  -- {table.row_count} rows"
    lines.append(tail)
    return "\n".join(lines)


def render_schema(tables: Sequence[Table]) -> str:
    """Kết xuất danh sách Table thành text DDL, ngăn bởi dòng trống.

    Đầu ra deterministic: thứ tự bảng và cột giữ nguyên như đầu vào, để
    prefix caching còn dùng được.
    """
    if not tables:
        return ""
    return "\n\n".join(_render_one(t) for t in tables)


def estimate_tokens(text: str) -> int:
    """Ước lượng thô số token. Dùng chung cho công tắc schema_mode (spec 3.3).

    Cố ý thô: mọi ngưỡng trong spec đều tính trên cùng ước lượng này, nên
    sai số hệ thống không làm lệch quyết định. Nếu đổi công thức thì phải
    đổi cả ngưỡng 6.000 ở mục 3.3.
    """
    return len(text) // 4
