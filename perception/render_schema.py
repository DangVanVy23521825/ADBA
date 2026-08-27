"""Biểu diễn schema dạng DDL để đưa vào prompt.

Vì sao DDL chứ không phải JSON: nhỏ hơn 2–3 lần, và Qwen2.5-Coder đã đọc
hàng triệu file SQL lúc pretrain trong khi JSON mô tả schema thì không.
Xem spec mục 3.4.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from perception.schema_model import Table

# Hình dạng một định danh Postgres AN TOÀN để KHÔNG quote: toàn chữ thường,
# số, `_`, không bắt đầu bằng số. Đây đúng là điều kiện Postgres tự áp dụng
# khi đọc một định danh không có ngoặc kép (nó hạ chữ hoa xuống thường rồi
# mới tra catalog) — bất cứ tên nào lệch khỏi hình dạng này mà để không
# quote sẽ bị Postgres đọc SAI đi so với tên thật trong catalog.
_UNQUOTED_SAFE = re.compile(r"^[a-z_][a-z0-9_]*$")

# Từ khoá "reserved" của Postgres (phụ lục C, cột "reserved" — không dùng
# được làm tên bảng/cột KHÔNG quote trong bất kỳ ngữ cảnh nào), liệt kê thủ
# công thay vì tái dùng bộ từ khoá của sqlparse: bộ đó rộng hơn nhiều (gồm
# cả từ khoá non-reserved như DATE, TEXT, LEVEL — quote thêm những tên đó
# là an toàn nhưng làm DDL khác đi không cần thiết so với hôm nay).
_RESERVED_WORDS = frozenset({
    "all", "analyse", "analyze", "and", "any", "array", "as", "asc",
    "asymmetric", "both", "case", "cast", "check", "collate", "column",
    "constraint", "create", "current_catalog", "current_date",
    "current_role", "current_time", "current_timestamp", "current_user",
    "default", "deferrable", "desc", "distinct", "do", "else", "end",
    "except", "false", "fetch", "for", "foreign", "from", "grant", "group",
    "having", "in", "initially", "intersect", "into", "lateral", "leading",
    "limit", "localtime", "localtimestamp", "not", "null", "offset", "on",
    "only", "or", "order", "placing", "primary", "references",
    "returning", "select", "session_user", "some", "symmetric", "table",
    "then", "to", "trailing", "true", "union", "unique", "user", "using",
    "variadic", "when", "where", "window", "with",
})


def _quote_ident(name: str) -> str:
    """Trả định danh sẵn sàng đưa thẳng vào DDL đưa cho model.

    Giữ nguyên, không quote, khi an toàn: chỉ khi đó DDL hôm nay không đổi
    một byte nào, và model chỉ cần chép nguyên tên vào câu SQL nó viết ra —
    Postgres đọc lại đúng y tên trong catalog (spec mục 2: unquoted fold về
    chữ thường, nên tên vốn đã toàn chữ thường không bị đổi nghĩa gì).

    Quote (và escape `"` bên trong thành `""`, quy tắc ngược của cách
    tables_in_sql giải mã nó) trong mọi trường hợp còn lại: có chữ hoa, có
    ký tự ngoài [a-z0-9_], bắt đầu bằng số, hoặc trùng một từ khoá dành
    riêng — bốn trường hợp mà một model chép tên KHÔNG quote vào `FROM ...`
    sẽ trỏ sai bảng (hoặc trỏ vào bảng không tồn tại) so với ý DDL này.
    """
    if name in _RESERVED_WORDS or not _UNQUOTED_SAFE.fullmatch(name):
        return '"' + name.replace('"', '""') + '"'
    return name


def _quote_fk_ref(ref: str) -> str:
    """Quote riêng từng phần của `"table(col)"` trong một dòng REFERENCES.

    KHÔNG được đổi giá trị lưu trong `Table.foreign_keys` — chuỗi đó còn bị
    `Table.references()` (`ref.split("(")[0]`) parse lại để lấy tên bảng
    RAW dùng trong `expand_by_foreign_keys`; quote nó ở đó sẽ làm sai lệch
    tra cứu. Việc quote chỉ xảy ra ở ĐÂY, tại điểm kết xuất text cuối cùng.
    """
    table_part, _, rest = ref.partition("(")
    col_part = rest[:-1] if rest.endswith(")") else rest
    return f"{_quote_ident(table_part)}({_quote_ident(col_part)})"

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
        # M4: gộp khoảng trắng (xuống dòng, tab, nhiều dấu cách liên tiếp)
        # thành một dấu cách — CÙNG cách xử lý mô tả CỘT ở dưới
        # (`collapsed_desc`), áp cho mô tả BẢNG. `schema.yaml` là escape
        # hatch sửa tay được tài liệu hoá, nên một block scalar YAML chứa
        # xuống dòng là input thực tế, không phải giả thuyết — không gộp
        # thì dòng 2 của mô tả trở thành text KHÔNG có tiền tố `-- `, nằm
        # ngay trước `CREATE TABLE`, phá hỏng cú pháp DDL đưa vào prompt
        # cho model SQL.
        collapsed_desc = " ".join(table.description.split())
        lines.append(f"-- {collapsed_desc}")
    lines.append(f"CREATE TABLE {_quote_ident(table.name)} (")

    pk = set(table.primary_key)
    body: list[str] = []
    for i, col in enumerate(table.columns):
        # So PRIMARY KEY / FK lookup dùng tên CỘT THẬT (col.name), chưa
        # quote — quote chỉ áp lúc kết xuất text (parts[0]), không được rò
        # vào các so khớp nội bộ này.
        parts = [_quote_ident(col.name), _short_type(col.data_type)]
        if col.name in pk:
            parts.append("PRIMARY KEY")
        if ref := table.foreign_keys.get(col.name):
            parts.append(f"REFERENCES {_quote_fk_ref(ref)}")
        if col.is_generated:
            parts.append("GENERATED")
        line = "  " + " ".join(parts)
        # Add comma if not the last column
        if i < len(table.columns) - 1:
            line += ","
        # Add description after the comma, with whitespace collapsed
        if col.description:
            collapsed_desc = " ".join(col.description.split())
            line += f"  -- {collapsed_desc}"
        body.append(line)
    lines.append("\n".join(body))

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
