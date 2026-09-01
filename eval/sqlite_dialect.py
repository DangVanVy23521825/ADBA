"""Dịch SQL vàng phương ngữ SQLite sang Postgres.

BIRD và Spider phát hành SQL vàng viết cho SQLite. Nạp DỮ LIỆU sang
Postgres (`eval/load_sqlite_to_postgres.py`) không làm cho SQL chạy được:
chỉ 531/1534 câu của BIRD chạy nổi trên Postgres nếu để nguyên.

Hai khác biệt gây ra gần hết số đó, và cả hai đều dịch được:

1. Định danh bọc bằng dấu huyền (`` `x` ``) thay vì nháy kép.
2. Định danh KHÔNG nháy giữ nguyên hoa thường trong SQLite, còn Postgres
   gấp về chữ thường. Bộ nạp giữ nguyên hoa thường của cột (`CDSCode`),
   nên `T1.CDSCode` không nháy gấp thành `cdscode` và không tìm thấy.

Đo thật trên 1.534 câu BIRD, từng bước một:

    chỉ đổi dấu huyền                       531  (34.6%)
    + bọc định danh, bỏ qua tên nhập nhằng  953  (62.1%)
    + phân giải nhập nhằng qua bí danh     1194  (77.8%)

340 câu còn lại KHÔNG dịch được ở đây và cố ý để lại: hàm chỉ SQLite có
(`strftime`, `julianday`), kiểu lỏng của SQLite (`text = integer`), cú
pháp `LIMIT #,#`, và bảng chưa nạp. Chúng cần dịch ngữ nghĩa chứ không
phải dịch định danh, nên chỗ của chúng không phải module này.

⚠️ 340 câu bị loại KHÔNG phải mẫu ngẫu nhiên. Câu dùng hàm ngày tháng của
SQLite thường là câu khó hơn trung bình, nên số đo trên 1.194 câu còn lại
lệch theo hướng LẠC QUAN. Ghi con số này cạnh mọi kết quả dùng nó.
"""

from __future__ import annotations

import collections
from collections.abc import Sequence

import sqlparse
from sqlparse.tokens import Keyword, Name, Punctuation

from perception.schema_model import Table

_SOURCE_KEYWORDS = {
    "FROM", "JOIN", "INNER JOIN", "LEFT JOIN", "RIGHT JOIN", "FULL JOIN",
    "CROSS JOIN", "LEFT OUTER JOIN", "RIGHT OUTER JOIN", "FULL OUTER JOIN",
}

# Từ khoá có thể đứng ngay sau một tên bảng mà KHÔNG phải bí danh của nó.
_NOT_AN_ALIAS = {
    "ON", "WHERE", "GROUP", "ORDER", "LIMIT", "HAVING", "USING", "JOIN",
    "INNER", "LEFT", "RIGHT", "FULL", "CROSS", "UNION", "AS", "SET",
    "AND", "OR", "EXCEPT", "INTERSECT",
}


class SchemaNames:
    """Tra tên thật (đúng hoa thường) của bảng và cột.

    Giữ hai lớp tra vì hai lớp giải được hai loại tham chiếu khác nhau:

    - `by_table`: cho tham chiếu CÓ ĐỊNH TÍNH (`T2.zip`). Biết bảng thì
      biết chính xác cột nào.
    - `unambiguous`: cho tham chiếu TRẦN (`zip`). Chỉ dùng được khi tên đó
      chỉ có MỘT biến thể hoa thường trong toàn schema.

    Lớp thứ hai bắt buộc phải bảo thủ. Schema BIRD gộp có `member.zip` và
    `schools.Zip` cùng tồn tại; đoán bừa một trong hai thì sửa được bảng
    này và phá bảng kia. Tên nhập nhằng mà không có định tính thì để
    nguyên, và câu đó rơi vào phần không dịch được — thà mất câu còn hơn
    dịch sai âm thầm.
    """

    def __init__(self, tables: Sequence[Table]):
        self.by_table: dict[str, dict[str, str]] = {}
        variants: dict[str, set[str]] = collections.defaultdict(set)
        for t in tables:
            self.by_table[t.name.lower()] = {c.name.lower(): c.name for c in t.columns}
            variants[t.name.lower()].add(t.name)
            for c in t.columns:
                variants[c.name.lower()].add(c.name)
        self.unambiguous = {k: next(iter(v)) for k, v in variants.items() if len(v) == 1}


def backticks_to_double_quotes(sql: str) -> str:
    """`x` -> "x", trừ khi dấu huyền nằm trong chuỗi nháy đơn.

    Chuỗi văn bản có thể chứa dấu huyền như dữ liệu thật; đổi nó thành
    nháy kép sẽ vừa hỏng chuỗi vừa đẻ ra một định danh không tồn tại.
    """
    out: list[str] = []
    in_string = False
    for ch in sql:
        if ch == "'":
            in_string = not in_string
            out.append(ch)
        elif ch == "`" and not in_string:
            out.append('"')
        else:
            out.append(ch)
    return "".join(out)


def _unquote(v: str) -> str:
    return v.strip('"')


def _alias_map(significant: list, known_tables: set[str]) -> dict[str, str]:
    """bí danh (chữ thường) -> tên bảng (chữ thường). Bảng tự trỏ chính nó."""
    out: dict[str, str] = {}
    for i, tok in enumerate(significant):
        if not (tok.ttype is Keyword and tok.normalized.upper() in _SOURCE_KEYWORDS):
            continue
        if i + 1 >= len(significant):
            continue
        nxt = significant[i + 1]
        if nxt.ttype not in Name:
            continue
        table = _unquote(nxt.value).lower()
        if table not in known_tables:
            continue
        out[table] = table
        j = i + 2
        if j < len(significant) and significant[j].ttype is Keyword \
                and significant[j].normalized.upper() == "AS":
            j += 1
        if j < len(significant) and significant[j].ttype in Name:
            alias = _unquote(significant[j].value)
            if alias.upper() not in _NOT_AN_ALIAS:
                out[alias.lower()] = table
    return out


def translate(sql: str, names: SchemaNames) -> str:
    """Đưa SQL vàng SQLite về dạng Postgres chạy được, hoặc trả gần đúng.

    Hàm này KHÔNG hứa SQL trả về sẽ chạy: nó chỉ sửa định danh. Câu dùng
    `strftime` vẫn hỏng, và hỏng ở tầng `execute` rồi vào bucket
    `gold_error` — đúng chỗ, vì đó là lỗi dữ liệu chứ không phải lỗi hệ
    thống.
    """
    sql = backticks_to_double_quotes(sql)

    toks = []
    for statement in sqlparse.parse(sql):
        toks.extend(statement.flatten())

    significant = [t for t in toks if not t.is_whitespace]
    aliases = _alias_map(significant, set(names.by_table))

    def _next_significant(i: int):
        for t in toks[i + 1:]:
            if not t.is_whitespace:
                return t
        return None

    def _qualifier_before(i: int) -> str | None:
        """Tên đứng trước `.` ngay trước vị trí i, nếu có."""
        seen_dot = False
        for t in reversed(toks[:i]):
            if t.is_whitespace:
                continue
            if not seen_dot:
                if t.ttype is Punctuation and t.value == ".":
                    seen_dot = True
                    continue
                return None
            return _unquote(t.value).lower() if t.ttype in Name else None
        return None

    out: list[str] = []
    for i, tok in enumerate(toks):
        value = tok.value
        if tok.ttype in Name and not value.startswith('"'):
            nxt = _next_significant(i)
            # Tên theo sau bởi '(' là LỜI GỌI HÀM, không phải cột. Bọc nháy
            # nó biến COUNT(...) thành "Count"(...) và Postgres đi tìm một
            # hàm tên đúng như vậy. Schema BIRD có cột tên `Count`, nên bẫy
            # này nổ thật: bản đầu của bộ dịch làm hỏng 347 câu vốn chạy tốt.
            if nxt is None or nxt.value != "(":
                real = None
                qualifier = _qualifier_before(i)
                if qualifier:
                    table = aliases.get(qualifier)
                    if table:
                        real = names.by_table[table].get(value.lower())
                if real is None:
                    real = names.unambiguous.get(value.lower())
                if real and real != real.lower():
                    value = f'"{real}"'
        out.append(value)
    return "".join(out)
