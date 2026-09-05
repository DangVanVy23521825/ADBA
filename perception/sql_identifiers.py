"""Bọc nháy lại định danh mà Postgres sẽ gấp sai.

Postgres gấp định danh KHÔNG nháy về chữ thường. Rất nhiều schema thật giữ
nguyên hoa thường — ORM sinh ra (`createdAt`), di cư từ SQL Server
(`MailStreet`), và toàn bộ BIRD/Spider. Trên những schema đó, `SELECT
MailStreet` gấp thành `mailstreet`, không tồn tại, và câu truy vấn hỏng.

`render_schema` đã in DDL đúng — `"MailStreet" TEXT` — nên model ĐƯỢC XEM
dạng có nháy. Nó vẫn viết trần. Thêm quy tắc vào prompt (quy tắc 11 của
`prompts/text_to_sql.txt`) chỉ giảm 6 lỗi xuống 5 trên mẫu 8 câu BIRD:
model 7B không tuân thủ ổn định, nên chỉ dẫn không đủ và cần một bước
sửa tất định.

**Không hoạt động trên schema toàn chữ thường.** Điều kiện sửa là tên thật
có ký tự hoa (`real != real.lower()`), nên với schema chữ thường — phần
lớn khách hàng — module này là no-op hoàn toàn. Đó là tính chất an toàn
quan trọng nhất của nó: nó không thể phá thứ đang chạy được.
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

    Hai lớp tra, vì hai lớp giải được hai loại tham chiếu khác nhau:

    - `by_table`: cho tham chiếu CÓ ĐỊNH TÍNH (`T2.zip`). Biết bảng thì
      biết chính xác cột nào.
    - `unambiguous`: cho tham chiếu TRẦN (`zip`). Chỉ dùng được khi tên đó
      có đúng MỘT biến thể hoa thường trong toàn schema.

    Lớp thứ hai bắt buộc bảo thủ. Schema BIRD gộp có `member.zip` và
    `schools.Zip` cùng tồn tại; đoán bừa thì sửa được bảng này và phá bảng
    kia. Tên nhập nhằng mà không có định tính thì để nguyên — thà để câu đó
    hỏng ồn ào còn hơn sửa sai âm thầm thành một câu chạy được nhưng đọc
    nhầm cột.
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

    def is_noop(self) -> bool:
        """Schema này có tên nào cần bọc nháy không?

        Toàn chữ thường thì `requote` không bao giờ đổi gì, và nơi gọi bỏ
        qua được cả bước parse.
        """
        return not any(
            real != real.lower()
            for real in list(self.unambiguous.values())
            + [c for cols in self.by_table.values() for c in cols.values()]
        )


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
        if (j < len(significant) and significant[j].ttype is Keyword
                and significant[j].normalized.upper() == "AS"):
            j += 1
        if j < len(significant) and significant[j].ttype in Name:
            alias = _unquote(significant[j].value)
            if alias.upper() not in _NOT_AN_ALIAS:
                out[alias.lower()] = table
    return out


def requote(sql: str, names: SchemaNames) -> str:
    """Bọc nháy những định danh trần mà tên thật có ký tự hoa.

    Chỉ động vào token mà schema xác nhận là bảng hoặc cột. Bí danh (`T1`),
    từ khoá, chuỗi văn bản, và tên đã có nháy đều không bị đụng.
    """
    toks: list = []
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
        """Tên đứng trước dấu `.` ngay trước vị trí i, nếu có."""
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
                    # Thu hẹp theo CHÍNH các bảng của câu truy vấn trước khi
                    # rơi về bản đồ toàn schema. `schools.Phone` và
                    # `member.phone` nhập nhằng trên toàn schema, nhưng
                    # trong một câu chỉ có `FROM schools` thì không — và
                    # đây đúng là cách Postgres phân giải cột không định
                    # tính. Chỉ nhận khi có ĐÚNG MỘT bảng khớp: hai bảng
                    # cùng có tên đó thì chính Postgres cũng báo nhập nhằng.
                    hits = {
                        names.by_table[t][value.lower()]
                        for t in set(aliases.values())
                        if value.lower() in names.by_table.get(t, {})
                    }
                    if len(hits) == 1:
                        real = next(iter(hits))
                    else:
                        real = names.unambiguous.get(value.lower())
                if real and real != real.lower():
                    value = f'"{real}"'
        out.append(value)
    return "".join(out)


def requote_for_tables(sql: str, tables: Sequence[Table]) -> str:
    """Tiện ích cho nơi gọi chỉ có sẵn danh sách bảng.

    Trả `sql` nguyên vẹn khi schema toàn chữ thường — tránh cả chi phí
    parse cho phần lớn khách hàng, và bảo đảm đường code đó không thể đổi
    gì.
    """
    if not tables:
        return sql
    names = SchemaNames(tables)
    if names.is_noop():
        return sql
    return requote(sql, names)


__all__ = ["SchemaNames", "requote", "requote_for_tables"]
