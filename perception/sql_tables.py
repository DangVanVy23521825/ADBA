"""Parse tập bảng thật ra khỏi một câu SQL mẫu.

Dùng cho eval tầng 1: so tập bảng mà retriever chọn với tập bảng mà SQL
mẫu thực sự chạm. Sau này còn được production SQL permission guard dùng
lại để kiểm tra câu SQL do agent sinh ra chỉ chạm các bảng được phép.
Không gọi LLM, không cần database.
"""

from __future__ import annotations

import sqlparse
from sqlparse.sql import Identifier, IdentifierList, Parenthesis, TokenList
from sqlparse.tokens import CTE, DML, Keyword

_SOURCE_KEYWORDS = {"FROM", "JOIN", "INNER JOIN", "LEFT JOIN", "RIGHT JOIN",
                    "FULL JOIN", "CROSS JOIN", "LEFT OUTER JOIN",
                    "RIGHT OUTER JOIN", "FULL OUTER JOIN"}


def _real_name(identifier: Identifier) -> str | None:
    """Tên bảng thật, bỏ alias và bỏ tiền tố schema."""
    name = identifier.get_real_name()
    if not name:
        return None
    return name.lower()


def _cte_names(statement: TokenList) -> set[str]:
    """Tên các CTE khai báo bằng WITH — chúng không phải bảng thật."""
    names: set[str] = set()
    tokens = list(statement.flatten())
    saw_with = any(t.ttype is CTE for t in tokens)
    if not saw_with:
        return names

    for token in statement.tokens:
        if token.ttype is CTE:  # WITH
            continue
        if isinstance(token, IdentifierList):
            for ident in token.get_identifiers():
                if isinstance(ident, Identifier) and (n := _real_name(ident)):
                    names.add(n)
        elif isinstance(token, Identifier):
            if n := _real_name(token):
                names.add(n)
        elif token.ttype is DML and token.value.upper() == "SELECT":
            break
    return names


def _collect(node: TokenList, out: set[str]) -> None:
    expecting_source = False
    for token in node.tokens:
        if token.is_whitespace:
            continue

        if token.ttype is Keyword and token.value.upper() in _SOURCE_KEYWORDS:
            expecting_source = True
            continue

        if expecting_source:
            if isinstance(token, IdentifierList):
                for ident in token.get_identifiers():
                    if isinstance(ident, Identifier) and (n := _real_name(ident)):
                        out.add(n)
                expecting_source = False
                continue
            if isinstance(token, Identifier):
                # `FROM (SELECT ...) alias` — đi tiếp vào trong, không lấy alias
                if any(isinstance(t, Parenthesis) for t in token.tokens):
                    _collect(token, out)
                elif n := _real_name(token):
                    out.add(n)
                expecting_source = False
                continue
            if isinstance(token, Parenthesis):
                _collect(token, out)
                expecting_source = False
                continue
            expecting_source = False

        if isinstance(token, TokenList):
            _collect(token, out)


def tables_in_sql(sql: str) -> frozenset[str]:
    """Trả tập tên bảng thật mà câu SQL chạm tới.

    Alias bị bỏ, tên CTE bị loại, tiền tố schema bị cắt, tên hạ về chữ thường.
    SQL rỗng hoặc không parse được trả về tập rỗng thay vì ném lỗi — eval
    cần đếm được, không cần dừng.
    """
    if not sql or not sql.strip():
        return frozenset()

    found: set[str] = set()
    ctes: set[str] = set()
    for statement in sqlparse.parse(sql):
        ctes |= _cte_names(statement)
        _collect(statement, found)

    return frozenset(found - ctes)
