"""Chọn bảng liên quan tới câu hỏi.

LƯU Ý — đây KHÔNG phải cơ chế phân quyền (spec 3.4.1). Nó chỉ thu hẹp thứ
model *nhìn thấy*, không thu hẹp thứ model *được phép chạm*. Quyền nằm ở
perception.connection_profile.permitted_tables().

LexicalRetriever là mốc so sánh, cố ý không phụ thuộc torch. Bản embedding
đến sau, và phải chứng minh nó thắng mốc này trên eval tầng 1 mới đáng
đánh đổi thêm 470MB artifact vào bundle on-prem.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Protocol

from perception.schema_model import Table

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    """Tách token, cắt cả snake_case: base_salary → {base, salary}."""
    return set(_WORD.findall(text.lower()))


class Retriever(Protocol):
    def search(self, question: str, k: int) -> list[str]:
        """Trả tối đa k tên bảng, xếp theo độ liên quan giảm dần."""
        ...


class FullRetriever:
    """Trả mọi bảng. Dùng cho schema_mode='full' và làm trần trên khi đo recall."""

    def __init__(self, tables: Sequence[Table]) -> None:
        self._names = [t.name for t in tables]

    def search(self, question: str, k: int) -> list[str]:  # noqa: ARG002
        return list(self._names)


class LexicalRetriever:
    """Chấm điểm bằng số token trùng giữa câu hỏi và mô tả bảng.

    Tên bảng có trọng số cao nhất, rồi mô tả, rồi tên cột — tên cột nhiều
    và nhiễu (id, name xuất hiện ở mọi bảng) nên không được lấn át.
    """

    _W_NAME = 3.0
    _W_DESC = 2.0
    _W_COLUMN = 1.0

    def __init__(self, tables: Sequence[Table]) -> None:
        self._index: list[tuple[str, set[str], set[str], set[str]]] = [
            (t.name, _tokens(t.name), _tokens(t.description),
             _tokens(" ".join(c.name for c in t.columns)))
            for t in tables
        ]

    def search(self, question: str, k: int) -> list[str]:
        q = _tokens(question)
        if not q:
            return []

        scored: list[tuple[float, int, str]] = []
        for rank, (name, n_tok, d_tok, c_tok) in enumerate(self._index):
            score = (
                self._W_NAME * len(q & n_tok)
                + self._W_DESC * len(q & d_tok)
                + self._W_COLUMN * len(q & c_tok)
            )
            if score > 0:
                # rank làm khóa phụ để hòa điểm vẫn ra thứ tự cố định
                scored.append((-score, rank, name))

        scored.sort()
        return [name for _, _, name in scored[:k]]


def expand_by_foreign_keys(names: Iterable[str], tables: Sequence[Table]) -> frozenset[str]:
    """Mở rộng đúng MỘT bước theo cạnh khóa ngoại, cả hai chiều.

    Hai chiều vì một câu hỏi nhắc `customers` thường vẫn cần `orders` để
    tính được gì đó, dù cạnh FK đi từ orders sang customers.

    Một bước vì hai bước trên schema doanh nghiệp sẽ kéo về gần hết schema
    và làm hỏng chính lợi ích của việc thu hẹp.
    """
    by_name = {t.name: t for t in tables}
    seed = {n for n in names if n in by_name}
    if not seed:
        return frozenset()

    out = set(seed)
    for name in seed:
        out |= by_name[name].references() & by_name.keys()
    for t in tables:
        if t.references() & seed:
            out.add(t.name)
    return frozenset(out)
