"""Logic của trang duyệt chú giải, tách khỏi Streamlit để test được.

Trang chỉ vẽ; mọi quyết định về trạng thái nằm ở đây.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from perception.annotations import HUMAN, Annotation, SchemaAnnotations
from perception.schema_model import Table


@dataclass(frozen=True)
class ReviewRow:
    table: str
    column: str | None
    current_text: str
    reviewed_by: str
    confidence: str
    type_hint: str


def review_rows(tables: Sequence[Table], ann: SchemaAnnotations) -> list[ReviewRow]:
    """Mọi thứ có thể chú giải, kể cả mục chưa có chú giải nào.

    Cột chưa được LLM mô tả vẫn phải hiện ra: chính những cột đó thường là
    thứ model bỏ qua vì không đoán nổi, tức là thứ cần người nhất.
    """
    rows: list[ReviewRow] = []
    for table in tables:
        t = ann.tables.get(table.name)
        rows.append(
            ReviewRow(
                table=table.name,
                column=None,
                current_text=t.text if t else "",
                reviewed_by=t.reviewed_by if t else "",
                confidence=t.confidence if t else "low",
                type_hint=f"{len(table.columns)} cột",
            )
        )
        cols = ann.columns.get(table.name, {})
        for c in table.columns:
            a = cols.get(c.name)
            rows.append(
                ReviewRow(
                    table=table.name,
                    column=c.name,
                    current_text=a.text if a else "",
                    reviewed_by=a.reviewed_by if a else "",
                    confidence=a.confidence if a else "low",
                    type_hint=c.data_type,
                )
            )
    return rows


def filter_rows(
    rows: Sequence[ReviewRow], only_pending: bool
) -> tuple[list[ReviewRow], int]:
    """Lọc hàng đợi duyệt, và đếm số mục bị GIẤU vì model tự tin.

    Trả `(hàng hiện ra, số mục model tự tin bị ẩn)`.

    Bộ lọc mặc định chỉ giữ mục `confidence == "low"`, nên chú giải do LLM
    sinh với `confidence == "high"` biến mất khỏi màn hình. Đó lại đúng là
    rủi ro lớn nhất của cả đường onboarding: một mô tả SAI mà model tự nhận
    là chắc chắn thì không ai kiểm lại, và mọi truy vấn dựa vào nó sai âm
    thầm.

    Không đảo mặc định — 150 bảng mà hiện hết thì không ai duyệt nổi, và
    một màn hình không dùng được còn tệ hơn. Thay vào đó trả về con số, để
    trang biến tập vô hình thành thứ analyst nhìn thấy và tự quyết.
    """
    if not only_pending:
        return list(rows), 0
    visible = [r for r in rows if r.reviewed_by != HUMAN and r.confidence == "low"]
    hidden_confident = sum(
        1 for r in rows if r.reviewed_by != HUMAN and r.confidence == "high"
    )
    return visible, hidden_confident


def apply_edit(
    ann: SchemaAnnotations, table: str, column: str | None, new_text: str
) -> SchemaAnnotations:
    """Ghi một sửa đổi của người và đánh dấu `reviewed_by="human"`.

    Đánh dấu là phần quan trọng: nó khiến mục này sống sót qua mọi lần làm
    mới về sau.
    """
    entry = Annotation(text=new_text, reviewed_by=HUMAN, confidence="high")
    if column is None:
        return SchemaAnnotations(
            tables={**ann.tables, table: entry},
            columns=ann.columns,
        )
    cols = {**ann.columns.get(table, {}), column: entry}
    return SchemaAnnotations(
        tables=ann.tables,
        columns={**ann.columns, table: cols},
    )


def review_progress(
    ann: SchemaAnnotations, tables: Sequence[Table] | None = None
) -> tuple[int, int]:
    """(số mục người đã duyệt, tổng số mục).

    Gọi `review_progress(ann)` KHÔNG kèm `tables` chỉ đo trong tập mục đã
    có chú giải (`ann.tables` + `ann.columns`) — nó không biết, và không
    thể biết, về những cột `review_rows` sẽ hiện ra mà chưa hề có annotation.
    Vì đúng những cột đó là thứ LLM bỏ qua vì không đoán nổi — tức là thứ
    cần người nhất — dùng dạng một-đối-số này để dựng cổng phát hành sẽ để
    lọt chúng ra khỏi mẫu số, và một khách hàng có thể chạm "100% đã duyệt"
    trong khi hàng chục cột vẫn trống.

    Truyền `tables` để mẫu số khớp đúng với những gì `review_rows(tables,
    ann)` liệt kê: mọi bảng và mọi cột trong `tables`, kể cả mục chưa có
    chú giải nào (đếm là chưa duyệt, tất nhiên).
    """
    if tables is None:
        entries = list(ann.tables.values()) + [
            a for cols in ann.columns.values() for a in cols.values()
        ]
        return sum(1 for a in entries if a.reviewed_by == HUMAN), len(entries)

    # Dẫn xuất thẳng từ `review_rows` chứ không đếm lại bằng một vòng lặp
    # song song: hai vòng viết tay chỉ TÌNH CỜ khớp nhau, và khi `review_rows`
    # đổi cách liệt kê thì mẫu số lặng lẽ lệch đi — đúng thứ mà tham số
    # `tables` sinh ra để ngăn. Ở đây thanh tiến độ và danh sách bên dưới nó
    # khớp nhau về mặt cấu trúc, không phải nhờ may mắn.
    rows = review_rows(tables, ann)
    return sum(1 for r in rows if r.reviewed_by == HUMAN), len(rows)
