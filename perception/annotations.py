"""Kho chú giải ngữ nghĩa cho một schema.

Chú giải là trần độ chính xác của hệ thống, và phần lớn nó do người viết.
Vì thế quy tắc chi phối module này là: **mục có `reviewed_by: human` không
bao giờ bị ghi đè.** Nếu một lần làm mới xoá công sức của khách thì họ sẽ
không sửa lần thứ hai, và trần đó tụt vĩnh viễn.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import yaml

from perception.schema_model import Table

HUMAN = "human"
LLM = "llm"

_REVIEWED_BY_VALUES = {LLM, HUMAN}
_CONFIDENCE_VALUES = {"high", "low"}


@dataclass(frozen=True)
class Annotation:
    text: str
    reviewed_by: str = LLM
    confidence: str = "high"


@dataclass(frozen=True)
class SchemaAnnotations:
    tables: Mapping[str, Annotation] = field(default_factory=dict)
    columns: Mapping[str, Mapping[str, Annotation]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Deep-freeze `tables` and `columns` so mutation raises TypeError.

        `frozen=True` on the dataclass only blocks reassigning the
        attributes themselves (`ann.tables = ...`); it does nothing to stop
        `ann.tables["orders"] = ...` from mutating the dict in place. That
        exact bug has shipped twice in this codebase already. Follow the
        same convention as `Table.__post_init__` in `schema_model.py`: wrap
        the outer mapping in `MappingProxyType`, and here — because
        `columns` is a mapping of mappings — wrap each inner per-table
        mapping too. Idempotent: wrapping an already-wrapped mapping is
        safe, which matters because `dataclasses.replace` re-runs
        `__post_init__`.
        """
        if not isinstance(self.tables, MappingProxyType):
            object.__setattr__(self, "tables", MappingProxyType(dict(self.tables)))
        if not isinstance(self.columns, MappingProxyType):
            object.__setattr__(
                self,
                "columns",
                MappingProxyType(
                    {
                        table: cols
                        if isinstance(cols, MappingProxyType)
                        else MappingProxyType(dict(cols))
                        for table, cols in self.columns.items()
                    }
                ),
            )


def _to_plain(ann: Annotation) -> dict:
    return {"text": ann.text, "reviewed_by": ann.reviewed_by, "confidence": ann.confidence}


def _from_plain(raw: Mapping, *, path: Path, table: str, column: str | None = None) -> Annotation:
    """Build an `Annotation` from a hand-editable YAML mapping.

    A missing `reviewed_by`/`confidence` key keeps the documented default
    (`LLM` / `"high"`) — absence is fine. An unrecognized *value* is not:
    silently defaulting a typo like `reviewed_by: Human` would make the
    system believe the LLM owns an annotation a human actually wrote, and
    the very next `merge_annotations` would delete it. Raise instead, and
    name exactly where the bad value lives so it costs the operator one
    character to fix.
    """
    reviewed_by = raw.get("reviewed_by", LLM)
    if reviewed_by not in _REVIEWED_BY_VALUES:
        where = f"table '{table}'" if column is None else f"table '{table}', column '{column}'"
        raise ValueError(
            f"{path}: {where}: field 'reviewed_by' has invalid value {reviewed_by!r}; "
            f"expected one of {sorted(_REVIEWED_BY_VALUES)}"
        )
    confidence = raw.get("confidence", "high")
    if confidence not in _CONFIDENCE_VALUES:
        where = f"table '{table}'" if column is None else f"table '{table}', column '{column}'"
        raise ValueError(
            f"{path}: {where}: field 'confidence' has invalid value {confidence!r}; "
            f"expected one of {sorted(_CONFIDENCE_VALUES)}"
        )
    return Annotation(text=raw.get("text", ""), reviewed_by=reviewed_by, confidence=confidence)


def save_annotations(ann: SchemaAnnotations, path: Path | str) -> None:
    """Ghi ra YAML đọc được bằng mắt.

    Định dạng người đọc được là có chủ ý: khi giao diện hỏng hoặc khách muốn
    sửa hàng loạt, file này vẫn là đường thoát.
    """
    payload = {
        "tables": {name: _to_plain(a) for name, a in sorted(ann.tables.items())},
        "columns": {
            table: {col: _to_plain(a) for col, a in sorted(cols.items())}
            for table, cols in sorted(ann.columns.items())
        },
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def load_annotations(path: Path | str) -> SchemaAnnotations:
    p = Path(path)
    if not p.exists():
        return SchemaAnnotations()
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return SchemaAnnotations(
        tables={
            n: _from_plain(a, path=p, table=n) for n, a in (raw.get("tables") or {}).items()
        },
        columns={
            t: {
                c: _from_plain(a, path=p, table=t, column=c)
                for c, a in (cols or {}).items()
            }
            for t, cols in (raw.get("columns") or {}).items()
        },
    )


def merge_annotations(
    existing: SchemaAnnotations, fresh: SchemaAnnotations
) -> SchemaAnnotations:
    """Ghép chú giải mới sinh vào chú giải đang có.

    Quy tắc: mục `reviewed_by == HUMAN` trong `existing` luôn thắng. Mục do
    LLM sinh bị thay bằng bản mới. Bảng và cột không còn trong `fresh` bị
    loại — giữ chú giải của thứ không còn tồn tại chỉ làm nhiễu bản duyệt.

    CẢNH BÁO: việc loại bỏ này dựa hoàn toàn vào `fresh`. `fresh` phải phủ
    toàn bộ schema hiện có — truyền một `fresh` thiếu hoặc rỗng sẽ âm thầm
    xoá sạch mọi chú giải do người viết, kể cả những mục `human` mà quy tắc
    ở trên đáng lẽ phải bảo vệ. Hàm này không tự kiểm tra điều đó; nơi gọi
    (lệnh refresh) chịu trách nhiệm đảm bảo `fresh` là đầy đủ.
    """
    tables = {}
    for name, new in fresh.tables.items():
        old = existing.tables.get(name)
        tables[name] = old if (old and old.reviewed_by == HUMAN) else new

    columns: dict[str, dict[str, Annotation]] = {}
    for table, new_cols in fresh.columns.items():
        old_cols = existing.columns.get(table, {})
        merged = {}
        for col, new in new_cols.items():
            old = old_cols.get(col)
            merged[col] = old if (old and old.reviewed_by == HUMAN) else new
        columns[table] = merged

    return SchemaAnnotations(tables=tables, columns=columns)


def apply_annotations(
    tables: Sequence[Table], ann: SchemaAnnotations
) -> tuple[Table, ...]:
    """Gắn mô tả vào `Table`/`Column` để `render_schema` in ra được."""
    out = []
    for table in tables:
        col_ann = ann.columns.get(table.name, {})
        out.append(
            dataclasses.replace(
                table,
                description=ann.tables[table.name].text if table.name in ann.tables else "",
                columns=tuple(
                    dataclasses.replace(
                        c, description=col_ann[c.name].text if c.name in col_ann else ""
                    )
                    for c in table.columns
                ),
            )
        )
    return tuple(out)


def pending_review(ann: SchemaAnnotations) -> list[tuple[str, str | None]]:
    """Những mục CẦN NHÌN TRƯỚC: LLM tự nhận không chắc, và chưa ai duyệt.

    Đây là hàng đợi ưu tiên, KHÔNG PHẢI thước đo độ phủ duyệt. Điều kiện là
    `reviewed_by != HUMAN and confidence == "low"` — nghĩa là một chú giải
    LLM sinh với `confidence: high` nhưng SAI hoàn toàn vẫn không bao giờ
    vào hàng đợi này, dù đó là rủi ro lớn nhất trong toàn bộ kế hoạch.

    `pending_review(ann) == []` KHÔNG có nghĩa là mọi thứ đã được duyệt; nó
    chỉ có nghĩa là không còn mục nào tự nhận là "chưa chắc". Đừng dùng hàm
    này để xây cổng phát hành dựa trên tiến độ duyệt — đó phải là một phép
    đo khác, dựa trên `reviewed_by == HUMAN` trên toàn bộ tập, không phải
    hàm này.

    Trả `(bảng, None)` cho chú giải bảng và `(bảng, cột)` cho chú giải cột,
    để giao diện xếp chúng vào cùng một hàng đợi.
    """
    out: list[tuple[str, str | None]] = []
    for name, a in ann.tables.items():
        if a.reviewed_by != HUMAN and a.confidence == "low":
            out.append((name, None))
    for table, cols in ann.columns.items():
        for col, a in cols.items():
            if a.reviewed_by != HUMAN and a.confidence == "low":
                out.append((table, col))
    return out
