"""Kiểu dữ liệu biểu diễn schema, bất biến, không phụ thuộc nguồn.

Cố ý KHÔNG mang `sample_rows`. Dữ liệu thật chỉ cần lúc sinh chú giải; nó
không được đi tiếp vào profile runtime, vì profile là thứ bị copy đi khi
hỗ trợ kỹ thuật. Xem spec mục 3.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

_EMPTY_FK: Mapping[str, str] = MappingProxyType({})


@dataclass(frozen=True)
class Column:
    name: str
    data_type: str
    is_generated: bool = False
    description: str = ""


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Column, ...]
    primary_key: tuple[str, ...] = ()
    foreign_keys: Mapping[str, str] = _EMPTY_FK
    row_count: int | None = None
    description: str = ""

    def __post_init__(self) -> None:
        """Wrap foreign_keys in MappingProxyType to ensure immutability.

        Works even with frozen=True via object.__setattr__.
        Idempotent: wrapping an already-wrapped mapping is safe.
        """
        if not isinstance(self.foreign_keys, MappingProxyType):
            object.__setattr__(self, "foreign_keys", MappingProxyType(self.foreign_keys))

    def references(self) -> frozenset[str]:
        """Tên các bảng mà bảng này trỏ tới qua khóa ngoại."""
        return frozenset(ref.split("(")[0] for ref in self.foreign_keys.values())


def tables_from_info_box(info_box: dict[str, Any]) -> tuple[Table, ...]:
    """Chuyển định dạng info_box JSON hiện có sang Table.

    Adapter tạm thời: nó tồn tại để pha 1 chạy được trên dữ liệu đang có.
    Khi đường onboarding (plan 3) hoàn thành, nguồn sẽ là schema.yaml.
    """
    fks: dict[str, dict[str, str]] = {}
    for raw in info_box.get("tables", []):
        for fk in raw.get("foreign_keys", []) or []:
            tbl, _, col = str(fk["references"]).partition(".")
            fks.setdefault(raw["table_name"], {})[fk["column"]] = f"{tbl}({col})"

    for hint in info_box.get("cross_domain_hints", []) or []:
        fks.setdefault(hint["from_table"], {})[hint["from_column"]] = (
            f"{hint['to_table']}({hint['to_column']})"
        )

    tables = []
    for raw in info_box.get("tables", []):
        name = raw["table_name"]
        tables.append(Table(
            name=name,
            columns=tuple(
                Column(
                    name=c["name"],
                    data_type=c["data_type"],
                    is_generated=bool(c.get("is_generated", False)),
                )
                for c in raw.get("columns", [])
            ),
            primary_key=tuple(raw.get("primary_key") or ()),
            foreign_keys=MappingProxyType(dict(fks.get(name, {}))),
            row_count=raw.get("row_count"),
            description=raw.get("description", ""),
        ))
    return tuple(tables)
