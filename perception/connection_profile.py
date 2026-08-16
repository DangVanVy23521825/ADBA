"""Hồ sơ kết nối: một object mang mọi thứ về "đang nói chuyện với DB nào".

Thay cho ba biến toàn cục rải rác trước đây (DATABASE_URL, _ALLOWED_TABLES,
info_box_*.json). Xem spec mục 3.4.

QUAN TRỌNG — hai tập bảng, khác bản chất, không được gộp (spec 3.4.1):

  permitted_tables(user)     BẢO MẬT.       Theo người dùng. Dẫn ra ở đây.
  retrieved_tables(question) NỘI DUNG PROMPT. Theo câu hỏi. Dẫn ra ở schema_context.

Bên thực thi SQL chỉ được dùng cái thứ nhất, và phải tự gọi hàm này chứ
không nhận tập quyền từ bên gọi. Nếu nhận từ bên gọi thì bên bị ràng buộc
đang tự khai ràng buộc của mình — guard tụt xuống thành lời hứa.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from perception.render_schema import estimate_tokens, render_schema
from perception.schema_model import Table

ALL_TABLES = "*"
DEFAULT_THRESHOLD_TOKENS = 6000


@dataclass(frozen=True)
class ConnectionProfile:
    dsn: str
    tables: tuple[Table, ...]
    grants: Mapping[str, frozenset[str]]
    schema_mode: str  # "full" | "retrieval"
    fingerprint: str

    def table_names(self) -> frozenset[str]:
        return frozenset(t.name for t in self.tables)

    def by_name(self) -> dict[str, Table]:
        return {t.name: t for t in self.tables}


def schema_fingerprint(tables: Sequence[Table]) -> str:
    """Hash của cấu trúc schema — tên bảng và tên/kiểu/is_generated của cột.

    Cố ý bỏ qua row_count và description: row_count đổi mỗi ngày và mô tả
    do người sửa; cả hai đều không phải thay đổi schema. Chỉ thay đổi cấu
    trúc mới được kích hoạt profile_stale (spec mục 5.2).

    is_generated CÓ tính vào hash — nó không cosmetic: một cột chuyển
    thành GENERATED đổi hẳn SQL nào là hợp lệ trên cột đó (prompt SQL có
    quy tắc riêng: cột GENERATED chỉ đọc, không được ghi), nên đó đúng là
    thay đổi cấu trúc mà profile_stale phải bắt được.
    """
    parts = []
    for t in sorted(tables, key=lambda x: x.name):
        cols = ",".join(
            f"{c.name}:{c.data_type}:{c.is_generated}"
            for c in sorted(t.columns, key=lambda c: c.name)
        )
        pk = ",".join(sorted(t.primary_key))
        fk = ",".join(f"{k}->{v}" for k, v in sorted(t.foreign_keys.items()))
        parts.append(f"{t.name}|{cols}|{pk}|{fk}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def build_profile(
    dsn: str,
    tables: Sequence[Table],
    grants: Mapping[str, frozenset[str]],
    threshold_tokens: int = DEFAULT_THRESHOLD_TOKENS,
) -> ConnectionProfile:
    """Dựng profile và quyết công tắc schema_mode MỘT LẦN, lúc cài đặt.

    Đây không phải fallback động: một nhánh code, một công tắc, quyết một
    lần. Xem spec mục 3.3 và mục 10.3.

    grants được đóng băng sâu — cả mapping ngoài (MappingProxyType) lẫn từng
    tập quyền bên trong (frozenset) — cùng cách Table.foreign_keys tự khóa
    ở schema_model.py. Nếu chỉ dict(grants) nông thì profile.grants vẫn là
    dict thường (gán thẳng phần tử được) và giá trị vẫn là object gốc của
    caller (caller mutate sau khi trả về thì permitted_tables đọc thấy ngay,
    không cần build lại profile) — ranh giới bảo mật tụt xuống lời hứa.
    """
    tables = tuple(tables)
    size = estimate_tokens(render_schema(tables))
    return ConnectionProfile(
        dsn=dsn,
        tables=tables,
        grants=MappingProxyType({u: frozenset(v) for u, v in grants.items()}),
        schema_mode="full" if size <= threshold_tokens else "retrieval",
        fingerprint=schema_fingerprint(tables),
    )


def permitted_tables(profile: ConnectionProfile, user: str) -> frozenset[str]:
    """Tập bảng người dùng được phép chạm. Ranh giới bảo mật thật.

    Mặc định đóng: người dùng không có mục trong grants thì không thấy gì.
    Tên bảng trong grant mà schema không có sẽ bị bỏ, để một grant cũ không
    mở ra thứ gì ngoài ý muốn khi schema đổi.
    """
    granted = profile.grants.get(user)
    if not granted:
        return frozenset()
    existing = profile.table_names()
    if ALL_TABLES in granted:
        return existing
    return frozenset(granted) & existing
