"""Đọc/ghi thư mục `profile/` — trạng thái onboarding của một khách hàng.

Bố cục:
    profile/
      profile.json    dsn (KHÔNG mật khẩu), grants, fingerprint, schema_mode
      structure.json  cấu trúc bảng/cột, KHÔNG có dữ liệu thật
      schema.yaml     chú giải — file người sửa được

Tách `structure.json` khỏi `schema.yaml` là có chủ ý: cấu trúc do máy sinh
và bị ghi đè mỗi lần làm mới, chú giải do người sửa và phải sống sót. Trộn
hai thứ vào một file sẽ khiến một lần làm mới xoá công sức của khách.

Mật khẩu DB không bao giờ chạm đĩa. `profile/` bị copy đi khi hỗ trợ kỹ
thuật — cùng lý do `perception/schema_model.py` cố ý không mang
`sample_rows`. `write_profile` bóc mật khẩu khỏi DSN trước khi ghi;
`read_profile` chắp lại từ biến môi trường `ADBA_DB_PASSWORD` nếu có. Nếu
biến đó không được set, `read_profile` KHÔNG lỗi — nó trả DSN không mật
khẩu nguyên vẹn, vì có nơi gọi `read_profile` chỉ để đọc cấu trúc/grants,
không bao giờ mở kết nối.
"""

from __future__ import annotations

import json
import os
import urllib.parse
from collections.abc import Mapping, Sequence
from pathlib import Path

from perception.annotations import (
    SchemaAnnotations,
    apply_annotations,
    load_annotations,
    save_annotations,
)
from perception.connection_profile import (
    DEFAULT_THRESHOLD_TOKENS,
    ConnectionProfile,
    build_profile,
    schema_fingerprint,
)
from perception.schema_model import Column, Table

PROFILE_JSON = "profile.json"
STRUCTURE_JSON = "structure.json"
SCHEMA_YAML = "schema.yaml"

PASSWORD_ENV_VAR = "ADBA_DB_PASSWORD"


def _host_port(parts: urllib.parse.SplitResult) -> str:
    host = parts.hostname or ""
    return f"{host}:{parts.port}" if parts.port is not None else host


def _strip_password(dsn: str) -> str:
    """Trả DSN không mật khẩu, dùng `urllib.parse` để tránh cắt nhầm '@'/':'
    thuộc về chính mật khẩu (str.split ngây thơ sẽ cắt sai)."""
    parts = urllib.parse.urlsplit(dsn)
    if parts.password is None:
        return dsn
    host_port = _host_port(parts)
    netloc = f"{parts.username}@{host_port}" if parts.username else host_port
    return urllib.parse.urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _restore_password(dsn: str, password: str | None) -> str:
    """Chắp mật khẩu (từ biến môi trường) vào DSN không mật khẩu.

    Không làm gì nếu `password` rỗng/None (biến môi trường chưa set), hoặc
    nếu `dsn` đã tự mang mật khẩu sẵn (không phải trường hợp bình thường,
    nhưng không nên ghi đè nếu xảy ra).
    """
    if not password:
        return dsn
    parts = urllib.parse.urlsplit(dsn)
    if parts.password is not None:
        return dsn
    host_port = _host_port(parts)
    userinfo = f"{parts.username}:{password}" if parts.username else f":{password}"
    netloc = f"{userinfo}@{host_port}"
    return urllib.parse.urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def structure_to_plain(tables: Sequence[Table]) -> list[dict]:
    return [
        {
            "name": t.name,
            "primary_key": list(t.primary_key),
            "foreign_keys": dict(t.foreign_keys),
            "row_count": t.row_count,
            "columns": [
                {"name": c.name, "data_type": c.data_type, "is_generated": c.is_generated}
                for c in t.columns
            ],
        }
        for t in tables
    ]


def structure_from_plain(raw: Sequence[Mapping]) -> tuple[Table, ...]:
    return tuple(
        Table(
            name=t["name"],
            columns=tuple(
                Column(
                    name=c["name"],
                    data_type=c["data_type"],
                    is_generated=bool(c.get("is_generated", False)),
                )
                for c in t.get("columns", [])
            ),
            primary_key=tuple(t.get("primary_key", ())),
            foreign_keys=dict(t.get("foreign_keys", {})),
            row_count=t.get("row_count"),
        )
        for t in raw
    )


def write_profile(
    directory: Path | str,
    *,
    dsn: str,
    tables: Sequence[Table],
    annotations: SchemaAnnotations,
    grants: Mapping[str, frozenset[str]],
    threshold_tokens: int = DEFAULT_THRESHOLD_TOKENS,
) -> None:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)

    (d / STRUCTURE_JSON).write_text(
        json.dumps(structure_to_plain(tables), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_annotations(annotations, d / SCHEMA_YAML)

    annotated = apply_annotations(tables, annotations)
    profile = build_profile(
        dsn=dsn, tables=annotated, grants=grants, threshold_tokens=threshold_tokens
    )
    (d / PROFILE_JSON).write_text(
        json.dumps(
            {
                "dsn": _strip_password(dsn),
                "grants": {u: sorted(t) for u, t in grants.items()},
                "fingerprint": profile.fingerprint,
                "schema_mode": profile.schema_mode,
                "threshold_tokens": threshold_tokens,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def read_profile(directory: Path | str) -> ConnectionProfile:
    """Dựng lại `ConnectionProfile` từ đĩa, chú giải đã gắn sẵn vào bảng.

    Mật khẩu được chắp lại từ biến môi trường `ADBA_DB_PASSWORD` (không có
    trên đĩa — xem module docstring). Nếu biến đó không được set, DSN trả
    về không có mật khẩu; hàm này không lỗi vì lý do đó.
    """
    d = Path(directory)
    meta_path = d / PROFILE_JSON
    if not meta_path.exists():
        raise FileNotFoundError(f"Không có {PROFILE_JSON} trong {d}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    tables = structure_from_plain(json.loads((d / STRUCTURE_JSON).read_text(encoding="utf-8")))
    annotated = apply_annotations(tables, load_annotations(d / SCHEMA_YAML))
    dsn = _restore_password(meta["dsn"], os.environ.get(PASSWORD_ENV_VAR))

    return build_profile(
        dsn=dsn,
        tables=annotated,
        grants={u: frozenset(t) for u, t in meta.get("grants", {}).items()},
        threshold_tokens=meta.get("threshold_tokens", DEFAULT_THRESHOLD_TOKENS),
    )


def profile_is_stale(directory: Path | str, tables: Sequence[Table]) -> bool:
    """Cấu trúc thật đã khác với lúc dựng profile chưa?

    So bằng fingerprint, vốn cố ý bỏ qua `row_count` và mô tả — chỉ thay đổi
    cấu trúc mới đáng khiến profile bị coi là cũ.
    """
    meta = json.loads((Path(directory) / PROFILE_JSON).read_text(encoding="utf-8"))
    return schema_fingerprint(tables) != meta["fingerprint"]
