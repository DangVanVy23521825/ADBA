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
`read_profile` chắp lại từ biến môi trường `ADBA_DB_PASSWORD`, NHƯNG chỉ
khi `profile.json` xác nhận có mật khẩu thật sự bị bóc lúc ghi (cờ
`dsn_password_stripped`) — không suy đoán từ hình dạng DSN hay từ việc
biến môi trường có được set hay không. Nếu DSN gốc vốn không có mật khẩu,
hoặc biến môi trường chưa set, `read_profile` KHÔNG lỗi — nó trả DSN
không mật khẩu nguyên vẹn, vì có nơi gọi `read_profile` chỉ để đọc cấu
trúc/grants, không bao giờ mở kết nối.

DSN đưa vào `write_profile` phải là một URI percent-encode HỢP LỆ theo RFC
3986. Một mật khẩu chứa '/', '?', hoặc '#' mà chưa percent-encode sẽ cắt
ngang authority và khiến `urlsplit` đọc sai vị trí mật khẩu; `write_profile`
phát hiện trường hợp đó và raise `ValueError` thay vì âm thầm ghi một DSN
bị đọc sai (và mật khẩu rơi nguyên dạng vào path/query/fragment) ra đĩa.
"""

from __future__ import annotations

import dataclasses
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


def dsn_host_for_error(dsn: str) -> str:
    """`host[:port]` của một DSN, KHÔNG BAO GIỜ username hay mật khẩu.

    Dùng bởi `onboard.py` khi một lệnh gọi DB thất bại và cần nêu DSN nào
    trong thông báo lỗi (traceback bị bọc thành `OnboardError`, xem C2):
    thông báo phải nói được HOST nào đang có vấn đề mà không lặp lại toàn
    bộ DSN — chính chỗ mật khẩu có thể lộ ra (libpq echo nguyên DSN vào lỗi
    parse khi nó không đúng dạng `key=value` cổ).

    Dùng CHUNG kỹ thuật với `_strip_password`: chỉ đọc `.hostname`/`.port`
    từ `urlsplit`, không bao giờ chạm `.username`/`.password`. Nếu DSN
    không parse được thành URI có host (đáng lẽ không tới đây, vì
    `onboard._resolve_dsn` đã chặn từ trước — nhưng hàm này không được tự
    nó là nguồn traceback mới khi gọi từ một khối `except`), trả `"?"` thay
    vì ném lỗi.
    """
    try:
        parts = urllib.parse.urlsplit(dsn)
        return _host_port(parts) or "?"
    except Exception:
        return "?"


def _strip_password(dsn: str) -> tuple[str, bool]:
    """Trả `(dsn không mật khẩu, đã_bóc_mật_khẩu_thật?)`.

    `parts.password is None` có HAI nguyên nhân khác nhau, và chỉ một
    trong hai là an toàn để coi là "không có mật khẩu":

    1. DSN thật sự không có mật khẩu (`postgresql://user@host/db` hoặc
       `postgresql://host/db`) — an toàn, trả nguyên DSN, `stripped=False`.
    2. Mật khẩu có '/', '?', hoặc '#' CHƯA percent-encode. Theo RFC 3986,
       ba ký tự đó cắt ngang authority, nên `urlsplit` đọc authority bị cụt
       và `.password` trả None dù DSN thật sự mang mật khẩu — phần còn lại
       của mật khẩu rơi nguyên dạng vào path/query/fragment. Coi trường
       hợp này là "không có mật khẩu" sẽ ghi cleartext ra đĩa qua đường
       vòng path/query/fragment thay vì qua netloc.

    Phân biệt hai trường hợp bằng một dấu hiệu gián tiếp nhưng đáng tin:
    authority hợp lệ không bao giờ để lọt '@' sang path/query/fragment
    (chính '@' đó là ký tự phân tách userinfo/host). Nếu nó xuất hiện ở
    đó, authority đã bị cắt cụt bởi '/', '?', hoặc '#' chưa encode — raise
    thay vì âm thầm coi là an toàn.
    """
    parts = urllib.parse.urlsplit(dsn)
    if parts.password is not None:
        host_port = _host_port(parts)
        netloc = f"{parts.username}@{host_port}" if parts.username else host_port
        stripped = urllib.parse.urlunsplit(
            (parts.scheme, netloc, parts.path, parts.query, parts.fragment)
        )
        return stripped, True

    if "@" in parts.path or "@" in parts.query or "@" in parts.fragment:
        # KHÔNG được nội suy `dsn` (hay bất kỳ phần nào rút ra từ authority
        # đã bị cắt cụt, như `parts.hostname` ở đây — nó thật ra là mảnh
        # username bị đọc nhầm, không đáng tin) vào thông điệp lỗi.
        # ValueError này sẽ đi vào traceback, log lỗi, dịch vụ theo dõi
        # lỗi — một nơi rò rỉ còn tệ hơn profile.json, vì profile.json chỉ
        # nằm trên máy khách còn log có thể bị gửi ra ngoài. Chỉ nêu
        # scheme (không mang thông tin nhạy cảm); operator đang cầm sẵn
        # DSN gốc, không cần được nhắc lại.
        raise ValueError(
            f"DSN dùng scheme {parts.scheme!r} có vẻ chứa mật khẩu chưa "
            "percent-encode. Ký tự '/', '?', hoặc '#' trong mật khẩu cắt "
            "ngang authority (RFC 3986), khiến vị trí mật khẩu bị đọc sai. "
            "Hãy percent-encode mật khẩu trước khi ghép vào DSN, ví dụ "
            "bằng urllib.parse.quote(password, safe='')."
        )
    return dsn, False


def _restore_password(dsn: str, password: str | None, *, stripped: bool) -> str:
    """Chắp mật khẩu (từ biến môi trường) vào DSN không mật khẩu.

    Chỉ chắp khi `stripped` đúng — nghĩa là `write_profile` đã XÁC NHẬN
    (cờ `dsn_password_stripped` trên đĩa) có bóc một mật khẩu thật ra khỏi
    DSN gốc lúc ghi. Không suy đoán từ hình dạng DSN hiện tại hay từ việc
    biến môi trường có được set hay không: một DSN vốn không có mật khẩu
    (không userinfo, hoặc có username nhưng không mật khẩu) phải giữ
    nguyên bất kể `ADBA_DB_PASSWORD` — nếu không, một biến môi trường đặt
    toàn host sẽ gắn nhầm mật khẩu vào những profile vốn dùng cách xác
    thực khác (vd. peer auth, IAM token).

    Mật khẩu được percent-encode bằng `urllib.parse.quote(safe="")` trước
    khi ghép vào netloc. Điều này đối xứng với việc `urlsplit` KHÔNG tự
    giải mã `.password` (nó trả nguyên chuỗi percent-encode nếu có): nếu
    ghép thẳng chuỗi mật khẩu thô vào netloc, một mật khẩu vốn được
    percent-encode đúng trong DSN gốc sẽ ghép lại SAI — ví dụ '%2F' của
    mật khẩu gốc tự nhiên biến thành '/' sống, dịch chuyển luôn ranh giới
    authority của DSN dựng lại.
    """
    if not stripped or not password:
        return dsn
    parts = urllib.parse.urlsplit(dsn)
    host_port = _host_port(parts)
    encoded_password = urllib.parse.quote(password, safe="")
    userinfo = (
        f"{parts.username}:{encoded_password}" if parts.username else f":{encoded_password}"
    )
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
    # Xác thực/bóc mật khẩu TRƯỚC khi ghi bất kỳ file nào: một DSN không
    # parse được đúng (xem _strip_password) không được để lại trạng thái
    # nửa vời (structure.json/schema.yaml đã ghi, profile.json thì chưa).
    stripped_dsn, password_was_stripped = _strip_password(dsn)

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
                "dsn": stripped_dsn,
                "dsn_password_stripped": password_was_stripped,
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

    Mật khẩu được chắp lại từ biến môi trường `ADBA_DB_PASSWORD`, CHỈ khi
    `profile.json` xác nhận (`dsn_password_stripped: true`) rằng có một
    mật khẩu thật đã bị bóc lúc ghi — không suy đoán từ hình dạng DSN hay
    từ việc biến môi trường có được set hay không (xem `_restore_password`).
    `profile.json` ghi từ trước khi trường này tồn tại mặc định về hướng
    an toàn: `False`, tức không chắp gì cả. Nếu không có gì để chắp — vì
    cờ là False, hoặc vì biến môi trường chưa set — hàm này không lỗi; nó
    trả DSN đọc từ đĩa nguyên vẹn, vì có nơi gọi `read_profile` chỉ để đọc
    cấu trúc/grants, không bao giờ mở kết nối.

    `schema_mode` LẤY TỪ `profile.json`, KHÔNG được recompute ở đây. `build`
    là nơi DUY NHẤT được quyết công tắc này (spec 3.3/10.3): `build_profile`
    ở trên vẫn được gọi (nó dựng `tables`/`fingerprint`/render), nhưng giá
    trị `schema_mode` nó trả về bị GHI ĐÈ bằng giá trị đọc từ đĩa ngay sau
    đó. Trước bản sửa này, mode bị tính lại từ kích thước RENDER — và vì
    `annotated` (tham số `tables` truyền vào `build_profile`) mang mô tả từ
    `schema.yaml` HIỆN TẠI, một lượt chú giải thêm mô tả sau khi `build` đã
    chạy có thể tự đẩy mode từ "full" (lúc build, lúc `verify` đo recall)
    sang "retrieval" (lúc chạy thật) mà không ai biết — `verify` khi đó
    chứng nhận một cấu hình khách hàng không hề dùng. Xem task C1/C3.

    `fingerprint` KHÔNG bị lỗi tương tự nên vẫn để `build_profile` tự tính:
    `schema_fingerprint` cố ý bỏ qua mô tả (chỉ tên bảng/cột/kiểu/PK/FK),
    nên `apply_annotations` (chỉ đổi trường `description`) không đổi được
    giá trị fingerprint dù chú giải có đổi bao nhiêu. Hơn nữa không nơi nào
    trong code sản phẩm đọc `ConnectionProfile.fingerprint` — chỉ
    `write_profile` ghi nó ra `profile.json`, còn `profile_is_stale` tự tính
    `schema_fingerprint` độc lập trên `tables` truyền vào, không đi qua
    `read_profile`. Recompute ở đây vô hại, và đúng ra còn phản ánh đúng
    hơn nếu `structure.json` đổi mà chưa `build` lại.

    Một `profile.json` ghi TRƯỚC bản sửa này (thiếu khoá `schema_mode`) mặc
    định về `"full"` — hướng an toàn được chọn có chủ đích, NGƯỢC với cách
    `dsn_password_stripped` mặc định `False`: đoán sai "retrieval" có thể
    âm thầm bỏ sót bảng JOIN (đúng thứ lỗi C3 đang sửa); đoán sai "full" chỉ
    tốn thêm token — hỏng ồn ào (context/token vượt ngưỡng) chứ không âm
    thầm cho ra câu trả lời sai.

    Cùng lý do đó, một giá trị CÓ MẶT nhưng không hợp lệ (`"Full"`, `""`,
    `null`, `"retrieval "` — file bị sửa tay hoặc cắt cụt) cũng phải rơi về
    `"full"`, KHÔNG được lọt xuống `schema_context.py`: ở đó phép so sánh là
    `== "full"` với `retrieval` là nhánh `else`, nên bất cứ giá trị lạ nào
    không được chặn ở đây sẽ tự động chọn nhánh retrieval — đúng thứ lỗi
    N1 đang chặn.
    """
    d = Path(directory)
    meta_path = d / PROFILE_JSON
    if not meta_path.exists():
        raise FileNotFoundError(f"Không có {PROFILE_JSON} trong {d}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    tables = structure_from_plain(json.loads((d / STRUCTURE_JSON).read_text(encoding="utf-8")))
    annotated = apply_annotations(tables, load_annotations(d / SCHEMA_YAML))
    dsn = _restore_password(
        meta["dsn"],
        os.environ.get(PASSWORD_ENV_VAR),
        stripped=bool(meta.get("dsn_password_stripped", False)),
    )

    profile = build_profile(
        dsn=dsn,
        tables=annotated,
        grants={u: frozenset(t) for u, t in meta.get("grants", {}).items()},
        threshold_tokens=meta.get("threshold_tokens", DEFAULT_THRESHOLD_TOKENS),
    )
    raw_mode = meta.get("schema_mode", "full")
    schema_mode = raw_mode if raw_mode in ("full", "retrieval") else "full"
    return dataclasses.replace(profile, schema_mode=schema_mode)


def profile_is_stale(directory: Path | str, tables: Sequence[Table]) -> bool:
    """Cấu trúc thật đã khác với lúc dựng profile chưa?

    So bằng fingerprint, vốn cố ý bỏ qua `row_count` và mô tả — chỉ thay đổi
    cấu trúc mới đáng khiến profile bị coi là cũ.
    """
    meta = json.loads((Path(directory) / PROFILE_JSON).read_text(encoding="utf-8"))
    return schema_fingerprint(tables) != meta["fingerprint"]
