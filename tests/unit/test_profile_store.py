import dataclasses
import json
import urllib.parse

import pytest

from perception.annotations import Annotation, SchemaAnnotations
from perception.connection_profile import ALL_TABLES, permitted_tables
from perception.profile_store import (
    PROFILE_JSON,
    SCHEMA_YAML,
    STRUCTURE_JSON,
    profile_is_stale,
    read_profile,
    write_profile,
)
from perception.schema_model import Column
from tests.fixtures.mini_schema import MINI_TABLES

ANN = SchemaAnnotations(
    tables={"orders": Annotation(text="Đơn hàng", reviewed_by="human")},
    columns={"orders": {"amount": Annotation(text="Số tiền")}},
)
GRANTS = {"analyst": frozenset({"orders", "customers"}), "admin": frozenset({ALL_TABLES})}

DSN = "postgresql://u:p@h:5432/d"


def _write(tmp_path, dsn=DSN):
    write_profile(tmp_path, dsn=dsn,
                  tables=MINI_TABLES, annotations=ANN, grants=GRANTS)
    return tmp_path


def test_round_trips_into_a_usable_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("ADBA_DB_PASSWORD", "p")
    profile = read_profile(_write(tmp_path))
    assert profile.dsn == DSN
    assert {t.name for t in profile.tables} == {t.name for t in MINI_TABLES}


def test_annotations_are_applied_to_the_loaded_tables(tmp_path, monkeypatch):
    monkeypatch.setenv("ADBA_DB_PASSWORD", "p")
    profile = read_profile(_write(tmp_path))
    orders = next(t for t in profile.tables if t.name == "orders")
    assert orders.description == "Đơn hàng"
    assert next(c for c in orders.columns if c.name == "amount").description == "Số tiền"


def test_grants_survive_the_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("ADBA_DB_PASSWORD", "p")
    profile = read_profile(_write(tmp_path))
    assert permitted_tables(profile, "analyst") == frozenset({"orders", "customers"})


def test_no_sample_row_data_is_written_anywhere(tmp_path, monkeypatch):
    """Ràng buộc spec: profile bị copy đi khi hỗ trợ kỹ thuật — dữ liệu MẪU
    (`sample_rows`, tức DÒNG DỮ LIỆU THẬT của khách) không bao giờ được ghi
    ra đĩa trong `profile/`.

    M3 — test cũ ở đây tìm literal `"sample_rows"` (TÊN HÀM Python) trong
    nội dung file, không phải dữ liệu hàng thật. Literal đó không bao giờ
    xuất hiện dù có rò rỉ dữ liệu thật hay không: `write_profile` (được
    gọi qua `_write` ở trên) không nhận tham số `samples` — không có
    ĐƯỜNG ĐI nào để chuỗi "sample_rows" lọt vào output, nên assertion cũ
    luôn đúng bất kể code có đúng hay không.

    Sửa: chạy CẢ pipeline thật (`extract` → `annotate` → `build`) với
    `sample_rows` (mock ở tầng `onboard`, nơi nó thật sự được TIÊU THỤ để
    dựng prompt cho LLM — xem `onboard.cmd_annotate`) trả về một giá trị
    SENTINEL đặc trưng, mô phỏng dữ liệu hàng thật (vd. một email/số điện
    thoại khách), rồi xác nhận sentinel đó — không phải tên hàm — không
    xuất hiện ở bất kỳ đâu trong `profile/` sau khi `build` ghi xong.
    """
    from onboard import cmd_annotate, cmd_build, cmd_extract

    monkeypatch.setattr("onboard.introspect_schema", lambda dsn, **kw: MINI_TABLES)  # noqa: ARG005
    sentinel = "SENTINEL-RAW-ROW-VALUE-9f3a7c21"
    monkeypatch.setattr(
        "onboard.sample_rows",
        lambda dsn, table, n=5: [{"some_column": sentinel}],  # noqa: ARG005
    )
    cmd_extract("postgresql://x", tmp_path)
    reply = '{"table": {"text": "mô tả", "confidence": "high"}, "columns": {}}'
    cmd_annotate(tmp_path, "postgresql://x", invoke=lambda s, u: reply)  # noqa: ARG005
    cmd_build(tmp_path, "postgresql://x", grants={"admin": frozenset({ALL_TABLES})})

    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    blob = "".join(p.read_text(encoding="utf-8", errors="ignore") for p in files)
    assert sentinel not in blob
    # Bất biến thật: chạy hết extract → annotate → build sinh đúng bốn file
    # này — ba file `write_profile` ghi, cộng `schema.yaml.bak` (bản sao
    # lưu nguyên tử của `cmd_annotate`, xem C1). Đây không phải trần cho
    # mọi thứ thư mục profile được phép chứa mãi mãi — một task sau ghi
    # thêm report.md vào cùng thư mục là hợp lệ.
    assert {p.name for p in files} == {
        PROFILE_JSON, STRUCTURE_JSON, SCHEMA_YAML, f"{SCHEMA_YAML}.bak",
    }


def test_the_schema_yaml_is_editable_by_hand(tmp_path):
    assert (_write(tmp_path) / "schema.yaml").exists()


def test_a_fresh_profile_is_not_stale(tmp_path):
    assert profile_is_stale(_write(tmp_path), MINI_TABLES) is False


def test_adding_a_column_makes_the_profile_stale(tmp_path):
    _write(tmp_path)
    grown = list(MINI_TABLES)
    grown[0] = dataclasses.replace(
        grown[0], columns=grown[0].columns + (Column("moi", "integer"),)
    )
    assert profile_is_stale(tmp_path, tuple(grown)) is True


def test_editing_a_description_does_not_make_it_stale(tmp_path):
    """Fingerprint theo dõi CẤU TRÚC. Chú giải đổi là chuyện bình thường."""
    _write(tmp_path)
    edited = tuple(dataclasses.replace(t, description="mô tả khác") for t in MINI_TABLES)
    assert profile_is_stale(tmp_path, edited) is False


def test_reading_a_directory_without_a_profile_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_profile(tmp_path / "khong-co")


# --- Amendment 1: mật khẩu DB không được ghi ra đĩa ---
#
# profile/ bị copy đi khi hỗ trợ kỹ thuật (cùng lý do test_no_sample_row_data
# ở trên tồn tại). Nếu đã cẩn thận không ghi sample rows vì lý do đó thì
# cũng không được ghi mật khẩu production ra cùng thư mục.

def test_password_is_not_written_to_disk_anywhere(tmp_path):
    """Mật khẩu chứa cả '@' và ':' để buộc code dùng urllib.parse thật sự,
    không phải str.split/rpartition tay — cả hai ký tự đó đều là ký tự
    phân tách trong cú pháp DSN.
    """
    dsn = "postgresql://u:p@ss:wd@h:5432/d"
    _write(tmp_path, dsn=dsn)
    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    blob = "".join(p.read_text(encoding="utf-8", errors="ignore") for p in files)
    assert "p@ss:wd" not in blob


def test_password_round_trips_via_environment_variable(tmp_path, monkeypatch):
    # DSN gốc phải là URI percent-encode HỢP LỆ — đây là điều write_profile
    # bây giờ đòi hỏi (xem test_write_profile_rejects_dsn_with_reserved_...
    # ở dưới). '@' và ':' vẫn parse đúng không cần encode (rpartition/
    # partition xử lý được), nhưng ta encode cho đúng chuẩn RFC 3986.
    password = "p@ss:wd"
    dsn = f"postgresql://u:{urllib.parse.quote(password, safe='')}@h:5432/d"
    monkeypatch.setenv("ADBA_DB_PASSWORD", password)
    profile = read_profile(_write(tmp_path, dsn=dsn))
    assert profile.dsn == dsn


def test_read_profile_without_password_env_var_returns_dsn_without_password(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("ADBA_DB_PASSWORD", raising=False)
    profile = read_profile(_write(tmp_path))
    assert profile.dsn == "postgresql://u@h:5432/d"


# --- Fix round 1: _strip_password conflated "cannot parse a password" with
# "there is no password" (Critical 1); _restore_password spliced the raw
# password back without re-encoding it (Critical 2); read_profile injected
# ADBA_DB_PASSWORD into DSNs that never had a password to begin with
# (Important 3). See task-5-report.md, "Fix round 1" section.

@pytest.mark.parametrize(
    "dsn",
    [
        "postgresql://u:sl/ash@h:5432/d",
        "postgresql://u:ha#sh@h/d",
        "postgresql://u:qu?ery@h/d",
    ],
)
def test_write_profile_rejects_dsn_with_reserved_chars_that_truncate_authority(
    tmp_path, dsn
):
    """'/', '?', '#' cắt ngang authority theo RFC 3986. Nếu mật khẩu chứa
    chúng mà chưa percent-encode, urlsplit đọc authority bị cụt và
    `parts.password` về None dù DSN thật sự mang mật khẩu — mật khẩu rơi
    nguyên dạng vào path/query/fragment. Phải raise, không được âm thầm
    ghi DSN đó ra đĩa coi như "không có mật khẩu".
    """
    with pytest.raises(ValueError):
        _write(tmp_path, dsn=dsn)


def test_write_profile_passes_through_dsn_that_has_username_but_no_password(tmp_path):
    dsn = "postgresql://user@host/db"
    _write(tmp_path, dsn=dsn)
    meta = json.loads((tmp_path / "profile.json").read_text(encoding="utf-8"))
    assert meta["dsn"] == dsn
    assert meta["dsn_password_stripped"] is False


def test_write_profile_passes_through_dsn_with_no_userinfo_at_all(tmp_path):
    dsn = "postgresql://host/db"
    _write(tmp_path, dsn=dsn)
    meta = json.loads((tmp_path / "profile.json").read_text(encoding="utf-8"))
    assert meta["dsn"] == dsn
    assert meta["dsn_password_stripped"] is False


def test_password_with_all_reserved_chars_round_trips_byte_identical(tmp_path, monkeypatch):
    """Mật khẩu chứa '@', ':', '/', '#', và một chuỗi đã percent-encode
    sẵn ('%'). _restore_password phải percent-encode LẠI mật khẩu thô lấy
    từ biến môi trường trước khi ghép vào netloc — ghép thẳng chuỗi thô sẽ
    đổi ranh giới authority và cho ra một DSN khác, sai."""
    password = "a@b:c/d#e%f"
    encoded = urllib.parse.quote(password, safe="")
    dsn = f"postgresql://u:{encoded}@h:5432/d"
    monkeypatch.setenv("ADBA_DB_PASSWORD", password)
    profile = read_profile(_write(tmp_path, dsn=dsn))
    assert profile.dsn.encode("utf-8") == dsn.encode("utf-8")


def test_password_free_dsn_stays_password_free_even_with_env_var_set(tmp_path, monkeypatch):
    """Biến môi trường đặt toàn host không được gắn mật khẩu vào một DSN
    vốn không có mật khẩu (spec: ghi lại SỰ THẬT là đã bóc, không đoán)."""
    dsn = "postgresql://user@host/db"
    monkeypatch.setenv("ADBA_DB_PASSWORD", "khong-duoc-xuat-hien")
    profile = read_profile(_write(tmp_path, dsn=dsn))
    assert profile.dsn == dsn


def test_no_userinfo_dsn_round_trips_unchanged_even_with_env_var_set(tmp_path, monkeypatch):
    dsn = "postgresql://host/db"
    monkeypatch.setenv("ADBA_DB_PASSWORD", "khong-duoc-xuat-hien")
    profile = read_profile(_write(tmp_path, dsn=dsn))
    assert profile.dsn == dsn


def test_missing_dsn_password_stripped_field_defaults_to_false(tmp_path, monkeypatch):
    """profile.json ghi trước khi trường này tồn tại phải mặc định về
    hướng an toàn: không chắp mật khẩu vào."""
    dsn = "postgresql://u:p@h:5432/d"
    _write(tmp_path, dsn=dsn)
    meta_path = tmp_path / "profile.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    stripped_dsn = meta["dsn"]
    del meta["dsn_password_stripped"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    monkeypatch.setenv("ADBA_DB_PASSWORD", "p")
    profile = read_profile(tmp_path)
    assert profile.dsn == stripped_dsn


# --- Fix round 2: thông điệp ValueError không được lặp lại mật khẩu; khôi
# phục một mật khẩu THÔ (chưa encode, như operator gõ tay) qua biến môi
# trường vẫn phải hoạt động đúng nghĩa, không chỉ đúng byte.

def test_reserved_char_error_does_not_leak_the_password(tmp_path):
    """Thông điệp lỗi từng lặp lại {dsn!r} — tức lặp lại mật khẩu — vào
    ValueError. Đó là một điểm rò rỉ TỆ HƠN profile.json: profile.json chỉ
    nằm trên máy khách, còn traceback/log lỗi có thể bị gửi ra ngoài qua
    dịch vụ theo dõi lỗi. Dùng mật khẩu đặc trưng để một match có ý nghĩa.
    """
    password = "distinctive-secret-xyz789/rest"
    dsn = f"postgresql://u:{password}@h:5432/d"
    with pytest.raises(ValueError) as exc_info:
        _write(tmp_path, dsn=dsn)
    assert "distinctive-secret-xyz789" not in str(exc_info.value)


def test_raw_password_round_trips_functionally_via_environment_variable(
    tmp_path, monkeypatch
):
    """Khác với test_password_with_all_reserved_chars_round_trips_byte_identical
    ở trên (so BYTE-FOR-BYTE toàn bộ DSN, xuất phát từ một DSN đã
    percent-encode SẴN — thứ một client DB thật sự tạo ra) — test này xuất
    phát từ một mật khẩu THÔ, như operator gõ tay: chứa '@' và ':' chưa
    percent-encode. Với đầu vào đó, `write_profile`/`read_profile` không
    hứa giữ nguyên HÌNH DẠNG chuỗi DSN gốc (DSN dựng lại sẽ percent-encode
    mật khẩu, khác chuỗi thô ban đầu) — chúng chỉ hứa khôi phục ĐÚNG mật
    khẩu. Vì vậy assertion ở đây percent-decode phần mật khẩu của DSN dựng
    lại rồi so với mật khẩu thô gốc, KHÔNG so cả chuỗi DSN.
    """
    password = "p@ss:wd"
    dsn = f"postgresql://u:{password}@h:5432/d"
    monkeypatch.setenv("ADBA_DB_PASSWORD", password)
    profile = read_profile(_write(tmp_path, dsn=dsn))
    recovered_password = urllib.parse.unquote(
        urllib.parse.urlsplit(profile.dsn).password
    )
    assert recovered_password == password


# ── C3: schema_mode chốt lúc build, không tính lại lúc đọc ──────────────────
#
# `read_profile` từng gọi lại `build_profile`, vốn TÍNH LẠI schema_mode từ
# kích thước bản render. Mà bản render đó gồm cả chú giải, nên mode trở thành
# hàm của việc analyst viết bao nhiêu chữ.
#
# Ngòi nổ là chính lời khuyên của hệ thống: `report.md` bảo operator "bổ sung
# mô tả cho các bảng bị thiếu rồi chạy lại". Thêm mô tả đúng là thứ đẩy một
# schema cận ngưỡng vượt qua nó. `verify` ĐẠT ở mode `full`, khách chạy mode
# `retrieval` với recall chưa ai đo. `profile_is_stale` không bắt được vì
# fingerprint cố ý bỏ qua description.


def test_annotating_a_built_profile_cannot_change_its_schema_mode(tmp_path):
    from perception.annotations import Annotation, save_annotations

    write_profile(
        tmp_path,
        dsn="postgresql://u:p@h:5432/d",
        tables=MINI_TABLES,
        annotations=SchemaAnnotations(),
        grants=GRANTS,
        threshold_tokens=200,
    )
    mode_at_build = json.loads(
        (tmp_path / "profile.json").read_text(encoding="utf-8")
    )["schema_mode"]
    mode_before = read_profile(tmp_path).schema_mode

    # Đúng thứ report.md khuyên analyst làm khi recall thấp.
    save_annotations(
        SchemaAnnotations(
            tables={
                t.name: Annotation(
                    "Mô tả nghiệp vụ rất dài để đẩy schema vượt ngưỡng token. " * 20,
                    reviewed_by="human",
                )
                for t in MINI_TABLES
            }
        ),
        tmp_path / "schema.yaml",
    )

    assert read_profile(tmp_path).schema_mode == mode_before == mode_at_build, (
        "chú giải thêm vào không được lật công tắc schema_mode mà build đã chốt"
    )


def test_schema_mode_comes_from_profile_json_not_from_recomputation(tmp_path):
    """Chứng minh trực tiếp nguồn sự thật là file, không phải phép tính lại."""
    write_profile(
        tmp_path,
        dsn="postgresql://u:p@h:5432/d",
        tables=MINI_TABLES,
        annotations=SchemaAnnotations(),
        grants=GRANTS,
    )
    meta_path = tmp_path / "profile.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["schema_mode"] == "full", "schema nhỏ nên build chốt full"

    # Sửa tay giá trị trên đĩa: nếu read_profile tính lại thì nó sẽ bỏ qua.
    meta["schema_mode"] = "retrieval"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    assert read_profile(tmp_path).schema_mode == "retrieval"


# ── N1: schema_mode không hợp lệ phải rơi về "full", không lọt xuống retrieval
#
# `schema_context.py` so `profile.schema_mode == "full"`, retrieval là nhánh
# `else` — nên bất cứ giá trị lạ nào (profile.json bị sửa tay/cắt cụt/hỏng)
# mà không bị chặn ở `read_profile` sẽ tự động chọn retrieval, có thể âm
# thầm bỏ sót bảng JOIN. Chỉ khoá bị thiếu mới được mặc định "full" — giá
# trị CÓ MẶT nhưng sai phải bị từ chối, không được đi qua nguyên trạng.
@pytest.mark.parametrize(
    "bad_value",
    ["Full", "FULL", "", None, 0, "retrieval "],
)
def test_read_profile_rejects_invalid_schema_mode_and_falls_back_to_full(
    tmp_path, bad_value
):
    write_profile(
        tmp_path,
        dsn="postgresql://u:p@h:5432/d",
        tables=MINI_TABLES,
        annotations=SchemaAnnotations(),
        grants=GRANTS,
    )
    meta_path = tmp_path / "profile.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["schema_mode"] = bad_value
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    assert read_profile(tmp_path).schema_mode == "full"
