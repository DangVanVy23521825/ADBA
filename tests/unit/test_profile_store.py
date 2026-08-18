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


def test_no_sample_row_data_is_written_anywhere(tmp_path):
    """Ràng buộc spec: profile bị copy đi khi hỗ trợ kỹ thuật."""
    _write(tmp_path)
    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    blob = "".join(p.read_text(encoding="utf-8", errors="ignore") for p in files)
    assert "sample_rows" not in blob
    # Bất biến thật: write_profile ở BƯỚC NÀY chỉ sinh đúng ba file này.
    # Đây không phải trần cho mọi thứ thư mục profile được phép chứa mãi
    # mãi — một task sau ghi thêm report.md vào cùng thư mục là hợp lệ.
    assert {p.name for p in files} == {PROFILE_JSON, STRUCTURE_JSON, SCHEMA_YAML}


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
