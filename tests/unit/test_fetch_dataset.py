"""Manifest-driven benchmark fetcher.

Trọng tâm của bộ test này không phải "tải được file về" mà là **không tải
được khi chưa có người xác nhận điều kiện sử dụng**. Đó là ràng buộc nghiệp
vụ, không phải chi tiết kỹ thuật: một bộ dữ liệu có license hạn chế nằm
trong git history hoặc trong bundle giao khách là việc rất khó gỡ.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from eval.fetch_dataset import (
    ChecksumMismatchError,
    DatasetSpec,
    LicenseNotConfirmedError,
    ManualDownloadRequiredError,
    UnknownDatasetError,
    fetch,
    load_manifest,
    provenance_text,
    require,
    require_commercial,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "manifest_sample.json"
REPO_MANIFEST = Path(__file__).resolve().parents[2] / "eval" / "benchmarks.json"


def _spec(**over) -> DatasetSpec:
    base = dict(
        name="demo",
        url="https://example.invalid/demo.zip",
        sha256="a" * 64,
        license="CC BY-SA 4.0",
        license_url="https://example.invalid/license",
        commercial_use=True,
        may_redistribute=True,
        fetch_mode="http",
        filename="",
        notes="",
    )
    base.update(over)
    return DatasetSpec(**base)


# ── manifest ────────────────────────────────────────────────────────────────

def test_load_manifest_reads_every_entry():
    manifest = load_manifest(FIXTURE)
    assert set(manifest) == {"confirmed", "pending_license", "no_ship", "research_only"}


def test_unknown_dataset_names_the_known_ones():
    manifest = load_manifest(FIXTURE)
    with pytest.raises(UnknownDatasetError) as exc:
        require("khong_ton_tai", manifest)
    assert "confirmed" in str(exc.value)


# ── cổng license: phần quan trọng nhất ──────────────────────────────────────

def test_entry_without_a_confirmed_license_is_refused():
    """Mặc định đóng: chưa điền thì không tải."""
    manifest = load_manifest(FIXTURE)
    with pytest.raises(LicenseNotConfirmedError):
        require("pending_license", manifest)


def test_license_named_but_axes_undecided_is_still_refused():
    """Biết tên giấy phép chưa đủ — phải quyết cả hai trục quyền."""
    spec = _spec(license="CC BY-SA 4.0", commercial_use=None, may_redistribute=None)
    with pytest.raises(LicenseNotConfirmedError):
        require("demo", {"demo": spec})


def test_confirmed_entry_passes():
    manifest = load_manifest(FIXTURE)
    spec = require("confirmed", manifest)
    assert spec.license == "CC BY-SA 4.0"
    assert spec.commercial_use is True


def test_fetch_refuses_before_download_when_license_unconfirmed(tmp_path):
    """Từ chối phải xảy ra TRƯỚC khi chạm mạng, không phải sau."""
    manifest = load_manifest(FIXTURE)
    calls: list[str] = []

    def spy(url, dest):  # noqa: ARG001
        calls.append(url)

    with pytest.raises(LicenseNotConfirmedError):
        fetch("pending_license", tmp_path, manifest=manifest, downloader=spy)
    assert calls == [], "đã gọi downloader dù license chưa xác nhận"


# ── hai trục quyền tách rời ─────────────────────────────────────────────────

def test_research_only_dataset_can_be_fetched_but_not_used_commercially():
    """Tải được để tham chiếu; chặn ở chỗ số đo dẫn tới quyết định sản phẩm."""
    manifest = load_manifest(FIXTURE)
    require("research_only", manifest)                       # tải được
    with pytest.raises(LicenseNotConfirmedError):
        require_commercial("research_only", manifest)        # không làm căn cứ được


def test_commercially_usable_dataset_passes_the_stricter_gate():
    manifest = load_manifest(FIXTURE)
    assert require_commercial("confirmed", manifest).commercial_use is True


def test_the_two_axes_are_independent():
    """`no_ship` dùng được cho sản phẩm nhưng không đóng gói được — và ngược lại."""
    manifest = load_manifest(FIXTURE)
    no_ship = manifest["no_ship"]
    research = manifest["research_only"]
    assert (no_ship.commercial_use, no_ship.may_redistribute) == (True, False)
    assert (research.commercial_use, research.may_redistribute) == (False, False)


# ── dấu vết đi kèm dữ liệu ──────────────────────────────────────────────────

def test_provenance_names_the_license_and_source():
    text = provenance_text(_spec())
    assert "CC BY-SA 4.0" in text
    assert "https://example.invalid/license" in text


def test_provenance_of_a_research_only_dataset_says_so_loudly():
    text = provenance_text(_spec(commercial_use=False))
    assert "KHÔNG DÙNG CHO MỤC ĐÍCH THƯƠNG MẠI" in text


def test_provenance_of_a_no_redistribute_dataset_says_so_loudly():
    text = provenance_text(_spec(may_redistribute=False))
    assert "KHÔNG ĐƯỢC ĐÓNG GÓI" in text


def test_provenance_of_an_unrestricted_dataset_carries_no_warning():
    text = provenance_text(_spec())
    assert "KHÔNG" not in text


def test_provenance_warns_when_the_checksum_is_unpinned():
    assert "checksum chưa được ghim" in provenance_text(_spec(sha256=""))


# ── tải + kiểm checksum ─────────────────────────────────────────────────────

def test_fetch_writes_data_and_provenance(tmp_path):
    payload = b"noi dung gia lap"
    digest = hashlib.sha256(payload).hexdigest()
    manifest = {"confirmed": _spec(name="confirmed", sha256=digest)}

    def fake_download(url, dest):  # noqa: ARG001
        Path(dest).write_bytes(payload)

    out = fetch("confirmed", tmp_path, manifest=manifest, downloader=fake_download)
    assert out.read_bytes() == payload
    assert (out.parent / "LICENSE.txt").exists()


def test_fetch_rejects_a_checksum_mismatch_and_removes_the_file(tmp_path):
    manifest = {"confirmed": _spec(name="confirmed", sha256="b" * 64)}

    def fake_download(url, dest):  # noqa: ARG001
        Path(dest).write_bytes(b"noi dung khac")

    with pytest.raises(ChecksumMismatchError):
        fetch("confirmed", tmp_path, manifest=manifest, downloader=fake_download)
    assert list(tmp_path.rglob("*.zip")) == [], "file hỏng phải bị xoá"


def test_fetch_skips_download_when_the_file_is_already_correct(tmp_path):
    payload = b"da co san"
    digest = hashlib.sha256(payload).hexdigest()
    manifest = {"confirmed": _spec(name="confirmed", sha256=digest)}
    calls: list[str] = []

    def counting(url, dest):  # noqa: ARG001
        calls.append(url)
        Path(dest).write_bytes(payload)

    fetch("confirmed", tmp_path, manifest=manifest, downloader=counting)
    fetch("confirmed", tmp_path, manifest=manifest, downloader=counting)
    assert len(calls) == 1, "lần thứ hai không được tải lại"


def test_manual_mode_never_downloads_and_says_where_to_put_the_file(tmp_path):
    manifest = {"m": _spec(name="m", fetch_mode="manual", filename="x.zip")}
    calls: list[str] = []

    def spy(url, dest):  # noqa: ARG001
        calls.append(url)

    with pytest.raises(ManualDownloadRequiredError) as exc:
        fetch("m", tmp_path, manifest=manifest, downloader=spy)
    assert calls == []
    assert "x.zip" in str(exc.value)


def test_manual_mode_accepts_a_file_the_user_placed(tmp_path):
    payload = b"tai tay"
    digest = hashlib.sha256(payload).hexdigest()
    manifest = {"m": _spec(name="m", fetch_mode="manual", filename="x.zip", sha256=digest)}
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "x.zip").write_bytes(payload)

    out = fetch("m", tmp_path, manifest=manifest, downloader=lambda u, d: None)
    assert out.read_bytes() == payload


# ── khoá ở tầng git ─────────────────────────────────────────────────────────

def test_downloaded_benchmark_data_is_gitignored():
    """Cổng license chặn việc TẢI; quy tắc này chặn việc COMMIT thứ đã tải.

    Hai lớp vì chúng hỏng theo hai cách khác nhau.
    """
    result = subprocess.run(
        ["git", "check-ignore", "-q", "data/benchmarks/beaver/questions.jsonl"],
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, "dữ liệu benchmark không bị gitignore"


def test_the_manifest_itself_is_not_gitignored():
    """Manifest là mã nguồn, không phải kết quả.

    Quy tắc `eval/*.json` có sẵn trong repo từng nuốt mất file này.
    """
    result = subprocess.run(
        ["git", "check-ignore", "-q", "eval/benchmarks.json"],
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0, "eval/benchmarks.json đang bị gitignore"


# ── manifest thật trong repo ────────────────────────────────────────────────

def test_repo_manifest_lists_the_three_benchmarks():
    manifest = load_manifest(REPO_MANIFEST)
    assert {"spider", "bird", "beaver"} <= set(manifest)


def test_spider_and_bird_are_confirmed_for_commercial_use():
    """Chủ repo đã xác minh cả hai là CC BY-SA 4.0."""
    manifest = load_manifest(REPO_MANIFEST)
    for name in ("spider", "bird"):
        spec = require_commercial(name, manifest)
        assert spec.license == "CC BY-SA 4.0"


def test_beaver_is_not_usable_until_its_license_is_settled():
    """Nguồn ghi mâu thuẫn (BY-NC-ND vs MIT) nên chưa dùng.

    Khi license được xác minh, test này sẽ đỏ — đó là chủ ý. Người sửa nó
    buộc phải đọc lại điều kiện, thay vì một thay đổi manifest lặng lẽ mở
    cổng.
    """
    manifest = load_manifest(REPO_MANIFEST)
    with pytest.raises(LicenseNotConfirmedError):
        require("beaver", manifest)


def test_no_benchmark_is_marked_redistributable():
    """Chính sách dự án: dữ liệu eval không đi kèm bản giao khách, bất kể license."""
    manifest = load_manifest(REPO_MANIFEST)
    for spec in manifest.values():
        assert spec.may_redistribute is not True, f"{spec.name} đang cho phép đóng gói"
