"""Manifest-driven benchmark fetcher.

Trọng tâm của bộ test này không phải "tải được file về" mà là **không tải
được khi chưa có người xác nhận license**. Đó là ràng buộc nghiệp vụ, không
phải chi tiết kỹ thuật: một bộ dữ liệu có license hạn chế nằm trong git
history hoặc trong bundle giao khách là việc rất khó gỡ.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from eval.fetch_dataset import (
    LicenseNotConfirmedError,
    ChecksumMismatchError,
    UnknownDatasetError,
    DatasetSpec,
    fetch,
    load_manifest,
    provenance_text,
    require,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "manifest_sample.json"


def _spec(**over) -> DatasetSpec:
    base = dict(
        name="demo",
        url="https://example.invalid/demo.zip",
        sha256="a" * 64,
        license="CC BY-SA 4.0",
        license_url="https://example.invalid/license",
        may_ship=True,
        notes="",
    )
    base.update(over)
    return DatasetSpec(**base)


# ── manifest ────────────────────────────────────────────────────────────────

def test_load_manifest_reads_every_entry():
    manifest = load_manifest(FIXTURE)
    assert set(manifest) == {"confirmed", "pending_license", "no_ship"}


def test_unknown_dataset_names_the_known_ones():
    manifest = load_manifest(FIXTURE)
    with pytest.raises(UnknownDatasetError) as exc:
        require("khong_ton_tai", manifest)
    assert "confirmed" in str(exc.value)


# ── cổng license: đây là phần quan trọng nhất ───────────────────────────────

def test_entry_without_a_confirmed_license_is_refused():
    """Mặc định đóng: license chưa điền thì không tải."""
    manifest = load_manifest(FIXTURE)
    with pytest.raises(LicenseNotConfirmedError):
        require("pending_license", manifest)


def test_entry_without_may_ship_decided_is_refused():
    """Biết license nhưng chưa quyết có được đóng gói hay không — vẫn từ chối."""
    manifest = load_manifest(FIXTURE)
    spec = manifest["pending_license"]
    assert spec.license is None or spec.may_ship is None


def test_confirmed_entry_passes():
    manifest = load_manifest(FIXTURE)
    spec = require("confirmed", manifest)
    assert spec.license == "CC BY-SA 4.0"
    assert spec.may_ship is True


def test_fetch_refuses_before_download_when_license_unconfirmed(tmp_path):
    """Từ chối phải xảy ra TRƯỚC khi chạm mạng, không phải sau."""
    manifest = load_manifest(FIXTURE)
    calls: list[str] = []

    def spy(url, dest):  # noqa: ARG001
        calls.append(url)

    with pytest.raises(LicenseNotConfirmedError):
        fetch("pending_license", tmp_path, manifest=manifest, downloader=spy)
    assert calls == [], "đã gọi downloader dù license chưa xác nhận"


# ── dấu vết đi kèm dữ liệu ──────────────────────────────────────────────────

def test_provenance_names_the_license_and_source():
    text = provenance_text(_spec())
    assert "CC BY-SA 4.0" in text
    assert "https://example.invalid/license" in text


def test_provenance_of_a_no_ship_dataset_says_so_loudly():
    text = provenance_text(_spec(name="x", may_ship=False))
    assert "KHÔNG ĐƯỢC ĐÓNG GÓI" in text


def test_provenance_of_a_shippable_dataset_does_not_say_so():
    text = provenance_text(_spec(may_ship=True))
    assert "KHÔNG ĐƯỢC ĐÓNG GÓI" not in text


# ── tải + kiểm checksum ─────────────────────────────────────────────────────

def _write_with_checksum(payload: bytes) -> tuple[str, bytes]:
    return hashlib.sha256(payload).hexdigest(), payload


def test_fetch_writes_data_and_provenance(tmp_path):
    payload = b"noi dung gia lap"
    digest, _ = _write_with_checksum(payload)
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
    digest, _ = _write_with_checksum(payload)
    manifest = {"confirmed": _spec(name="confirmed", sha256=digest)}
    calls: list[str] = []

    def counting(url, dest):  # noqa: ARG001
        calls.append(url)
        Path(dest).write_bytes(payload)

    fetch("confirmed", tmp_path, manifest=manifest, downloader=counting)
    fetch("confirmed", tmp_path, manifest=manifest, downloader=counting)
    assert len(calls) == 1, "lần thứ hai không được tải lại"


# ── manifest thật trong repo ────────────────────────────────────────────────

def test_repo_manifest_parses_and_lists_the_three_benchmarks():
    manifest = load_manifest(Path("eval/benchmarks.json"))
    assert {"spider", "bird", "beaver"} <= set(manifest)


def test_downloaded_benchmark_data_is_gitignored():
    """Khoá ở tầng git, không chỉ ở tầng quy ước.

    Cổng license chặn việc tải; quy tắc này chặn việc commit thứ đã tải.
    Hai lớp vì chúng hỏng theo hai cách khác nhau.
    """
    import subprocess

    probe = "data/benchmarks/beaver/questions.jsonl"
    result = subprocess.run(
        ["git", "check-ignore", "-q", probe], capture_output=True, check=False
    )
    assert result.returncode == 0, f"{probe} không bị gitignore — dữ liệu có thể lọt vào repo"


def test_the_manifest_itself_is_not_gitignored():
    """Manifest là mã nguồn, không phải kết quả.

    Quy tắc `eval/*.json` có sẵn trong repo từng nuốt mất file này; nếu ai
    đó gỡ dòng phủ định thì test này đỏ thay vì manifest lặng lẽ biến mất
    khỏi bản clone của người khác.
    """
    import subprocess

    result = subprocess.run(
        ["git", "check-ignore", "-q", "eval/benchmarks.json"],
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0, "eval/benchmarks.json đang bị gitignore"


def test_repo_manifest_has_no_confirmed_license_yet():
    """Chốt trạng thái hiện tại: chưa bộ nào được xác nhận, nên chưa tải được bộ nào.

    Khi license được xác nhận, test này sẽ đỏ — đó là chủ ý. Người sửa nó
    buộc phải đọc lại điều kiện và cập nhật có ý thức, thay vì để một thay
    đổi manifest lặng lẽ mở cổng tải.
    """
    manifest = load_manifest(Path("eval/benchmarks.json"))
    for name in ("spider", "bird", "beaver"):
        with pytest.raises(LicenseNotConfirmedError):
            require(name, manifest)
