"""Tải bộ dữ liệu benchmark ngoài, có cổng license chặn trước.

Nguyên tắc chi phối: **manifest là nơi duy nhất giữ sự thật về license.**
Một dòng trong tài liệu thì không ai đọc lúc chạy script; một trường trong
manifest thì script đọc được và từ chối được.

Hai trục quyền, cố ý tách rời vì chúng là hai câu hỏi khác nhau:

  commercial_use     Được dùng để đo/đánh giá phục vụ một sản phẩm bán ra?
                     Đây là trục mà điều khoản NonCommercial (NC) chạm tới.
  may_redistribute   Được đóng gói kèm bản giao khách hàng?
                     Đây là trục phân phối.

Một bộ có thể cho phép trục này mà cấm trục kia. Gộp làm một trường sẽ
buộc phải chọn bừa một trong hai khi chúng khác nhau.

Mặc định đóng: mục chưa có cả hai trục do người điền sẽ bị từ chối **trước
khi chạm mạng**. Dữ liệu có license hạn chế nằm trong git history hoặc trong
bundle giao khách là việc rất khó gỡ, nên chỗ rẻ nhất để chặn là trước khi
nó tồn tại trên đĩa.

Không thêm phụ thuộc: tải bằng `urllib` của thư viện chuẩn.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MANIFEST = Path("eval/benchmarks.json")
DEFAULT_ROOT = Path("data/benchmarks")
_CHUNK = 1 << 20


class UnknownDatasetError(KeyError):
    """Tên bộ dữ liệu không có trong manifest."""


class LicenseNotConfirmedError(PermissionError):
    """Mục manifest chưa được người xác nhận điều kiện sử dụng."""


class ManifestIncompleteError(ValueError):
    """License đã xác nhận nhưng thiếu thông tin kỹ thuật để tải."""


class ManualDownloadRequiredError(RuntimeError):
    """Bộ này phải tải tay; script không tự lấy được."""


class ChecksumMismatchError(RuntimeError):
    """File không khớp sha256 đã khai."""


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    url: str
    sha256: str
    license: str | None
    license_url: str
    commercial_use: bool | None
    may_redistribute: bool | None
    fetch_mode: str = "http"
    filename: str = ""
    notes: str = ""

    @property
    def confirmed(self) -> bool:
        """Đã có người điền license và cả hai trục quyền."""
        return (
            self.license is not None
            and self.commercial_use is not None
            and self.may_redistribute is not None
        )

    @property
    def target_name(self) -> str:
        return self.filename or Path(self.url).name


def load_manifest(path: Path | str = DEFAULT_MANIFEST) -> dict[str, DatasetSpec]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        name: DatasetSpec(
            name=name,
            url=entry.get("url", ""),
            sha256=entry.get("sha256", ""),
            license=entry.get("license"),
            license_url=entry.get("license_url", ""),
            commercial_use=entry.get("commercial_use"),
            may_redistribute=entry.get("may_redistribute"),
            fetch_mode=entry.get("fetch_mode", "http"),
            filename=entry.get("filename", ""),
            notes=entry.get("notes", ""),
        )
        for name, entry in raw.items()
    }


def require(name: str, manifest: Mapping[str, DatasetSpec]) -> DatasetSpec:
    """Trả về spec, hoặc ném lỗi nếu chưa đủ điều kiện dùng.

    Đây là cổng. Mọi đường dẫn tới việc tải dữ liệu phải đi qua đây.
    """
    if name not in manifest:
        raise UnknownDatasetError(
            f"Không có bộ '{name}' trong manifest. Đang khai báo: {sorted(manifest)}"
        )

    spec = manifest[name]
    if not spec.confirmed:
        raise LicenseNotConfirmedError(
            f"Bộ '{name}' chưa được xác nhận điều kiện sử dụng.\n"
            f"  Đọc điều khoản tại: {spec.license_url or '(chưa khai)'}\n"
            f"  Rồi điền cho mục '{name}' trong manifest:\n"
            f"    license          — tên giấy phép\n"
            f"    commercial_use   — được dùng để đo phục vụ sản phẩm bán ra?\n"
            f"    may_redistribute — được đóng gói kèm bản giao khách?\n"
            f"  Ghi chú hiện có: {spec.notes or '(không)'}"
        )
    return spec


def require_commercial(name: str, manifest: Mapping[str, DatasetSpec]) -> DatasetSpec:
    """Như `require`, nhưng thêm điều kiện được dùng cho mục đích thương mại.

    Dùng ở những chỗ con số đo được sẽ dẫn tới một quyết định về sản phẩm
    bán ra — chẳng hạn quyết định fine-tune có chuyển giao được sang schema
    khách hay không. Bộ chỉ cho phép nghiên cứu vẫn tải được để tham khảo,
    nhưng không được đi vào đường ra quyết định đó.
    """
    spec = require(name, manifest)
    if not spec.commercial_use:
        raise LicenseNotConfirmedError(
            f"Bộ '{name}' ({spec.license}) không được dùng cho mục đích thương mại.\n"
            f"  Chỉ dùng tham chiếu nội bộ. Không lấy số từ bộ này làm căn cứ\n"
            f"  cho quyết định về sản phẩm bán ra.\n"
            f"  Điều khoản: {spec.license_url}"
        )
    return spec


def provenance_text(spec: DatasetSpec) -> str:
    """Nội dung LICENSE.txt đặt cạnh dữ liệu.

    Tồn tại để người mở thư mục sáu tháng sau biết dữ liệu ở đâu ra và được
    làm gì với nó, mà không phải tìm ngược về manifest.
    """
    lines = [
        f"Bộ dữ liệu : {spec.name}",
        f"Nguồn      : {spec.url}",
        f"License    : {spec.license}",
        f"Điều khoản : {spec.license_url}",
        "",
    ]
    if spec.commercial_use is False:
        lines += [
            "*** CHỈ DÙNG THAM CHIẾU NỘI BỘ — KHÔNG DÙNG CHO MỤC ĐÍCH THƯƠNG MẠI ***",
            "",
            "Không lấy số đo từ bộ này làm căn cứ cho quyết định về sản phẩm",
            "bán ra. Xem trường commercial_use trong eval/benchmarks.json.",
            "",
        ]
    if spec.may_redistribute is False:
        lines += [
            "*** KHÔNG ĐƯỢC ĐÓNG GÓI KÈM BẢN GIAO KHÁCH HÀNG ***",
            "",
            "Bước đóng gói on-prem phải loại trừ thư mục này.",
            "",
        ]
    if not spec.sha256:
        lines += [
            "CẢNH BÁO: checksum chưa được ghim trong manifest — nội dung file",
            "không được xác minh. Chạy lại script để lấy hash và điền vào.",
            "",
        ]
    if spec.notes:
        lines += [f"Ghi chú    : {spec.notes}", ""]
    return "\n".join(lines)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _urllib_download(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url) as response, Path(dest).open("wb") as fh:  # noqa: S310
        while chunk := response.read(_CHUNK):
            fh.write(chunk)


def fetch(
    name: str,
    root: Path | str = DEFAULT_ROOT,
    manifest: Mapping[str, DatasetSpec] | None = None,
    downloader: Callable[[str, Path], None] = _urllib_download,
) -> Path:
    """Tải một bộ về `root/<name>/`, kiểm checksum, ghi kèm dấu vết license.

    Bỏ qua việc tải nếu file đã có và khớp checksum. Xoá file nếu lệch
    checksum — một bản tải hỏng để lại trên đĩa sẽ được lần chạy sau coi là
    hợp lệ nếu ai đó nới lỏng kiểm tra.

    `fetch_mode="manual"` dành cho nguồn script không lấy được (ví dụ link
    Google Drive có trang trung gian): nó không tải, chỉ xác minh file bạn
    đã đặt sẵn và ghi dấu vết.
    """
    manifest = manifest if manifest is not None else load_manifest()
    spec = require(name, manifest)

    target_dir = Path(root) / name
    target_dir.mkdir(parents=True, exist_ok=True)

    if not spec.target_name:
        raise ManifestIncompleteError(
            f"Bộ '{name}' đã xác nhận license nhưng chưa có 'url' hoặc 'filename'."
        )
    dest = target_dir / spec.target_name

    if dest.exists() and spec.sha256 and _sha256(dest) == spec.sha256:
        (target_dir / "LICENSE.txt").write_text(provenance_text(spec), encoding="utf-8")
        return dest

    if spec.fetch_mode == "manual":
        if not dest.exists():
            raise ManualDownloadRequiredError(
                f"Bộ '{name}' phải tải tay — script không lấy tự động được.\n"
                f"  Tải từ : {spec.url}\n"
                f"  Đặt vào: {dest}\n"
                f"  Rồi chạy lại lệnh này để xác minh và ghi dấu vết."
            )
    else:
        if not spec.url:
            raise ManifestIncompleteError(f"Bộ '{name}' thiếu 'url'.")
        downloader(spec.url, dest)

    actual = _sha256(dest)
    if spec.sha256 and actual != spec.sha256:
        dest.unlink(missing_ok=True)
        raise ChecksumMismatchError(
            f"'{name}' lệch checksum — đã xoá file.\n"
            f"  khai báo: {spec.sha256}\n"
            f"  thực tế : {actual}"
        )

    (target_dir / "LICENSE.txt").write_text(provenance_text(spec), encoding="utf-8")
    if not spec.sha256:
        print(f"  [chưa ghim] sha256 của {spec.target_name}:\n    {actual}")
        print("  Điền vào manifest để lần sau được xác minh.")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Tải bộ dữ liệu benchmark ngoài (có cổng xác nhận license)"
    )
    ap.add_argument("name", nargs="?", help="tên bộ; bỏ trống để liệt kê")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)

    if not args.name:
        for spec in manifest.values():
            if not spec.confirmed:
                print(f"{spec.name:10s} CHỜ XÁC NHẬN LICENSE")
                continue
            flags = []
            if not spec.commercial_use:
                flags.append("chỉ nội bộ")
            if not spec.may_redistribute:
                flags.append("không đóng gói")
            if spec.fetch_mode == "manual":
                flags.append("tải tay")
            suffix = ("  [" + ", ".join(flags) + "]") if flags else ""
            print(f"{spec.name:10s} {spec.license}{suffix}")
        return

    path = fetch(args.name, args.root, manifest=manifest)
    print(f"{args.name}: {path}")


if __name__ == "__main__":
    main()
