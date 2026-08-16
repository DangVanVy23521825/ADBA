"""Tải bộ dữ liệu benchmark ngoài, có cổng license chặn trước.

Nguyên tắc chi phối: **manifest là nơi duy nhất giữ sự thật về license.**
Một dòng trong tài liệu thì không ai đọc lúc chạy script; một trường trong
manifest thì script đọc được và từ chối được.

Mặc định đóng. Một mục chưa có `license` và `may_ship` do người điền sẽ bị
từ chối **trước khi chạm mạng**. Lý do: dữ liệu có license hạn chế nằm trong
git history hoặc trong bundle giao khách là việc rất khó gỡ, nên chỗ rẻ nhất
để chặn là trước khi nó tồn tại trên đĩa.

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
    """Mục manifest chưa được người xác nhận license và quyền đóng gói."""


class ChecksumMismatchError(RuntimeError):
    """File tải về không khớp sha256 đã khai."""


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    url: str
    sha256: str
    license: str | None
    license_url: str
    may_ship: bool | None
    notes: str = ""

    @property
    def confirmed(self) -> bool:
        """Đã có người điền cả license lẫn quyết định đóng gói."""
        return self.license is not None and self.may_ship is not None


def load_manifest(path: Path | str = DEFAULT_MANIFEST) -> dict[str, DatasetSpec]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        name: DatasetSpec(
            name=name,
            url=entry.get("url", ""),
            sha256=entry.get("sha256", ""),
            license=entry.get("license"),
            license_url=entry.get("license_url", ""),
            may_ship=entry.get("may_ship"),
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
            f"Bộ '{name}' chưa được xác nhận license.\n"
            f"  Đọc điều khoản tại: {spec.license_url or '(chưa khai)'}\n"
            f"  Rồi điền 'license' và 'may_ship' cho mục '{name}' trong manifest.\n"
            f"  may_ship = có được đóng gói kèm bản giao khách hàng hay không.\n"
            f"  Ghi chú hiện có: {spec.notes or '(không)'}"
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
    if spec.may_ship is False:
        lines += [
            "*** KHÔNG ĐƯỢC ĐÓNG GÓI KÈM BẢN GIAO KHÁCH HÀNG ***",
            "",
            "Bộ này chỉ dùng để đo nội bộ. Bước đóng gói on-prem phải loại trừ",
            "thư mục này. Xem trường may_ship trong eval/benchmarks.json.",
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
    """
    manifest = manifest if manifest is not None else load_manifest()
    spec = require(name, manifest)

    target_dir = Path(root) / name
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / Path(spec.url).name

    if dest.exists() and _sha256(dest) == spec.sha256:
        (target_dir / "LICENSE.txt").write_text(provenance_text(spec), encoding="utf-8")
        return dest

    downloader(spec.url, dest)

    actual = _sha256(dest)
    if actual != spec.sha256:
        dest.unlink(missing_ok=True)
        raise ChecksumMismatchError(
            f"'{name}' lệch checksum — đã xoá file.\n"
            f"  khai báo: {spec.sha256}\n"
            f"  thực tế : {actual}"
        )

    (target_dir / "LICENSE.txt").write_text(provenance_text(spec), encoding="utf-8")
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
            state = "sẵn sàng" if spec.confirmed else "CHỜ XÁC NHẬN LICENSE"
            ship = "" if not spec.confirmed else ("" if spec.may_ship else "  [không đóng gói]")
            print(f"{spec.name:10s} {state}{ship}")
        return

    path = fetch(args.name, args.root, manifest=manifest)
    print(f"{args.name}: {path}")


if __name__ == "__main__":
    main()
