"""
Wrapper chạy MLX-LM LoRA training cho ADBA.

Usage:
  python training/train_mlx.py
  python training/train_mlx.py --config training/mlx_config.yaml --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "training" / "mlx_config.yaml"


def _read_config_value(config_text: str, key: str) -> str | None:
    prefix = f"{key}:"
    for line in config_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return None


def _assert_prerequisites(config_path: Path) -> tuple[Path, Path]:
    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy config: {config_path}")

    text = config_path.read_text(encoding="utf-8")
    data_dir_raw = _read_config_value(text, "data")
    if not data_dir_raw:
        raise ValueError("Config thiếu trường 'data'.")

    data_dir = (ROOT / data_dir_raw).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy data dir: {data_dir}")

    train_file = data_dir / "train.jsonl"
    valid_file = data_dir / "valid.jsonl"
    if not train_file.exists():
        raise FileNotFoundError(f"Thiếu file train: {train_file}")
    if not valid_file.exists():
        raise FileNotFoundError(f"Thiếu file valid: {valid_file}")

    return train_file, valid_file


def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _check_mlx_lm_installed() -> None:
    try:
        import mlx_lm  # noqa: F401
    except Exception:
        print("[ERROR] Chưa cài package 'mlx-lm' trong virtualenv hiện tại.")
        print("Hãy cài dependencies trước khi train:")
        print(f"  {sys.executable} -m pip install -U mlx mlx-lm")
        print("Sau đó chạy lại: python training/train_mlx.py")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="ADBA MLX-LM training wrapper")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Đường dẫn file config yaml cho mlx_lm.lora",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ validate điều kiện đầu vào, không chạy train",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()

    try:
        train_file, valid_file = _assert_prerequisites(config_path)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    train_n = _line_count(train_file)
    valid_n = _line_count(valid_file)

    print("=== ADBA MLX Training Wrapper ===")
    print(f"Config      : {config_path}")
    print(f"Train file  : {train_file} ({train_n} lines)")
    print(f"Valid file  : {valid_file} ({valid_n} lines)")
    print("Command     : python -m mlx_lm.lora --config training/mlx_config.yaml")
    print("=================================")

    if args.dry_run:
        print("[DRY-RUN] Điều kiện đầu vào hợp lệ. Sẵn sàng chạy training.")
        return

    _check_mlx_lm_installed()

    cmd = [sys.executable, "-m", "mlx_lm.lora", "--config", str(config_path)]
    print(f"[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
