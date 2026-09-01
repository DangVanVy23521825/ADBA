#!/usr/bin/env bash
# ADBA — cài git hook đồng bộ tài liệu.
#
#   bash scripts/install_docs_hooks.sh            # cài
#   bash scripts/install_docs_hooks.sh --uninstall # gỡ
#
# Dùng core.hooksPath để hook nằm trong repo (versioned), không phải .git/hooks.

set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"

if [[ "${1:-}" == "--uninstall" ]]; then
    git config --unset core.hooksPath || true
    echo "Đã gỡ: core.hooksPath về mặc định (.git/hooks)."
    exit 0
fi

chmod +x scripts/hooks/* 2>/dev/null || true
git config core.hooksPath scripts/hooks

echo "Đã cài hook: core.hooksPath = scripts/hooks"
echo
echo "Từ giờ mỗi 'git commit' sẽ:"
echo "  1. chạy scripts/update_docs.py"
echo "  2. nếu docs/ đổi → tạo commit phụ 'docs(auto): đồng bộ tài liệu theo commit <sha>'"
echo
echo "Chạy thử ngay:  python3 scripts/update_docs.py --check"
