"""Đường onboarding một khách hàng: extract → annotate → build → verify.

Chạy TẠI CHỖ KHÁCH, offline. Bước `annotate` gọi model local; không có bước
nào gọi API ngoài — xem `_local_invoke`.

DSN không bao giờ nên xuất hiện trên dòng lệnh: nó lộ trong `ps aux` cho
mọi user cục bộ và ở lại trong lịch sử shell. Mỗi lệnh con nhận DSN từ cờ
`--dsn` HOẶC từ biến môi trường `ADBA_DSN`; cờ (nếu có) thắng biến môi
trường. Không có cả hai là lỗi rõ ràng, không phải một DSN rỗng âm thầm
trôi xuống tầng introspect.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

from model.model_client import ModelClient
from perception.annotate import annotate_schema
from perception.annotations import (
    SchemaAnnotations,
    load_annotations,
    merge_annotations,
    pending_review,
    save_annotations,
)
from perception.connection_profile import ConnectionProfile
from perception.introspect import introspect_schema, sample_rows
from perception.profile_store import (
    SCHEMA_YAML,
    STRUCTURE_JSON,
    read_profile,
    structure_from_plain,
    structure_to_plain,
    write_profile,
)
from perception.schema_model import Table

DSN_ENV_VAR = "ADBA_DSN"

# Xem model/model_config.py — đây là biến operator cần kiểm tra khi toàn
# bộ lượt annotate thất bại, dấu hiệu của một địa chỉ Ollama sai chứ không
# phải một vấn đề của schema khách.
OLLAMA_ENV_VAR = "OLLAMA_BASE_URL"


def cmd_extract(dsn: str, profile_dir: Path | str) -> tuple[Table, ...]:
    """Đọc cấu trúc DB, ghi `structure.json`. Không đụng tới chú giải."""
    tables = introspect_schema(dsn)
    d = Path(profile_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / STRUCTURE_JSON).write_text(
        json.dumps(structure_to_plain(tables), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"{len(tables)} bảng → {d / STRUCTURE_JSON}")
    return tables


def _load_structure(profile_dir: Path) -> tuple[Table, ...]:
    return structure_from_plain(
        json.loads((profile_dir / STRUCTURE_JSON).read_text(encoding="utf-8"))
    )


def _local_invoke(system: str, user: str) -> str:
    """Gọi model local qua `ModelClient`.

    `enable_openai_fallback=False` được truyền TƯỜNG MINH, không dựa vào
    mặc định của `ModelClient` (vốn bật fallback qua biến môi trường
    `ENABLE_OPENAI_FALLBACK`) hay vào việc `agent_type="insight"` hiện
    không nằm trong tập agent được phép fallback của `ModelClient.invoke`.
    Prompt annotate mang tên bảng, tên cột, và DÒNG DỮ LIỆU THẬT của khách
    — nó không được rời khỏi mạng của họ dù tập agent được phép fallback
    đổi ở nơi khác trong tương lai.
    """
    return ModelClient(agent_type="insight", enable_openai_fallback=False).invoke(
        system_prompt=system, user_prompt=user
    )


def _with_progress(
    invoke: Callable[[str, str], str], tables: tuple[Table, ...]
) -> Callable[[str, str], str]:
    """Bọc `invoke` để in tiến độ theo từng bảng.

    `annotate_schema` gọi `invoke` đúng một lần cho mỗi bảng, theo đúng
    thứ tự của `tables` (kể cả khi `invoke` ném lỗi — lỗi đó bị nó tự bắt
    và tính là một lần thất bại, không lặp lại bảng đó). Nhờ vậy một bộ
    đếm tuần tự ở đây khớp chính xác với bảng đang được xử lý mà không cần
    `annotate_schema` phải lộ ra một tham số callback riêng.
    """
    total = len(tables)
    state = {"i": 0}

    def wrapped(system: str, user: str) -> str:
        state["i"] += 1
        name = tables[state["i"] - 1].name if state["i"] - 1 < total else "?"
        print(f"  [{state['i']}/{total}] {name}", flush=True)
        return invoke(system, user)

    return wrapped


def cmd_annotate(
    profile_dir: Path | str,
    dsn: str,
    invoke: Callable[[str, str], str] = _local_invoke,
) -> SchemaAnnotations:
    """Sinh chú giải và ghép vào bản đang có, giữ nguyên mục người sửa.

    Một model 7B local mất vài giây mỗi bảng, nên 150 bảng là 8-25 phút im
    lặng nếu không có gì được in ra — operator không phân biệt được "đang
    chạy" với "đã treo". Vì `annotate_schema` cố ý nuốt lỗi hạ tầng (một
    bảng lỗi không được phép giết cả lượt chạy 150 bảng), hàm này phải tự
    báo cáo tiến độ VÀ số lượng thất bại; im lặng ở đây có thể biến một địa
    chỉ Ollama gõ sai thành 150 chú giải rỗng được báo là thành công.
    """
    d = Path(profile_dir)
    tables = _load_structure(d)
    total = len(tables)
    samples = {t.name: sample_rows(dsn, t.name, n=3) for t in tables}

    tracked_invoke = _with_progress(invoke, tables)
    fresh, failures = annotate_schema(tables, samples, tracked_invoke)
    merged = merge_annotations(load_annotations(d / SCHEMA_YAML), fresh)
    save_annotations(merged, d / SCHEMA_YAML)

    pending = pending_review(merged)
    print(f"Chú giải xong {total} bảng. {len(pending)} mục cần người duyệt.")
    print(f"Thất bại: {failures}/{total} bảng.")
    if total > 0 and failures == total:
        print(
            "CẢNH BÁO: toàn bộ bảng chú giải thất bại — đây là dấu hiệu "
            f"model local không kết nối được (sai địa chỉ, chưa chạy Ollama), "
            "không phải vấn đề của schema. Kiểm tra biến môi trường "
            f"{OLLAMA_ENV_VAR} (và rằng Ollama đang chạy tại chỗ khách)."
        )
    return merged


class UnreviewedAnnotationsError(RuntimeError):
    """Quá ít chú giải được người duyệt để dựng profile dùng được."""


def cmd_build(
    profile_dir: Path | str,
    dsn: str,
    grants: Mapping[str, frozenset[str]],
    min_reviewed: float = 0.0,
) -> ConnectionProfile:
    """Ghép cấu trúc + chú giải thành `profile/` đọc được lúc chạy.

    `min_reviewed` là cổng chặn bàn giao sớm. Một profile gần như không có
    chú giải do người duyệt sẽ cho recall thấp, và người vận hành sẽ đổ lỗi
    cho model thay vì cho chú giải — mặc định 0.0 để không cản lúc phát
    triển, nhưng bản giao khách phải đặt ngưỡng thật.

    Mẫu số của cổng dùng dạng hai-đối-số của `review_progress` — cùng tập
    `tables` vừa đọc ở trên — chứ không phải dạng một-đối-số vốn chỉ đếm
    trong những mục ĐÃ có chú giải. Một cột LLM bỏ qua vì không đoán nổi
    thì không có mục trong `ann` để đếm; đó chính xác là cột cần người
    nhất, và dạng một-đối-số sẽ để nó lọt khỏi mẫu số, cho một khách hàng
    thấy "100% đã duyệt" trong khi hàng chục cột vẫn trống.
    """
    from perception.review_state import review_progress

    d = Path(profile_dir)
    tables = _load_structure(d)
    ann = load_annotations(d / SCHEMA_YAML)

    done, total = review_progress(ann, tables)
    ratio = (done / total) if total else 0.0
    if ratio < min_reviewed:
        raise UnreviewedAnnotationsError(
            f"Mới {done}/{total} mục ({ratio:.0%}) được người duyệt, "
            f"ngưỡng là {min_reviewed:.0%}. Mở trang 'Chú giải schema' để duyệt tiếp."
        )

    if not grants:
        print(
            "CẢNH BÁO: không có --grant nào — profile này mặc định đóng, "
            "không ai xem được bảng nào. Thêm --grant user=* (hoặc "
            "--grant user=bang1,bang2) để cấp quyền, hoặc cấp sau bằng một "
            "lượt `build` khác."
        )

    write_profile(d, dsn=dsn, tables=tables, annotations=ann, grants=grants)
    profile = read_profile(d)
    print(f"profile → {d}  ({len(profile.tables)} bảng, chế độ {profile.schema_mode})")
    return profile


def _resolve_dsn(cli_dsn: str | None) -> str:
    dsn = cli_dsn or os.environ.get(DSN_ENV_VAR)
    if not dsn:
        print(
            f"Thiếu DSN: truyền --dsn hoặc đặt biến môi trường {DSN_ENV_VAR}. "
            "Không truyền DSN qua dòng lệnh trên máy khách nếu tránh được — "
            "nó lộ ra trong `ps aux` và lịch sử shell.",
            file=sys.stderr,
        )
        sys.exit(1)
    return dsn


def _add_dsn_profile_args(p: argparse.ArgumentParser) -> None:
    """Cờ dùng chung cho mọi lệnh con: DSN tuỳ chọn + thư mục profile.

    Tách riêng để các lệnh con sau này (build, verify, refresh — task 9,
    10, 11) thêm vào bằng một lời gọi nữa, không phải chép lại hai dòng
    `add_argument` này.
    """
    p.add_argument(
        "--dsn",
        default=None,
        help=f"DSN kết nối database; nếu bỏ trống, lấy từ biến môi trường {DSN_ENV_VAR}",
    )
    p.add_argument("--profile", type=Path, default=Path("profile"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Đường onboarding một khách hàng")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_extract = sub.add_parser("extract", help="đọc cấu trúc database")
    _add_dsn_profile_args(p_extract)

    p_annotate = sub.add_parser("annotate", help="sinh chú giải bằng model local")
    _add_dsn_profile_args(p_annotate)

    p_build = sub.add_parser("build", help="dựng profile từ cấu trúc + chú giải")
    _add_dsn_profile_args(p_build)
    p_build.add_argument(
        "--grant",
        action="append",
        default=[],
        help="user=bang1,bang2 hoặc user=* (lặp lại được)",
    )
    p_build.add_argument("--min-reviewed", type=float, default=0.0)

    # Task 10, 11 thêm `verify`, `refresh` vào đây theo cùng khuôn: một
    # `sub.add_parser(...)`, một `_add_dsn_profile_args(...)` (hoặc biến thể
    # của nó nếu lệnh đó không cần DSN), và một nhánh mới trong khối
    # `if/elif` bên dưới. Không cần viết lại gì ở trên.

    args = ap.parse_args()
    dsn = _resolve_dsn(args.dsn)

    if args.cmd == "extract":
        cmd_extract(dsn, args.profile)
    elif args.cmd == "annotate":
        cmd_annotate(args.profile, dsn)
    elif args.cmd == "build":
        grants = {}
        for spec in args.grant:
            if "=" not in spec:
                # `--grant admin` (no '=') would otherwise silently become
                # `{"admin": frozenset()}` — a phantom user granted nothing,
                # AND a non-empty `grants` mapping that suppresses the
                # separate "no --grant at all" warning below. Both failures
                # look identical from the operator's chair (empty context on
                # the first question), so refuse outright instead of
                # guessing the operator meant an empty grant.
                print(
                    f"Cờ --grant sai cú pháp: {spec!r} thiếu dấu '='. "
                    "Cú pháp đúng: user=bang1,bang2 hoặc user=* "
                    "(hoặc user= để cấp rỗng có chủ đích).",
                    file=sys.stderr,
                )
                sys.exit(1)
            user, _, tables_csv = spec.partition("=")
            grants[user] = frozenset(t.strip() for t in tables_csv.split(",") if t.strip())
        cmd_build(args.profile, dsn, grants, min_reviewed=args.min_reviewed)


if __name__ == "__main__":
    main()
