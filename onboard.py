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
import urllib.parse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import psycopg2

from model.model_client import ModelClient
from perception.annotate import annotate_schema
from perception.annotations import (
    HUMAN,
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
    dsn_host_for_error,
    read_profile,
    structure_from_plain,
    structure_to_plain,
    write_profile,
)
from perception.schema_model import Table

_T = TypeVar("_T")

DSN_ENV_VAR = "ADBA_DSN"


class OnboardError(RuntimeError):
    """Lỗi vận hành mà người dùng sửa được, không phải bug của chương trình.

    `main()` bắt loại này và in nguyên câu thông báo rồi thoát mã 1. Đừng
    dùng nó cho lỗi lập trình: traceback của những lỗi đó vẫn cần hiện ra
    đầy đủ.

    Thông báo không được chứa DSN — nó mang mật khẩu, và cả đường onboarding
    này giữ mật khẩu ra khỏi file lẫn thông báo lỗi.
    """

# Xem model/model_config.py — đây là biến operator cần kiểm tra khi toàn
# bộ lượt annotate thất bại, dấu hiệu của một địa chỉ Ollama sai chứ không
# phải một vấn đề của schema khách.
OLLAMA_ENV_VAR = "OLLAMA_BASE_URL"


def _run_db_call(dsn: str, step: str, call: Callable[[], _T]) -> _T:
    """Chạy một lệnh gọi DB (`introspect_schema`/`sample_rows`), bọc lỗi kết
    nối thành `OnboardError` chỉ nêu HOST — không bao giờ DSN hay mật khẩu.

    `main()` chỉ bắt `OnboardError` (xem docstring class đó); không có gì
    ở đây thì một lỗi kết nối từ `psycopg2` thoát ra thành traceback trần.
    Với DSN không đúng dạng URI, libpq/psycopg2 echo NGUYÊN VĂN chuỗi DSN
    vào thông báo lỗi parse (`invalid dsn: missing "=" after "..." in
    connection info string`) — nếu DSN đó mang mật khẩu, mật khẩu lộ ra
    console/lịch sử shell/log CI/ticket hỗ trợ. `onboard._resolve_dsn` đã
    chặn hình dạng đó từ trước khi tới đây (C2), nhưng khối `except` này
    vẫn tồn tại làm lớp phòng thủ thứ hai cho MỌI lỗi `psycopg2` khác (host
    không resolve được, sai port, bị từ chối kết nối, ...) — những lỗi đó
    tự thân thường không mang mật khẩu, nhưng vẫn phải biến thành
    `OnboardError` để không thoát ra thành traceback trần.

    Dùng `perception.profile_store.dsn_host_for_error` để lấy host — CHUNG
    kỹ thuật với `_strip_password` (chỉ đọc `.hostname`/`.port` từ
    `urlsplit`), không phát minh cách bóc credential thứ hai.
    """
    try:
        return call()
    except psycopg2.Error as e:
        host = dsn_host_for_error(dsn)
        raise OnboardError(
            f"Không kết nối được database lúc {step} (host {host}): "
            f"{type(e).__name__}. Kiểm tra host/port, mạng, và rằng "
            "Postgres đang chạy tại đó. Không in DSN ở đây vì nó có thể "
            "mang mật khẩu."
        ) from e


def cmd_extract(dsn: str, profile_dir: Path | str) -> tuple[Table, ...]:
    """Đọc cấu trúc DB, ghi `structure.json`. Không đụng tới chú giải."""
    tables = _run_db_call(dsn, "extract", lambda: introspect_schema(dsn))
    d = Path(profile_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / STRUCTURE_JSON).write_text(
        json.dumps(structure_to_plain(tables), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"{len(tables)} bảng → {d / STRUCTURE_JSON}")
    return tables


def _load_structure(profile_dir: Path) -> tuple[Table, ...]:
    """Đọc `structure.json`, hoặc báo lỗi nói được phải làm gì tiếp.

    Chạy sai thứ tự lệnh, hoặc trỏ `--profile` nhầm thư mục, là việc đầu
    tiên người vận hành làm sai. Để `FileNotFoundError` nguyên dạng thoát
    ra thành traceback thì họ nhận một trang stack Python thay vì câu
    "chạy extract trước" — trong khi trang Streamlit vốn đã xử lý đúng
    tình huống này.
    """
    path = profile_dir / STRUCTURE_JSON
    if not path.exists():
        raise OnboardError(
            f"Không có {STRUCTURE_JSON} trong {profile_dir}. "
            f"Chạy `onboard.py extract --profile {profile_dir}` trước."
        )
    return structure_from_plain(json.loads(path.read_text(encoding="utf-8")))


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


# Ngưỡng "co ngót nhiều" cho `_guarded_merge` (C1). `fresh` phải, theo đúng
# tinh thần docstring của `merge_annotations`, phủ toàn bộ schema hiện có —
# nếu nó chỉ còn phủ CHƯA ĐẾN MỘT NỬA số bảng mà `schema.yaml` đang biết
# tới, đó gần như chắc chắn là một lượt `extract` hỏng (ADBA_DSN trỏ nhầm
# DB rỗng/staging, nhầm schema ngoài `public`), không phải khách hàng vừa
# xoá thật hơn một nửa số bảng của họ trong một lượt. 0.5 chọn có chủ đích:
# một đợt dọn dẹp/loại bỏ bảng deprecated thật sự hiếm khi chạm mốc này
# trong một lần chạy, còn trỏ nhầm DB thì luôn chạm (thường về 0). So trên
# TỔNG SỐ BẢNG trong `schema.yaml` hiện có — không lọc riêng mục `human` —
# vì đây là một chốt chặn RẺ, chạy TRƯỚC khi ghép, lúc chưa biết trong số
# bị mất có bao nhiêu là `human`; biết được điều đó đòi hỏi đã ghép xong,
# tức đã quá muộn để "chặn trước khi ghi" (yêu cầu C1).
SHRINK_ABORT_RATIO = 0.5


def _human_entries(ann: SchemaAnnotations) -> tuple[frozenset[str], frozenset[tuple[str, str]]]:
    """Tập (bảng, cột) có `reviewed_by == human` trong `ann`."""
    tables = frozenset(name for name, a in ann.tables.items() if a.reviewed_by == HUMAN)
    columns = frozenset(
        (table, col)
        for table, cols in ann.columns.items()
        for col, a in cols.items()
        if a.reviewed_by == HUMAN
    )
    return tables, columns


def _preserved_and_dropped(before: SchemaAnnotations, merged: SchemaAnnotations) -> tuple[int, int]:
    """Đếm mục `human` của `before` còn sống sót (và bị loại) trong `merged`.

    Một mục `human` trong `before` mà tên bảng/cột của nó vẫn còn mặt trong
    `merged` CHẮC CHẮN vẫn là `human` sau khi ghép — `merge_annotations`
    luôn giữ nguyên mục `existing` khi nó là `human` (quy tắc ghép, không
    đổi ở đây) — nên chỉ cần kiểm tra còn mặt hay không, không cần đọc lại
    `reviewed_by` của `merged`.
    """
    before_tables, before_columns = _human_entries(before)
    merged_table_names = frozenset(merged.tables)
    merged_columns = frozenset(
        (table, col) for table, cols in merged.columns.items() for col in cols
    )
    preserved = (
        len(before_tables & merged_table_names) + len(before_columns & merged_columns)
    )
    dropped = len(before_tables) + len(before_columns) - preserved
    return preserved, dropped


def _report_dropped(dropped: int) -> None:
    if dropped > 0:
        print(
            f"CẢNH BÁO: {dropped} mục do người duyệt bị loại vì bảng/cột "
            "tương ứng không còn trong schema."
        )


def _guarded_merge(
    existing: SchemaAnnotations, fresh: SchemaAnnotations, *, force: bool
) -> tuple[SchemaAnnotations, int, int]:
    """Ghép chú giải, CHẶN LẠI trước khi ghi nếu `fresh` trông như một lượt
    extract hỏng chứ không phải một schema thật sự co lại. Trả
    `(merged, preserved, dropped)`.

    `merge_annotations` (không đổi ở đây — xem Constraints) đòi hỏi `fresh`
    phủ toàn bộ schema và TỰ NÓ không kiểm tra điều đó; nó xoá mọi mục
    (kể cả `human`) vắng mặt trong `fresh`, theo đúng docstring của nó. Một
    `ADBA_DSN` trỏ nhầm (DB rỗng, staging, sai schema ngoài `public` —
    `introspect_schema` hardcode `schema="public"`) khiến `extract` ghi
    `structure.json` rỗng hoặc thiếu, và từ đó `fresh` cũng rỗng/thiếu theo
    — hàm này là chốt chặn cho đúng chuỗi nhân quả đó, KHÔNG thay quy tắc
    ghép.

    Hai trường hợp bị chặn khi `force=False` VÀ `existing` không rỗng:
      1. `fresh` rỗng hoàn toàn (`structure.json` rỗng) — luôn chặn.
      2. `fresh` phủ chưa tới `SHRINK_ABORT_RATIO` số bảng `existing` đang
         có — chặn ("co ngót nhiều", xem hằng số ở trên).
    `force=True` bỏ qua cả hai, cho operator biết chủ đích một đường đi
    qua — đúng lúc họ THẬT SỰ vừa xoá phần lớn schema.
    """
    existing_table_count = len(existing.tables)
    fresh_table_count = len(fresh.tables)

    if not force and existing_table_count > 0:
        if fresh_table_count == 0:
            raise OnboardError(
                "Từ chối ghép chú giải: structure.json hiện có 0 bảng, "
                f"trong khi schema.yaml đang có {existing_table_count} bảng "
                "đã chú giải (kể cả mục người duyệt). Ghép sẽ XOÁ TOÀN BỘ "
                "chú giải hiện có. Đây gần như chắc chắn là ADBA_DSN trỏ "
                "nhầm (DB rỗng, môi trường staging, hoặc schema ngoài "
                "'public') chứ không phải khách hàng vừa xoá sạch bảng. "
                f"Kiểm tra biến môi trường {DSN_ENV_VAR}, chạy lại `extract` "
                "rồi `annotate`. Nếu bảng thật sự đã bị xoá hết, chạy lại "
                "`annotate` với --force."
            )
        if fresh_table_count < existing_table_count * SHRINK_ABORT_RATIO:
            raise OnboardError(
                f"Từ chối ghép chú giải: structure.json chỉ còn "
                f"{fresh_table_count} bảng, giảm hơn một nửa so với "
                f"{existing_table_count} bảng schema.yaml đang có chú giải. "
                "Ghép sẽ xoá chú giải của mọi bảng biến mất, kể cả mục "
                f"người duyệt. Kiểm tra biến môi trường {DSN_ENV_VAR} — đây "
                "thường là dấu hiệu trỏ nhầm DB/schema, không phải một đợt "
                "xoá bảng thật sự lớn. Nếu đây đúng là chủ đích, chạy lại "
                "`annotate` với --force."
            )

    merged = merge_annotations(existing, fresh)
    preserved, dropped = _preserved_and_dropped(existing, merged)
    return merged, preserved, dropped


def cmd_annotate(
    profile_dir: Path | str,
    dsn: str,
    invoke: Callable[[str, str], str] = _local_invoke,
    *,
    force: bool = False,
) -> SchemaAnnotations:
    """Sinh chú giải và ghép vào bản đang có, giữ nguyên mục người sửa.

    Một model 7B local mất vài giây mỗi bảng, nên 150 bảng là 8-25 phút im
    lặng nếu không có gì được in ra — operator không phân biệt được "đang
    chạy" với "đã treo". Vì `annotate_schema` cố ý nuốt lỗi hạ tầng (một
    bảng lỗi không được phép giết cả lượt chạy 150 bảng), hàm này phải tự
    báo cáo tiến độ VÀ số lượng thất bại; im lặng ở đây có thể biến một địa
    chỉ Ollama gõ sai thành 150 chú giải rỗng được báo là thành công.

    `force`: bỏ qua chốt chặn "fresh co ngót nhiều" của `_guarded_merge`
    (C1) — xem docstring hàm đó. Mặc định `False`: an toàn hơn là tiện lợi,
    vì hậu quả của việc đoán sai (ghi đè hàng tuần công sức duyệt) không
    đảo ngược được, còn hậu quả của việc chặn nhầm chỉ là chạy lại với
    `--force`.
    """
    d = Path(profile_dir)
    tables = _load_structure(d)
    total = len(tables)
    samples = {
        t.name: _run_db_call(dsn, f"lấy mẫu bảng {t.name}", lambda t=t: sample_rows(dsn, t.name, n=3))
        for t in tables
    }

    tracked_invoke = _with_progress(invoke, tables)
    fresh, failures = annotate_schema(tables, samples, tracked_invoke)
    existing = load_annotations(d / SCHEMA_YAML)
    merged, _preserved, dropped = _guarded_merge(existing, fresh, force=force)
    save_annotations(merged, d / SCHEMA_YAML)

    pending = pending_review(merged)
    print(f"Chú giải xong {total} bảng. {len(pending)} mục cần người duyệt.")
    print(f"Thất bại: {failures}/{total} bảng.")
    _report_dropped(dropped)
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


HANDOVER_RECALL = 0.95  # spec 6.5 — ngưỡng chặn bàn giao.


@dataclass
class VerifyReport:
    total: int
    recall: float
    avg_context_tables: float
    passed: bool
    misses: list[tuple[str, frozenset[str]]]


def _read_golden(golden_path: Path) -> tuple[list[dict], int]:
    """Đọc golden set JSONL, trả `(bản ghi, số dòng đọc được)`.

    File này do khách viết tay. Một dòng thiếu `question` hoặc `sql`, hay
    không phải JSON hợp lệ, phải báo lỗi nêu ĐÚNG số dòng — không phải một
    `KeyError`/`JSONDecodeError` trần trụi mà người vận hành phải tự mò
    ngược lại xem dòng nào trong file.
    """
    if not golden_path.is_file():
        # `.is_file()` (không phải `.exists()`) cố ý: `--golden` trỏ nhầm vào
        # một thư mục vẫn "exists", nhưng `.read_text()` trên nó ném
        # `IsADirectoryError` trần trụi — cùng loại traceback mà nhánh này
        # tồn tại để tránh.
        raise OnboardError(
            f"Không có golden set tại {golden_path}. Cần một file JSONL, mỗi "
            'dòng {"question": ..., "sql": ...}.'
        )

    rows: list[dict] = []
    lines_read = 0
    for lineno, raw in enumerate(golden_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise OnboardError(
                f"{golden_path}: dòng {lineno} không phải JSON hợp lệ ({e}). "
                "File golden là JSONL viết tay — sửa dòng đó rồi chạy lại."
            ) from e
        lines_read += 1
        # `row` có thể là JSON hợp lệ nhưng không phải object (`42`, `[1,2]`,
        # `"chuỗi"`) — `isinstance` phải đứng trước `in`, nếu không toán tử
        # `in` trên một int ném `TypeError` trần trụi thay vì `OnboardError`
        # nêu đúng số dòng.
        if not isinstance(row, dict) or "question" not in row or "sql" not in row:
            raise OnboardError(
                f"{golden_path}: dòng {lineno} thiếu trường 'question' hoặc "
                "'sql'. File golden là JSONL viết tay — sửa dòng đó rồi chạy lại."
            )
        rows.append(row)
    return rows, lines_read


def cmd_verify(
    profile_dir: Path | str,
    golden_path: Path | str,
    user: str,
    k: int = 8,
) -> VerifyReport:
    """Chấm recall chọn bảng trên golden set của khách, ghi `report.md`.

    Dùng lại đúng định nghĩa recall của eval tầng 1: một câu tính là đạt khi
    context chứa ĐỦ tập bảng đúng. Thiếu một bảng JOIN là SQL sai chắc chắn,
    nên không có điểm từng phần.

    `measure_recall` (eval.tier1_recall) CỐ Ý bỏ qua những bản ghi mà SQL
    mẫu không parse ra bảng nào — coi đó là lỗi dữ liệu golden, không phải
    lỗi retriever. Hệ quả là `report.total` là số câu CHẤM ĐƯỢC, không phải
    số dòng trong file golden. Nếu hai số đó khác nhau, report.md phải nói
    rõ cả hai và lý do — im lặng ở đây sẽ khiến người vận hành tưởng golden
    set nhỏ hơn nó thật sự là, và cổng bàn giao lặng lẽ bỏ qua đúng nửa khó
    của bộ câu hỏi khách viết.

    Không cần DSN: hàm này chỉ đọc `profile/` đã dựng sẵn (qua `read_profile`,
    không bao giờ mở kết nối DB thật — xem `perception/profile_store.py`),
    không chạm database của khách.
    """
    from eval.tier1_recall import measure_recall
    from eval.datasets import EvalRecord
    from perception.connection_profile import permitted_tables
    from perception.retrieval import LexicalRetriever
    from perception.schema_context import resolve_schema_context

    d = Path(profile_dir)
    try:
        profile = read_profile(d)
    except FileNotFoundError as e:
        # Mirror `_load_structure`: verify chạy sai thứ tự (trước `build`,
        # hoặc `--profile` trỏ nhầm thư mục) là lỗi vận hành đầu tiên người
        # ta gặp ở BƯỚC CUỐI của pipeline — để `FileNotFoundError` nguyên
        # dạng thoát ra sẽ cho một trang traceback thay vì câu "chạy build
        # trước". `str(e)` an toàn để lộ ra: nó tới từ `read_profile`, chỉ
        # nêu tên file thiếu và đường dẫn thư mục, không bao giờ mang DSN.
        raise OnboardError(
            f"Không đọc được profile trong {d} ({e}). "
            f"Chạy `onboard.py build --profile {d}` trước khi verify."
        ) from e
    permitted = permitted_tables(profile, user)
    retriever = LexicalRetriever(profile.tables)

    rows, lines_read = _read_golden(Path(golden_path))
    records = [
        EvalRecord(question=row["question"], gold_sql=row["sql"],
                   db_id="khach", tables=profile.tables)
        for row in rows
    ]

    def resolve(rec: EvalRecord) -> frozenset[str]:
        ctx = resolve_schema_context(
            profile, rec.question, permitted, retriever=retriever, k=k
        )
        return frozenset(ctx.retrieved_tables)

    r = measure_recall(records, resolve)
    report = VerifyReport(
        total=r.total,
        recall=r.recall,
        avg_context_tables=r.avg_context_tables,
        passed=r.recall >= HANDOVER_RECALL,
        misses=r.misses,
    )

    lines = ["# Báo cáo kiểm profile", ""]
    if lines_read != report.total:
        skipped = lines_read - report.total
        lines += [
            f"- Dòng golden đọc được: **{lines_read}**",
            f"- Câu chấm được: **{report.total}** — bỏ qua {skipped} câu vì "
            "SQL mẫu không parse ra bảng nào (lỗi dữ liệu golden set, "
            "KHÔNG tính là trượt). Recall dưới đây chỉ tính trên số câu "
            "chấm được, không phải trên toàn bộ file golden.",
        ]
    else:
        lines += [f"- Câu chấm được: **{report.total}**"]
    lines += [
        f"- Recall: **{report.recall:.3f}**  (ngưỡng bàn giao: {HANDOVER_RECALL:.0%})",
        f"- Bảng/context trung bình: {report.avg_context_tables:.1f}",
        f"- Kết luận: **{'ĐẠT' if report.passed else 'CHƯA ĐẠT'}**",
        "",
    ]
    # Phân biệt "trượt vì quyền" với "trượt vì chú giải" — hai nguyên nhân
    # đòi hỏi hai cách sửa khác hẳn nhau, và gộp chúng vào một lời khuyên
    # chung sẽ đưa operator đi sửa nhầm thứ. `permitted` đã tính sẵn ở trên:
    # một bảng có thật trong schema (`profile.table_names()`) nhưng KHÔNG
    # nằm trong `permitted` sẽ không bao giờ được retriever chọn dù chú
    # giải có hoàn hảo tới đâu — sửa chú giải cho bảng đó là công vô ích.
    existing_names = profile.table_names()
    all_missing = frozenset(t for _, missing in report.misses for t in missing)
    grant_blocked = frozenset(
        t for t in all_missing if t in existing_names and t not in permitted
    )
    # "Chắc chắn do quyền": hoặc user không có bảng nào được cấp (không cần
    # xét từng bảng để biết lý do), hoặc MỌI bảng thiếu trên MỌI câu trượt
    # đều là bảng có thật nhưng ngoài quyền — không còn khoảng trống nào để
    # đổ cho chú giải.
    fully_grants_caused = bool(report.misses) and (not permitted or grant_blocked == all_missing)

    if not permitted:
        lines += [
            f"CẢNH BÁO: user `{user}` không được cấp quyền trên bảng nào. "
            "Mọi câu hỏi trượt vì lý do này — KHÔNG PHẢI do chú giải.",
            f"Thêm quyền bằng `--grant {user}=bang1,bang2` (hoặc "
            f"`--grant {user}=*`), chạy lại `build`, rồi `verify` lại.",
            "",
        ]

    if report.misses:
        lines += ["## Câu chưa đạt", ""]
        shown = report.misses[:20]
        lines += [f"- {q} — thiếu `{sorted(m)}`" for q, m in shown]
        omitted = len(report.misses) - len(shown)
        if omitted > 0:
            lines += [f"- … và {omitted} câu khác không hiện ở đây."]
        if permitted and grant_blocked:
            lines += [
                "",
                f"{len(grant_blocked)} bảng trong các câu trượt trên là do quyền, "
                f"không phải do chú giải: `{sorted(grant_blocked)}` không nằm "
                f"trong quyền của user `{user}`. Xem lại `--grant` nếu đây không "
                "phải chủ đích, rồi chạy lại.",
            ]
        if not fully_grants_caused:
            lines += [
                "",
                "Recall thấp hầu như luôn là do chú giải, không phải do model.",
                "Mở trang 'Chú giải schema', bổ sung mô tả cho các bảng bị thiếu, rồi chạy lại.",
            ]
    (d / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"recall {report.recall:.3f} — {'ĐẠT' if report.passed else 'CHƯA ĐẠT'}")
    return report


def cmd_refresh(
    profile_dir: Path | str,
    dsn: str,
    invoke: Callable[[str, str], str] = _local_invoke,
    *,
    force: bool = False,
) -> tuple[int, int]:
    """Đọc lại cấu trúc, chú giải lại, ghép vào bản cũ.

    Trả `(số bảng đã chú giải, số mục người sửa được giữ nguyên)`. Con số thứ
    hai được in ra có chủ ý: nó là bằng chứng cho khách rằng công sức duyệt
    của họ không mất, và đó là thứ quyết định họ có duyệt lần thứ hai không —
    nên nó không được phóng đại.

    Vì lẽ đó, `preserved` được đếm từ KẾT QUẢ SAU KHI GHÉP (`merged`), không
    phải từ `before` (chú giải trước khi ghép) — xem `_preserved_and_dropped`
    (shared với `cmd_annotate`, C1). `merge_annotations` loại bỏ những mục mà
    bảng/cột của chúng không còn trong schema mới, kể cả mục `human` — đếm
    từ `before` sẽ báo "giữ nguyên" cho những mục vừa bị xoá, đúng lúc thông
    báo này tồn tại để làm bằng chứng KHÔNG bị mất.

    Dòng cảnh báo "N mục bị loại" KHÔNG được in ở đây nữa — nó đã in bên
    trong `cmd_annotate` (được gọi ngay dưới), qua `_report_dropped`, dùng
    CHÍNH giá trị `existing`/`merged` này. In lại ở đây sẽ trùng lặp dòng
    đó trong output của một lượt `refresh`.

    `force` được CHUYỂN THẲNG xuống `cmd_annotate`: `refresh` tự nó cũng có
    thể trúng đúng lỗi C1 (nếu ADBA_DSN trỏ nhầm ngay lúc refresh, `extract`
    bên trong ghi `structure.json` rỗng/thiếu) — không có lý do để `refresh`
    miễn nhiễm với chốt chặn mà `annotate` đứng riêng phải tuân theo.

    Đọc `before` PHẢI xảy ra trước `cmd_extract`: `cmd_extract` ghi đè
    `structure.json`, và sau đó không còn cách nào so sánh với schema cũ.
    `merge_annotations` đòi hỏi `fresh` phủ toàn bộ schema — `cmd_annotate`
    (được gọi qua `annotate_schema`) ghi một mục cho mọi bảng vô điều kiện,
    kể cả khi annotate thất bại (mục đó có `text` rỗng), nên truyền thẳng
    kết quả của nó là đúng; không được lọc/rút gọn `fresh` trước khi ghép.
    """

    d = Path(profile_dir)
    before = load_annotations(d / SCHEMA_YAML)

    cmd_extract(dsn, d)
    merged = cmd_annotate(d, dsn, invoke=invoke, force=force)

    preserved, _dropped = _preserved_and_dropped(before, merged)
    print(f"Giữ nguyên {preserved} mục do người duyệt.")
    return len(merged.tables), preserved


def _resolve_dsn(cli_dsn: str | None) -> str:
    """Lấy DSN từ `--dsn` hoặc `ADBA_DSN`, và xác nhận nó ĐÚNG DẠNG URI.

    Kiểm tra hình dạng ở ĐÂY, TRƯỚC khi bất kỳ lệnh con nào đưa DSN xuống
    `psycopg2.connect` (C2). Một DSN không đúng dạng URI (thiếu `scheme://`
    — vd. lỗi gõ `postgres//...` thay vì `postgresql://...`, một khoảng
    trắng thừa, một artefact copy-paste) khiến libpq/psycopg2 dùng cú pháp
    `key=value` cổ để parse, và khi cú pháp đó cũng không khớp, nó echo
    NGUYÊN VĂN chuỗi DSN vào thông báo lỗi (`invalid dsn: missing "=" after
    "..." in connection info string`) — nếu DSN mang mật khẩu, mật khẩu đó
    lộ ra console, lịch sử shell, log CI, và bất cứ đâu operator paste
    thông báo lỗi đó vào (vd. một ticket hỗ trợ). Một DSN URI-shaped với
    port/query sai KHÔNG lộ theo cách này (xem test); nên chỉ hình dạng
    không phải URI mới cần chặn ở đây.

    Thông báo lỗi không lặp lại DSN (đang cầm sẵn nó mới gọi lệnh, không
    cần nhắc lại) và không đoán mật khẩu là gì — chỉ nói ĐÚNG biến môi
    trường cần kiểm tra và hình dạng đúng phải có.
    """
    dsn = cli_dsn or os.environ.get(DSN_ENV_VAR)
    if not dsn:
        print(
            f"Thiếu DSN: truyền --dsn hoặc đặt biến môi trường {DSN_ENV_VAR}. "
            "Không truyền DSN qua dòng lệnh trên máy khách nếu tránh được — "
            "nó lộ ra trong `ps aux` và lịch sử shell.",
            file=sys.stderr,
        )
        sys.exit(1)

    parts = urllib.parse.urlsplit(dsn)
    if not parts.scheme or not parts.hostname:
        raise OnboardError(
            f"DSN không đúng dạng URI (thiếu 'scheme://' hoặc host). Kiểm "
            f"tra biến môi trường {DSN_ENV_VAR} (hoặc cờ --dsn nếu dùng nó): "
            "DSN phải có dạng 'postgresql://user:password@host:port/dbname'. "
            "Không in lại DSN ở đây vì nó có thể mang mật khẩu — lỗi gõ "
            "thường gặp nhất là thiếu dấu ':' trong 'postgresql://' "
            "(vd. gõ nhầm 'postgres//...')."
        )
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
    p_annotate.add_argument(
        "--force",
        action="store_true",
        help=(
            "bỏ qua chốt chặn khi structure.json rỗng hoặc co ngót nhiều so "
            "với schema.yaml đang có (xem OnboardError khi không có cờ này) — "
            "chỉ dùng khi bảng thật sự đã bị xoá phần lớn, có chủ đích"
        ),
    )

    p_build = sub.add_parser("build", help="dựng profile từ cấu trúc + chú giải")
    _add_dsn_profile_args(p_build)
    p_build.add_argument(
        "--grant",
        action="append",
        default=[],
        help="user=bang1,bang2 hoặc user=* (lặp lại được)",
    )
    p_build.add_argument("--min-reviewed", type=float, default=0.0)

    # `verify` không nhận `--dsn`: nó chỉ đọc `profile/` đã dựng sẵn qua
    # `read_profile` (không bao giờ mở kết nối DB thật), nên không có gì để
    # đòi DSN — bắt operator set ADBA_DSN chỉ để chạy verify là phiền vô ích.
    p_verify = sub.add_parser("verify", help="chấm recall trên golden set của khách")
    p_verify.add_argument("--profile", type=Path, default=Path("profile"))
    p_verify.add_argument("--golden", type=Path, required=True)
    p_verify.add_argument("--user", required=True)
    p_verify.add_argument("--k", type=int, default=8)

    p_refresh = sub.add_parser("refresh", help="đọc lại schema và chú giải phần mới")
    _add_dsn_profile_args(p_refresh)
    p_refresh.add_argument(
        "--force",
        action="store_true",
        help="như --force của `annotate` — refresh gọi cmd_annotate bên trong, cùng chốt chặn",
    )

    args = ap.parse_args()
    try:
        _dispatch(args)
    except OnboardError as e:
        # Lỗi người vận hành sửa được: in câu thông báo, không in traceback.
        print(str(e), file=sys.stderr)
        sys.exit(1)


def _dispatch(args) -> None:
    # `verify` không có cờ `--dsn` (xem p_verify ở main()) và không cần kết
    # nối DB thật, nên không gọi `_resolve_dsn` cho nó — nhánh else của
    # ternary này không được evaluate khi cmd == "verify", nên `args.dsn`
    # (vốn không tồn tại trên namespace của verify) không bao giờ bị chạm.
    dsn = None if args.cmd == "verify" else _resolve_dsn(args.dsn)

    if args.cmd == "extract":
        cmd_extract(dsn, args.profile)
    elif args.cmd == "annotate":
        cmd_annotate(args.profile, dsn, force=args.force)
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
    elif args.cmd == "verify":
        report = cmd_verify(args.profile, args.golden, args.user, k=args.k)
        raise SystemExit(0 if report.passed else 1)
    elif args.cmd == "refresh":
        cmd_refresh(args.profile, dsn, force=args.force)


if __name__ == "__main__":
    main()
