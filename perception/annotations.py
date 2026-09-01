"""Kho chú giải ngữ nghĩa cho một schema.

Chú giải là trần độ chính xác của hệ thống, và phần lớn nó do người viết.
Vì thế quy tắc chi phối module này là: **mục có `reviewed_by: human` không
bao giờ bị ghi đè.** Nếu một lần làm mới xoá công sức của khách thì họ sẽ
không sửa lần thứ hai, và trần đó tụt vĩnh viễn.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import yaml

from perception.schema_model import Table

HUMAN = "human"
LLM = "llm"

_REVIEWED_BY_VALUES = {LLM, HUMAN}
_CONFIDENCE_VALUES = {"high", "low"}


@dataclass(frozen=True)
class Annotation:
    text: str
    reviewed_by: str = LLM
    confidence: str = "high"

    def __post_init__(self) -> None:
        """Từ chối giá trị ngoài miền ngay lúc dựng, không đợi tới lúc đọc lại.

        `_from_plain` đã kiểm hai trường này, nhưng nó chỉ chắn được đường
        VÀO TỪ FILE. Code nội bộ gọi thẳng `Annotation(...)` đi vòng qua
        nó, nên một giá trị sai miền vẫn ghi ra `schema.yaml` được — rồi
        `load_annotations` từ chối đọc chính file mình vừa ghi. Ghi được,
        đọc lại không được, và lỗi nổ ở LẦN CHẠY SAU, cách xa dòng gây ra
        nó. Chặn ở constructor thì nó nổ đúng chỗ.

        `_from_plain` vẫn giữ kiểm tra riêng chứ không uỷ thác xuống đây:
        nó biết đường dẫn file và vị trí bảng/cột, còn constructor thì
        không — mà đúng thông tin đó mới cho người vận hành sửa được file
        của họ. Ở đây chỉ cần nêu trường và giá trị.
        """
        if self.reviewed_by not in _REVIEWED_BY_VALUES:
            raise ValueError(
                f"Annotation.reviewed_by có giá trị không hợp lệ "
                f"{self.reviewed_by!r}; cần một trong {sorted(_REVIEWED_BY_VALUES)}"
            )
        if self.confidence not in _CONFIDENCE_VALUES:
            raise ValueError(
                f"Annotation.confidence có giá trị không hợp lệ "
                f"{self.confidence!r}; cần một trong {sorted(_CONFIDENCE_VALUES)}"
            )


@dataclass(frozen=True)
class SchemaAnnotations:
    tables: Mapping[str, Annotation] = field(default_factory=dict)
    columns: Mapping[str, Mapping[str, Annotation]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Deep-freeze `tables` and `columns` so mutation raises TypeError.

        `frozen=True` on the dataclass only blocks reassigning the
        attributes themselves (`ann.tables = ...`); it does nothing to stop
        `ann.tables["orders"] = ...` from mutating the dict in place. That
        exact bug has shipped twice in this codebase already. Follow the
        same convention as `Table.__post_init__` in `schema_model.py`: wrap
        the outer mapping in `MappingProxyType`, and here — because
        `columns` is a mapping of mappings — wrap each inner per-table
        mapping too. Idempotent: wrapping an already-wrapped mapping is
        safe, which matters because `dataclasses.replace` re-runs
        `__post_init__`.
        """
        if not isinstance(self.tables, MappingProxyType):
            object.__setattr__(self, "tables", MappingProxyType(dict(self.tables)))
        if not isinstance(self.columns, MappingProxyType):
            object.__setattr__(
                self,
                "columns",
                MappingProxyType(
                    {
                        table: cols
                        if isinstance(cols, MappingProxyType)
                        else MappingProxyType(dict(cols))
                        for table, cols in self.columns.items()
                    }
                ),
            )


def _to_plain(ann: Annotation) -> dict:
    return {"text": ann.text, "reviewed_by": ann.reviewed_by, "confidence": ann.confidence}


def _from_plain(raw: Mapping, *, path: Path, table: str, column: str | None = None) -> Annotation:
    """Build an `Annotation` from a hand-editable YAML mapping.

    A missing `reviewed_by`/`confidence` key keeps the documented default
    (`LLM` / `"high"`) — absence is fine. An unrecognized *value* is not:
    silently defaulting a typo like `reviewed_by: Human` would make the
    system believe the LLM owns an annotation a human actually wrote, and
    the very next `merge_annotations` would delete it. Raise instead, and
    name exactly where the bad value lives so it costs the operator one
    character to fix.
    """
    reviewed_by = raw.get("reviewed_by", LLM)
    if reviewed_by not in _REVIEWED_BY_VALUES:
        where = f"table '{table}'" if column is None else f"table '{table}', column '{column}'"
        raise ValueError(
            f"{path}: {where}: field 'reviewed_by' has invalid value {reviewed_by!r}; "
            f"expected one of {sorted(_REVIEWED_BY_VALUES)}"
        )
    confidence = raw.get("confidence", "high")
    if confidence not in _CONFIDENCE_VALUES:
        where = f"table '{table}'" if column is None else f"table '{table}', column '{column}'"
        raise ValueError(
            f"{path}: {where}: field 'confidence' has invalid value {confidence!r}; "
            f"expected one of {sorted(_CONFIDENCE_VALUES)}"
        )
    return Annotation(text=raw.get("text", ""), reviewed_by=reviewed_by, confidence=confidence)


def _atomic_write_with_backup(path: Path, content: str) -> None:
    """Ghi `content` ra `path` không bao giờ để lộ trạng thái nửa vời.

    `schema.yaml` mang trần độ chính xác của hệ thống. Một `write_text`
    thường ghi trực tiếp vào `path` — nếu tiến trình chết giữa chừng, file
    bị cắt cụt, và không có cách nào phân biệt "vừa bị cắt cụt" với "khách
    thật sự chỉ chú giải nửa vời". Ghi vào một file tạm CÙNG THƯ MỤC (bắt
    buộc: `os.replace` giữa hai đường dẫn trên cùng filesystem là một
    syscall rename, nguyên tử trên mọi hệ POSIX/NT) rồi thay bằng
    `os.replace` — tại mọi thời điểm trước lần gọi đó, file đích cũ (nếu
    có) vẫn nguyên vẹn.

    Đây là atomicity chống TIẾN TRÌNH chết, không chống MÁY chết: có
    `fsync` cả file tạm lẫn thư mục cha trước khi `os.replace`, để nội
    dung và việc rename thật sự nằm trên đĩa trước khi hàm trả về — thiếu
    hai lần `fsync` đó, `os.replace` vẫn nguyên tử theo nghĩa "không bao
    giờ thấy trạng thái nửa vời", nhưng một lần mất điện đúng lúc có thể
    khiến `path` trỏ tới một rename chưa kịp ghi xuống đĩa (rơi về nội
    dung cũ hoặc file rỗng tuỳ filesystem/kernel) — chi phí I/O nhỏ (một
    file YAML vài chục KB) đổi lấy durability qua mất điện, không chỉ qua
    process crash.

    Giữ đúng MỘT bản `.bak` — bản ngay trước lần ghi này, không phải một
    lịch sử tích luỹ (mỗi lần ghi đè `.bak` cũ bằng nội dung cũ hiện tại).
    Chỉ tạo `.bak` khi `path` đã tồn tại từ trước: lần ghi đầu tiên (file
    chưa từng có) không có gì để sao lưu, và không được tự bịa ra một file
    thừa trong `profile/` (xem test_no_sample_row_data_is_written_anywhere
    ở tests/unit/test_profile_store.py — nó đếm đúng số file trong đó).

    `mkstemp` tạo file tạm với quyền 0600 bất kể umask hay quyền của
    `path` hiện có — cố ý, vì nội dung tạm không nên đọc được bởi ai khác
    trong lúc đang ghi. Nhưng `os.replace` mang nguyên mode đó sang đích,
    nên nếu không sửa lại trước khi replace, MỖI lần lưu sẽ âm thầm siết
    một `schema.yaml` 0644 (file "cửa thoát hiểm" cho việc sửa tay, có thể
    thuộc một user khác chạy `annotate`) xuống 0600. Khôi phục mode của
    `path` cũ nếu nó đã tồn tại; nếu đây là lần ghi đầu tiên, dùng mode mà
    umask của tiến trình đáng lẽ tạo ra cho một file mới (0o666 & ~umask),
    không ép cứng 0600.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_mode: int | None = None
    if path.exists():
        existing_mode = stat.S_IMODE(os.stat(path).st_mode)
        # copy2 (không phải move/replace): bản GỐC phải còn nguyên tại
        # `path` cho tới khi `os.replace` bên dưới hoán đổi nội dung mới
        # vào — nếu backup thất bại giữa chừng, `path` vẫn là bản cũ hợp lệ.
        shutil.copy2(path, path.with_name(path.name + ".bak"))

    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        if existing_mode is not None:
            os.chmod(tmp_name, existing_mode)
        else:
            umask = os.umask(0)
            os.umask(umask)
            os.chmod(tmp_name, 0o666 & ~umask)
        os.replace(tmp_name, path)
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def save_annotations(ann: SchemaAnnotations, path: Path | str) -> None:
    """Ghi ra YAML đọc được bằng mắt.

    Định dạng người đọc được là có chủ ý: khi giao diện hỏng hoặc khách muốn
    sửa hàng loạt, file này vẫn là đường thoát.

    Ghi NGUYÊN TỬ, kèm giữ một bản `.bak` — xem `_atomic_write_with_backup`.
    """
    payload = {
        "tables": {name: _to_plain(a) for name, a in sorted(ann.tables.items())},
        "columns": {
            table: {col: _to_plain(a) for col, a in sorted(cols.items())}
            for table, cols in sorted(ann.columns.items())
        },
    }
    content = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
    _atomic_write_with_backup(Path(path), content)


def load_annotations(path: Path | str) -> SchemaAnnotations:
    p = Path(path)
    if not p.exists():
        return SchemaAnnotations()
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return SchemaAnnotations(
        tables={
            n: _from_plain(a, path=p, table=n) for n, a in (raw.get("tables") or {}).items()
        },
        columns={
            t: {
                c: _from_plain(a, path=p, table=t, column=c)
                for c, a in (cols or {}).items()
            }
            for t, cols in (raw.get("columns") or {}).items()
        },
    )


def merge_annotations(
    existing: SchemaAnnotations, fresh: SchemaAnnotations
) -> SchemaAnnotations:
    """Ghép chú giải mới sinh vào chú giải đang có.

    Quy tắc: mục `reviewed_by == HUMAN` trong `existing` luôn thắng. Mục do
    LLM sinh bị thay bằng bản mới. Bảng và cột không còn trong `fresh` bị
    loại — giữ chú giải của thứ không còn tồn tại chỉ làm nhiễu bản duyệt.

    Vòng ghép cột lặp theo HỢP của `fresh.tables` và `fresh.columns` — tập
    bảng còn tồn tại, mà `annotate_schema` bảo đảm ghi đủ cho mọi bảng —
    chứ không phải theo riêng `fresh.columns` (hợp với `fresh.columns` chỉ
    để không phá vỡ cách gọi đã có, nơi ai đó truyền `columns` mà không kèm
    `tables` tương ứng). Lý do phải nhìn sang `fresh.tables`: model được
    `SYSTEM_PROMPT` (trong `perception/annotate.py`) chỉ đạo bỏ qua cột
    hiển nhiên như id, và có thể trả `"columns": {}` cho cả một bảng khi
    không cột nào đáng chú giải. Cả hai đều là hoạt động bình thường, không
    phải dấu hiệu bảng hay cột đó đã biến mất khỏi schema thật — nên không
    được dùng sự vắng mặt ấy làm căn cứ xoá chú giải `human`. Với mỗi bảng
    còn trong hợp đó, tập cột được ghép là HỢP của tên cột trong
    `fresh.columns.get(table, {})` và các cột `reviewed_by == HUMAN` trong
    `existing.columns.get(table, {})`; cột chỉ có trong `existing` mà
    không phải `human` thì bị bỏ. Nếu hợp đó rỗng, bảng không được thêm
    vào `columns` — giữ đúng hình dạng hiện tại của đầu ra.

    CẢNH BÁO: việc loại bỏ này dựa hoàn toàn vào `fresh`. `fresh` phải phủ
    toàn bộ schema hiện có — truyền một `fresh` thiếu hoặc rỗng sẽ âm thầm
    xoá sạch mọi chú giải do người viết, kể cả những mục `human` mà quy tắc
    ở trên đáng lẽ phải bảo vệ. Hàm này không tự kiểm tra điều đó; nơi gọi
    (lệnh refresh) chịu trách nhiệm đảm bảo `fresh` là đầy đủ.
    """
    tables = {}
    for name, new in fresh.tables.items():
        old = existing.tables.get(name)
        tables[name] = old if (old and old.reviewed_by == HUMAN) else new

    columns: dict[str, dict[str, Annotation]] = {}
    for table in sorted(set(fresh.tables) | set(fresh.columns)):
        new_cols = fresh.columns.get(table, {})
        old_cols = existing.columns.get(table, {})
        col_names = sorted(
            set(new_cols)
            | {col for col, old in old_cols.items() if old.reviewed_by == HUMAN}
        )
        merged = {}
        for col in col_names:
            old = old_cols.get(col)
            new = new_cols.get(col)
            merged[col] = old if (old and old.reviewed_by == HUMAN) else new
        if merged:
            columns[table] = merged

    return SchemaAnnotations(tables=tables, columns=columns)


def apply_annotations(
    tables: Sequence[Table], ann: SchemaAnnotations
) -> tuple[Table, ...]:
    """Gắn mô tả vào `Table`/`Column` để `render_schema` in ra được."""
    out = []
    for table in tables:
        col_ann = ann.columns.get(table.name, {})
        out.append(
            dataclasses.replace(
                table,
                description=ann.tables[table.name].text if table.name in ann.tables else "",
                columns=tuple(
                    dataclasses.replace(
                        c, description=col_ann[c.name].text if c.name in col_ann else ""
                    )
                    for c in table.columns
                ),
            )
        )
    return tuple(out)


def pending_review(ann: SchemaAnnotations) -> list[tuple[str, str | None]]:
    """Những mục CẦN NHÌN TRƯỚC: LLM tự nhận không chắc, và chưa ai duyệt.

    Đây là hàng đợi ưu tiên, KHÔNG PHẢI thước đo độ phủ duyệt. Điều kiện là
    `reviewed_by != HUMAN and confidence == "low"` — nghĩa là một chú giải
    LLM sinh với `confidence: high` nhưng SAI hoàn toàn vẫn không bao giờ
    vào hàng đợi này, dù đó là rủi ro lớn nhất trong toàn bộ kế hoạch.

    `pending_review(ann) == []` KHÔNG có nghĩa là mọi thứ đã được duyệt; nó
    chỉ có nghĩa là không còn mục nào tự nhận là "chưa chắc". Đừng dùng hàm
    này để xây cổng phát hành dựa trên tiến độ duyệt — đó phải là một phép
    đo khác, dựa trên `reviewed_by == HUMAN` trên toàn bộ tập, không phải
    hàm này.

    Trả `(bảng, None)` cho chú giải bảng và `(bảng, cột)` cho chú giải cột,
    để giao diện xếp chúng vào cùng một hàng đợi.
    """
    out: list[tuple[str, str | None]] = []
    for name, a in ann.tables.items():
        if a.reviewed_by != HUMAN and a.confidence == "low":
            out.append((name, None))
    for table, cols in ann.columns.items():
        for col, a in cols.items():
            if a.reviewed_by != HUMAN and a.confidence == "low":
                out.append((table, col))
    return out
