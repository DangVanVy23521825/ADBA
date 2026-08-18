"""Sinh chú giải ngữ nghĩa cho từng bảng bằng model LOCAL.

Ràng buộc tuyệt đối: model local, không gọi API ngoài. Schema của khách
không được rời khỏi mạng của họ ở bước đầu tiên của quá trình cài đặt —
đó là thứ duy nhất khiến họ chọn on-prem.

Mọi thứ sinh ra ở đây đều mang `reviewed_by="llm"`. Chỉ giao diện duyệt mới
được đặt `"human"`. Nhầm chỗ này sẽ khiến chú giải máy đoán được coi là đã
xác nhận, và không ai nhìn lại nó nữa.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence

from perception.annotations import LLM, Annotation, SchemaAnnotations
from perception.schema_model import Table

_ALLOWED_CONFIDENCE = {"high", "low"}
_FENCE = re.compile(r"```(?:json)?\s*|\s*```")
_MAX_CELL_LEN = 200

SYSTEM_PROMPT = """\
Bạn mô tả ý nghĩa nghiệp vụ của bảng và cột trong một cơ sở dữ liệu.

Trả về DUY NHẤT một object JSON, không kèm giải thích:

{"table": {"text": "<mô tả bảng>", "confidence": "high|low"},
 "columns": {"<tên cột>": {"text": "<mô tả cột>", "confidence": "high|low"}}}

Quy tắc:
- Viết bằng tiếng Việt, mỗi mô tả một câu ngắn.
- confidence = "low" khi bạn phải đoán. Tên viết tắt không rõ nghĩa, cột cờ
  không có ngữ cảnh, mã nội bộ — đều là "low".
- Thà nhận không chắc còn hơn đoán nghe hợp lý. Một mô tả sai được đánh dấu
  "high" sẽ không ai kiểm lại, và mọi truy vấn dựa vào nó sẽ sai âm thầm.
- Chỉ mô tả cột nào bạn thực sự suy ra được; bỏ qua cột hiển nhiên như id.
"""


def _truncate(value):
    """Cắt một giá trị cell về tối đa `_MAX_CELL_LEN` ký tự.

    Bảng ERP thật có cột note/xml/json chứa hàng chục KB. Ba dòng mẫu như
    vậy tràn context của một model 7B. Chỉ cắt chuỗi — số, bool, None giữ
    nguyên vì chúng vốn đã ngắn và json.dumps xử lý được trực tiếp.
    """
    if isinstance(value, str) and len(value) > _MAX_CELL_LEN:
        return value[:_MAX_CELL_LEN] + "…"
    return value


def _truncate_row(row: Mapping) -> dict:
    return {k: _truncate(v) for k, v in row.items()}


def build_annotation_prompt(table: Table, samples: Sequence[Mapping]) -> str:
    """Prompt cho một bảng: cấu trúc cộng vài dòng thật.

    Dòng mẫu là thứ phân biệt "cột boolean tên flg_tt" với "cờ đánh dấu đơn
    đã thanh toán" — không có chúng thì model chỉ đoán từ tên.
    """
    cols = "\n".join(
        f"  {c.name} {c.data_type}"
        + (" [GENERATED]" if c.is_generated else "")
        + (f" [FK → {table.foreign_keys[c.name]}]" if c.name in table.foreign_keys else "")
        for c in table.columns
    )
    pk = ", ".join(table.primary_key) or "(không có)"
    row_count = "(chưa biết)" if table.row_count is None else table.row_count
    truncated_rows = [_truncate_row(row) for row in list(samples)[:3]]
    rows = json.dumps(truncated_rows, ensure_ascii=False, default=str, indent=1)
    return (
        f"Bảng: {table.name}\n"
        f"Khoá chính: {pk}\n"
        f"Số dòng (ước lượng): {row_count}\n"
        f"Cột:\n{cols}\n\n"
        f"Vài dòng dữ liệu thật:\n{rows}\n\n"
        f"Hãy mô tả bằng tiếng Việt.\n"
    )


def _parse(raw: str) -> dict:
    """Tách một object JSON khỏi phần văn xuôi có thể bao quanh nó.

    Model local hay trả kiểu "Đây là kết quả: {...} Hy vọng giúp ích." —
    `json.loads(text[start:])` đòi mọi thứ sau dấu `{` đầu tiên phải là JSON
    hợp lệ nên sẽ ném lỗi vì đoạn văn xuôi cuối câu. `raw_decode` chỉ đọc
    đúng một giá trị JSON và bỏ qua phần còn lại.

    Không dừng lại ở dấu `{` đầu tiên trong chuỗi: một câu dẫn kiểu
    "Kết quả (dạng {key:val}): {"table": ...}" có một dấu `{` giả trong
    phần văn xuôi trước JSON thật. Nếu khoá vào dấu đó, `raw_decode` sẽ
    hỏng và cả câu trả lời — vốn chứa JSON hợp lệ — bị vứt bỏ oan. Thay vào
    đó, quét từng dấu `{` từ trái sang phải và dùng kết quả đầu tiên giải
    mã thành công (thành một object, không phải số hay chuỗi); chỉ trả về
    `{}` khi mọi ứng viên đều thất bại.
    """
    text = _FENCE.sub("", raw).strip()
    decoder = json.JSONDecoder()
    pos = text.find("{")
    while pos >= 0:
        try:
            obj, _end = decoder.raw_decode(text, pos)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        pos = text.find("{", pos + 1)
    return {}


def _confidence(value) -> str:
    return value if value in _ALLOWED_CONFIDENCE else "low"


def annotate_schema(
    tables: Sequence[Table],
    samples: Mapping[str, list[dict]],
    invoke: Callable[[str, str], str],
) -> tuple[SchemaAnnotations, int]:
    """Chú giải từng bảng một.

    Từng bảng chứ không phải cả schema một lượt: 150 bảng không vừa context,
    và một lần trả lời hỏng chỉ làm hỏng một bảng thay vì cả lượt chạy. Điều
    đó chỉ đúng nếu `invoke` ném lỗi (Ollama sập, timeout, hết bộ nhớ) cũng
    được xử lý như một lần trả lời không đọc được — nếu không, một bảng lỗi
    hạ tầng ở giữa lượt chạy sẽ làm mất công 36 bảng đã xong trước đó.

    Một bảng tính là lỗi khi mô tả bảng thu được rỗng — bất kể nguyên nhân
    là `invoke` ném lỗi, câu trả lời không đọc được, hay câu trả lời là JSON
    hợp lệ nhưng rỗng (`{"table": {}, "columns": {}}`). Model local suy
    thoái về JSON hợp lệ-nhưng-rỗng là một kiểu lỗi thật, khác lỗi kết nối
    nhưng cùng hậu quả: bảng đó không được chú giải. Đếm theo văn bản rỗng
    thay vì theo "parse được hay không" mới trung thực với đúng câu hỏi mà
    con số này trả lời cho người vận hành.
    """
    table_ann: dict[str, Annotation] = {}
    column_ann: dict[str, dict[str, Annotation]] = {}
    failures = 0

    for table in tables:
        prompt = build_annotation_prompt(table, samples.get(table.name, []))
        try:
            raw = invoke(SYSTEM_PROMPT, prompt)
        except Exception:
            raw = ""
        parsed = _parse(raw)

        t = parsed.get("table") or {}
        text = t.get("text", "")
        table_ann[table.name] = Annotation(
            text=text,
            reviewed_by=LLM,
            confidence=_confidence(t.get("confidence")) if text else "low",
        )
        if not text:
            failures += 1

        cols = parsed.get("columns") or {}
        known = {c.name for c in table.columns}
        entries = {
            name: Annotation(
                text=body.get("text", ""),
                reviewed_by=LLM,
                confidence=_confidence(body.get("confidence")),
            )
            for name, body in cols.items()
            if name in known and isinstance(body, dict) and body.get("text")
        }
        if entries:
            column_ann[table.name] = entries

    return SchemaAnnotations(tables=table_ann, columns=column_ann), failures
