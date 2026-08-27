"""Dạng bản ghi chung cho mọi bộ dữ liệu eval.

Spider, BIRD, BEAVER có cấu trúc thư mục khác nhau nhưng cùng cho bộ ba
(câu hỏi, SQL mẫu, schema). Quy về một dạng để harness tầng 1 không phải
biết bộ nào.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from perception.schema_model import Table, tables_from_info_box


@dataclass(frozen=True)
class EvalRecord:
    question: str
    gold_sql: str
    db_id: str
    tables: tuple[Table, ...]


def load_adba_golden(path: Path, info_box_path: Path) -> list[EvalRecord]:
    """Đọc golden set của chính ADBA.

    Định dạng: JSONL, mỗi dòng {"question": str, "sql": str}.
    Mọi bản ghi dùng chung một schema, đọc từ info_box.
    """
    tables = tables_from_info_box(json.loads(Path(info_box_path).read_text()))
    records: list[EvalRecord] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        records.append(EvalRecord(
            question=row["question"],
            gold_sql=row["sql"],
            db_id="adba",
            tables=tables,
        ))
    return records


def load_normalized(jsonl_path: Path, schemas_path: Path) -> list[EvalRecord]:
    """Đọc một bộ dữ liệu đã chuẩn hóa về hai file questions.jsonl + schemas.json.

    Bản ghi trỏ tới db_id không có trong schemas.json sẽ bị bỏ qua — bộ dữ
    liệu ngoài thường có câu hỏi cho database không kèm theo bản tải.
    """
    raw = json.loads(Path(schemas_path).read_text())
    tables_by_db = {db_id: tables_from_info_box(box) for db_id, box in raw.items()}
    return load_jsonl_generic(Path(jsonl_path), tables_by_db)


def load_jsonl_generic(path: Path, tables_by_db: dict[str, Sequence[Table]]) -> list[EvalRecord]:
    """Đọc bộ đã chuẩn hóa sẵn: {"question", "sql", "db_id"} mỗi dòng.

    Loader riêng cho Spider/BIRD/BEAVER sẽ chuyển bộ gốc về định dạng này
    rồi gọi hàm này. Việc chuyển đổi đó nằm ở Task 11, vì nó phụ thuộc vào
    bố cục thư mục thực tế của từng bộ.
    """
    records: list[EvalRecord] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        db_id = row["db_id"]
        if db_id not in tables_by_db:
            continue
        records.append(EvalRecord(
            question=row["question"],
            gold_sql=row["sql"],
            db_id=db_id,
            tables=tuple(tables_by_db[db_id]),
        ))
    return records
