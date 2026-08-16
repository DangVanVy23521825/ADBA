# Đường onboarding — Implementation Plan (Plan A)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Biến một database Postgres lạ thành một `profile/` mà ADBA trả lời được câu hỏi trên đó, với chú giải ngữ nghĩa do LLM sinh và người duyệt qua giao diện.

**Architecture:** Một chuỗi năm bước chạy tại chỗ khách: introspect → chú giải bằng LLM local → người duyệt qua trang Streamlit → dựng profile → kiểm và chấm điểm. Chú giải lưu ở `schema.yaml` khoá theo bảng/cột, và mục nào có `reviewed_by: human` không bao giờ bị ghi đè khi làm mới. Toàn bộ trạng thái nằm trong một thư mục `profile/` đọc được bằng `read_profile()`.

**Tech Stack:** Python 3.12, psycopg2, PyYAML, Streamlit, pytest. Không thêm phụ thuộc mới.

**Spec:** `docs/superpowers/specs/2026-08-15-adba-multi-schema-onprem-design.md` — pha 3.
**Lộ trình:** `docs/superpowers/plans/2026-08-16-lo-trinh-toi-dong-goi.md`

## Global Constraints

- **Chú giải sinh bằng model LOCAL, tuyệt đối không gọi OpenAI.** Nếu không thì schema của khách rời khỏi mạng ngay ở thao tác đầu tiên của quá trình cài đặt — thứ duy nhất khiến họ chọn on-prem. (spec 3.2)
- **`sample_rows` không được ghi vào `profile/`.** Nó cần cho bước chú giải để đoán ý nghĩa cột, nhưng `profile/` là thứ bị copy đi khi hỗ trợ kỹ thuật. (spec 3.2, tiêu chí 9)
- **`refresh` phải giữ nguyên 100% mục `reviewed_by: human`.** Nếu làm mới xoá công sức của khách thì họ không sửa lần thứ hai, và chất lượng chú giải — tức trần accuracy — tụt vĩnh viễn. (spec 5.2, tiêu chí 10)
- **Không bao giờ dựng `ConnectionProfile(...)` trực tiếp — luôn qua `build_profile(...)`.** Deep-freeze bảo vệ `grants` chỉ nằm trong `build_profile`.
- **`permitted_tables` là ranh giới bảo mật; `retrieved_tables` chỉ là nội dung prompt.** Không gộp. (spec 3.4.1)
- Giữ ngưỡng `schema_mode` ở **6.000 token**. Không quy đổi sang số bảng. (spec 3.3)
- Không hồi quy: **227 test** hiện tại (201 unit + 26 integration) không được giảm, không test nào đang pass được phép fail.
- Chạy test: `"/Users/dangvanvy/Documents/ADBA — Autonomous Data & Business Intelligence Agent/.venv/bin/python" -m pytest <dir> -q`, chạy **ở foreground**, hai lệnh riêng cho `tests/unit` và `tests/integration`. Không đẩy vào nền. Nếu `tests/integration` quá ~30 giây thì có lời gọi model thật — dừng và báo.
- Container `adba-postgres` chạy ở `localhost:5432` với dữ liệu đã seed. Không dừng nó, **không chạy `docker compose` trong worktree**.
- 19 cảnh báo `multiprocessing` DeprecationWarning có sẵn từ trước, không phải finding.

---

## File Structure

**Tạo mới:**

| File | Trách nhiệm |
|---|---|
| `perception/introspect.py` | Đọc Postgres → `tuple[Table, ...]`. Đường duy nhất từ DB thật sang kiểu dữ liệu nội bộ. |
| `perception/annotations.py` | Kiểu `Annotation`/`SchemaAnnotations`, đọc/ghi `schema.yaml`, **merge giữ mục người sửa**, gắn mô tả vào `Table`. Hàm thuần. |
| `perception/annotate.py` | Sinh chú giải bằng LLM local kèm cờ độ tự tin. Nhận `invoke` để test không cần model. |
| `perception/profile_store.py` | Ghi/đọc thư mục `profile/`. `read_profile()` trả `ConnectionProfile`. |
| `onboard.py` | CLI một lệnh, năm lệnh con: `extract`, `annotate`, `build`, `verify`, `refresh`. |
| `pages/1_Chu_giai_schema.py` | Trang Streamlit để analyst duyệt và sửa chú giải. |
| `tests/unit/test_*.py` | Một file test cho mỗi module trên. |
| `tests/integration/test_onboard_flow.py` | Chạy trọn chuỗi trên `adba-postgres`. |

**Sửa:**

| File | Sửa gì |
|---|---|
| `perception/schema_model.py` | `Column` thêm `description: str = ""` |
| `perception/render_schema.py` | Render mô tả cột thành comment cuối dòng |
| `perception/retrieval.py` | `Retriever.search` thêm `candidates`; lọc quyền **trước** khi tìm |
| `perception/schema_context.py` | Truyền `permitted` xuống `search` làm nguồn ứng viên |
| `app.py` | Đọc profile từ đĩa thay vì dựng lại mỗi câu hỏi |

---

## Task 1: `introspect_schema` — đọc Postgres thành `Table`

**Files:**
- Create: `perception/introspect.py`
- Test: `tests/unit/test_introspect.py`

**Interfaces:**
- Consumes: `perception.schema_model.Column`, `Table`
- Produces:
  - `perception.introspect.introspect_schema(dsn: str, schema: str = "public") -> tuple[Table, ...]`
  - `perception.introspect.sample_rows(dsn: str, table: str, n: int = 5) -> list[dict]`

`perception/extract_info_box.py` đã có logic introspect nhưng đi qua JSON trung gian và gắn với ba domain cứng. Đường mới trả thẳng `Table` — bớt một tầng và bỏ được sự phụ thuộc vào cấu trúc domain.

- [ ] **Step 1: Viết test thất bại**

Tạo `tests/unit/test_introspect.py`:

```python
import os

import pytest

from perception.introspect import introspect_schema, sample_rows

DSN = os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")
pytestmark = pytest.mark.skipif(not DSN, reason="cần DATABASE_URL trỏ Postgres thật")


def test_finds_every_public_table():
    names = {t.name for t in introspect_schema(DSN)}
    assert {"orders", "customers", "products", "payroll"} <= names


def test_columns_carry_name_and_type():
    orders = next(t for t in introspect_schema(DSN) if t.name == "orders")
    by_name = {c.name: c for c in orders.columns}
    assert "id" in by_name
    assert by_name["id"].data_type


def test_primary_key_is_detected():
    orders = next(t for t in introspect_schema(DSN) if t.name == "orders")
    assert orders.primary_key


def test_foreign_keys_render_as_table_paren_column():
    orders = next(t for t in introspect_schema(DSN) if t.name == "orders")
    assert any(ref.startswith("customers(") for ref in orders.foreign_keys.values())


def test_generated_columns_are_flagged():
    orders = next(t for t in introspect_schema(DSN) if t.name == "orders")
    assert any(c.is_generated for c in orders.columns), "orders có cột GENERATED"


def test_descriptions_start_empty():
    """Introspect chỉ ra cấu trúc. Ý nghĩa đến từ bước chú giải."""
    tables = introspect_schema(DSN)
    assert all(t.description == "" for t in tables)
    assert all(c.description == "" for t in tables for c in t.columns)


def test_sample_rows_returns_dicts_capped_at_n():
    rows = sample_rows(DSN, "customers", n=3)
    assert len(rows) <= 3
    assert all(isinstance(r, dict) for r in rows)


def test_sample_rows_rejects_a_table_name_that_is_not_an_identifier():
    with pytest.raises(ValueError):
        sample_rows(DSN, "orders; DROP TABLE customers", n=1)
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `.venv/bin/python -m pytest tests/unit/test_introspect.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'perception.introspect'`

- [ ] **Step 3: Hiện thực**

Tạo `perception/introspect.py`:

```python
"""Đọc cấu trúc một database Postgres thành kiểu dữ liệu nội bộ.

Chỉ ra CẤU TRÚC. Ý nghĩa — mô tả bảng và cột — đến từ bước chú giải sau đó;
introspect không đoán, vì đoán sai ở đây sẽ lặng lẽ sai suốt chuỗi.
"""

from __future__ import annotations

import re

import psycopg2
import psycopg2.extras

from perception.schema_model import Column, Table

_IDENT = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

_COLUMNS_SQL = """
SELECT table_name, column_name, data_type, is_generated
FROM information_schema.columns
WHERE table_schema = %s
ORDER BY table_name, ordinal_position
"""

_PK_SQL = """
SELECT tc.table_name, kcu.column_name
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu
  ON tc.constraint_name = kcu.constraint_name
 AND tc.table_schema = kcu.table_schema
WHERE tc.constraint_type = 'PRIMARY KEY' AND tc.table_schema = %s
ORDER BY kcu.ordinal_position
"""

_FK_SQL = """
SELECT tc.table_name, kcu.column_name,
       ccu.table_name AS ref_table, ccu.column_name AS ref_column
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu
  ON tc.constraint_name = kcu.constraint_name
 AND tc.table_schema = kcu.table_schema
JOIN information_schema.constraint_column_usage ccu
  ON ccu.constraint_name = tc.constraint_name
WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = %s
"""

_COUNT_SQL = "SELECT reltuples::bigint FROM pg_class WHERE relname = %s"


def introspect_schema(dsn: str, schema: str = "public") -> tuple[Table, ...]:
    """Trả về mọi bảng trong `schema`, kèm cột, khoá chính và khoá ngoại.

    `row_count` lấy ước lượng từ `pg_class` chứ không `COUNT(*)`: trên bảng
    hàng chục triệu dòng, một lần đếm thật có thể mất hàng phút và con số đó
    chỉ dùng để tham khảo trong prompt.
    """
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(_COLUMNS_SQL, (schema,))
            cols: dict[str, list[Column]] = {}
            for table, name, dtype, generated in cur.fetchall():
                cols.setdefault(table, []).append(
                    Column(name=name, data_type=dtype, is_generated=(generated == "ALWAYS"))
                )

            cur.execute(_PK_SQL, (schema,))
            pks: dict[str, list[str]] = {}
            for table, column in cur.fetchall():
                pks.setdefault(table, []).append(column)

            cur.execute(_FK_SQL, (schema,))
            fks: dict[str, dict[str, str]] = {}
            for table, column, ref_table, ref_column in cur.fetchall():
                fks.setdefault(table, {})[column] = f"{ref_table}({ref_column})"

            counts: dict[str, int] = {}
            for table in cols:
                cur.execute(_COUNT_SQL, (table,))
                row = cur.fetchone()
                counts[table] = int(row[0]) if row and row[0] is not None else 0
    finally:
        conn.close()

    return tuple(
        Table(
            name=table,
            columns=tuple(columns),
            primary_key=tuple(pks.get(table, ())),
            foreign_keys=dict(fks.get(table, {})),
            row_count=counts.get(table),
            description="",
        )
        for table, columns in sorted(cols.items())
    )


def sample_rows(dsn: str, table: str, n: int = 5) -> list[dict]:
    """Vài dòng mẫu để bước chú giải đoán ý nghĩa cột.

    KHÔNG được ghi vào `profile/` — xem Global Constraints. Tên bảng đi qua
    regex định danh vì nó bị nội suy vào SQL (không tham số hoá được tên
    bảng trong Postgres).
    """
    if not _IDENT.match(table):
        raise ValueError(f"Tên bảng không hợp lệ: {table!r}")

    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"SELECT * FROM {table} LIMIT %s", (n,))  # noqa: S608
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()
```

- [ ] **Step 4: Chạy test để xác nhận pass**

Run: `set -a && . ./.env && set +a && DATABASE_URL="$POSTGRES_URL" .venv/bin/python -m pytest tests/unit/test_introspect.py -q`
Expected: PASS, 8 passed

- [ ] **Step 5: Chạy cả hai thư mục test và commit**

```bash
git add perception/introspect.py tests/unit/test_introspect.py
git commit -m "feat(perception): introspect_schema đọc Postgres thẳng ra Table"
```

---

## Task 2: `Column.description` và render thành comment

**Files:**
- Modify: `perception/schema_model.py`
- Modify: `perception/render_schema.py`
- Test: `tests/unit/test_render_schema.py` (bổ sung)

**Interfaces:**
- Produces: `Column(name, data_type, is_generated=False, description="")`; `render_schema` in mô tả cột thành comment cuối dòng.

Chú giải cấp cột là thứ analyst sẽ sửa nhiều nhất — `flg_tt` mới là chỗ cần giải thích, không phải tên bảng. Thêm trường có giá trị mặc định nên tương thích ngược với mọi nơi đang dựng `Column`.

- [ ] **Step 1: Viết test thất bại**

Thêm vào `tests/unit/test_render_schema.py`:

```python
def test_column_description_renders_as_a_trailing_comment():
    t = Table(
        name="orders",
        columns=(
            Column("id", "integer"),
            Column("flg_tt", "boolean", description="Cờ đã thanh toán"),
        ),
    )
    out = render_schema([t])
    assert "flg_tt BOOL  -- Cờ đã thanh toán" in out


def test_column_without_a_description_gets_no_comment():
    t = Table(name="x", columns=(Column("a", "integer"),))
    assert "--" not in render_schema([t])


def test_column_description_does_not_break_the_700_byte_ceiling():
    """Chú giải làm bảng to lên; trần vẫn phải giữ."""
    import dataclasses

    from tests.fixtures.mini_schema import MINI_TABLES

    fat = tuple(
        dataclasses.replace(
            t,
            columns=tuple(
                dataclasses.replace(c, description="Mô tả nghiệp vụ dài vừa phải cho cột này")
                for c in t.columns
            ),
        )
        for t in MINI_TABLES
    )
    out = render_schema(fat)
    assert len(out.encode()) / len(fat) <= 700
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `.venv/bin/python -m pytest tests/unit/test_render_schema.py -q`
Expected: FAIL — `Column.__init__() got an unexpected keyword argument 'description'`

- [ ] **Step 3: Thêm trường vào `Column`**

Trong `perception/schema_model.py`, thêm vào dataclass `Column`:

```python
@dataclass(frozen=True)
class Column:
    name: str
    data_type: str
    is_generated: bool = False
    description: str = ""
```

- [ ] **Step 4: Render mô tả cột**

Trong `perception/render_schema.py`, hàm `_render_one`, sau khi ghép `parts` cho mỗi cột:

```python
        line = "  " + " ".join(parts)
        if col.description:
            line += f"  -- {col.description}"
        body.append(line)
```

- [ ] **Step 5: Chạy test và commit**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: mọi test pass; số test tăng 3.

```bash
git add perception/schema_model.py perception/render_schema.py tests/unit/test_render_schema.py
git commit -m "feat(perception): Column mang mô tả, render thành comment cuối dòng"
```

---

## Task 3: Kho chú giải và phép merge giữ công sức người

Đây là task quan trọng nhất về mặt sản phẩm. Nếu merge sai, khách sửa một lần rồi không sửa nữa.

**Files:**
- Create: `perception/annotations.py`
- Test: `tests/unit/test_annotations.py`

**Interfaces:**
- Consumes: `perception.schema_model.Column`, `Table`
- Produces:
  - `Annotation(text: str, reviewed_by: str, confidence: str)` — `reviewed_by ∈ {"llm","human"}`, `confidence ∈ {"high","low"}`
  - `SchemaAnnotations(tables: Mapping[str, Annotation], columns: Mapping[str, Mapping[str, Annotation]])`
  - `load_annotations(path) -> SchemaAnnotations`
  - `save_annotations(ann, path) -> None`
  - `merge_annotations(existing, fresh) -> SchemaAnnotations`
  - `apply_annotations(tables, ann) -> tuple[Table, ...]`
  - `pending_review(ann) -> list[tuple[str, str | None]]`

- [ ] **Step 1: Viết test thất bại**

Tạo `tests/unit/test_annotations.py`:

```python
from pathlib import Path

from perception.annotations import (
    Annotation,
    SchemaAnnotations,
    apply_annotations,
    load_annotations,
    merge_annotations,
    pending_review,
    save_annotations,
)
from perception.schema_model import Column, Table


def _ann(text, reviewed_by="llm", confidence="high") -> Annotation:
    return Annotation(text=text, reviewed_by=reviewed_by, confidence=confidence)


def _sample() -> SchemaAnnotations:
    return SchemaAnnotations(
        tables={"orders": _ann("Đơn hàng", reviewed_by="human")},
        columns={"orders": {"flg_tt": _ann("Cờ thanh toán", confidence="low")}},
    )


# ── vòng đời file ───────────────────────────────────────────────────────────

def test_save_then_load_round_trips(tmp_path: Path):
    path = tmp_path / "schema.yaml"
    save_annotations(_sample(), path)
    back = load_annotations(path)
    assert back.tables["orders"].text == "Đơn hàng"
    assert back.tables["orders"].reviewed_by == "human"
    assert back.columns["orders"]["flg_tt"].confidence == "low"


def test_loading_a_missing_file_gives_empty_annotations(tmp_path: Path):
    empty = load_annotations(tmp_path / "khong-co.yaml")
    assert empty.tables == {}
    assert empty.columns == {}


def test_saved_file_is_human_readable_yaml(tmp_path: Path):
    path = tmp_path / "schema.yaml"
    save_annotations(_sample(), path)
    text = path.read_text(encoding="utf-8")
    assert "orders" in text and "Đơn hàng" in text


# ── merge: phần quan trọng nhất ─────────────────────────────────────────────

def test_human_edits_survive_a_refresh():
    existing = SchemaAnnotations(
        tables={"orders": _ann("Mô tả do NGƯỜI sửa", reviewed_by="human")}, columns={}
    )
    fresh = SchemaAnnotations(
        tables={"orders": _ann("Mô tả do LLM sinh lại", reviewed_by="llm")}, columns={}
    )
    merged = merge_annotations(existing, fresh)
    assert merged.tables["orders"].text == "Mô tả do NGƯỜI sửa"
    assert merged.tables["orders"].reviewed_by == "human"


def test_llm_entries_are_replaced_by_a_refresh():
    existing = SchemaAnnotations(tables={"orders": _ann("Bản cũ")}, columns={})
    fresh = SchemaAnnotations(tables={"orders": _ann("Bản mới")}, columns={})
    assert merge_annotations(existing, fresh).tables["orders"].text == "Bản mới"


def test_human_column_edits_survive_too():
    existing = SchemaAnnotations(
        tables={}, columns={"orders": {"flg_tt": _ann("Người sửa", reviewed_by="human")}}
    )
    fresh = SchemaAnnotations(
        tables={}, columns={"orders": {"flg_tt": _ann("LLM sinh lại")}}
    )
    merged = merge_annotations(existing, fresh)
    assert merged.columns["orders"]["flg_tt"].text == "Người sửa"


def test_new_tables_from_a_refresh_are_added():
    existing = SchemaAnnotations(tables={"orders": _ann("A")}, columns={})
    fresh = SchemaAnnotations(tables={"orders": _ann("B"), "invoices": _ann("C")}, columns={})
    merged = merge_annotations(existing, fresh)
    assert merged.tables["invoices"].text == "C"


def test_a_table_dropped_from_the_schema_is_dropped_from_annotations():
    """Giữ lại chú giải của bảng không còn tồn tại chỉ làm nhiễu bản duyệt."""
    existing = SchemaAnnotations(
        tables={"orders": _ann("A", reviewed_by="human"), "cu": _ann("B", reviewed_by="human")},
        columns={},
    )
    fresh = SchemaAnnotations(tables={"orders": _ann("A2")}, columns={})
    merged = merge_annotations(existing, fresh)
    assert "cu" not in merged.tables


def test_no_human_entry_is_ever_lost_across_a_full_refresh():
    """Ràng buộc của spec, phát biểu trực tiếp: 100% mục human sống sót."""
    existing = SchemaAnnotations(
        tables={f"t{i}": _ann(f"người {i}", reviewed_by="human") for i in range(20)},
        columns={f"t{i}": {"c": _ann(f"cột {i}", reviewed_by="human")} for i in range(20)},
    )
    fresh = SchemaAnnotations(
        tables={f"t{i}": _ann("llm ghi đè") for i in range(20)},
        columns={f"t{i}": {"c": _ann("llm ghi đè")} for i in range(20)},
    )
    merged = merge_annotations(existing, fresh)
    assert all(merged.tables[f"t{i}"].text == f"người {i}" for i in range(20))
    assert all(merged.columns[f"t{i}"]["c"].text == f"cột {i}" for i in range(20))


# ── gắn vào Table ───────────────────────────────────────────────────────────

def test_apply_puts_descriptions_onto_tables_and_columns():
    tables = (Table(name="orders", columns=(Column("flg_tt", "boolean"),)),)
    out = apply_annotations(tables, _sample())
    assert out[0].description == "Đơn hàng"
    assert out[0].columns[0].description == "Cờ thanh toán"


def test_apply_leaves_unannotated_entries_empty():
    tables = (Table(name="khac", columns=(Column("x", "integer"),)),)
    out = apply_annotations(tables, _sample())
    assert out[0].description == ""
    assert out[0].columns[0].description == ""


# ── hàng đợi duyệt ──────────────────────────────────────────────────────────

def test_pending_review_lists_low_confidence_and_unreviewed():
    ann = SchemaAnnotations(
        tables={
            "a": _ann("ok", reviewed_by="human"),
            "b": _ann("chưa chắc", confidence="low"),
        },
        columns={"a": {"c1": _ann("chưa chắc", confidence="low")}},
    )
    pending = pending_review(ann)
    assert ("b", None) in pending
    assert ("a", "c1") in pending
    assert ("a", None) not in pending, "mục người đã duyệt không nằm trong hàng đợi"
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `.venv/bin/python -m pytest tests/unit/test_annotations.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'perception.annotations'`

- [ ] **Step 3: Hiện thực**

Tạo `perception/annotations.py`:

```python
"""Kho chú giải ngữ nghĩa cho một schema.

Chú giải là trần độ chính xác của hệ thống, và phần lớn nó do người viết.
Vì thế quy tắc chi phối module này là: **mục có `reviewed_by: human` không
bao giờ bị ghi đè.** Nếu một lần làm mới xoá công sức của khách thì họ sẽ
không sửa lần thứ hai, và trần đó tụt vĩnh viễn.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from perception.schema_model import Table

HUMAN = "human"
LLM = "llm"


@dataclass(frozen=True)
class Annotation:
    text: str
    reviewed_by: str = LLM
    confidence: str = "high"


@dataclass(frozen=True)
class SchemaAnnotations:
    tables: Mapping[str, Annotation] = field(default_factory=dict)
    columns: Mapping[str, Mapping[str, Annotation]] = field(default_factory=dict)


def _to_plain(ann: Annotation) -> dict:
    return {"text": ann.text, "reviewed_by": ann.reviewed_by, "confidence": ann.confidence}


def _from_plain(raw: Mapping) -> Annotation:
    return Annotation(
        text=raw.get("text", ""),
        reviewed_by=raw.get("reviewed_by", LLM),
        confidence=raw.get("confidence", "high"),
    )


def save_annotations(ann: SchemaAnnotations, path: Path | str) -> None:
    """Ghi ra YAML đọc được bằng mắt.

    Định dạng người đọc được là có chủ ý: khi giao diện hỏng hoặc khách muốn
    sửa hàng loạt, file này vẫn là đường thoát.
    """
    payload = {
        "tables": {name: _to_plain(a) for name, a in sorted(ann.tables.items())},
        "columns": {
            table: {col: _to_plain(a) for col, a in sorted(cols.items())}
            for table, cols in sorted(ann.columns.items())
        },
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def load_annotations(path: Path | str) -> SchemaAnnotations:
    p = Path(path)
    if not p.exists():
        return SchemaAnnotations()
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return SchemaAnnotations(
        tables={n: _from_plain(a) for n, a in (raw.get("tables") or {}).items()},
        columns={
            t: {c: _from_plain(a) for c, a in (cols or {}).items()}
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
    """
    tables = {}
    for name, new in fresh.tables.items():
        old = existing.tables.get(name)
        tables[name] = old if (old and old.reviewed_by == HUMAN) else new

    columns: dict[str, dict[str, Annotation]] = {}
    for table, new_cols in fresh.columns.items():
        old_cols = existing.columns.get(table, {})
        merged = {}
        for col, new in new_cols.items():
            old = old_cols.get(col)
            merged[col] = old if (old and old.reviewed_by == HUMAN) else new
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
    """Những mục cần người nhìn: LLM tự nhận không chắc, và chưa ai duyệt.

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
```

- [ ] **Step 4: Chạy test để xác nhận pass**

Run: `.venv/bin/python -m pytest tests/unit/test_annotations.py -q`
Expected: PASS, 13 passed

- [ ] **Step 5: Commit**

```bash
git add perception/annotations.py tests/unit/test_annotations.py
git commit -m "feat(perception): kho chú giải + merge giữ nguyên mục người sửa"
```

---

## Task 4: Sinh chú giải bằng LLM local

**Files:**
- Create: `perception/annotate.py`
- Test: `tests/unit/test_annotate.py`

**Interfaces:**
- Consumes: `Table`, `SchemaAnnotations`, `Annotation`
- Produces: `annotate_schema(tables, samples, invoke) -> SchemaAnnotations`
  - `samples: Mapping[str, list[dict]]` — bảng → vài dòng mẫu
  - `invoke: Callable[[str, str], str]` — `(system_prompt, user_prompt) -> raw`, tiêm vào để test không cần model

- [ ] **Step 1: Viết test thất bại**

Tạo `tests/unit/test_annotate.py`:

```python
import json

from perception.annotate import annotate_schema, build_annotation_prompt
from perception.schema_model import Column, Table

TABLES = (
    Table(
        name="orders",
        columns=(Column("id", "integer"), Column("flg_tt", "boolean")),
        primary_key=("id",),
    ),
)
SAMPLES = {"orders": [{"id": 1, "flg_tt": True}]}


def _reply(payload: dict):
    return lambda system, user: json.dumps(payload, ensure_ascii=False)  # noqa: ARG005


def test_produces_a_table_description():
    ann = annotate_schema(
        TABLES, SAMPLES,
        _reply({"table": {"text": "Đơn hàng bán", "confidence": "high"},
                "columns": {}}),
    )
    assert ann.tables["orders"].text == "Đơn hàng bán"


def test_produces_column_descriptions():
    ann = annotate_schema(
        TABLES, SAMPLES,
        _reply({"table": {"text": "Đơn hàng", "confidence": "high"},
                "columns": {"flg_tt": {"text": "Cờ đã thanh toán", "confidence": "low"}}}),
    )
    assert ann.columns["orders"]["flg_tt"].text == "Cờ đã thanh toán"
    assert ann.columns["orders"]["flg_tt"].confidence == "low"


def test_everything_it_generates_is_marked_as_llm_not_human():
    ann = annotate_schema(
        TABLES, SAMPLES,
        _reply({"table": {"text": "x", "confidence": "high"},
                "columns": {"flg_tt": {"text": "y", "confidence": "high"}}}),
    )
    assert ann.tables["orders"].reviewed_by == "llm"
    assert ann.columns["orders"]["flg_tt"].reviewed_by == "llm"


def test_an_unparseable_reply_becomes_low_confidence_not_a_crash():
    """Onboarding chạy trên 150 bảng; một lần model trả rác không được làm hỏng cả lượt."""
    ann = annotate_schema(TABLES, SAMPLES, lambda s, u: "không phải json")  # noqa: ARG005
    assert ann.tables["orders"].confidence == "low"
    assert ann.tables["orders"].text == ""


def test_a_confidence_value_outside_the_allowed_set_is_treated_as_low():
    ann = annotate_schema(
        TABLES, SAMPLES,
        _reply({"table": {"text": "x", "confidence": "rất chắc chắn"}, "columns": {}}),
    )
    assert ann.tables["orders"].confidence == "low"


def test_the_prompt_carries_column_names_and_sample_values():
    prompt = build_annotation_prompt(TABLES[0], SAMPLES["orders"])
    assert "flg_tt" in prompt
    assert "orders" in prompt


def test_the_prompt_asks_for_vietnamese():
    assert "tiếng Việt" in build_annotation_prompt(TABLES[0], SAMPLES["orders"])
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `.venv/bin/python -m pytest tests/unit/test_annotate.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'perception.annotate'`

- [ ] **Step 3: Hiện thực**

Tạo `perception/annotate.py`:

```python
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
    rows = json.dumps(list(samples)[:3], ensure_ascii=False, default=str, indent=1)
    return (
        f"Bảng: {table.name}\n"
        f"Khoá chính: {pk}\n"
        f"Số dòng (ước lượng): {table.row_count}\n"
        f"Cột:\n{cols}\n\n"
        f"Vài dòng dữ liệu thật:\n{rows}\n"
    )


def _parse(raw: str) -> dict:
    text = _FENCE.sub("", raw).strip()
    start = text.find("{")
    if start < 0:
        return {}
    try:
        return json.loads(text[start:])
    except json.JSONDecodeError:
        return {}


def _confidence(value) -> str:
    return value if value in _ALLOWED_CONFIDENCE else "low"


def annotate_schema(
    tables: Sequence[Table],
    samples: Mapping[str, list[dict]],
    invoke: Callable[[str, str], str],
) -> SchemaAnnotations:
    """Chú giải từng bảng một.

    Từng bảng chứ không phải cả schema một lượt: 150 bảng không vừa context,
    và một lần trả lời hỏng chỉ làm hỏng một bảng thay vì cả lượt chạy.
    """
    table_ann: dict[str, Annotation] = {}
    column_ann: dict[str, dict[str, Annotation]] = {}

    for table in tables:
        prompt = build_annotation_prompt(table, samples.get(table.name, []))
        parsed = _parse(invoke(SYSTEM_PROMPT, prompt))

        t = parsed.get("table") or {}
        table_ann[table.name] = Annotation(
            text=t.get("text", ""),
            reviewed_by=LLM,
            confidence=_confidence(t.get("confidence")) if t.get("text") else "low",
        )

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

    return SchemaAnnotations(tables=table_ann, columns=column_ann)
```

- [ ] **Step 4: Chạy test để xác nhận pass**

Run: `.venv/bin/python -m pytest tests/unit/test_annotate.py -q`
Expected: PASS, 7 passed

- [ ] **Step 5: Commit**

```bash
git add perception/annotate.py tests/unit/test_annotate.py
git commit -m "feat(perception): sinh chú giải bằng model local, có cờ độ tự tin"
```

---

## Task 5: Thư mục `profile/` — ghi và đọc

**Files:**
- Create: `perception/profile_store.py`
- Test: `tests/unit/test_profile_store.py`

**Interfaces:**
- Consumes: `Table`, `SchemaAnnotations`, `build_profile`, `schema_fingerprint`
- Produces:
  - `write_profile(dir, *, dsn, tables, annotations, grants, threshold_tokens=6000) -> None`
  - `read_profile(dir) -> ConnectionProfile`
  - `profile_is_stale(dir, tables) -> bool`

- [ ] **Step 1: Viết test thất bại**

Tạo `tests/unit/test_profile_store.py`:

```python
import dataclasses

import pytest

from perception.annotations import Annotation, SchemaAnnotations
from perception.connection_profile import ALL_TABLES, permitted_tables
from perception.profile_store import profile_is_stale, read_profile, write_profile
from perception.schema_model import Column
from tests.fixtures.mini_schema import MINI_TABLES

ANN = SchemaAnnotations(
    tables={"orders": Annotation(text="Đơn hàng", reviewed_by="human")},
    columns={"orders": {"amount": Annotation(text="Số tiền")}},
)
GRANTS = {"analyst": frozenset({"orders", "customers"}), "admin": frozenset({ALL_TABLES})}


def _write(tmp_path):
    write_profile(tmp_path, dsn="postgresql://u:p@h:5432/d",
                  tables=MINI_TABLES, annotations=ANN, grants=GRANTS)
    return tmp_path


def test_round_trips_into_a_usable_profile(tmp_path):
    profile = read_profile(_write(tmp_path))
    assert profile.dsn == "postgresql://u:p@h:5432/d"
    assert {t.name for t in profile.tables} == {t.name for t in MINI_TABLES}


def test_annotations_are_applied_to_the_loaded_tables(tmp_path):
    profile = read_profile(_write(tmp_path))
    orders = next(t for t in profile.tables if t.name == "orders")
    assert orders.description == "Đơn hàng"
    assert next(c for c in orders.columns if c.name == "amount").description == "Số tiền"


def test_grants_survive_the_round_trip(tmp_path):
    profile = read_profile(_write(tmp_path))
    assert permitted_tables(profile, "analyst") == frozenset({"orders", "customers"})


def test_no_sample_row_data_is_written_anywhere(tmp_path):
    """Ràng buộc spec: profile bị copy đi khi hỗ trợ kỹ thuật."""
    _write(tmp_path)
    blob = "".join(
        p.read_text(encoding="utf-8", errors="ignore")
        for p in tmp_path.rglob("*") if p.is_file()
    )
    assert "sample_rows" not in blob


def test_the_schema_yaml_is_editable_by_hand(tmp_path):
    assert (_write(tmp_path) / "schema.yaml").exists()


def test_a_fresh_profile_is_not_stale(tmp_path):
    assert profile_is_stale(_write(tmp_path), MINI_TABLES) is False


def test_adding_a_column_makes_the_profile_stale(tmp_path):
    _write(tmp_path)
    grown = list(MINI_TABLES)
    grown[0] = dataclasses.replace(
        grown[0], columns=grown[0].columns + (Column("moi", "integer"),)
    )
    assert profile_is_stale(tmp_path, tuple(grown)) is True


def test_editing_a_description_does_not_make_it_stale(tmp_path):
    """Fingerprint theo dõi CẤU TRÚC. Chú giải đổi là chuyện bình thường."""
    _write(tmp_path)
    edited = tuple(dataclasses.replace(t, description="mô tả khác") for t in MINI_TABLES)
    assert profile_is_stale(tmp_path, edited) is False


def test_reading_a_directory_without_a_profile_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_profile(tmp_path / "khong-co")
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `.venv/bin/python -m pytest tests/unit/test_profile_store.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'perception.profile_store'`

- [ ] **Step 3: Hiện thực**

Tạo `perception/profile_store.py`:

```python
"""Đọc/ghi thư mục `profile/` — trạng thái onboarding của một khách hàng.

Bố cục:
    profile/
      profile.json    dsn, grants, fingerprint, schema_mode
      structure.json  cấu trúc bảng/cột, KHÔNG có dữ liệu thật
      schema.yaml     chú giải — file người sửa được

Tách `structure.json` khỏi `schema.yaml` là có chủ ý: cấu trúc do máy sinh
và bị ghi đè mỗi lần làm mới, chú giải do người sửa và phải sống sót. Trộn
hai thứ vào một file sẽ khiến một lần làm mới xoá công sức của khách.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from perception.annotations import (
    SchemaAnnotations,
    apply_annotations,
    load_annotations,
    save_annotations,
)
from perception.connection_profile import (
    DEFAULT_THRESHOLD_TOKENS,
    ConnectionProfile,
    build_profile,
    schema_fingerprint,
)
from perception.schema_model import Column, Table

PROFILE_JSON = "profile.json"
STRUCTURE_JSON = "structure.json"
SCHEMA_YAML = "schema.yaml"


def structure_to_plain(tables: Sequence[Table]) -> list[dict]:
    return [
        {
            "name": t.name,
            "primary_key": list(t.primary_key),
            "foreign_keys": dict(t.foreign_keys),
            "row_count": t.row_count,
            "columns": [
                {"name": c.name, "data_type": c.data_type, "is_generated": c.is_generated}
                for c in t.columns
            ],
        }
        for t in tables
    ]


def structure_from_plain(raw: Sequence[Mapping]) -> tuple[Table, ...]:
    return tuple(
        Table(
            name=t["name"],
            columns=tuple(
                Column(
                    name=c["name"],
                    data_type=c["data_type"],
                    is_generated=bool(c.get("is_generated", False)),
                )
                for c in t.get("columns", [])
            ),
            primary_key=tuple(t.get("primary_key", ())),
            foreign_keys=dict(t.get("foreign_keys", {})),
            row_count=t.get("row_count"),
        )
        for t in raw
    )


def write_profile(
    directory: Path | str,
    *,
    dsn: str,
    tables: Sequence[Table],
    annotations: SchemaAnnotations,
    grants: Mapping[str, frozenset[str]],
    threshold_tokens: int = DEFAULT_THRESHOLD_TOKENS,
) -> None:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)

    (d / STRUCTURE_JSON).write_text(
        json.dumps(structure_to_plain(tables), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_annotations(annotations, d / SCHEMA_YAML)

    annotated = apply_annotations(tables, annotations)
    profile = build_profile(
        dsn=dsn, tables=annotated, grants=grants, threshold_tokens=threshold_tokens
    )
    (d / PROFILE_JSON).write_text(
        json.dumps(
            {
                "dsn": dsn,
                "grants": {u: sorted(t) for u, t in grants.items()},
                "fingerprint": profile.fingerprint,
                "schema_mode": profile.schema_mode,
                "threshold_tokens": threshold_tokens,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def read_profile(directory: Path | str) -> ConnectionProfile:
    """Dựng lại `ConnectionProfile` từ đĩa, chú giải đã gắn sẵn vào bảng."""
    d = Path(directory)
    meta_path = d / PROFILE_JSON
    if not meta_path.exists():
        raise FileNotFoundError(f"Không có {PROFILE_JSON} trong {d}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    tables = structure_from_plain(json.loads((d / STRUCTURE_JSON).read_text(encoding="utf-8")))
    annotated = apply_annotations(tables, load_annotations(d / SCHEMA_YAML))

    return build_profile(
        dsn=meta["dsn"],
        tables=annotated,
        grants={u: frozenset(t) for u, t in meta.get("grants", {}).items()},
        threshold_tokens=meta.get("threshold_tokens", DEFAULT_THRESHOLD_TOKENS),
    )


def profile_is_stale(directory: Path | str, tables: Sequence[Table]) -> bool:
    """Cấu trúc thật đã khác với lúc dựng profile chưa?

    So bằng fingerprint, vốn cố ý bỏ qua `row_count` và mô tả — chỉ thay đổi
    cấu trúc mới đáng khiến profile bị coi là cũ.
    """
    meta = json.loads((Path(directory) / PROFILE_JSON).read_text(encoding="utf-8"))
    return schema_fingerprint(tables) != meta["fingerprint"]
```

- [ ] **Step 4: Chạy test để xác nhận pass**

Run: `.venv/bin/python -m pytest tests/unit/test_profile_store.py -q`
Expected: PASS, 9 passed

- [ ] **Step 5: Commit**

```bash
git add perception/profile_store.py tests/unit/test_profile_store.py
git commit -m "feat(perception): đọc/ghi thư mục profile, tách cấu trúc khỏi chú giải"
```

---

## Task 6: Retriever lọc quyền TRƯỚC khi tìm

Sửa lỗi thiết kế ghi ở spec mục 4.1. Làm ở đây vì onboarding sinh ra profile nhiều người dùng, tức là lúc lỗi này bắt đầu biểu hiện.

**Files:**
- Modify: `perception/retrieval.py`
- Modify: `perception/schema_context.py`
- Test: `tests/unit/test_retrieval.py`, `tests/unit/test_schema_context.py` (bổ sung)

**Interfaces:**
- Produces: `Retriever.search(question: str, k: int, candidates: frozenset[str] | None = None) -> list[str]`

- [ ] **Step 1: Viết test thất bại**

Thêm vào `tests/unit/test_retrieval.py`:

```python
def test_lexical_search_only_returns_candidates_when_given():
    r = LexicalRetriever(MINI_TABLES)
    hits = r.search("bảng lương nhân viên", k=4, candidates=frozenset({"orders"}))
    assert set(hits) <= {"orders"}


def test_lexical_search_without_candidates_searches_everything():
    r = LexicalRetriever(MINI_TABLES)
    assert "payroll" in r.search("bảng lương nhân viên", k=4)


def test_candidates_are_applied_before_top_k_not_after():
    """Bảng hợp lệ không được bị đẩy khỏi top-K bởi bảng người dùng không thấy."""
    r = LexicalRetriever(MINI_TABLES)
    hits = r.search("id", k=1, candidates=frozenset({"payroll"}))
    assert hits == ["payroll"]


def test_full_retriever_honours_candidates():
    r = FullRetriever(MINI_TABLES)
    assert set(r.search("gì đó", k=9, candidates=frozenset({"orders"}))) == {"orders"}
```

Thêm vào `tests/unit/test_schema_context.py`:

```python
def test_narrow_grants_still_produce_a_non_empty_context():
    """Lỗi spec 4.1: trước bản sửa, user quyền hẹp nhận context rỗng."""
    p = _profile(threshold=1)
    ctx = resolve_schema_context(
        p, "bảng lương nhân viên", permitted=frozenset({"payroll"}),
        retriever=LexicalRetriever(MINI_TABLES), k=1,
    )
    assert ctx.retrieved_tables == ("payroll",)
    assert "payroll" in ctx.rendered_text
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `.venv/bin/python -m pytest tests/unit/test_retrieval.py tests/unit/test_schema_context.py -q`
Expected: FAIL — `search() got an unexpected keyword argument 'candidates'`

- [ ] **Step 3: Sửa `perception/retrieval.py`**

Đổi Protocol:

```python
class Retriever(Protocol):
    def search(
        self, question: str, k: int, candidates: frozenset[str] | None = None
    ) -> list[str]:
        """Trả tối đa k tên bảng, xếp theo độ liên quan giảm dần.

        `candidates` giới hạn nguồn ứng viên TRƯỚC khi xếp hạng. Lọc sau khi
        xếp hạng sẽ để bảng người dùng không được xem đẩy bảng hợp lệ của họ
        ra khỏi top-K — xem spec mục 4.1.
        """
        ...
```

`FullRetriever.search`:

```python
    def search(self, question: str, k: int, candidates=None) -> list[str]:  # noqa: ARG002
        if candidates is None:
            return list(self._names)
        return [n for n in self._names if n in candidates]
```

`LexicalRetriever.search` — thêm bộ lọc ngay đầu vòng lặp chấm điểm:

```python
        for rank, (name, n_tok, d_tok, c_tok) in enumerate(self._index):
            if candidates is not None and name not in candidates:
                continue
            score = (...)
```

- [ ] **Step 4: Sửa `perception/schema_context.py`**

Trong nhánh `retrieval`, truyền `permitted` xuống:

```python
        hits = retriever.search(question, k=k, candidates=permitted)
```

Giữ nguyên `chosen &= permitted` ở cuối. Phép giao vẫn là thao tác tập hợp
cuối cùng — mở rộng theo FK có thể kéo vào bảng ngoài quyền, và cận trên đó
là thứ không được bỏ.

- [ ] **Step 5: Chạy test và commit**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: mọi test pass.

```bash
git add perception/retrieval.py perception/schema_context.py tests/unit/test_retrieval.py tests/unit/test_schema_context.py
git commit -m "fix(perception): lọc quyền trước khi tìm, không phải sau (spec 4.1)"
```

---

## Task 7: CLI `onboard extract` và `onboard annotate`

**Files:**
- Create: `onboard.py`
- Test: `tests/unit/test_onboard_cli.py`

**Interfaces:**
- Produces:
  - `onboard.cmd_extract(dsn, out_dir) -> tuple[Table, ...]`
  - `onboard.cmd_annotate(out_dir, dsn, invoke) -> SchemaAnnotations`
  - CLI: `python onboard.py extract --dsn ... --profile ./profile`
  - CLI: `python onboard.py annotate --profile ./profile --dsn ...`

- [ ] **Step 1: Viết test thất bại**

Tạo `tests/unit/test_onboard_cli.py`:

```python
import json

from onboard import cmd_annotate, cmd_extract
from perception.annotations import load_annotations
from perception.profile_store import STRUCTURE_JSON

from tests.fixtures.mini_schema import MINI_TABLES


def test_extract_writes_structure_without_annotations(tmp_path, monkeypatch):
    monkeypatch.setattr("onboard.introspect_schema", lambda dsn, **kw: MINI_TABLES)  # noqa: ARG005
    cmd_extract("postgresql://x", tmp_path)
    raw = json.loads((tmp_path / STRUCTURE_JSON).read_text(encoding="utf-8"))
    assert {t["name"] for t in raw} == {t.name for t in MINI_TABLES}
    assert not (tmp_path / "schema.yaml").exists()


def test_annotate_writes_schema_yaml(tmp_path, monkeypatch):
    monkeypatch.setattr("onboard.introspect_schema", lambda dsn, **kw: MINI_TABLES)  # noqa: ARG005
    monkeypatch.setattr("onboard.sample_rows", lambda dsn, table, n=5: [])  # noqa: ARG005
    cmd_extract("postgresql://x", tmp_path)

    reply = '{"table": {"text": "mô tả", "confidence": "high"}, "columns": {}}'
    cmd_annotate(tmp_path, "postgresql://x", invoke=lambda s, u: reply)  # noqa: ARG005

    ann = load_annotations(tmp_path / "schema.yaml")
    assert ann.tables["orders"].text == "mô tả"


def test_annotate_preserves_human_edits_on_a_second_run(tmp_path, monkeypatch):
    monkeypatch.setattr("onboard.introspect_schema", lambda dsn, **kw: MINI_TABLES)  # noqa: ARG005
    monkeypatch.setattr("onboard.sample_rows", lambda dsn, table, n=5: [])  # noqa: ARG005
    cmd_extract("postgresql://x", tmp_path)

    from perception.annotations import Annotation, SchemaAnnotations, save_annotations
    save_annotations(
        SchemaAnnotations(tables={"orders": Annotation("NGƯỜI VIẾT", reviewed_by="human")}),
        tmp_path / "schema.yaml",
    )

    reply = '{"table": {"text": "LLM ghi đè", "confidence": "high"}, "columns": {}}'
    cmd_annotate(tmp_path, "postgresql://x", invoke=lambda s, u: reply)  # noqa: ARG005

    assert load_annotations(tmp_path / "schema.yaml").tables["orders"].text == "NGƯỜI VIẾT"
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `.venv/bin/python -m pytest tests/unit/test_onboard_cli.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'onboard'`

- [ ] **Step 3: Hiện thực**

Tạo `onboard.py`:

```python
"""Đường onboarding một khách hàng: extract → annotate → build → verify.

Chạy TẠI CHỖ KHÁCH, offline. Bước `annotate` gọi model local; không có bước
nào gọi API ngoài.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

from perception.annotate import annotate_schema
from perception.annotations import (
    SchemaAnnotations,
    load_annotations,
    merge_annotations,
    save_annotations,
)
from perception.introspect import introspect_schema, sample_rows
from perception.profile_store import SCHEMA_YAML, STRUCTURE_JSON, structure_to_plain
from perception.schema_model import Table


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
    from perception.profile_store import structure_from_plain

    return structure_from_plain(
        json.loads((profile_dir / STRUCTURE_JSON).read_text(encoding="utf-8"))
    )


def _local_invoke(system: str, user: str) -> str:
    from model.model_client import ModelClient

    return ModelClient(agent_type="insight").invoke(system_prompt=system, user_prompt=user)


def cmd_annotate(
    profile_dir: Path | str,
    dsn: str,
    invoke: Callable[[str, str], str] = _local_invoke,
) -> SchemaAnnotations:
    """Sinh chú giải và ghép vào bản đang có, giữ nguyên mục người sửa."""
    d = Path(profile_dir)
    tables = _load_structure(d)
    samples = {t.name: sample_rows(dsn, t.name, n=3) for t in tables}

    fresh = annotate_schema(tables, samples, invoke)
    merged = merge_annotations(load_annotations(d / SCHEMA_YAML), fresh)
    save_annotations(merged, d / SCHEMA_YAML)

    from perception.annotations import pending_review

    pending = pending_review(merged)
    print(f"Chú giải xong {len(tables)} bảng. {len(pending)} mục cần người duyệt.")
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description="Đường onboarding một khách hàng")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("extract", help="đọc cấu trúc database")
    p.add_argument("--dsn", required=True)
    p.add_argument("--profile", type=Path, default=Path("profile"))

    p = sub.add_parser("annotate", help="sinh chú giải bằng model local")
    p.add_argument("--dsn", required=True)
    p.add_argument("--profile", type=Path, default=Path("profile"))

    args = ap.parse_args()
    if args.cmd == "extract":
        cmd_extract(args.dsn, args.profile)
    elif args.cmd == "annotate":
        cmd_annotate(args.profile, args.dsn)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Chạy test và commit**

Run: `.venv/bin/python -m pytest tests/unit/test_onboard_cli.py -q`
Expected: PASS, 3 passed

```bash
git add onboard.py tests/unit/test_onboard_cli.py
git commit -m "feat(onboard): lệnh extract và annotate"
```

---

## Task 8: Trang duyệt chú giải

Đây là mặt tiền sản phẩm với người duyệt. Người dùng là **analyst nghiệp vụ**, không phải kỹ sư — họ không quen YAML và không dùng git.

**Files:**
- Create: `pages/1_Chu_giai_schema.py`
- Create: `perception/review_state.py`
- Test: `tests/unit/test_review_state.py`

Tách logic ra `review_state.py` để test được mà không cần chạy Streamlit; trang chỉ còn phần vẽ.

**Interfaces:**
- Produces:
  - `review_rows(tables, ann) -> list[ReviewRow]`
  - `ReviewRow(table, column, current_text, reviewed_by, confidence, sample_hint)`
  - `apply_edit(ann, table, column, new_text) -> SchemaAnnotations` — đặt `reviewed_by="human"`
  - `review_progress(ann) -> tuple[int, int]` — (đã duyệt, tổng)

- [ ] **Step 1: Viết test thất bại**

Tạo `tests/unit/test_review_state.py`:

```python
from perception.annotations import Annotation, SchemaAnnotations
from perception.review_state import apply_edit, review_progress, review_rows
from perception.schema_model import Column, Table

TABLES = (
    Table(name="orders", columns=(Column("flg_tt", "boolean"), Column("id", "integer"))),
)
ANN = SchemaAnnotations(
    tables={"orders": Annotation("Đơn hàng", confidence="low")},
    columns={"orders": {"flg_tt": Annotation("Cờ gì đó", confidence="low")}},
)


def test_rows_cover_the_table_and_each_column():
    rows = review_rows(TABLES, ANN)
    keys = {(r.table, r.column) for r in rows}
    assert ("orders", None) in keys
    assert ("orders", "flg_tt") in keys
    assert ("orders", "id") in keys, "cột chưa có chú giải vẫn phải hiện ra để sửa"


def test_rows_carry_the_current_text_and_state():
    row = next(r for r in review_rows(TABLES, ANN) if r.column == "flg_tt")
    assert row.current_text == "Cờ gì đó"
    assert row.confidence == "low"
    assert row.reviewed_by == "llm"


def test_applying_an_edit_marks_it_reviewed_by_human():
    after = apply_edit(ANN, "orders", "flg_tt", "Cờ đã thanh toán")
    entry = after.columns["orders"]["flg_tt"]
    assert entry.text == "Cờ đã thanh toán"
    assert entry.reviewed_by == "human"


def test_applying_an_edit_to_a_table_description_works_too():
    after = apply_edit(ANN, "orders", None, "Đơn bán hàng")
    assert after.tables["orders"].reviewed_by == "human"


def test_editing_a_column_that_had_no_annotation_creates_one():
    after = apply_edit(ANN, "orders", "id", "Khoá chính")
    assert after.columns["orders"]["id"].text == "Khoá chính"


def test_an_edit_never_touches_other_entries():
    after = apply_edit(ANN, "orders", "flg_tt", "mới")
    assert after.tables["orders"].text == "Đơn hàng"


def test_progress_counts_human_reviewed_over_total():
    done, total = review_progress(ANN)
    assert (done, total) == (0, 2)
    after = apply_edit(ANN, "orders", None, "x")
    assert review_progress(after)[0] == 1
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `.venv/bin/python -m pytest tests/unit/test_review_state.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'perception.review_state'`

- [ ] **Step 3: Hiện thực `perception/review_state.py`**

```python
"""Logic của trang duyệt chú giải, tách khỏi Streamlit để test được.

Trang chỉ vẽ; mọi quyết định về trạng thái nằm ở đây.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from perception.annotations import HUMAN, Annotation, SchemaAnnotations
from perception.schema_model import Table


@dataclass(frozen=True)
class ReviewRow:
    table: str
    column: str | None
    current_text: str
    reviewed_by: str
    confidence: str
    type_hint: str


def review_rows(tables: Sequence[Table], ann: SchemaAnnotations) -> list[ReviewRow]:
    """Mọi thứ có thể chú giải, kể cả mục chưa có chú giải nào.

    Cột chưa được LLM mô tả vẫn phải hiện ra: chính những cột đó thường là
    thứ model bỏ qua vì không đoán nổi, tức là thứ cần người nhất.
    """
    rows: list[ReviewRow] = []
    for table in tables:
        t = ann.tables.get(table.name)
        rows.append(
            ReviewRow(
                table=table.name,
                column=None,
                current_text=t.text if t else "",
                reviewed_by=t.reviewed_by if t else "",
                confidence=t.confidence if t else "low",
                type_hint=f"{len(table.columns)} cột",
            )
        )
        cols = ann.columns.get(table.name, {})
        for c in table.columns:
            a = cols.get(c.name)
            rows.append(
                ReviewRow(
                    table=table.name,
                    column=c.name,
                    current_text=a.text if a else "",
                    reviewed_by=a.reviewed_by if a else "",
                    confidence=a.confidence if a else "low",
                    type_hint=c.data_type,
                )
            )
    return rows


def apply_edit(
    ann: SchemaAnnotations, table: str, column: str | None, new_text: str
) -> SchemaAnnotations:
    """Ghi một sửa đổi của người và đánh dấu `reviewed_by="human"`.

    Đánh dấu là phần quan trọng: nó khiến mục này sống sót qua mọi lần làm
    mới về sau.
    """
    entry = Annotation(text=new_text, reviewed_by=HUMAN, confidence="high")
    if column is None:
        return SchemaAnnotations(
            tables={**ann.tables, table: entry},
            columns=ann.columns,
        )
    cols = {**ann.columns.get(table, {}), column: entry}
    return SchemaAnnotations(
        tables=ann.tables,
        columns={**ann.columns, table: cols},
    )


def review_progress(ann: SchemaAnnotations) -> tuple[int, int]:
    """(số mục người đã duyệt, tổng số mục có chú giải)."""
    entries = list(ann.tables.values()) + [
        a for cols in ann.columns.values() for a in cols.values()
    ]
    return sum(1 for a in entries if a.reviewed_by == HUMAN), len(entries)
```

- [ ] **Step 4: Hiện thực trang Streamlit**

Tạo `pages/1_Chu_giai_schema.py`:

```python
"""Trang duyệt chú giải schema — dành cho analyst nghiệp vụ, không phải kỹ sư."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from perception.annotations import load_annotations, save_annotations
from perception.profile_store import SCHEMA_YAML, STRUCTURE_JSON
from perception.review_state import apply_edit, review_progress, review_rows

st.set_page_config(page_title="Chú giải schema", page_icon="📝", layout="wide")
st.title("📝 Chú giải schema")
st.caption(
    "Mô tả bảng và cột bằng ngôn ngữ nghiệp vụ. Chất lượng ở đây quyết định "
    "hệ thống trả lời đúng tới đâu — quan trọng hơn mọi thiết lập kỹ thuật."
)

profile_dir = Path(os.environ.get("ADBA_PROFILE_DIR", "profile"))

if not (profile_dir / STRUCTURE_JSON).exists():
    st.warning(
        f"Chưa có `{STRUCTURE_JSON}` trong `{profile_dir}`. "
        "Chạy `python onboard.py extract --dsn ...` trước."
    )
    st.stop()

from perception.profile_store import structure_from_plain  # noqa: E402
import json  # noqa: E402

tables = structure_from_plain(
    json.loads((profile_dir / STRUCTURE_JSON).read_text(encoding="utf-8"))
)
ann = load_annotations(profile_dir / SCHEMA_YAML)

done, total = review_progress(ann)
st.progress(done / total if total else 0.0, text=f"Đã duyệt {done}/{total} mục")

only_pending = st.checkbox("Chỉ hiện mục cần duyệt", value=True)
rows = review_rows(tables, ann)
if only_pending:
    rows = [r for r in rows if r.reviewed_by != "human" and r.confidence == "low"]

if not rows:
    st.success("Không còn mục nào cần duyệt.")
    st.stop()

by_table: dict[str, list] = {}
for r in rows:
    by_table.setdefault(r.table, []).append(r)

for table_name, table_rows in by_table.items():
    with st.expander(f"**{table_name}**  ({len(table_rows)} mục)", expanded=True):
        for r in table_rows:
            label = f"Bảng `{r.table}`" if r.column is None else f"Cột `{r.column}` — {r.type_hint}"
            key = f"{r.table}::{r.column or ''}"
            col_input, col_btn = st.columns([5, 1])
            with col_input:
                new = st.text_input(label, value=r.current_text, key=key)
            with col_btn:
                st.write("")
                if st.button("Lưu", key=f"btn::{key}"):
                    ann = apply_edit(ann, r.table, r.column, new)
                    save_annotations(ann, profile_dir / SCHEMA_YAML)
                    st.rerun()
            if r.confidence == "low" and r.reviewed_by != "human":
                st.caption("⚠️ Model không chắc về mục này")
```

- [ ] **Step 5: Chạy test, kiểm trang bằng mắt, commit**

Run: `.venv/bin/python -m pytest tests/unit/test_review_state.py -q`
Expected: PASS, 7 passed

Kiểm bằng mắt: `.venv/bin/streamlit run app.py`, mở trang "Chú giải schema" ở thanh bên, sửa một mục, xác nhận `schema.yaml` đổi và `reviewed_by` thành `human`.

```bash
git add pages/ perception/review_state.py tests/unit/test_review_state.py
git commit -m "feat(onboard): trang duyệt chú giải cho analyst nghiệp vụ"
```

---

## Task 9: `onboard build` — dựng profile hoàn chỉnh

**Files:**
- Modify: `onboard.py`
- Test: `tests/unit/test_onboard_cli.py` (bổ sung)

**Interfaces:**
- Produces: `onboard.cmd_build(profile_dir, dsn, grants) -> ConnectionProfile`

- [ ] **Step 1: Viết test thất bại**

Thêm vào `tests/unit/test_onboard_cli.py`:

```python
def test_build_writes_a_loadable_profile(tmp_path, monkeypatch):
    monkeypatch.setattr("onboard.introspect_schema", lambda dsn, **kw: MINI_TABLES)  # noqa: ARG005
    cmd_extract("postgresql://x", tmp_path)

    from onboard import cmd_build
    from perception.connection_profile import ALL_TABLES, permitted_tables
    from perception.profile_store import read_profile

    cmd_build(tmp_path, "postgresql://x", grants={"admin": frozenset({ALL_TABLES})})
    profile = read_profile(tmp_path)
    assert permitted_tables(profile, "admin") == {t.name for t in MINI_TABLES}


def test_build_refuses_when_annotations_are_mostly_unreviewed(tmp_path, monkeypatch):
    """Chặn bàn giao sớm: profile không chú giải sẽ cho recall thấp và không ai biết vì sao."""
    monkeypatch.setattr("onboard.introspect_schema", lambda dsn, **kw: MINI_TABLES)  # noqa: ARG005
    cmd_extract("postgresql://x", tmp_path)

    from onboard import UnreviewedAnnotationsError, cmd_build

    with pytest.raises(UnreviewedAnnotationsError):
        cmd_build(tmp_path, "postgresql://x", grants={}, min_reviewed=0.5)
```

Thêm `import pytest` vào đầu file test nếu chưa có.

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `.venv/bin/python -m pytest tests/unit/test_onboard_cli.py -q`
Expected: FAIL — `ImportError: cannot import name 'cmd_build'`

- [ ] **Step 3: Hiện thực**

Thêm vào `onboard.py`:

```python
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
    """
    from perception.annotations import load_annotations
    from perception.review_state import review_progress

    d = Path(profile_dir)
    tables = _load_structure(d)
    ann = load_annotations(d / SCHEMA_YAML)

    done, total = review_progress(ann)
    ratio = (done / total) if total else 0.0
    if ratio < min_reviewed:
        raise UnreviewedAnnotationsError(
            f"Mới {done}/{total} mục ({ratio:.0%}) được người duyệt, "
            f"ngưỡng là {min_reviewed:.0%}. Mở trang 'Chú giải schema' để duyệt tiếp."
        )

    write_profile(d, dsn=dsn, tables=tables, annotations=ann, grants=grants)
    profile = read_profile(d)
    print(f"profile → {d}  ({len(profile.tables)} bảng, chế độ {profile.schema_mode})")
    return profile
```

Thêm import ở đầu file: `from collections.abc import Mapping`, `from perception.connection_profile import ConnectionProfile`, `from perception.profile_store import read_profile, write_profile`.

Thêm lệnh con vào `main()`:

```python
    p = sub.add_parser("build", help="dựng profile từ cấu trúc + chú giải")
    p.add_argument("--dsn", required=True)
    p.add_argument("--profile", type=Path, default=Path("profile"))
    p.add_argument("--grant", action="append", default=[],
                   help="user=bang1,bang2 hoặc user=* (lặp lại được)")
    p.add_argument("--min-reviewed", type=float, default=0.0)
```

Và xử lý:

```python
    elif args.cmd == "build":
        grants = {}
        for spec in args.grant:
            user, _, tables_csv = spec.partition("=")
            grants[user] = frozenset(t.strip() for t in tables_csv.split(",") if t.strip())
        cmd_build(args.profile, args.dsn, grants, min_reviewed=args.min_reviewed)
```

- [ ] **Step 4: Chạy test và commit**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: mọi test pass.

```bash
git add onboard.py tests/unit/test_onboard_cli.py
git commit -m "feat(onboard): lệnh build, có cổng chặn bàn giao khi chú giải chưa duyệt"
```

---

## Task 10: `onboard verify` — chấm điểm và ra `report.md`

**Files:**
- Modify: `onboard.py`
- Test: `tests/unit/test_onboard_verify.py`

**Interfaces:**
- Produces:
  - `onboard.cmd_verify(profile_dir, golden_path, user) -> VerifyReport`
  - `VerifyReport(total, recall, avg_context_tables, passed, misses)`
  - Ngưỡng chặn bàn giao: recall ≥ 0.95 (spec 6.5)

- [ ] **Step 1: Viết test thất bại**

Tạo `tests/unit/test_onboard_verify.py`:

```python
import json

import pytest

from onboard import cmd_verify
from perception.connection_profile import ALL_TABLES
from perception.profile_store import write_profile
from perception.annotations import SchemaAnnotations
from tests.fixtures.mini_schema import MINI_TABLES

GOLDEN = [
    {"question": "doanh thu theo khách", "sql": "SELECT * FROM orders JOIN customers ON 1=1"},
    {"question": "lương nhân viên", "sql": "SELECT * FROM payroll"},
]


def _setup(tmp_path):
    write_profile(tmp_path, dsn="postgresql://x", tables=MINI_TABLES,
                  annotations=SchemaAnnotations(),
                  grants={"admin": frozenset({ALL_TABLES})})
    golden = tmp_path / "golden.jsonl"
    golden.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in GOLDEN),
                      encoding="utf-8")
    return golden


def test_reports_recall_over_the_golden_set(tmp_path):
    report = cmd_verify(tmp_path, _setup(tmp_path), user="admin")
    assert report.total == 2
    assert 0.0 <= report.recall <= 1.0


def test_full_mode_reaches_perfect_recall(tmp_path):
    """4 bảng nằm dưới ngưỡng nên profile ở chế độ full — mọi bảng đều có mặt."""
    report = cmd_verify(tmp_path, _setup(tmp_path), user="admin")
    assert report.recall == 1.0
    assert report.passed is True


def test_a_user_without_grants_fails_every_question(tmp_path):
    golden = _setup(tmp_path)
    report = cmd_verify(tmp_path, golden, user="nguoi_la")
    assert report.recall == 0.0
    assert report.passed is False


def test_misses_name_the_tables_that_were_absent(tmp_path):
    golden = _setup(tmp_path)
    report = cmd_verify(tmp_path, golden, user="nguoi_la")
    assert any("orders" in missing for _, missing in report.misses)


def test_writes_a_markdown_report(tmp_path):
    cmd_verify(tmp_path, _setup(tmp_path), user="admin")
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "recall" in text.lower()


def test_the_report_states_the_handover_threshold(tmp_path):
    cmd_verify(tmp_path, _setup(tmp_path), user="admin")
    assert "95" in (tmp_path / "report.md").read_text(encoding="utf-8")
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `.venv/bin/python -m pytest tests/unit/test_onboard_verify.py -q`
Expected: FAIL — `ImportError: cannot import name 'cmd_verify'`

- [ ] **Step 3: Hiện thực**

Thêm vào `onboard.py`:

```python
HANDOVER_RECALL = 0.95


@dataclass
class VerifyReport:
    total: int
    recall: float
    avg_context_tables: float
    passed: bool
    misses: list[tuple[str, frozenset[str]]]


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
    """
    from eval.tier1_recall import measure_recall
    from eval.datasets import EvalRecord
    from perception.connection_profile import permitted_tables
    from perception.retrieval import LexicalRetriever
    from perception.schema_context import resolve_schema_context

    d = Path(profile_dir)
    profile = read_profile(d)
    permitted = permitted_tables(profile, user)
    retriever = LexicalRetriever(profile.tables)

    records = []
    for line in Path(golden_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        records.append(
            EvalRecord(question=row["question"], gold_sql=row["sql"],
                       db_id="khach", tables=profile.tables)
        )

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

    lines = [
        "# Báo cáo kiểm profile",
        "",
        f"- Câu chấm được: **{report.total}**",
        f"- Recall: **{report.recall:.3f}**  (ngưỡng bàn giao: {HANDOVER_RECALL:.0%} = 95%)",
        f"- Bảng/context trung bình: {report.avg_context_tables:.1f}",
        f"- Kết luận: **{'ĐẠT' if report.passed else 'CHƯA ĐẠT'}**",
        "",
    ]
    if report.misses:
        lines += ["## Câu chưa đạt", ""]
        lines += [f"- {q} — thiếu `{sorted(m)}`" for q, m in report.misses[:20]]
        lines += [
            "",
            "Recall thấp hầu như luôn là do chú giải, không phải do model.",
            "Mở trang 'Chú giải schema', bổ sung mô tả cho các bảng bị thiếu, rồi chạy lại.",
        ]
    (d / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"recall {report.recall:.3f} — {'ĐẠT' if report.passed else 'CHƯA ĐẠT'}")
    return report
```

Thêm `from dataclasses import dataclass` vào đầu file.

Thêm lệnh con vào `main()`:

```python
    p = sub.add_parser("verify", help="chấm recall trên golden set của khách")
    p.add_argument("--profile", type=Path, default=Path("profile"))
    p.add_argument("--golden", type=Path, required=True)
    p.add_argument("--user", required=True)
    p.add_argument("--k", type=int, default=8)
```

```python
    elif args.cmd == "verify":
        report = cmd_verify(args.profile, args.golden, args.user, k=args.k)
        raise SystemExit(0 if report.passed else 1)
```

- [ ] **Step 4: Chạy test và commit**

Run: `.venv/bin/python -m pytest tests/unit/test_onboard_verify.py -q`
Expected: PASS, 6 passed

```bash
git add onboard.py tests/unit/test_onboard_verify.py
git commit -m "feat(onboard): lệnh verify, ngưỡng bàn giao recall 95%"
```

---

## Task 11: `onboard refresh` — làm mới khi schema đổi

**Files:**
- Modify: `onboard.py`
- Test: `tests/unit/test_onboard_refresh.py`

**Interfaces:**
- Produces: `onboard.cmd_refresh(profile_dir, dsn, invoke) -> tuple[int, int]` — (số bảng mới chú giải, số mục human giữ nguyên)

- [ ] **Step 1: Viết test thất bại**

Tạo `tests/unit/test_onboard_refresh.py`:

```python
import dataclasses
import json

from onboard import cmd_extract, cmd_refresh
from perception.annotations import Annotation, SchemaAnnotations, load_annotations, save_annotations
from perception.schema_model import Column
from tests.fixtures.mini_schema import MINI_TABLES

REPLY = '{"table": {"text": "llm mới", "confidence": "high"}, "columns": {}}'


def test_refresh_keeps_every_human_entry(tmp_path, monkeypatch):
    monkeypatch.setattr("onboard.introspect_schema", lambda dsn, **kw: MINI_TABLES)  # noqa: ARG005
    monkeypatch.setattr("onboard.sample_rows", lambda dsn, table, n=5: [])  # noqa: ARG005
    cmd_extract("postgresql://x", tmp_path)
    save_annotations(
        SchemaAnnotations(
            tables={t.name: Annotation(f"người {t.name}", reviewed_by="human")
                    for t in MINI_TABLES}
        ),
        tmp_path / "schema.yaml",
    )

    cmd_refresh(tmp_path, "postgresql://x", invoke=lambda s, u: REPLY)  # noqa: ARG005

    ann = load_annotations(tmp_path / "schema.yaml")
    assert all(ann.tables[t.name].text == f"người {t.name}" for t in MINI_TABLES)


def test_refresh_picks_up_a_new_table(tmp_path, monkeypatch):
    grown = MINI_TABLES + (
        dataclasses.replace(MINI_TABLES[0], name="invoices"),
    )
    monkeypatch.setattr("onboard.introspect_schema", lambda dsn, **kw: grown)  # noqa: ARG005
    monkeypatch.setattr("onboard.sample_rows", lambda dsn, table, n=5: [])  # noqa: ARG005
    cmd_extract("postgresql://x", tmp_path)

    cmd_refresh(tmp_path, "postgresql://x", invoke=lambda s, u: REPLY)  # noqa: ARG005
    assert "invoices" in load_annotations(tmp_path / "schema.yaml").tables


def test_refresh_updates_the_structure_file(tmp_path, monkeypatch):
    monkeypatch.setattr("onboard.introspect_schema", lambda dsn, **kw: MINI_TABLES)  # noqa: ARG005
    monkeypatch.setattr("onboard.sample_rows", lambda dsn, table, n=5: [])  # noqa: ARG005
    cmd_extract("postgresql://x", tmp_path)

    grown = list(MINI_TABLES)
    grown[0] = dataclasses.replace(
        grown[0], columns=grown[0].columns + (Column("cot_moi", "integer"),)
    )
    monkeypatch.setattr("onboard.introspect_schema", lambda dsn, **kw: tuple(grown))  # noqa: ARG005
    cmd_refresh(tmp_path, "postgresql://x", invoke=lambda s, u: REPLY)  # noqa: ARG005

    raw = json.loads((tmp_path / "structure.json").read_text(encoding="utf-8"))
    cols = {c["name"] for t in raw if t["name"] == grown[0].name for c in t["columns"]}
    assert "cot_moi" in cols


def test_refresh_reports_how_many_human_entries_it_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr("onboard.introspect_schema", lambda dsn, **kw: MINI_TABLES)  # noqa: ARG005
    monkeypatch.setattr("onboard.sample_rows", lambda dsn, table, n=5: [])  # noqa: ARG005
    cmd_extract("postgresql://x", tmp_path)
    save_annotations(
        SchemaAnnotations(tables={"orders": Annotation("người", reviewed_by="human")}),
        tmp_path / "schema.yaml",
    )
    _, preserved = cmd_refresh(tmp_path, "postgresql://x", invoke=lambda s, u: REPLY)  # noqa: ARG005
    assert preserved == 1
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `.venv/bin/python -m pytest tests/unit/test_onboard_refresh.py -q`
Expected: FAIL — `ImportError: cannot import name 'cmd_refresh'`

- [ ] **Step 3: Hiện thực**

Thêm vào `onboard.py`:

```python
def cmd_refresh(
    profile_dir: Path | str,
    dsn: str,
    invoke: Callable[[str, str], str] = _local_invoke,
) -> tuple[int, int]:
    """Đọc lại cấu trúc, chú giải lại, ghép vào bản cũ.

    Trả `(số bảng đã chú giải, số mục người sửa được giữ nguyên)`. Con số thứ
    hai được in ra có chủ ý: nó là bằng chứng cho khách rằng công sức duyệt
    của họ không mất, và đó là thứ quyết định họ có duyệt lần thứ hai không.
    """
    from perception.annotations import HUMAN, load_annotations

    d = Path(profile_dir)
    before = load_annotations(d / SCHEMA_YAML)
    preserved = sum(1 for a in before.tables.values() if a.reviewed_by == HUMAN) + sum(
        1 for cols in before.columns.values()
        for a in cols.values() if a.reviewed_by == HUMAN
    )

    cmd_extract(dsn, d)
    merged = cmd_annotate(d, dsn, invoke=invoke)

    print(f"Giữ nguyên {preserved} mục do người duyệt.")
    return len(merged.tables), preserved
```

Thêm lệnh con vào `main()`:

```python
    p = sub.add_parser("refresh", help="đọc lại schema và chú giải phần mới")
    p.add_argument("--dsn", required=True)
    p.add_argument("--profile", type=Path, default=Path("profile"))
```

```python
    elif args.cmd == "refresh":
        cmd_refresh(args.profile, args.dsn)
```

- [ ] **Step 4: Chạy test và commit**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: mọi test pass.

```bash
git add onboard.py tests/unit/test_onboard_refresh.py
git commit -m "feat(onboard): lệnh refresh, báo số mục người duyệt được giữ"
```

---

## Task 12: `app.py` đọc profile từ đĩa

**Files:**
- Modify: `app.py`
- Test: `tests/integration/test_onboard_flow.py`

**Interfaces:**
- Consumes: `read_profile`, `profile_is_stale`

- [ ] **Step 1: Viết test thất bại**

Tạo `tests/integration/test_onboard_flow.py`:

```python
import json
import os

import pytest

from onboard import cmd_build, cmd_extract, cmd_verify
from perception.connection_profile import ALL_TABLES, permitted_tables
from perception.profile_store import profile_is_stale, read_profile

DSN = os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")
pytestmark = pytest.mark.skipif(not DSN, reason="cần DATABASE_URL trỏ Postgres thật")


def test_extract_build_verify_runs_end_to_end(tmp_path):
    cmd_extract(DSN, tmp_path)
    cmd_build(tmp_path, DSN, grants={"admin": frozenset({ALL_TABLES})})

    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        json.dumps({"question": "doanh thu theo region",
                    "sql": "SELECT region, SUM(amount) FROM orders GROUP BY region"},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    report = cmd_verify(tmp_path, golden, user="admin")
    assert report.total == 1


def test_the_built_profile_matches_the_live_schema(tmp_path):
    from perception.introspect import introspect_schema

    cmd_extract(DSN, tmp_path)
    cmd_build(tmp_path, DSN, grants={})
    assert profile_is_stale(tmp_path, introspect_schema(DSN)) is False


def test_a_profile_loaded_from_disk_carries_grants(tmp_path):
    cmd_extract(DSN, tmp_path)
    cmd_build(tmp_path, DSN, grants={"sales": frozenset({"orders"})})
    profile = read_profile(tmp_path)
    assert permitted_tables(profile, "sales") == frozenset({"orders"})
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `set -a && . ./.env && set +a && DATABASE_URL="$POSTGRES_URL" .venv/bin/python -m pytest tests/integration/test_onboard_flow.py -q`
Expected: FAIL — `ModuleNotFoundError` hoặc `ImportError` tuỳ task nào chưa xong.

- [ ] **Step 3: Sửa `app.py`**

Thay khối dựng profile mỗi câu hỏi bằng đọc từ đĩa, có cache:

```python
@st.cache_resource
def _load_profile():
    """Đọc profile một lần cho cả phiên.

    Trước đây khối này dựng lại profile mỗi câu hỏi — introspect, hash, dựng
    index — tức là lặp công việc onboarding trên mỗi lần gõ Enter, và quyết
    lại `schema_mode` mỗi lượt dù spec nói quyết một lần lúc cài đặt.
    """
    from perception.profile_store import read_profile

    return read_profile(Path(os.environ.get("ADBA_PROFILE_DIR", "profile")))


@st.cache_resource
def _load_retriever(_profile):
    from perception.retrieval import LexicalRetriever

    return LexicalRetriever(_profile.tables)
```

Chỗ xử lý câu hỏi:

```python
try:
    profile = _load_profile()
except FileNotFoundError:
    st.error(
        "Chưa có profile. Chạy:\n\n"
        "```\npython onboard.py extract --dsn ...\n"
        "python onboard.py annotate --dsn ...\n"
        "python onboard.py build --dsn ... --grant local=*\n```"
    )
    st.stop()

user = "local"  # TODO(plan-4): thay bằng danh tính từ lớp xác thực
ctx = resolve_schema_context(
    profile, prompt, permitted_tables(profile, user),
    retriever=_load_retriever(profile),
)
result = run_graph(query=prompt, schema_context=ctx, profile=profile, user=user)
```

- [ ] **Step 4: Chạy cả hai thư mục test**

Run: `.venv/bin/python -m pytest tests/unit -q` rồi `... tests/integration -q`
Expected: mọi test pass.

- [ ] **Step 5: Chạy thử thật một vòng trọn**

```bash
set -a && . ./.env && set +a
export DATABASE_URL="$POSTGRES_URL" ADBA_PROFILE_DIR=./profile-adba
.venv/bin/python onboard.py extract  --dsn "$DATABASE_URL" --profile ./profile-adba
.venv/bin/python onboard.py annotate --dsn "$DATABASE_URL" --profile ./profile-adba
.venv/bin/python onboard.py build    --dsn "$DATABASE_URL" --profile ./profile-adba --grant local=*
.venv/bin/streamlit run app.py
```

Mở trang "Chú giải schema", sửa vài mục, quay lại trang chính và hỏi một câu. Xác nhận SQL sinh ra dùng đúng bảng.

- [ ] **Step 6: Commit**

```bash
git add app.py tests/integration/test_onboard_flow.py
git commit -m "feat(app): đọc profile từ đĩa thay vì dựng lại mỗi câu hỏi"
```

---

## Điều kiện ra của plan này

- [ ] `pytest tests/unit -q` và `pytest tests/integration -q` đều pass, không hồi quy so với 227 test
- [ ] Chạy trọn `extract → annotate → duyệt trên UI → build → verify` trên **một schema BIRD**, ra được `report.md`
- [ ] `refresh` giữ nguyên **100%** mục `reviewed_by: human` — có test trên 20 mục
- [ ] Người dùng có quyền hẹp không còn nhận context rỗng — có test riêng
- [ ] Không có `sample_rows` trong `profile/` — kiểm bằng test
- [ ] `app.py` trả lời được một câu hỏi thật từ profile đọc trên đĩa

**Chưa thuộc plan này, đừng làm:** sinh few-shot theo khách, kiểm `profile_stale` lúc khởi động, train lại LoRA, cứng hóa L1/L3, đóng gói on-prem.

### Một chỗ plan này đi lệch spec, có chủ ý

Spec mục 7 xếp **retriever embedding** vào pha 3, cùng lúc với đường onboarding. Plan này làm phần giao thức (Task 6 — thêm `candidates`, lọc quyền trước khi tìm) nhưng **hoãn bản embedding**.

Lý do: con số recall 0,250 hiện tại là của một schema **không có chú giải nào**. Sau khi đường onboarding chạy, mô tả bảng và cột sẽ là tiếng Việt, và câu hỏi cũng tiếng Việt — `LexicalRetriever` giờ đã xử lý dấu và `đ`, nên nó có thể tự khá lên đáng kể mà không cần embedding.

Đo trước, rồi mới quyết có đáng ship thêm 470 MB artifact hay không. Đây là cùng một nguyên tắc đã áp cho việc train lại: không trả tiền cho một thứ chưa biết có thiếu hay không.

Giao thức `Retriever` vẫn được chốt ngay trong plan này để bản embedding sau cắm vào được mà không phải sửa lại interface khi đã có hai hiện thực phụ thuộc nó.

**Kích hoạt bản embedding khi:** chạy `verify` sau khi chú giải xong mà recall vẫn dưới 0,95.

**Rủi ro lớn nhất:** chất lượng chú giải do LLM sinh. Nếu nó đoán sai ý nghĩa một cột và tự đánh dấu `confidence: high`, sẽ không ai kiểm lại, và mọi truy vấn dựa vào cột đó sai **âm thầm**. Hai lớp chống: prompt yêu cầu thà nhận không chắc còn hơn đoán nghe hợp lý, và cổng recall ở `verify`. Cả hai đều không hoàn hảo — đây là chỗ cần người thật nhìn.
