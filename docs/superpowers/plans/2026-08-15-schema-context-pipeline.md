# Schema Context Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thay `info_box` JSON tĩnh bằng một đường ống schema-context động, để ADBA trả lời được câu hỏi trên một schema chưa từng thấy, và đo được chất lượng chọn bảng.

**Architecture:** Ba lớp tách bạch. (1) `perception/` giữ biểu diễn schema, hồ sơ kết nối, và việc chọn bảng — toàn bộ là hàm thuần, không gọi LLM. (2) `graph/` tiêu thụ một `SchemaContext` đã dựng sẵn thay vì tự đọc file JSON. (3) `eval/` đo recall chọn bảng bằng cách so với tập bảng parse từ SQL mẫu — không cần LLM, không cần GPU, chạy hết vài giây.

Ranh giới quan trọng nhất: **`permitted_tables(user)` là bảo mật, `retrieved_tables(question)` là nội dung prompt.** Chúng được dẫn ra độc lập và không bao giờ gộp.

**Tech Stack:** Python 3.12, pytest 9, sqlparse (mới), PyYAML, pandas, psycopg2. Không thêm torch/sentence-transformers trong plan này.

**Spec:** `docs/superpowers/specs/2026-08-15-adba-multi-schema-onprem-design.md` (commit `9bfeb37`) — pha 0 và pha 1 ở mục 7.

## Global Constraints

- **Ngưỡng `schema_mode`:** 6.000 token schema đã render. Dưới → `full`, trên → `retrieval`. Ước lượng token = `len(text) // 4`. (spec 3.3)
- **`render_schema()` ≤ 700 B/bảng** ở mức chi tiết mặc định. (spec tiêu chí 3)
- **Không tên bảng hardcode** trong `graph/`, `prompts/`, `model/`. (spec tiêu chí 1, 2)
- **`execute_sql` không được có tham số nào cho phép bên gọi nới rộng tập bảng được chạm.** Tập quyền chỉ dẫn từ profile + danh tính. (spec tiêu chí 11, mục 3.4.1)
- **`sample_rows` không được ghi vào profile runtime.** (spec tiêu chí 9, mục 3.2)
- **Không hồi quy:** `pytest tests/ -q` phải giữ nguyên 59 test pass ở mọi commit.
- Toàn bộ code mới trong plan này **không gọi LLM**. Nếu một hàm cần LLM, nó thuộc plan khác.
- Chạy test: `.venv/bin/python -m pytest <path> -q` từ thư mục gốc repo.

---

## File Structure

**Tạo mới:**

| File | Trách nhiệm |
|---|---|
| `perception/schema_model.py` | `Column`, `Table` — kiểu dữ liệu schema, bất biến. Adapter từ `info_box` JSON. |
| `perception/render_schema.py` | `render_schema()` — Table → text DDL. Hàm thuần, không phụ thuộc gì. |
| `perception/connection_profile.py` | `ConnectionProfile`, `permitted_tables()`, `schema_fingerprint()` |
| `perception/retrieval.py` | `Retriever` protocol, `FullRetriever`, `LexicalRetriever` |
| `perception/schema_context.py` | `SchemaContext`, `resolve_schema_context()` — nơi công tắc `full`/`retrieval` sống |
| `eval/gold_tables.py` | `tables_in_sql()` — parse SQL mẫu → tập bảng, bằng sqlparse |
| `eval/datasets.py` | `EvalRecord` + loader cho Spider/BIRD/BEAVER về một dạng chung |
| `eval/tier1_recall.py` | `measure_recall()` + CLI. Đây là vòng lặp đo nhanh của cả dự án. |
| `eval/README_multischema.md` | Ghi cấu hình thực tế của ba bộ dữ liệu (số DB, số bảng, license) |
| `tests/unit/test_*.py` | Một file test cho mỗi module trên |
| `tests/fixtures/` | Fixture schema + SQL mẫu nhỏ, dùng chung cho test |

**Sửa:**

| File | Sửa gì |
|---|---|
| `requirements.txt` | thêm `sqlparse>=0.5.0` |
| `graph/tools/sql_tool.py:27-31` | bỏ `_ALLOWED_TABLES` tĩnh; nhận `permitted` từ tham số |
| `graph/state.py:14,45` | `info_box: dict` → `schema_context: SchemaContext` |
| `graph/multi_agent.py:94` | `run_graph(query, info_box)` → `run_graph(query, schema_context)` |
| `graph/agents/supervisor.py:36` | `json.dumps(info_box)` → `schema_context.rendered_text` |
| `graph/agents/sql_agent.py:69` | như trên, cộng few-shots |
| `prompts/text_to_sql.txt` | tách ba đường; `{info_box}` xuống cuối |
| `prompts/supervisor_routing.txt` | như trên |
| `app.py:225-234` | đọc profile thay vì `info_box_latest.json` |

---

## Task 1: `tables_in_sql()` — parse SQL mẫu thành tập bảng

Đây là nền của toàn bộ eval tầng 1. Nếu nó sai thì mọi con số recall về sau đều vô nghĩa.

**Files:**
- Create: `eval/gold_tables.py`
- Create: `tests/unit/test_gold_tables.py`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: (không có — task đầu tiên)
- Produces: `eval.gold_tables.tables_in_sql(sql: str) -> frozenset[str]` — trả tên bảng thật, đã bỏ alias và bỏ tên CTE.

- [ ] **Step 1: Cài sqlparse và ghi vào requirements**

```bash
.venv/bin/pip install 'sqlparse>=0.5.0'
```

Sửa `requirements.txt`, thêm sau dòng `sqlalchemy>=2.0.0`:

```
sqlparse>=0.5.0
```

- [ ] **Step 2: Viết test thất bại**

Tạo `tests/unit/test_gold_tables.py`:

```python
import pytest

from eval.gold_tables import tables_in_sql


def test_single_table():
    assert tables_in_sql("SELECT id FROM orders") == frozenset({"orders"})


def test_join_with_aliases_returns_real_names_not_aliases():
    sql = """
        SELECT o.id, p.name
        FROM orders o
        JOIN products p ON o.product_id = p.id
    """
    assert tables_in_sql(sql) == frozenset({"orders", "products"})


def test_cte_name_is_not_a_table():
    sql = """
        WITH q4 AS (SELECT product_id, SUM(amount) AS s FROM orders GROUP BY product_id)
        SELECT p.name, q.s FROM q4 q JOIN products p ON q.product_id = p.id
    """
    # q4 là CTE, không phải bảng thật
    assert tables_in_sql(sql) == frozenset({"orders", "products"})


def test_subquery_tables_are_included():
    sql = "SELECT * FROM (SELECT id FROM employees) e JOIN payroll p ON e.id = p.employee_id"
    assert tables_in_sql(sql) == frozenset({"employees", "payroll"})


def test_schema_qualified_name_keeps_only_table():
    assert tables_in_sql("SELECT * FROM public.orders") == frozenset({"orders"})


def test_case_is_normalised_to_lower():
    assert tables_in_sql("SELECT * FROM Orders") == frozenset({"orders"})


def test_empty_or_unparseable_returns_empty_set():
    assert tables_in_sql("") == frozenset()
    assert tables_in_sql("not sql at all") == frozenset()
```

- [ ] **Step 3: Chạy test để xác nhận nó fail**

Run: `.venv/bin/python -m pytest tests/unit/test_gold_tables.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval.gold_tables'`

- [ ] **Step 4: Hiện thực tối thiểu**

Tạo `eval/gold_tables.py`:

```python
"""Parse tập bảng thật ra khỏi một câu SQL mẫu.

Dùng cho eval tầng 1: so tập bảng mà retriever chọn với tập bảng mà SQL
mẫu thực sự chạm. Không gọi LLM, không cần database.
"""

from __future__ import annotations

import sqlparse
from sqlparse.sql import Identifier, IdentifierList, Parenthesis, TokenList
from sqlparse.tokens import CTE, DML, Keyword

_SOURCE_KEYWORDS = {"FROM", "JOIN", "INNER JOIN", "LEFT JOIN", "RIGHT JOIN",
                    "FULL JOIN", "CROSS JOIN", "LEFT OUTER JOIN",
                    "RIGHT OUTER JOIN", "FULL OUTER JOIN"}


def _real_name(identifier: Identifier) -> str | None:
    """Tên bảng thật, bỏ alias và bỏ tiền tố schema."""
    name = identifier.get_real_name()
    if not name:
        return None
    return name.lower()


def _cte_names(statement: TokenList) -> set[str]:
    """Tên các CTE khai báo bằng WITH — chúng không phải bảng thật."""
    names: set[str] = set()
    tokens = list(statement.flatten())
    saw_with = any(t.ttype is CTE for t in tokens)
    if not saw_with:
        return names

    for token in statement.tokens:
        if token.ttype is CTE:  # WITH
            continue
        if isinstance(token, IdentifierList):
            for ident in token.get_identifiers():
                if isinstance(ident, Identifier) and (n := _real_name(ident)):
                    names.add(n)
        elif isinstance(token, Identifier):
            if n := _real_name(token):
                names.add(n)
        elif token.ttype is DML and token.value.upper() == "SELECT":
            break
    return names


def _collect(node: TokenList, out: set[str]) -> None:
    expecting_source = False
    for token in node.tokens:
        if token.is_whitespace:
            continue

        if token.ttype is Keyword and token.value.upper() in _SOURCE_KEYWORDS:
            expecting_source = True
            continue

        if expecting_source:
            if isinstance(token, IdentifierList):
                for ident in token.get_identifiers():
                    if isinstance(ident, Identifier) and (n := _real_name(ident)):
                        out.add(n)
                expecting_source = False
                continue
            if isinstance(token, Identifier):
                # `FROM (SELECT ...) alias` — đi tiếp vào trong, không lấy alias
                if any(isinstance(t, Parenthesis) for t in token.tokens):
                    _collect(token, out)
                elif n := _real_name(token):
                    out.add(n)
                expecting_source = False
                continue
            if isinstance(token, Parenthesis):
                _collect(token, out)
                expecting_source = False
                continue
            expecting_source = False

        if isinstance(token, TokenList):
            _collect(token, out)


def tables_in_sql(sql: str) -> frozenset[str]:
    """Trả tập tên bảng thật mà câu SQL chạm tới.

    Alias bị bỏ, tên CTE bị loại, tiền tố schema bị cắt, tên hạ về chữ thường.
    SQL rỗng hoặc không parse được trả về tập rỗng thay vì ném lỗi — eval
    cần đếm được, không cần dừng.
    """
    if not sql or not sql.strip():
        return frozenset()

    found: set[str] = set()
    ctes: set[str] = set()
    for statement in sqlparse.parse(sql):
        ctes |= _cte_names(statement)
        _collect(statement, found)

    return frozenset(found - ctes)
```

- [ ] **Step 5: Chạy test để xác nhận pass**

Run: `.venv/bin/python -m pytest tests/unit/test_gold_tables.py -q`
Expected: PASS, 7 passed

- [ ] **Step 6: Xác nhận không hồi quy**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: 66 passed (59 cũ + 7 mới)

- [ ] **Step 7: Commit**

```bash
git add eval/gold_tables.py tests/unit/test_gold_tables.py requirements.txt
git commit -m "feat(eval): parse tập bảng thật từ SQL mẫu cho eval tầng 1"
```

---

## Task 2: Kiểu dữ liệu schema và adapter từ `info_box`

**Files:**
- Create: `perception/schema_model.py`
- Create: `tests/unit/test_schema_model.py`
- Create: `tests/fixtures/__init__.py`
- Create: `tests/fixtures/mini_schema.py`

**Interfaces:**
- Consumes: (không có)
- Produces:
  - `perception.schema_model.Column(name: str, data_type: str, is_generated: bool = False)`
  - `perception.schema_model.Table(name: str, columns: tuple[Column, ...], primary_key: tuple[str, ...] = (), foreign_keys: Mapping[str, str] = {}, row_count: int | None = None, description: str = "")` — `foreign_keys` ánh xạ tên cột → chuỗi `"bảng(cột)"`
  - `perception.schema_model.tables_from_info_box(info_box: dict) -> tuple[Table, ...]`
  - `tests.fixtures.mini_schema.MINI_TABLES: tuple[Table, ...]` — 4 bảng dùng chung cho mọi test về sau

- [ ] **Step 1: Đảm bảo import từ gốc repo hoạt động**

Repo hiện **không có** `__init__.py` ở `perception/`, `eval/`, `tests/`, và không có `conftest.py`. Import hiện chạy được là nhờ `python -m pytest` tự thêm thư mục hiện tại vào `sys.path`, cộng namespace package của Python 3. Điều đó vỡ ngay khi ai đó gọi `pytest` trần.

Tạo `conftest.py` ở gốc repo (file rỗng là đủ — sự tồn tại của nó ghim rootdir và đảm bảo gốc repo nằm trong `sys.path` bất kể cách gọi):

```python
# Ghim rootdir cho pytest để `from perception... import` và
# `from tests.fixtures... import` chạy được ở mọi cách gọi.
```

**Không** tạo `tests/__init__.py` — biến `tests/` thành package thật sẽ đổi cách pytest thu thập và có thể làm hỏng 59 test đang chạy.

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: 66 passed (không đổi so với sau Task 1)

- [ ] **Step 2: Viết test thất bại**

Tạo `tests/unit/test_schema_model.py`:

```python
from perception.schema_model import Column, Table, tables_from_info_box


def _info_box() -> dict:
    return {
        "domain": "test",
        "tables": [
            {
                "table_name": "customers",
                "row_count": 500,
                "primary_key": ["id"],
                "columns": [
                    {"name": "id", "data_type": "integer", "is_generated": False},
                    {"name": "name", "data_type": "character varying", "is_generated": False},
                ],
                "foreign_keys": [],
                "sample_rows": [{"id": 1, "name": "bí mật của khách"}],
            },
            {
                "table_name": "orders",
                "row_count": 29830,
                "primary_key": ["id"],
                "columns": [
                    {"name": "id", "data_type": "integer", "is_generated": False},
                    {"name": "customer_id", "data_type": "integer", "is_generated": False},
                    {"name": "year", "data_type": "smallint", "is_generated": True},
                ],
                "foreign_keys": [{"column": "customer_id", "references": "customers.id"}],
                "sample_rows": [],
            },
        ],
        "cross_domain_hints": [
            {"from_table": "orders", "from_column": "id",
             "to_table": "customers", "to_column": "id", "note": "bịa cho test"},
        ],
    }


def test_maps_basic_fields():
    tables = tables_from_info_box(_info_box())
    by_name = {t.name: t for t in tables}
    assert set(by_name) == {"customers", "orders"}
    assert by_name["customers"].row_count == 500
    assert by_name["customers"].primary_key == ("id",)


def test_foreign_keys_render_as_table_paren_column():
    tables = tables_from_info_box(_info_box())
    orders = next(t for t in tables if t.name == "orders")
    assert orders.foreign_keys["customer_id"] == "customers(id)"


def test_cross_domain_hints_become_foreign_keys():
    tables = tables_from_info_box(_info_box())
    orders = next(t for t in tables if t.name == "orders")
    assert orders.foreign_keys["id"] == "customers(id)"


def test_generated_columns_are_flagged():
    tables = tables_from_info_box(_info_box())
    orders = next(t for t in tables if t.name == "orders")
    year = next(c for c in orders.columns if c.name == "year")
    assert year.is_generated is True


def test_sample_rows_are_never_carried_into_the_model():
    """Tiêu chí 9 của spec: dữ liệu thật không được lọt vào profile."""
    tables = tables_from_info_box(_info_box())
    blob = repr(tables)
    assert "sample_rows" not in blob
    assert "bí mật của khách" not in blob


def test_tables_are_immutable():
    t = Table(name="x", columns=(Column("a", "integer"),))
    try:
        t.name = "y"
    except Exception as exc:
        assert isinstance(exc, (AttributeError, TypeError))
    else:
        raise AssertionError("Table phải bất biến")
```

Tạo `tests/fixtures/__init__.py` (file rỗng).

Tạo `tests/fixtures/mini_schema.py`:

```python
"""Schema 4 bảng dùng chung cho test. Cố ý nhỏ và có FK bắc cầu."""

from perception.schema_model import Column, Table

MINI_TABLES = (
    Table(
        name="customers",
        columns=(Column("id", "integer"), Column("name", "character varying"),
                 Column("segment", "character varying")),
        primary_key=("id",),
        row_count=500,
        description="Khách hàng doanh nghiệp và cá nhân",
    ),
    Table(
        name="products",
        columns=(Column("id", "integer"), Column("name", "character varying"),
                 Column("category", "character varying"), Column("price", "numeric")),
        primary_key=("id",),
        row_count=80,
        description="Danh mục sản phẩm bán ra",
    ),
    Table(
        name="orders",
        columns=(Column("id", "integer"), Column("customer_id", "integer"),
                 Column("product_id", "integer"), Column("amount", "numeric"),
                 Column("order_date", "date"), Column("year", "smallint", is_generated=True)),
        primary_key=("id",),
        foreign_keys={"customer_id": "customers(id)", "product_id": "products(id)"},
        row_count=29830,
        description="Đơn hàng bán cho khách",
    ),
    Table(
        name="payroll",
        columns=(Column("id", "integer"), Column("employee_id", "integer"),
                 Column("base_salary", "numeric"), Column("month", "smallint")),
        primary_key=("id",),
        row_count=5897,
        description="Bảng lương nhân viên theo tháng",
    ),
)
```

- [ ] **Step 3: Chạy test để xác nhận fail**

Run: `.venv/bin/python -m pytest tests/unit/test_schema_model.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'perception.schema_model'`

- [ ] **Step 4: Hiện thực**

Tạo `perception/schema_model.py`:

```python
"""Kiểu dữ liệu biểu diễn schema, bất biến, không phụ thuộc nguồn.

Cố ý KHÔNG mang `sample_rows`. Dữ liệu thật chỉ cần lúc sinh chú giải; nó
không được đi tiếp vào profile runtime, vì profile là thứ bị copy đi khi
hỗ trợ kỹ thuật. Xem spec mục 3.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

_EMPTY_FK: Mapping[str, str] = MappingProxyType({})


@dataclass(frozen=True)
class Column:
    name: str
    data_type: str
    is_generated: bool = False


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Column, ...]
    primary_key: tuple[str, ...] = ()
    foreign_keys: Mapping[str, str] = _EMPTY_FK
    row_count: int | None = None
    description: str = ""

    def references(self) -> frozenset[str]:
        """Tên các bảng mà bảng này trỏ tới qua khóa ngoại."""
        return frozenset(ref.split("(")[0] for ref in self.foreign_keys.values())


def tables_from_info_box(info_box: dict[str, Any]) -> tuple[Table, ...]:
    """Chuyển định dạng info_box JSON hiện có sang Table.

    Adapter tạm thời: nó tồn tại để pha 1 chạy được trên dữ liệu đang có.
    Khi đường onboarding (plan 3) hoàn thành, nguồn sẽ là schema.yaml.
    """
    fks: dict[str, dict[str, str]] = {}
    for raw in info_box.get("tables", []):
        for fk in raw.get("foreign_keys", []) or []:
            tbl, _, col = str(fk["references"]).partition(".")
            fks.setdefault(raw["table_name"], {})[fk["column"]] = f"{tbl}({col})"

    for hint in info_box.get("cross_domain_hints", []) or []:
        fks.setdefault(hint["from_table"], {})[hint["from_column"]] = (
            f"{hint['to_table']}({hint['to_column']})"
        )

    tables = []
    for raw in info_box.get("tables", []):
        name = raw["table_name"]
        tables.append(Table(
            name=name,
            columns=tuple(
                Column(
                    name=c["name"],
                    data_type=c["data_type"],
                    is_generated=bool(c.get("is_generated", False)),
                )
                for c in raw.get("columns", [])
            ),
            primary_key=tuple(raw.get("primary_key") or ()),
            foreign_keys=MappingProxyType(dict(fks.get(name, {}))),
            row_count=raw.get("row_count"),
            description=raw.get("description", ""),
        ))
    return tuple(tables)
```

- [ ] **Step 5: Chạy test để xác nhận pass**

Run: `.venv/bin/python -m pytest tests/unit/test_schema_model.py -q`
Expected: PASS, 6 passed

- [ ] **Step 6: Xác nhận adapter chạy được trên dữ liệu thật**

Run:

```bash
.venv/bin/python -c "
import json
from perception.schema_model import tables_from_info_box
tb = tables_from_info_box(json.load(open('perception/info_box_all.json')))
print(len(tb), 'bảng')
print([t.name for t in tb])
print('orders FK:', dict(next(t for t in tb if t.name=='orders').foreign_keys))
"
```

Expected: `9 bảng`, danh sách 9 tên, và `orders FK` chứa ít nhất `customer_id` và `product_id`.

- [ ] **Step 7: Commit**

```bash
git add conftest.py perception/schema_model.py tests/unit/test_schema_model.py tests/fixtures/
git commit -m "feat(perception): kiểu Table/Column bất biến + adapter từ info_box"
```

---

## Task 3: `render_schema()` — Table thành text DDL

**Files:**
- Create: `perception/render_schema.py`
- Create: `tests/unit/test_render_schema.py`

**Interfaces:**
- Consumes: `perception.schema_model.Table`, `Column`; `tests.fixtures.mini_schema.MINI_TABLES`
- Produces:
  - `perception.render_schema.render_schema(tables: Sequence[Table]) -> str`
  - `perception.render_schema.estimate_tokens(text: str) -> int` — `len(text) // 4`, dùng chung cho công tắc `schema_mode`

- [ ] **Step 1: Viết test thất bại**

Tạo `tests/unit/test_render_schema.py`:

```python
from perception.render_schema import estimate_tokens, render_schema
from perception.schema_model import Column, Table
from tests.fixtures.mini_schema import MINI_TABLES


def test_renders_create_table_per_table():
    out = render_schema(MINI_TABLES)
    for name in ("customers", "products", "orders", "payroll"):
        assert f"CREATE TABLE {name} (" in out


def test_primary_key_is_marked():
    out = render_schema([MINI_TABLES[0]])
    assert "id INT PRIMARY KEY" in out


def test_foreign_key_renders_as_references():
    orders = next(t for t in MINI_TABLES if t.name == "orders")
    out = render_schema([orders])
    assert "REFERENCES customers(id)" in out
    assert "REFERENCES products(id)" in out


def test_generated_column_is_marked():
    orders = next(t for t in MINI_TABLES if t.name == "orders")
    assert "GENERATED" in render_schema([orders])


def test_description_renders_as_leading_comment():
    out = render_schema([MINI_TABLES[0]])
    assert "-- Khách hàng doanh nghiệp và cá nhân" in out


def test_verbose_postgres_types_are_shortened():
    out = render_schema(MINI_TABLES)
    assert "character varying" not in out
    assert "VARCHAR" in out


def test_unknown_type_is_passed_through_uppercased():
    t = Table(name="x", columns=(Column("c", "jsonb"),))
    assert "JSONB" in render_schema([t])


def test_stays_under_700_bytes_per_table():
    """Tiêu chí 3 của spec."""
    out = render_schema(MINI_TABLES)
    assert len(out.encode()) / len(MINI_TABLES) <= 700


def test_output_is_deterministic():
    assert render_schema(MINI_TABLES) == render_schema(MINI_TABLES)


def test_empty_input_gives_empty_string():
    assert render_schema([]) == ""


def test_estimate_tokens_is_quarter_of_length():
    assert estimate_tokens("a" * 400) == 100
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `.venv/bin/python -m pytest tests/unit/test_render_schema.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'perception.render_schema'`

- [ ] **Step 3: Hiện thực**

Tạo `perception/render_schema.py`:

```python
"""Biểu diễn schema dạng DDL để đưa vào prompt.

Vì sao DDL chứ không phải JSON: nhỏ hơn 2–3 lần, và Qwen2.5-Coder đã đọc
hàng triệu file SQL lúc pretrain trong khi JSON mô tả schema thì không.
Xem spec mục 3.4.
"""

from __future__ import annotations

from collections.abc import Sequence

from perception.schema_model import Table

_TYPE_SHORT = {
    "character varying": "VARCHAR",
    "character": "CHAR",
    "timestamp without time zone": "TIMESTAMP",
    "timestamp with time zone": "TIMESTAMPTZ",
    "double precision": "FLOAT",
    "integer": "INT",
    "smallint": "SMALLINT",
    "bigint": "BIGINT",
    "numeric": "NUMERIC",
    "boolean": "BOOL",
    "date": "DATE",
    "text": "TEXT",
}


def _short_type(data_type: str) -> str:
    return _TYPE_SHORT.get(data_type.lower(), data_type.upper())


def _render_one(table: Table) -> str:
    lines: list[str] = []
    if table.description:
        lines.append(f"-- {table.description}")
    lines.append(f"CREATE TABLE {table.name} (")

    pk = set(table.primary_key)
    body: list[str] = []
    for col in table.columns:
        parts = [col.name, _short_type(col.data_type)]
        if col.name in pk:
            parts.append("PRIMARY KEY")
        if ref := table.foreign_keys.get(col.name):
            parts.append(f"REFERENCES {ref}")
        if col.is_generated:
            parts.append("GENERATED")
        body.append("  " + " ".join(parts))
    lines.append(",\n".join(body))

    tail = ");"
    if table.row_count is not None:
        tail += f"  -- {table.row_count} rows"
    lines.append(tail)
    return "\n".join(lines)


def render_schema(tables: Sequence[Table]) -> str:
    """Kết xuất danh sách Table thành text DDL, ngăn bởi dòng trống.

    Đầu ra deterministic: thứ tự bảng và cột giữ nguyên như đầu vào, để
    prefix caching còn dùng được.
    """
    if not tables:
        return ""
    return "\n\n".join(_render_one(t) for t in tables)


def estimate_tokens(text: str) -> int:
    """Ước lượng thô số token. Dùng chung cho công tắc schema_mode (spec 3.3).

    Cố ý thô: mọi ngưỡng trong spec đều tính trên cùng ước lượng này, nên
    sai số hệ thống không làm lệch quyết định. Nếu đổi công thức thì phải
    đổi cả ngưỡng 6.000 ở mục 3.3.
    """
    return len(text) // 4
```

- [ ] **Step 4: Chạy test để xác nhận pass**

Run: `.venv/bin/python -m pytest tests/unit/test_render_schema.py -q`
Expected: PASS, 11 passed

- [ ] **Step 5: Đo trên schema thật và ghi lại con số**

Run:

```bash
.venv/bin/python -c "
import json
from perception.schema_model import tables_from_info_box
from perception.render_schema import render_schema, estimate_tokens
tb = tables_from_info_box(json.load(open('perception/info_box_all.json')))
out = render_schema(tb)
print('bytes:', len(out.encode()), '| B/bảng:', len(out.encode())//len(tb), '| token:', estimate_tokens(out))
"
```

Expected: B/bảng ≤ 700. Ghi con số thực tế vào commit message — spec mục 3.4 yêu cầu đo lại giả định 140 token/bảng ở pha 1.

- [ ] **Step 6: Commit**

```bash
git add perception/render_schema.py tests/unit/test_render_schema.py
git commit -m "feat(perception): render_schema() DDL thay cho json.dumps info_box"
```

---

## Task 4: `ConnectionProfile` + `permitted_tables()` + fingerprint

Đây là ranh giới bảo mật của cả hệ thống. Xem spec mục 3.4.1.

**Files:**
- Create: `perception/connection_profile.py`
- Create: `tests/unit/test_connection_profile.py`

**Interfaces:**
- Consumes: `perception.schema_model.Table`
- Produces:
  - `perception.connection_profile.ConnectionProfile(dsn: str, tables: tuple[Table, ...], grants: Mapping[str, frozenset[str]], schema_mode: str, fingerprint: str)`
  - `perception.connection_profile.permitted_tables(profile: ConnectionProfile, user: str) -> frozenset[str]`
  - `perception.connection_profile.schema_fingerprint(tables: Sequence[Table]) -> str`
  - `perception.connection_profile.build_profile(dsn, tables, grants, threshold_tokens=6000) -> ConnectionProfile`
  - Hằng `ALL_TABLES = "*"` — giá trị grant nghĩa là toàn quyền

- [ ] **Step 1: Viết test thất bại**

Tạo `tests/unit/test_connection_profile.py`:

```python
import pytest

from perception.connection_profile import (
    ALL_TABLES,
    build_profile,
    permitted_tables,
    schema_fingerprint,
)
from perception.schema_model import Column, Table
from tests.fixtures.mini_schema import MINI_TABLES

DSN = "postgresql://u:p@localhost:5432/db"


def _profile(grants=None, tables=MINI_TABLES):
    return build_profile(dsn=DSN, tables=tables, grants=grants or {})


def test_unknown_user_gets_nothing():
    """Mặc định đóng. Không có grant nghĩa là không thấy gì."""
    assert permitted_tables(_profile(), "nguoi_la") == frozenset()


def test_wildcard_grant_gives_every_table():
    p = _profile({"admin": frozenset({ALL_TABLES})})
    assert permitted_tables(p, "admin") == {t.name for t in MINI_TABLES}


def test_explicit_grant_gives_exactly_those_tables():
    p = _profile({"sales": frozenset({"orders", "customers"})})
    assert permitted_tables(p, "sales") == frozenset({"orders", "customers"})


def test_grant_naming_a_table_that_does_not_exist_is_dropped():
    p = _profile({"sales": frozenset({"orders", "bang_khong_ton_tai"})})
    assert permitted_tables(p, "sales") == frozenset({"orders"})


def test_payroll_is_not_reachable_without_an_explicit_grant():
    p = _profile({"sales": frozenset({"orders", "customers", "products"})})
    assert "payroll" not in permitted_tables(p, "sales")


def test_fingerprint_is_stable_across_calls():
    assert schema_fingerprint(MINI_TABLES) == schema_fingerprint(MINI_TABLES)


def test_fingerprint_ignores_table_ordering():
    assert schema_fingerprint(MINI_TABLES) == schema_fingerprint(tuple(reversed(MINI_TABLES)))


def test_fingerprint_changes_when_a_column_is_added():
    before = schema_fingerprint(MINI_TABLES)
    grown = list(MINI_TABLES)
    t = grown[0]
    grown[0] = Table(
        name=t.name,
        columns=t.columns + (Column("phone", "character varying"),),
        primary_key=t.primary_key,
        foreign_keys=t.foreign_keys,
        row_count=t.row_count,
        description=t.description,
    )
    assert schema_fingerprint(tuple(grown)) != before


def test_fingerprint_ignores_row_count():
    """row_count đổi mỗi ngày; nó không phải thay đổi schema."""
    before = schema_fingerprint(MINI_TABLES)
    t = MINI_TABLES[0]
    same = (Table(name=t.name, columns=t.columns, primary_key=t.primary_key,
                  foreign_keys=t.foreign_keys, row_count=999999,
                  description=t.description),) + MINI_TABLES[1:]
    assert schema_fingerprint(same) == before


def test_small_schema_gets_full_mode():
    assert _profile().schema_mode == "full"


def test_large_schema_gets_retrieval_mode():
    many = tuple(
        Table(name=f"t{i}", columns=tuple(Column(f"c{j}", "integer") for j in range(12)))
        for i in range(200)
    )
    assert build_profile(dsn=DSN, tables=many, grants={}).schema_mode == "retrieval"


def test_threshold_is_configurable():
    assert build_profile(dsn=DSN, tables=MINI_TABLES, grants={},
                         threshold_tokens=1).schema_mode == "retrieval"
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `.venv/bin/python -m pytest tests/unit/test_connection_profile.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'perception.connection_profile'`

- [ ] **Step 3: Hiện thực**

Tạo `perception/connection_profile.py`:

```python
"""Hồ sơ kết nối: một object mang mọi thứ về "đang nói chuyện với DB nào".

Thay cho ba biến toàn cục rải rác trước đây (DATABASE_URL, _ALLOWED_TABLES,
info_box_*.json). Xem spec mục 3.4.

QUAN TRỌNG — hai tập bảng, khác bản chất, không được gộp (spec 3.4.1):

  permitted_tables(user)     BẢO MẬT.       Theo người dùng. Dẫn ra ở đây.
  retrieved_tables(question) NỘI DUNG PROMPT. Theo câu hỏi. Dẫn ra ở schema_context.

Bên thực thi SQL chỉ được dùng cái thứ nhất, và phải tự gọi hàm này chứ
không nhận tập quyền từ bên gọi. Nếu nhận từ bên gọi thì bên bị ràng buộc
đang tự khai ràng buộc của mình — guard tụt xuống thành lời hứa.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from perception.render_schema import estimate_tokens, render_schema
from perception.schema_model import Table

ALL_TABLES = "*"
DEFAULT_THRESHOLD_TOKENS = 6000


@dataclass(frozen=True)
class ConnectionProfile:
    dsn: str
    tables: tuple[Table, ...]
    grants: Mapping[str, frozenset[str]]
    schema_mode: str  # "full" | "retrieval"
    fingerprint: str

    def table_names(self) -> frozenset[str]:
        return frozenset(t.name for t in self.tables)

    def by_name(self) -> dict[str, Table]:
        return {t.name: t for t in self.tables}


def schema_fingerprint(tables: Sequence[Table]) -> str:
    """Hash của cấu trúc schema — tên bảng và tên/kiểu cột.

    Cố ý bỏ qua row_count và description: row_count đổi mỗi ngày và mô tả
    do người sửa; cả hai đều không phải thay đổi schema. Chỉ thay đổi cấu
    trúc mới được kích hoạt profile_stale (spec mục 5.2).
    """
    parts = []
    for t in sorted(tables, key=lambda x: x.name):
        cols = ",".join(f"{c.name}:{c.data_type}" for c in sorted(t.columns, key=lambda c: c.name))
        pk = ",".join(sorted(t.primary_key))
        fk = ",".join(f"{k}->{v}" for k, v in sorted(t.foreign_keys.items()))
        parts.append(f"{t.name}|{cols}|{pk}|{fk}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def build_profile(
    dsn: str,
    tables: Sequence[Table],
    grants: Mapping[str, frozenset[str]],
    threshold_tokens: int = DEFAULT_THRESHOLD_TOKENS,
) -> ConnectionProfile:
    """Dựng profile và quyết công tắc schema_mode MỘT LẦN, lúc cài đặt.

    Đây không phải fallback động: một nhánh code, một công tắc, quyết một
    lần. Xem spec mục 3.3 và mục 10.3.
    """
    tables = tuple(tables)
    size = estimate_tokens(render_schema(tables))
    return ConnectionProfile(
        dsn=dsn,
        tables=tables,
        grants=dict(grants),
        schema_mode="full" if size <= threshold_tokens else "retrieval",
        fingerprint=schema_fingerprint(tables),
    )


def permitted_tables(profile: ConnectionProfile, user: str) -> frozenset[str]:
    """Tập bảng người dùng được phép chạm. Ranh giới bảo mật thật.

    Mặc định đóng: người dùng không có mục trong grants thì không thấy gì.
    Tên bảng trong grant mà schema không có sẽ bị bỏ, để một grant cũ không
    mở ra thứ gì ngoài ý muốn khi schema đổi.
    """
    granted = profile.grants.get(user)
    if not granted:
        return frozenset()
    existing = profile.table_names()
    if ALL_TABLES in granted:
        return existing
    return frozenset(granted) & existing
```

- [ ] **Step 4: Chạy test để xác nhận pass**

Run: `.venv/bin/python -m pytest tests/unit/test_connection_profile.py -q`
Expected: PASS, 12 passed

- [ ] **Step 5: Commit**

```bash
git add perception/connection_profile.py tests/unit/test_connection_profile.py
git commit -m "feat(perception): ConnectionProfile + permitted_tables là ranh giới bảo mật"
```

---

## Task 5: `Retriever` — `FullRetriever` và `LexicalRetriever`

**Files:**
- Create: `perception/retrieval.py`
- Create: `tests/unit/test_retrieval.py`

**Interfaces:**
- Consumes: `perception.schema_model.Table`
- Produces:
  - `perception.retrieval.Retriever` — Protocol với `search(question: str, k: int) -> list[str]`
  - `perception.retrieval.FullRetriever(tables: Sequence[Table])` — trả mọi tên bảng, dùng làm trần trên khi đo recall
  - `perception.retrieval.LexicalRetriever(tables: Sequence[Table])` — chấm điểm bằng trùng token trên tên bảng, tên cột và mô tả
  - `perception.retrieval.expand_by_foreign_keys(names: Iterable[str], tables: Sequence[Table]) -> frozenset[str]` — mở rộng 1 bước, cả hai chiều

- [ ] **Step 1: Viết test thất bại**

Tạo `tests/unit/test_retrieval.py`:

```python
from perception.retrieval import FullRetriever, LexicalRetriever, expand_by_foreign_keys
from tests.fixtures.mini_schema import MINI_TABLES


def test_full_retriever_returns_everything():
    r = FullRetriever(MINI_TABLES)
    assert set(r.search("bất kỳ câu gì", k=2)) == {t.name for t in MINI_TABLES}


def test_lexical_matches_on_table_name():
    r = LexicalRetriever(MINI_TABLES)
    assert "orders" in r.search("tổng doanh thu theo orders", k=2)


def test_lexical_matches_on_column_name():
    r = LexicalRetriever(MINI_TABLES)
    assert "payroll" in r.search("thống kê base_salary", k=2)


def test_lexical_matches_on_description():
    r = LexicalRetriever(MINI_TABLES)
    assert "payroll" in r.search("bảng lương nhân viên", k=1)


def test_lexical_respects_k():
    r = LexicalRetriever(MINI_TABLES)
    assert len(r.search("orders customers products payroll", k=2)) == 2


def test_lexical_returns_empty_on_no_overlap():
    r = LexicalRetriever(MINI_TABLES)
    assert r.search("zzzzz qqqqq", k=3) == []


def test_lexical_is_deterministic_on_ties():
    r = LexicalRetriever(MINI_TABLES)
    assert r.search("id", k=4) == r.search("id", k=4)


def test_fk_expansion_follows_outgoing_edges():
    # orders → customers, products
    assert expand_by_foreign_keys(["orders"], MINI_TABLES) == frozenset(
        {"orders", "customers", "products"}
    )


def test_fk_expansion_follows_incoming_edges():
    # customers ← orders
    assert "orders" in expand_by_foreign_keys(["customers"], MINI_TABLES)


def test_fk_expansion_is_one_hop_only():
    """customers → orders → products. products KHÔNG được kéo vào từ customers."""
    got = expand_by_foreign_keys(["customers"], MINI_TABLES)
    assert "products" not in got


def test_fk_expansion_of_isolated_table_returns_itself():
    assert expand_by_foreign_keys(["payroll"], MINI_TABLES) == frozenset({"payroll"})


def test_fk_expansion_ignores_unknown_names():
    assert expand_by_foreign_keys(["khong_co"], MINI_TABLES) == frozenset()
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `.venv/bin/python -m pytest tests/unit/test_retrieval.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'perception.retrieval'`

- [ ] **Step 3: Hiện thực**

Tạo `perception/retrieval.py`:

```python
"""Chọn bảng liên quan tới câu hỏi.

LƯU Ý — đây KHÔNG phải cơ chế phân quyền (spec 3.4.1). Nó chỉ thu hẹp thứ
model *nhìn thấy*, không thu hẹp thứ model *được phép chạm*. Quyền nằm ở
perception.connection_profile.permitted_tables().

LexicalRetriever là mốc so sánh, cố ý không phụ thuộc torch. Bản embedding
đến sau, và phải chứng minh nó thắng mốc này trên eval tầng 1 mới đáng
đánh đổi thêm 470MB artifact vào bundle on-prem.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Protocol

from perception.schema_model import Table

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    """Tách token, cắt cả snake_case: base_salary → {base, salary}."""
    return set(_WORD.findall(text.lower()))


class Retriever(Protocol):
    def search(self, question: str, k: int) -> list[str]:
        """Trả tối đa k tên bảng, xếp theo độ liên quan giảm dần."""
        ...


class FullRetriever:
    """Trả mọi bảng. Dùng cho schema_mode='full' và làm trần trên khi đo recall."""

    def __init__(self, tables: Sequence[Table]) -> None:
        self._names = [t.name for t in tables]

    def search(self, question: str, k: int) -> list[str]:  # noqa: ARG002
        return list(self._names)


class LexicalRetriever:
    """Chấm điểm bằng số token trùng giữa câu hỏi và mô tả bảng.

    Tên bảng có trọng số cao nhất, rồi mô tả, rồi tên cột — tên cột nhiều
    và nhiễu (id, name xuất hiện ở mọi bảng) nên không được lấn át.
    """

    _W_NAME = 3.0
    _W_DESC = 2.0
    _W_COLUMN = 1.0

    def __init__(self, tables: Sequence[Table]) -> None:
        self._index: list[tuple[str, set[str], set[str], set[str]]] = [
            (t.name, _tokens(t.name), _tokens(t.description),
             _tokens(" ".join(c.name for c in t.columns)))
            for t in tables
        ]

    def search(self, question: str, k: int) -> list[str]:
        q = _tokens(question)
        if not q:
            return []

        scored: list[tuple[float, int, str]] = []
        for rank, (name, n_tok, d_tok, c_tok) in enumerate(self._index):
            score = (
                self._W_NAME * len(q & n_tok)
                + self._W_DESC * len(q & d_tok)
                + self._W_COLUMN * len(q & c_tok)
            )
            if score > 0:
                # rank làm khóa phụ để hòa điểm vẫn ra thứ tự cố định
                scored.append((-score, rank, name))

        scored.sort()
        return [name for _, _, name in scored[:k]]


def expand_by_foreign_keys(names: Iterable[str], tables: Sequence[Table]) -> frozenset[str]:
    """Mở rộng đúng MỘT bước theo cạnh khóa ngoại, cả hai chiều.

    Hai chiều vì một câu hỏi nhắc `customers` thường vẫn cần `orders` để
    tính được gì đó, dù cạnh FK đi từ orders sang customers.

    Một bước vì hai bước trên schema doanh nghiệp sẽ kéo về gần hết schema
    và làm hỏng chính lợi ích của việc thu hẹp.
    """
    by_name = {t.name: t for t in tables}
    seed = {n for n in names if n in by_name}
    if not seed:
        return frozenset()

    out = set(seed)
    for name in seed:
        out |= by_name[name].references() & by_name.keys()
    for t in tables:
        if t.references() & seed:
            out.add(t.name)
    return frozenset(out)
```

- [ ] **Step 4: Chạy test để xác nhận pass**

Run: `.venv/bin/python -m pytest tests/unit/test_retrieval.py -q`
Expected: PASS, 12 passed

- [ ] **Step 5: Commit**

```bash
git add perception/retrieval.py tests/unit/test_retrieval.py
git commit -m "feat(perception): Retriever + FullRetriever/LexicalRetriever + mở rộng FK 1 bước"
```

---

## Task 6: `SchemaContext` + `resolve_schema_context()`

Đây là nơi công tắc `full`/`retrieval` sống và là điểm cắm duy nhất mà `graph/` gọi.

**Files:**
- Create: `perception/schema_context.py`
- Create: `tests/unit/test_schema_context.py`

**Interfaces:**
- Consumes: `ConnectionProfile`, `permitted_tables`, `render_schema`, `Retriever`, `expand_by_foreign_keys`
- Produces:
  - `perception.schema_context.SchemaContext(retrieved_tables: tuple[str, ...], rendered_text: str, few_shots: tuple[dict, ...])`
  - `perception.schema_context.resolve_schema_context(profile, question, permitted, retriever=None, k=8, must_include=()) -> SchemaContext`

- [ ] **Step 1: Viết test thất bại**

Tạo `tests/unit/test_schema_context.py`:

```python
import pytest

from perception.connection_profile import ALL_TABLES, build_profile
from perception.retrieval import LexicalRetriever
from perception.schema_context import resolve_schema_context
from tests.fixtures.mini_schema import MINI_TABLES

DSN = "postgresql://u:p@localhost:5432/db"


def _profile(threshold=6000):
    return build_profile(dsn=DSN, tables=MINI_TABLES,
                         grants={"admin": frozenset({ALL_TABLES})},
                         threshold_tokens=threshold)


ALL = frozenset(t.name for t in MINI_TABLES)


def test_full_mode_returns_every_permitted_table_regardless_of_question():
    ctx = resolve_schema_context(_profile(), "bất kỳ câu gì", permitted=ALL)
    assert set(ctx.retrieved_tables) == ALL


def test_full_mode_never_leaks_a_table_outside_permitted():
    ctx = resolve_schema_context(_profile(), "lương nhân viên",
                                 permitted=frozenset({"orders", "customers"}))
    assert set(ctx.retrieved_tables) == {"orders", "customers"}
    assert "payroll" not in ctx.rendered_text


def test_retrieval_mode_narrows_by_question():
    p = _profile(threshold=1)  # ép sang retrieval
    ctx = resolve_schema_context(p, "bảng lương nhân viên", permitted=ALL,
                                 retriever=LexicalRetriever(MINI_TABLES), k=1)
    assert "payroll" in ctx.retrieved_tables
    assert set(ctx.retrieved_tables) != ALL


def test_retrieval_mode_expands_along_foreign_keys():
    p = _profile(threshold=1)
    ctx = resolve_schema_context(p, "orders", permitted=ALL,
                                 retriever=LexicalRetriever(MINI_TABLES), k=1)
    # orders kéo theo customers và products qua FK
    assert {"orders", "customers", "products"} <= set(ctx.retrieved_tables)


def test_permitted_filter_is_applied_after_fk_expansion():
    """Mở rộng FK không được vượt qua hàng rào quyền."""
    p = _profile(threshold=1)
    ctx = resolve_schema_context(p, "orders", permitted=frozenset({"orders"}),
                                 retriever=LexicalRetriever(MINI_TABLES), k=1)
    assert set(ctx.retrieved_tables) == {"orders"}


def test_must_include_forces_a_table_in():
    p = _profile(threshold=1)
    ctx = resolve_schema_context(p, "zzzz không khớp gì", permitted=ALL,
                                 retriever=LexicalRetriever(MINI_TABLES), k=1,
                                 must_include=["payroll"])
    assert "payroll" in ctx.retrieved_tables


def test_must_include_cannot_bypass_permitted():
    """must_include là công cụ sửa lỗi retrieval, KHÔNG phải cửa hậu quyền."""
    p = _profile(threshold=1)
    ctx = resolve_schema_context(p, "gì đó", permitted=frozenset({"orders"}),
                                 retriever=LexicalRetriever(MINI_TABLES), k=1,
                                 must_include=["payroll"])
    assert "payroll" not in ctx.retrieved_tables


def test_retrieval_mode_without_a_retriever_is_an_error():
    p = _profile(threshold=1)
    with pytest.raises(ValueError, match="retriever"):
        resolve_schema_context(p, "gì đó", permitted=ALL)


def test_rendered_text_contains_only_retrieved_tables():
    p = _profile(threshold=1)
    ctx = resolve_schema_context(p, "bảng lương nhân viên", permitted=ALL,
                                 retriever=LexicalRetriever(MINI_TABLES), k=1)
    for name in ALL - set(ctx.retrieved_tables):
        assert f"CREATE TABLE {name} (" not in ctx.rendered_text
    for name in ctx.retrieved_tables:
        assert f"CREATE TABLE {name} (" in ctx.rendered_text


def test_empty_permitted_gives_empty_context():
    ctx = resolve_schema_context(_profile(), "gì đó", permitted=frozenset())
    assert ctx.retrieved_tables == ()
    assert ctx.rendered_text == ""


def test_table_order_is_stable_for_prefix_caching():
    p = _profile()
    a = resolve_schema_context(p, "câu một", permitted=ALL)
    b = resolve_schema_context(p, "câu hai khác hẳn", permitted=ALL)
    assert a.rendered_text == b.rendered_text
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `.venv/bin/python -m pytest tests/unit/test_schema_context.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'perception.schema_context'`

- [ ] **Step 3: Hiện thực**

Tạo `perception/schema_context.py`:

```python
"""Dựng lát schema đưa vào prompt cho một câu hỏi cụ thể.

Hàm thuần: không gọi LLM, không chạm database, không có kiểu lỗi cần retry.
Vì thế nó KHÔNG phải node LangGraph — biến nó thành node là mua thêm một
hop và một nhánh lỗi để đổi lấy không gì cả (spec mục 10.4). Nó chạy đúng
một lần mỗi câu hỏi, trước make_initial_state().
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from perception.connection_profile import ConnectionProfile
from perception.render_schema import render_schema
from perception.retrieval import Retriever, expand_by_foreign_keys


@dataclass(frozen=True)
class SchemaContext:
    """Nội dung prompt cho một câu hỏi. KHÔNG mang thông tin quyền.

    Cố ý không có trường `allowed_tables`: nếu có, ai đó sẽ truyền nó xuống
    execute_sql và biến guard thành thứ do bên gọi tự khai (spec 3.4.1).
    """

    retrieved_tables: tuple[str, ...] = ()
    rendered_text: str = ""
    few_shots: tuple[dict, ...] = field(default_factory=tuple)


def resolve_schema_context(
    profile: ConnectionProfile,
    question: str,
    permitted: frozenset[str],
    retriever: Retriever | None = None,
    k: int = 8,
    must_include: Sequence[str] = (),
) -> SchemaContext:
    """Chọn bảng và kết xuất text schema cho một câu hỏi.

    Args:
        profile: hồ sơ kết nối, mang schema và công tắc schema_mode.
        question: câu hỏi của người dùng.
        permitted: tập bảng người dùng được phép — dẫn ra ĐỘC LẬP bởi
            connection_profile.permitted_tables(). Mọi đường ra khỏi hàm
            này đều bị chặn bởi tập đó, kể cả must_include.
        retriever: bắt buộc khi profile.schema_mode == "retrieval".
        k: số bảng lấy trước khi mở rộng theo khóa ngoại.
        must_include: bảng buộc phải có mặt. Dùng khi reflector báo
            schema_mismatch (spec mục 5.2). Không vượt được `permitted`.

    Raises:
        ValueError: khi schema_mode là "retrieval" mà không có retriever.
    """
    if not permitted:
        return SchemaContext()

    if profile.schema_mode == "full":
        chosen = permitted
    else:
        if retriever is None:
            raise ValueError(
                "schema_mode='retrieval' cần một retriever. "
                "Dùng FullRetriever nếu muốn hành vi của chế độ full."
            )
        hits = retriever.search(question, k=k)
        chosen = expand_by_foreign_keys(hits, profile.tables) | set(must_include)
        chosen &= permitted

    by_name = profile.by_name()
    # Giữ thứ tự theo profile, không theo điểm số: đầu ra phải ổn định
    # giữa các câu hỏi thì prefix caching mới dùng được.
    ordered = tuple(t.name for t in profile.tables if t.name in chosen)

    return SchemaContext(
        retrieved_tables=ordered,
        rendered_text=render_schema([by_name[n] for n in ordered]),
        few_shots=(),
    )
```

- [ ] **Step 4: Chạy test để xác nhận pass**

Run: `.venv/bin/python -m pytest tests/unit/test_schema_context.py -q`
Expected: PASS, 11 passed

- [ ] **Step 5: Chạy toàn bộ test**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: 118 passed (59 cũ + 59 mới)

- [ ] **Step 6: Commit**

```bash
git add perception/schema_context.py tests/unit/test_schema_context.py
git commit -m "feat(perception): resolve_schema_context với công tắc full/retrieval"
```

---

## Task 7: Harness đo recall tầng 1

Đây là vòng lặp đo nhanh sẽ chạy hàng trăm lần khi chỉnh retriever. Nó phải chạy trong vài giây và không cần LLM, không cần GPU, không cần database.

**Files:**
- Create: `eval/datasets.py`
- Create: `eval/tier1_recall.py`
- Create: `tests/unit/test_tier1_recall.py`
- Create: `eval/README_multischema.md`

**Interfaces:**
- Consumes: `eval.gold_tables.tables_in_sql`, `perception.schema_model.Table`, `perception.schema_context.resolve_schema_context`
- Produces:
  - `eval.datasets.EvalRecord(question: str, gold_sql: str, db_id: str, tables: tuple[Table, ...])`
  - `eval.datasets.load_adba_golden(path: Path) -> list[EvalRecord]`
  - `eval.tier1_recall.RecallReport(total: int, full_hits: int, recall: float, avg_context_tables: float, misses: list[tuple[str, frozenset[str]]])`
  - `eval.tier1_recall.measure_recall(records, resolve) -> RecallReport` — `resolve` là callable `(EvalRecord) -> frozenset[str]`

- [ ] **Step 1: Viết test thất bại**

Tạo `tests/unit/test_tier1_recall.py`:

```python
from eval.datasets import EvalRecord
from eval.tier1_recall import measure_recall
from tests.fixtures.mini_schema import MINI_TABLES


def _rec(question: str, sql: str) -> EvalRecord:
    return EvalRecord(question=question, gold_sql=sql, db_id="mini", tables=MINI_TABLES)


def test_perfect_retriever_scores_one():
    records = [_rec("q", "SELECT * FROM orders JOIN customers ON 1=1")]
    report = measure_recall(records, lambda r: frozenset({"orders", "customers"}))
    assert report.recall == 1.0
    assert report.full_hits == 1


def test_missing_one_table_counts_as_a_full_miss():
    """Không có điểm từng phần: thiếu một bảng JOIN là SQL sai chắc chắn."""
    records = [_rec("q", "SELECT * FROM orders JOIN customers ON 1=1")]
    report = measure_recall(records, lambda r: frozenset({"orders"}))
    assert report.recall == 0.0
    assert report.full_hits == 0


def test_extra_tables_do_not_hurt_recall():
    records = [_rec("q", "SELECT * FROM orders")]
    report = measure_recall(records, lambda r: frozenset({"orders", "payroll", "products"}))
    assert report.recall == 1.0


def test_avg_context_tables_is_reported_so_precision_is_visible():
    records = [_rec("q", "SELECT * FROM orders")]
    report = measure_recall(records, lambda r: frozenset({"orders", "payroll", "products"}))
    assert report.avg_context_tables == 3.0


def test_misses_record_which_tables_were_absent():
    records = [_rec("câu hỏi X", "SELECT * FROM orders JOIN payroll ON 1=1")]
    report = measure_recall(records, lambda r: frozenset({"orders"}))
    assert report.misses == [("câu hỏi X", frozenset({"payroll"}))]


def test_records_whose_gold_sql_has_no_tables_are_skipped():
    records = [_rec("q", "not sql"), _rec("q2", "SELECT * FROM orders")]
    report = measure_recall(records, lambda r: frozenset({"orders"}))
    assert report.total == 1
    assert report.recall == 1.0


def test_empty_record_list_gives_zero_without_dividing_by_zero():
    report = measure_recall([], lambda r: frozenset())
    assert report.total == 0
    assert report.recall == 0.0
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `.venv/bin/python -m pytest tests/unit/test_tier1_recall.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval.datasets'`

- [ ] **Step 3: Hiện thực `eval/datasets.py`**

```python
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


def load_jsonl_generic(path: Path, tables_by_db: dict[str, Sequence[Table]]) -> list[EvalRecord]:
    """Đọc bộ đã chuẩn hóa sẵn: {"question", "sql", "db_id"} mỗi dòng.

    Loader riêng cho Spider/BIRD/BEAVER sẽ chuyển bộ gốc về định dạng này
    rồi gọi hàm này. Việc chuyển đổi đó nằm ở Step 6 dưới đây, vì nó phụ
    thuộc vào bố cục thư mục thực tế của từng bộ.
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
```

- [ ] **Step 4: Hiện thực `eval/tier1_recall.py`**

```python
"""Eval tầng 1: retriever có chọn ĐỦ tập bảng đúng không?

Không LLM, không GPU, không database. Tập bảng đúng parse từ SQL mẫu.
Chạy hết vài giây cho hàng nghìn câu — đây là vòng lặp để chỉnh retriever.

"Đủ" chứ không phải "có": thiếu một bảng JOIN là SQL sai chắc chắn, nên
không tính điểm từng phần. Xem spec mục 6.1 và 6.5.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from eval.datasets import EvalRecord, load_adba_golden
from eval.gold_tables import tables_in_sql
from perception.connection_profile import ALL_TABLES, build_profile
from perception.retrieval import FullRetriever, LexicalRetriever
from perception.schema_context import resolve_schema_context


@dataclass
class RecallReport:
    total: int
    full_hits: int
    recall: float
    avg_context_tables: float
    misses: list[tuple[str, frozenset[str]]] = field(default_factory=list)

    def as_text(self) -> str:
        lines = [
            f"câu chấm được   : {self.total}",
            f"phủ trọn vẹn    : {self.full_hits}",
            f"recall          : {self.recall:.3f}",
            f"bảng/context TB : {self.avg_context_tables:.1f}",
        ]
        if self.misses:
            lines.append(f"\n{len(self.misses)} câu trượt (10 câu đầu):")
            for q, missing in self.misses[:10]:
                lines.append(f"  - {q[:70]}  ← thiếu {sorted(missing)}")
        return "\n".join(lines)


def measure_recall(
    records: Sequence[EvalRecord],
    resolve: Callable[[EvalRecord], frozenset[str]],
) -> RecallReport:
    """Đo tỉ lệ câu mà context chứa TOÀN BỘ tập bảng đúng.

    Bản ghi có SQL mẫu không parse ra bảng nào sẽ bị bỏ qua chứ không tính
    là trượt — đó là lỗi dữ liệu, không phải lỗi retriever.
    """
    total = 0
    hits = 0
    ctx_sizes: list[int] = []
    misses: list[tuple[str, frozenset[str]]] = []

    for rec in records:
        gold = tables_in_sql(rec.gold_sql)
        if not gold:
            continue
        total += 1
        got = resolve(rec)
        ctx_sizes.append(len(got))
        missing = gold - got
        if missing:
            misses.append((rec.question, missing))
        else:
            hits += 1

    return RecallReport(
        total=total,
        full_hits=hits,
        recall=(hits / total) if total else 0.0,
        avg_context_tables=(sum(ctx_sizes) / len(ctx_sizes)) if ctx_sizes else 0.0,
        misses=misses,
    )


def _resolver(strategy: str, k: int) -> Callable[[EvalRecord], frozenset[str]]:
    def resolve(rec: EvalRecord) -> frozenset[str]:
        permitted = frozenset(t.name for t in rec.tables)
        threshold = 10**9 if strategy == "full" else 1
        profile = build_profile(
            dsn="", tables=rec.tables,
            grants={"eval": frozenset({ALL_TABLES})},
            threshold_tokens=threshold,
        )
        retriever = (FullRetriever(rec.tables) if strategy == "full"
                     else LexicalRetriever(rec.tables))
        ctx = resolve_schema_context(profile, rec.question, permitted,
                                     retriever=retriever, k=k)
        return frozenset(ctx.retrieved_tables)

    return resolve


def main() -> None:
    ap = argparse.ArgumentParser(description="Đo recall chọn bảng (eval tầng 1)")
    ap.add_argument("--golden", type=Path, required=True, help="JSONL {question, sql}")
    ap.add_argument("--info-box", type=Path, default=Path("perception/info_box_all.json"))
    ap.add_argument("--strategy", choices=["full", "lexical"], default="lexical")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--json", type=Path, help="ghi báo cáo dạng JSON ra đây")
    args = ap.parse_args()

    records = load_adba_golden(args.golden, args.info_box)
    report = measure_recall(records, _resolver(args.strategy, args.k))
    print(f"[{args.strategy}, k={args.k}]")
    print(report.as_text())

    if args.json:
        args.json.write_text(json.dumps({
            "strategy": args.strategy, "k": args.k,
            "total": report.total, "full_hits": report.full_hits,
            "recall": report.recall,
            "avg_context_tables": report.avg_context_tables,
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Chạy test để xác nhận pass**

Run: `.venv/bin/python -m pytest tests/unit/test_tier1_recall.py -q`
Expected: PASS, 7 passed

- [ ] **Step 6: Dựng golden set ADBA và đo mốc thật**

Tạo `eval/golden_adba.jsonl` với 12 câu (một dòng JSON mỗi câu). Dùng đúng các câu đã có trong `tests/integration/`:

```json
{"question": "Tổng doanh thu theo region năm 2024", "sql": "SELECT region, SUM(amount) FROM orders WHERE year = 2024 GROUP BY region"}
{"question": "Top 5 sản phẩm bán chạy nhất theo doanh thu", "sql": "SELECT p.name, SUM(o.amount) AS rev FROM orders o JOIN products p ON o.product_id = p.id GROUP BY p.name ORDER BY rev DESC LIMIT 5"}
{"question": "Số lượng nhân viên theo từng phòng ban", "sql": "SELECT d.name, COUNT(*) FROM employees e JOIN departments d ON e.department_id = d.id GROUP BY d.name"}
{"question": "Doanh thu theo tháng trong năm 2024", "sql": "SELECT EXTRACT(MONTH FROM order_date) AS m, SUM(amount) FROM orders WHERE year = 2024 GROUP BY m"}
{"question": "Giá trị đơn hàng trung bình theo region", "sql": "SELECT region, AVG(amount) FROM orders GROUP BY region"}
{"question": "Số lượng khách hàng theo phân khúc", "sql": "SELECT segment, COUNT(*) FROM customers GROUP BY segment"}
{"question": "Sản phẩm có tồn kho dưới ngưỡng tối thiểu", "sql": "SELECT p.name, s.quantity FROM stock s JOIN products p ON s.product_id = p.id WHERE s.quantity < s.min_threshold"}
{"question": "Tổng doanh thu theo quý năm 2024", "sql": "SELECT quarter, SUM(amount) FROM orders WHERE year = 2024 GROUP BY quarter"}
{"question": "Lương trung bình theo phòng ban", "sql": "SELECT d.name, AVG(p.base_salary) FROM payroll p JOIN employees e ON p.employee_id = e.id JOIN departments d ON e.department_id = d.id GROUP BY d.name"}
{"question": "So sánh doanh thu Q4 2024 vs Q4 2023 theo region", "sql": "SELECT region, year, SUM(amount) FROM orders WHERE quarter = 4 AND year IN (2023, 2024) GROUP BY region, year"}
{"question": "Lượng hàng luân chuyển theo kho", "sql": "SELECT w.name, SUM(m.quantity) FROM stock_movements m JOIN warehouses w ON m.warehouse_id = w.id GROUP BY w.name"}
{"question": "Sản phẩm chưa bán trong 180 ngày qua", "sql": "SELECT p.name FROM products p LEFT JOIN orders o ON o.product_id = p.id AND o.order_date > CURRENT_DATE - 180 WHERE o.id IS NULL"}
```

> **Lưu ý cho người thực hiện:** kiểm tra tên cột trong `perception/info_box_all.json` trước khi chấp nhận file trên. Nếu một câu SQL tham chiếu cột không tồn tại, sửa câu SQL — đừng sửa `tables_in_sql`. Chỉ cần đúng *tên bảng*; câu SQL không cần chạy được ở tầng 1.

Chạy cả hai chiến lược:

```bash
.venv/bin/python -m eval.tier1_recall --golden eval/golden_adba.jsonl --strategy full --json eval/tier1_full.json
.venv/bin/python -m eval.tier1_recall --golden eval/golden_adba.jsonl --strategy lexical --k 8 --json eval/tier1_lexical.json
```

Expected: `full` cho recall = 1.000 với `bảng/context TB = 9.0` (trần trên, đúng theo định nghĩa). `lexical` cho recall thấp hơn nhưng `bảng/context TB` nhỏ hơn hẳn. **Ghi hai con số này vào commit message** — đây là mốc mà mọi retriever về sau phải vượt.

- [ ] **Step 7: Viết `eval/README_multischema.md`**

```markdown
# Eval đa schema — cấu hình thực tế

## Tầng 1 — recall chọn bảng

Không LLM, không GPU, không database. Tập bảng đúng parse từ SQL mẫu bằng
`eval/gold_tables.py`.

Chạy:

    python -m eval.tier1_recall --golden <file.jsonl> --strategy lexical --k 8

Định nghĩa recall: tỉ lệ câu mà context chứa **toàn bộ** tập bảng đúng.
Không có điểm từng phần.

### Mốc trên golden set ADBA (12 câu, 9 bảng)

Điền hai dòng dưới bằng đúng số in ra từ hai lệnh ở trên. Đây là mốc mà
mọi retriever về sau phải vượt.

| Chiến lược | recall | bảng/context TB |
|---|---|---|
| `full` | | 9.0 |
| `lexical`, k=8 | | |

`full` luôn cho recall 1.0 theo định nghĩa — nó là trần trên, không phải
kết quả. Con số đáng nhìn là `bảng/context TB`: mọi retriever phải giữ
recall gần 1.0 trong khi kéo con số đó xuống.

## Ba bộ dữ liệu ngoài

Xem Task 11 của plan. Bảng cấu hình nằm ở đó và được `eval/describe_dataset.py`
sinh ra, không viết tay.

## Tầng 2 và 3

Xem spec mục 6.1. Chưa hiện thực trong plan này.
```

- [ ] **Step 8: Chạy toàn bộ test**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: 125 passed

- [ ] **Step 9: Commit**

```bash
git add eval/datasets.py eval/tier1_recall.py eval/golden_adba.jsonl \
        eval/README_multischema.md eval/tier1_full.json eval/tier1_lexical.json \
        tests/unit/test_tier1_recall.py
git commit -m "feat(eval): harness recall tầng 1 + mốc lexical vs full trên golden ADBA"
```

---

## Task 8: `execute_sql` thực thi `permitted_tables`, bỏ whitelist tĩnh

**Files:**
- Modify: `graph/tools/sql_tool.py:19-60`
- Create: `tests/unit/test_sql_tool_guard.py`

**Interfaces:**
- Consumes: `eval.gold_tables.tables_in_sql`, `perception.connection_profile.ConnectionProfile`, `permitted_tables`
- Produces:
  - `graph.tools.sql_tool.execute_sql(sql: str, profile: ConnectionProfile, user: str, params=None, timeout_ms=30000) -> pd.DataFrame` — chữ ký đổi: **không** còn tham số nào nhận tập bảng
  - `graph.tools.sql_tool.TableNotPermittedError` — ngoại lệ riêng

- [ ] **Step 1: Viết test thất bại**

Tạo `tests/unit/test_sql_tool_guard.py`:

```python
import inspect

import pytest

from graph.tools import sql_tool
from perception.connection_profile import ALL_TABLES, build_profile
from tests.fixtures.mini_schema import MINI_TABLES


def _profile(grants):
    return build_profile(dsn="postgresql://u:p@h:5432/d", tables=MINI_TABLES, grants=grants)


def test_query_touching_a_forbidden_table_is_refused():
    p = _profile({"sales": frozenset({"orders"})})
    with pytest.raises(sql_tool.TableNotPermittedError, match="payroll"):
        sql_tool.assert_tables_permitted("SELECT * FROM payroll", p, "sales")


def test_query_within_permitted_tables_is_accepted():
    p = _profile({"sales": frozenset({"orders", "customers"})})
    sql_tool.assert_tables_permitted(
        "SELECT * FROM orders o JOIN customers c ON o.customer_id = c.id", p, "sales")


def test_permitted_but_not_retrieved_is_allowed():
    """spec 3.4.1: retrieval không phải cơ chế bảo mật.

    Model đoán đúng tên một bảng mà retriever bỏ sót thì câu đó phải chạy,
    vì bảng đó vốn nằm trong quyền của người dùng.
    """
    p = _profile({"admin": frozenset({ALL_TABLES})})
    sql_tool.assert_tables_permitted("SELECT * FROM payroll", p, "admin")


def test_user_without_grants_is_refused_everything():
    p = _profile({})
    with pytest.raises(sql_tool.TableNotPermittedError):
        sql_tool.assert_tables_permitted("SELECT * FROM orders", p, "nguoi_la")


def test_cte_name_is_not_mistaken_for_a_forbidden_table():
    p = _profile({"sales": frozenset({"orders"})})
    sql_tool.assert_tables_permitted(
        "WITH tmp AS (SELECT * FROM orders) SELECT * FROM tmp", p, "sales")


def test_sql_that_parses_to_no_tables_is_refused():
    """Fail closed: không đọc được bảng nào thì từ chối, không cho qua."""
    p = _profile({"admin": frozenset({ALL_TABLES})})
    with pytest.raises(sql_tool.TableNotPermittedError):
        sql_tool.assert_tables_permitted("khong phai sql", p, "admin")


def test_execute_sql_has_no_parameter_that_widens_table_access():
    """Tiêu chí 11 của spec — kiểm bằng chữ ký hàm, không bằng quy ước."""
    params = set(inspect.signature(sql_tool.execute_sql).parameters)
    forbidden = {"allowed_tables", "permitted", "permitted_tables", "tables", "allow"}
    assert not (params & forbidden), f"execute_sql không được nhận: {params & forbidden}"


def test_module_no_longer_exposes_a_static_whitelist():
    """Tiêu chí 1 của spec: không tên bảng hardcode."""
    assert not hasattr(sql_tool, "_ALLOWED_TABLES")
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `.venv/bin/python -m pytest tests/unit/test_sql_tool_guard.py -q`
Expected: FAIL — `AttributeError: module 'graph.tools.sql_tool' has no attribute 'TableNotPermittedError'`

- [ ] **Step 3: Sửa `graph/tools/sql_tool.py`**

Xóa khối `_ALLOWED_TABLES` ở dòng 27–31 (giữ `_TABLE_NAME_RE`, nó còn dùng cho `get_table_sample`). Thêm vào sau phần import:

```python
from eval.gold_tables import tables_in_sql
from perception.connection_profile import ConnectionProfile, permitted_tables


class TableNotPermittedError(PermissionError):
    """Câu SQL chạm bảng ngoài quyền của người dùng."""


def assert_tables_permitted(sql: str, profile: ConnectionProfile, user: str) -> frozenset[str]:
    """Chặn trước khi chạy. Ném TableNotPermittedError nếu vi phạm.

    Tập quyền dẫn ra TẠI ĐÂY từ profile + danh tính. Nó không phải tham số
    của hàm, và không được biến thành tham số: khi tool tách qua MCP, một
    tham số như vậy sẽ nằm trong payload lời gọi, tức là bên bị ràng buộc
    tự khai ràng buộc của mình. Xem spec mục 3.4.1.

    Fail closed: SQL không parse ra bảng nào cũng bị từ chối.
    """
    touched = tables_in_sql(sql)
    if not touched:
        raise TableNotPermittedError(
            "Không đọc được tên bảng nào từ câu SQL — từ chối để an toàn."
        )
    allowed = permitted_tables(profile, user)
    if forbidden := (touched - allowed):
        raise TableNotPermittedError(
            f"Câu hỏi này chạm dữ liệu bạn không được cấp quyền. "
            f"(nội bộ: {sorted(forbidden)})"
        )
    return touched
```

Đổi chữ ký `execute_sql`:

```python
def execute_sql(
    sql: str,
    profile: ConnectionProfile,
    user: str,
    params: tuple | None = None,
    timeout_ms: int = SQL_TIMEOUT_MS,
) -> pd.DataFrame:
    """Chạy một câu SELECT, trả DataFrame.

    Kiểm quyền trước khi mở kết nối. Không có tham số nào cho phép bên gọi
    nới rộng tập bảng được chạm (spec tiêu chí 11).
    """
    assert_tables_permitted(sql, profile, user)
    conn = psycopg2.connect(profile.dsn)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SET LOCAL statement_timeout = %s", (timeout_ms,))
            cur.execute(sql, params)
            rows = cur.fetchall()
            if not rows:
                columns = [desc[0] for desc in cur.description] if cur.description else []
                return pd.DataFrame(columns=columns)
            return pd.DataFrame(rows)
    finally:
        conn.close()
```

> **Lưu ý:** thông báo lỗi hướng tới người dùng cố ý **không nêu tên bảng** (spec mục 5.2, `permission_denied`) — tên bảng chỉ nằm trong ngoặc "nội bộ" để ghi log, và `app.py` phải cắt phần đó trước khi hiển thị. Việc cắt đó thuộc Task 10.

- [ ] **Step 4: Chạy test để xác nhận pass**

Run: `.venv/bin/python -m pytest tests/unit/test_sql_tool_guard.py -q`
Expected: PASS, 8 passed

- [ ] **Step 5: Sửa hai nơi gọi cũ**

Đã khảo sát: toàn repo chỉ có **hai** nơi gọi `execute_sql`.

**`graph/tools/sql_tool.py:72`**, bên trong `get_table_sample` — đang gọi bằng tham số vị trí `execute_sql(query, (limit,))`, sẽ nhận nhầm `(limit,)` làm `profile`. Đổi chữ ký hàm bao ngoài:

```python
def get_table_sample(
    table: str,
    profile: ConnectionProfile,
    user: str,
    limit: int = 5,
) -> pd.DataFrame:
    if not _TABLE_NAME_RE.match(table):
        raise ValueError(f"Tên bảng không hợp lệ: {table!r}")
    query = f"SELECT * FROM {table} LIMIT %s"  # noqa: S608 — tên bảng đã qua regex
    return execute_sql(query, profile=profile, user=user, params=(limit,))
```

**`graph/agents/sql_agent.py:101`** — `df = execute_sql(sql)`. Đổi thành:

```python
meta = state.get("shared_metadata", {})
df = execute_sql(sql, profile=meta["profile"], user=meta["user"])
```

`profile` và `user` được `run_graph` đặt vào `shared_metadata` ở Task 10 Step 6. Cho tới khi Task 10 xong, test của `sql_agent` phải mock `shared_metadata` — cập nhật `tests/unit/test_sql_agent.py` cho khớp.

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: mọi test pass. Nếu một test cũ fail vì chữ ký, sửa test — **đừng** khôi phục chữ ký cũ.

- [ ] **Step 6: Commit**

```bash
git add graph/tools/sql_tool.py tests/
git commit -m "feat(sql): execute_sql thực thi permitted_tables, bỏ whitelist 9 bảng hardcode"
```

---

## Task 9: Tách prompt template ba đường

**Files:**
- Modify: `prompts/text_to_sql.txt` (toàn bộ)
- Modify: `prompts/supervisor_routing.txt:59`
- Create: `tests/unit/test_prompts_are_schema_agnostic.py`

**Interfaces:**
- Consumes: (không)
- Produces: template với placeholder `{schema}`, `{few_shots}`, `{task}` — `{info_box}` không còn tồn tại

- [ ] **Step 1: Viết test thất bại**

Tạo `tests/unit/test_prompts_are_schema_agnostic.py`:

```python
from pathlib import Path

import pytest

PROMPTS = Path("prompts")
ADBA_TABLES = ["orders", "products", "customers", "payroll", "employees",
               "departments", "stock", "stock_movements", "warehouses"]
ADBA_COLUMNS = ["order_date", "base_salary", "min_threshold", "sku", "segment"]


@pytest.mark.parametrize("name", ["text_to_sql.txt", "supervisor_routing.txt"])
def test_template_names_no_adba_table(name):
    """Tiêu chí 2 của spec: template không được mô tả một schema cụ thể."""
    text = (PROMPTS / name).read_text().lower()
    found = [t for t in ADBA_TABLES if t in text]
    assert not found, f"{name} còn nhắc bảng ADBA: {found}"


@pytest.mark.parametrize("name", ["text_to_sql.txt", "supervisor_routing.txt"])
def test_template_names_no_adba_column(name):
    text = (PROMPTS / name).read_text().lower()
    found = [c for c in ADBA_COLUMNS if c in text]
    assert not found, f"{name} còn nhắc cột ADBA: {found}"


def test_text_to_sql_has_the_new_placeholders():
    text = (PROMPTS / "text_to_sql.txt").read_text()
    assert "{schema}" in text
    assert "{few_shots}" in text
    assert "{task}" in text
    assert "{info_box}" not in text


def test_schema_placeholder_sits_after_the_static_rules():
    """Phần cố định phải đứng trước phần biến thiên thì prefix cache mới dùng được.

    Xem spec mục 3.5.
    """
    text = (PROMPTS / "text_to_sql.txt").read_text()
    assert text.index("## RULES") < text.index("{schema}")
    assert text.index("{schema}") < text.index("{task}")


def test_supervisor_schema_placeholder_is_near_the_end():
    text = (PROMPTS / "supervisor_routing.txt").read_text()
    assert "{schema}" in text
    assert "{info_box}" not in text
    assert text.index("{schema}") > len(text) * 0.6
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `.venv/bin/python -m pytest tests/unit/test_prompts_are_schema_agnostic.py -q`
Expected: FAIL — cả 6 test, vì template hiện có 45/71 dòng nói về schema ADBA

- [ ] **Step 3: Viết lại `prompts/text_to_sql.txt`**

Thay toàn bộ file bằng:

```
## ROLE
You are a PostgreSQL specialist agent. Your only job is to write a single, correct,
executable SQL query that answers the given task.

## RULES
1. Return ONLY the SQL query. No explanation, no markdown fences, no comments.
2. The first token of your output MUST be `SELECT` or `WITH`.
3. Never output prose such as "You are...", "Here is...", or explanations.
4. Use ONLY tables and columns that appear in the SCHEMA section below.
   If the task needs something not in the schema, write the closest correct
   query using what IS there — never invent a table or column name.
5. For multi-step logic, prefer CTEs (WITH clauses) over nested subqueries.
6. Always use table aliases in JOINs.
7. Join only along foreign keys declared in the schema.
8. Include LIMIT 1000 on exploratory `SELECT *` queries; omit it for aggregations.
9. Columns marked GENERATED are read-only — never write to them.
10. When filtering by a part of a date, use EXTRACT(...) unless the schema
    already declares a dedicated column for that part.

## EXAMPLES
{few_shots}

## SCHEMA
{schema}

## NOW GENERATE

Task: {task}

Output:
```

> **Ba đường của template** (spec mục 3.5): mục `RULES` là kỹ năng chung, ở lại đây. Đặc thù schema (ví dụ "cột `quarter` là GENERATED", "doanh thu chỉ tính status completed") đi vào `schema.yaml` và được render vào `{schema}` dưới dạng comment — plan 3 làm việc đó. Few-shot sinh riêng cho từng khách, đổ vào `{few_shots}` — cũng plan 3. Trong plan này `{few_shots}` được thay bằng chuỗi rỗng.

- [ ] **Step 4: Sửa `prompts/supervisor_routing.txt`**

Đã khảo sát: file này chỉ có **hai** dòng nhắc tên bảng ADBA, cả hai nằm trong ví dụ phản diện về việc không được nhét SQL vào trường `task`.

Dòng 30:

```
  ✗ "SELECT region, SUM(amount) FROM orders WHERE year=2024 ..."  ← SQL thuộc về sql agent
```

đổi thành:

```
  ✗ "SELECT <cols> FROM <table> WHERE ..."  ← SQL thuộc về sql agent
```

Dòng 40:

```
  "task": "SELECT region, SUM(amount) FROM orders WHERE year=2024 GROUP BY region",
```

đổi thành:

```
  "task": "SELECT <cols>, SUM(<measure>) FROM <table> WHERE ... GROUP BY <cols>",
```

Ví dụ đối chiếu ngay dưới (dòng ~48, phần "Corrected step") cũng nhắc `orders` trong trường `task` bằng tiếng Việt tự nhiên — đó là **mô tả ý định**, không phải tên bảng trong SQL, nhưng nó vẫn gắn template vào một schema. Đổi thành một câu trung tính, ví dụ `"Tổng hợp <chỉ số> theo <chiều> cho <kỳ>"`.

Sau đó: đổi `{info_box}` ở dòng 59 thành `{schema}`, và **chuyển cả khối `## SCHEMA CONTEXT` xuống cuối file**, ngay trước phần sinh output — lý do ở spec mục 3.5.

Kiểm bằng grep trước khi chạy test:

```bash
grep -niE "orders|products|customers|payroll|employees|departments|stock|warehouses" prompts/supervisor_routing.txt
```

Expected: không ra dòng nào.

- [ ] **Step 5: Chạy test để xác nhận pass**

Run: `.venv/bin/python -m pytest tests/unit/test_prompts_are_schema_agnostic.py -q`
Expected: PASS, 7 passed (hai test có parametrize 2 giá trị)

- [ ] **Step 6: Commit**

```bash
git add prompts/text_to_sql.txt prompts/supervisor_routing.txt \
        tests/unit/test_prompts_are_schema_agnostic.py
git commit -m "refactor(prompts): tách kỹ năng chung khỏi đặc thù schema; schema xuống cuối prompt"
```

---

## Task 10: Nối `graph/` và `app.py` vào `SchemaContext`

Task cuối. Sau task này hệ thống chạy end-to-end trên đường mới.

**Files:**
- Modify: `graph/state.py:14,45-68`
- Modify: `graph/multi_agent.py:94-97`
- Modify: `graph/agents/supervisor.py:34-36,109-116`
- Modify: `graph/agents/sql_agent.py:68-69`
- Modify: `app.py:225-234`
- Create: `tests/integration/test_schema_context_wiring.py`

**Interfaces:**
- Consumes: mọi thứ từ Task 1–9
- Produces: `graph.multi_agent.run_graph(query: str, schema_context: SchemaContext, profile: ConnectionProfile, user: str) -> MultiAgentState`

- [ ] **Step 1: Viết test thất bại**

Tạo `tests/integration/test_schema_context_wiring.py`:

```python
import inspect

from graph import multi_agent
from graph.agents import sql_agent, supervisor
from graph.state import make_initial_state
from perception.connection_profile import ALL_TABLES, build_profile
from perception.schema_context import resolve_schema_context
from tests.fixtures.mini_schema import MINI_TABLES

ALL = frozenset(t.name for t in MINI_TABLES)


def _ctx():
    p = build_profile(dsn="postgresql://u:p@h:5432/d", tables=MINI_TABLES,
                      grants={"admin": frozenset({ALL_TABLES})})
    return resolve_schema_context(p, "doanh thu", permitted=ALL)


def test_run_graph_takes_a_schema_context_not_an_info_box():
    params = list(inspect.signature(multi_agent.run_graph).parameters)
    assert "schema_context" in params
    assert "info_box" not in params


def test_initial_state_carries_the_schema_context():
    state = make_initial_state("câu hỏi", _ctx())
    assert state["schema_context"].rendered_text.startswith("--") or \
           state["schema_context"].rendered_text.startswith("CREATE TABLE")
    assert "info_box" not in state


def test_sql_prompt_contains_rendered_ddl_not_json():
    prompt = sql_agent.build_system_prompt(_ctx())
    assert "CREATE TABLE orders (" in prompt
    assert '"table_name"' not in prompt  # không còn json.dumps


def test_supervisor_prompt_contains_rendered_ddl_not_json():
    prompt = supervisor.build_system_prompt(_ctx())
    assert "CREATE TABLE orders (" in prompt
    assert '"table_name"' not in prompt


def test_prompt_omits_tables_outside_the_context():
    p = build_profile(dsn="d", tables=MINI_TABLES,
                      grants={"sales": frozenset({"orders", "customers"})})
    ctx = resolve_schema_context(p, "doanh thu", permitted=frozenset({"orders", "customers"}))
    prompt = sql_agent.build_system_prompt(ctx)
    assert "payroll" not in prompt
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `.venv/bin/python -m pytest tests/integration/test_schema_context_wiring.py -q`
Expected: FAIL — `AssertionError` ở test đầu, `info_box` vẫn còn trong chữ ký

- [ ] **Step 3: Sửa `graph/state.py`**

Dòng 14: đổi `info_box: dict[str, Any]` thành `schema_context: SchemaContext`.
Dòng 45–49: đổi chữ ký và phần khởi tạo:

```python
def make_initial_state(query: str, schema_context: SchemaContext) -> MultiAgentState:
    """Trả state mới cho một câu hỏi."""
    return MultiAgentState(
        query=query,
        schema_context=schema_context,
        execution_plan=[],
        # ... phần còn lại giữ nguyên
    )
```

Thêm import: `from perception.schema_context import SchemaContext`.

- [ ] **Step 4: Sửa `graph/agents/sql_agent.py`**

Thay dòng 68–69:

```python
def build_system_prompt(schema_context: SchemaContext) -> str:
    """Render template với lát schema của lượt này.

    Few-shot để rỗng trong pha 1; plan 3 (onboarding) sẽ đổ vào.
    """
    few = "\n\n".join(
        f"Task: {fs['question']}\nOutput:\n{fs['sql']}"
        for fs in schema_context.few_shots
    )
    return (SQL_SYSTEM_PROMPT
            .replace("{few_shots}", few)
            .replace("{schema}", schema_context.rendered_text))


# trong sql_agent_node():
schema_context = state["schema_context"]
system_prompt = build_system_prompt(schema_context)
```

Bỏ `import json` nếu không còn dùng.

- [ ] **Step 5: Sửa `graph/agents/supervisor.py`**

Thay `_build_system_prompt` ở dòng 34–36 (đổi tên thành công khai để test gọi được):

```python
def build_system_prompt(schema_context: SchemaContext) -> str:
    return SUPERVISOR_SYSTEM_PROMPT.replace("{schema}", schema_context.rendered_text)
```

Dòng 109 và 116: đổi `info_box = state.get("info_box", {})` thành `schema_context = state["schema_context"]`, và `system_prompt = build_system_prompt(schema_context)`.

- [ ] **Step 6: Sửa `graph/multi_agent.py`**

```python
def run_graph(
    query: str,
    schema_context: SchemaContext,
    profile: ConnectionProfile,
    user: str,
) -> MultiAgentState:
    initial = make_initial_state(query, schema_context)
    initial["shared_metadata"] = {"profile": profile, "user": user}
    # phần còn lại giữ nguyên
```

`profile` và `user` đi qua `shared_metadata` để node SQL truyền xuống `execute_sql`. Sửa chỗ gọi `execute_sql` trong `sql_agent.py` cho khớp.

- [ ] **Step 7: Sửa `app.py:225-234`**

```python
from perception.connection_profile import ALL_TABLES, build_profile, permitted_tables
from perception.retrieval import LexicalRetriever
from perception.schema_context import resolve_schema_context
from perception.schema_model import tables_from_info_box

# ... trong nhánh xử lý câu hỏi:
info_box_path = Path(__file__).parent / "perception" / "info_box_all.json"
tables = tables_from_info_box(json.loads(info_box_path.read_text()))
profile = build_profile(
    dsn=os.environ["POSTGRES_URL"],
    tables=tables,
    grants={"local": frozenset({ALL_TABLES})},   # chưa có auth — plan 4 thay chỗ này
)
user = "local"
ctx = resolve_schema_context(
    profile, prompt, permitted_tables(profile, user),
    retriever=LexicalRetriever(tables),
)
result = run_graph(query=prompt, schema_context=ctx, profile=profile, user=user)
```

Bọc phần hiển thị lỗi để cắt chi tiết nội bộ:

```python
except TableNotPermittedError as exc:
    st.error(str(exc).split("(nội bộ:")[0].strip())
```

> **Lưu ý:** `grants={"local": {ALL_TABLES}}` là chỗ giữ tạm, có chủ ý. Điểm cắm xác thực thật thuộc plan 4 (spec mục 4). Ghi `# TODO(plan-4)` ngay tại dòng đó.

- [ ] **Step 8: Chạy test wiring**

Run: `.venv/bin/python -m pytest tests/integration/test_schema_context_wiring.py -q`
Expected: PASS, 5 passed

- [ ] **Step 9: Chạy toàn bộ test — cổng chống hồi quy**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: mọi test pass. Test cũ nào fail vì chữ ký đổi thì sửa test; **không** khôi phục đường `info_box`.

- [ ] **Step 10: Chạy thử thật một câu**

```bash
docker compose up -d postgres
set -a && . ./.env && set +a
.venv/bin/streamlit run app.py
```

Hỏi "Tổng doanh thu theo region năm 2024". Expected: trả lời được, và SQL sinh ra chỉ chạm bảng có trong context.

- [ ] **Step 11: Commit**

```bash
git add graph/ app.py tests/
git commit -m "feat(graph): nối SchemaContext vào state, supervisor, sql_agent và app"
```

---

## Task 11: Nạp ba bộ dữ liệu ngoài và ghi cấu hình thực tế

Đây là hạng mục pha 0 mà spec mục 6.2 yêu cầu, và là hạng mục duy nhất trong plan có **cổng con người**: license phải được đọc trước khi tải và trước khi commit bất cứ gì.

**Files:**
- Modify: `eval/datasets.py` (thêm `load_normalized`)
- Create: `eval/describe_dataset.py`
- Create: `tests/unit/test_describe_dataset.py`
- Create: `tests/fixtures/mini_dataset/questions.jsonl`
- Create: `tests/fixtures/mini_dataset/schemas.json`
- Modify: `eval/README_multischema.md`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `eval.datasets.EvalRecord`, `load_jsonl_generic`, `perception.schema_model.tables_from_info_box`
- Produces:
  - `eval.datasets.load_normalized(jsonl_path: Path, schemas_path: Path) -> list[EvalRecord]`
  - `eval.describe_dataset.DatasetProfile(name: str, n_databases: int, n_questions: int, avg_tables_per_db: float, max_tables_per_db: int)`
  - `eval.describe_dataset.describe(name: str, records: Sequence[EvalRecord]) -> DatasetProfile`

**Định dạng chuẩn hóa** — mỗi bộ ngoài phải được chuyển về đúng hai file này trước khi vào harness:

| File | Nội dung |
|---|---|
| `questions.jsonl` | mỗi dòng `{"question": str, "sql": str, "db_id": str}` |
| `schemas.json` | `{db_id: <dict đúng dạng info_box>}` — tái dùng `tables_from_info_box` |

- [ ] **Step 1: Cổng license — làm trước khi tải bất cứ gì**

Với từng bộ Spider, BIRD, BEAVER: mở trang gốc, đọc license, ghi lại tên license vào bảng ở Step 7. Trả lời dứt khoát hai câu:

1. Có được phép tải và dùng cho mục đích phát triển nội bộ không?
2. Có được phép **commit dữ liệu vào repo này** không?

Nếu câu 2 là không — phần lớn trường hợp sẽ là không — thì thêm vào `.gitignore`:

```
data/benchmarks/
```

và để dữ liệu ở `data/benchmarks/<tên bộ>/`, ngoài git. Ghi đường dẫn gốc vào `.env` dưới khóa `BENCHMARK_DIR`.

> **Không được bỏ qua bước này.** Commit nhầm một bộ dữ liệu có license hạn chế vào git là việc rất khó gỡ, vì nó nằm lại trong lịch sử.

- [ ] **Step 2: Viết test thất bại**

Tạo `tests/fixtures/mini_dataset/schemas.json`:

```json
{
  "db_a": {
    "tables": [
      {"table_name": "t1", "primary_key": ["id"], "row_count": 10,
       "columns": [{"name": "id", "data_type": "integer", "is_generated": false}],
       "foreign_keys": []},
      {"table_name": "t2", "primary_key": ["id"], "row_count": 20,
       "columns": [{"name": "id", "data_type": "integer", "is_generated": false}],
       "foreign_keys": []}
    ],
    "cross_domain_hints": []
  },
  "db_b": {
    "tables": [
      {"table_name": "u1", "primary_key": ["id"], "row_count": 5,
       "columns": [{"name": "id", "data_type": "integer", "is_generated": false}],
       "foreign_keys": []}
    ],
    "cross_domain_hints": []
  }
}
```

Tạo `tests/fixtures/mini_dataset/questions.jsonl`:

```json
{"question": "hỏi một", "sql": "SELECT * FROM t1", "db_id": "db_a"}
{"question": "hỏi hai", "sql": "SELECT * FROM t1 JOIN t2 ON 1=1", "db_id": "db_a"}
{"question": "hỏi ba", "sql": "SELECT * FROM u1", "db_id": "db_b"}
{"question": "hỏi bốn", "sql": "SELECT * FROM zz", "db_id": "db_khong_ton_tai"}
```

Tạo `tests/unit/test_describe_dataset.py`:

```python
from pathlib import Path

from eval.datasets import load_normalized
from eval.describe_dataset import describe

FIX = Path("tests/fixtures/mini_dataset")


def _records():
    return load_normalized(FIX / "questions.jsonl", FIX / "schemas.json")


def test_records_without_a_known_db_are_dropped():
    recs = _records()
    assert len(recs) == 3
    assert all(r.db_id in {"db_a", "db_b"} for r in recs)


def test_each_record_carries_its_own_schema():
    by_db = {r.db_id: r for r in _records()}
    assert {t.name for t in by_db["db_a"].tables} == {"t1", "t2"}
    assert {t.name for t in by_db["db_b"].tables} == {"u1"}


def test_describe_counts_databases_and_questions():
    p = describe("mini", _records())
    assert p.n_databases == 2
    assert p.n_questions == 3


def test_describe_reports_table_counts_per_database():
    p = describe("mini", _records())
    assert p.max_tables_per_db == 2
    assert p.avg_tables_per_db == 1.5


def test_describe_of_empty_input_does_not_divide_by_zero():
    p = describe("rỗng", [])
    assert p.n_databases == 0
    assert p.avg_tables_per_db == 0.0


def test_markdown_row_is_pipe_delimited():
    row = describe("mini", _records()).as_markdown_row()
    assert row.startswith("| mini |")
    assert row.count("|") == 7
```

- [ ] **Step 3: Chạy test để xác nhận fail**

Run: `.venv/bin/python -m pytest tests/unit/test_describe_dataset.py -q`
Expected: FAIL — `ImportError: cannot import name 'load_normalized' from 'eval.datasets'`

- [ ] **Step 4: Thêm `load_normalized` vào `eval/datasets.py`**

```python
def load_normalized(jsonl_path: Path, schemas_path: Path) -> list[EvalRecord]:
    """Đọc một bộ dữ liệu đã chuẩn hóa về hai file questions.jsonl + schemas.json.

    Bản ghi trỏ tới db_id không có trong schemas.json sẽ bị bỏ qua — bộ dữ
    liệu ngoài thường có câu hỏi cho database không kèm theo bản tải.
    """
    raw = json.loads(Path(schemas_path).read_text())
    tables_by_db = {db_id: tables_from_info_box(box) for db_id, box in raw.items()}
    return load_jsonl_generic(Path(jsonl_path), tables_by_db)
```

- [ ] **Step 5: Hiện thực `eval/describe_dataset.py`**

```python
"""Đo cấu hình thực tế của một bộ dữ liệu eval.

Tồn tại để bảng cấu hình trong eval/README_multischema.md được SINH RA chứ
không viết tay. Spec mục 6.2 nói rõ: các con số này không được chốt từ trí
nhớ.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from eval.datasets import EvalRecord, load_normalized


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    n_databases: int
    n_questions: int
    avg_tables_per_db: float
    max_tables_per_db: int

    def as_markdown_row(self) -> str:
        return (f"| {self.name} | {self.n_databases} | {self.n_questions} | "
                f"{self.avg_tables_per_db:.1f} | {self.max_tables_per_db} | |")


def describe(name: str, records: Sequence[EvalRecord]) -> DatasetProfile:
    sizes = {r.db_id: len(r.tables) for r in records}
    return DatasetProfile(
        name=name,
        n_databases=len(sizes),
        n_questions=len(records),
        avg_tables_per_db=(sum(sizes.values()) / len(sizes)) if sizes else 0.0,
        max_tables_per_db=max(sizes.values()) if sizes else 0,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Đo cấu hình một bộ dữ liệu eval")
    ap.add_argument("--name", required=True)
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--schemas", type=Path, required=True)
    args = ap.parse_args()

    profile = describe(args.name, load_normalized(args.questions, args.schemas))
    print(profile.as_markdown_row())


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Chạy test để xác nhận pass**

Run: `.venv/bin/python -m pytest tests/unit/test_describe_dataset.py -q`
Expected: PASS, 6 passed

- [ ] **Step 7: Chuẩn hóa từng bộ và sinh bảng cấu hình**

Với mỗi bộ đã tải ở Step 1, viết một script chuyển đổi một lần trong `eval/convert/<tên bộ>.py` xuất ra `questions.jsonl` + `schemas.json` theo định dạng ở đầu task. Bố cục thư mục gốc của ba bộ khác nhau nên script chuyển đổi phải viết sau khi nhìn thấy dữ liệu thật — **đó là lý do task này nằm sau cổng license, không phải trước.**

Rồi chạy:

```bash
for name in spider bird beaver; do
  .venv/bin/python -m eval.describe_dataset \
    --name "$name" \
    --questions "data/benchmarks/$name/questions.jsonl" \
    --schemas   "data/benchmarks/$name/schemas.json"
done
```

Dán ba dòng in ra vào `eval/README_multischema.md`, dưới tiêu đề bảng:

```markdown
| Bộ | Số DB | Số câu | Bảng/DB (TB) | Bảng/DB (max) | License |
|---|---|---|---|---|---|
```

Điền cột License bằng tên license đọc được ở Step 1.

- [ ] **Step 8: Đo recall tầng 1 trên cả ba bộ**

```bash
for name in spider bird beaver; do
  echo "=== $name ==="
  .venv/bin/python -m eval.tier1_recall \
    --normalized "data/benchmarks/$name" --strategy full
  .venv/bin/python -m eval.tier1_recall \
    --normalized "data/benchmarks/$name" --strategy lexical --k 8
done
```

Thêm cờ `--normalized <dir>` vào `eval/tier1_recall.py` — nó đọc `<dir>/questions.jsonl` và `<dir>/schemas.json` qua `load_normalized`, thay cho `--golden`:

```python
ap.add_argument("--normalized", type=Path,
                help="thư mục chứa questions.jsonl + schemas.json")
# trong main():
if args.normalized:
    records = load_normalized(args.normalized / "questions.jsonl",
                              args.normalized / "schemas.json")
else:
    records = load_adba_golden(args.golden, args.info_box)
```

Đổi `--golden` thành không bắt buộc (`required=False`) và thêm kiểm tra: phải có đúng một trong hai cờ.

Ghi kết quả vào README. **Đây là điều kiện ra của pha 1** theo spec mục 7.

- [ ] **Step 9: Chạy toàn bộ test**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: mọi test pass

- [ ] **Step 10: Commit**

```bash
git add eval/datasets.py eval/describe_dataset.py eval/convert/ \
        eval/README_multischema.md tests/unit/test_describe_dataset.py \
        tests/fixtures/mini_dataset/ .gitignore
git commit -m "feat(eval): nạp Spider/BIRD/BEAVER chuẩn hóa + sinh bảng cấu hình thực tế"
```

> **Kiểm trước khi commit:** `git status --short` không được liệt kê file nào dưới `data/benchmarks/`.

---

## Điều kiện ra của plan này

Đối chiếu với điều kiện ra pha 1 trong spec mục 7:

- [ ] `pytest tests/ -q` toàn bộ pass, không hồi quy so với 59 test ban đầu
- [ ] Recall tầng 1 đo được và ghi lại cho cả `full` lẫn `lexical` trên golden set ADBA (Task 7) **và trên cả ba benchmark ngoài** (Task 11)
- [ ] `grep -rn "_ALLOWED_TABLES" graph/` không ra kết quả
- [ ] `grep -rniE "orders|payroll|products" prompts/` không ra kết quả
- [ ] `render_schema()` ≤ 700 B/bảng, đo trên `info_box_all.json`
- [ ] `app.py` trả lời được một câu hỏi thật qua đường mới
- [ ] `eval/README_multischema.md` có bảng cấu hình ba bộ với số **đo được**, không phải số nhớ được

**Một tiêu chí của spec chỉ đạt được một phần ở plan này.** Tiêu chí 4 (mục 9 spec) đòi recall ≥ 95%. Plan này *đo* được recall nhưng không hứa *đạt* — `LexicalRetriever` là mốc so sánh, và nhiều khả năng nó không tới 95% trên schema doanh nghiệp. Đưa recall lên ngưỡng là việc của `EmbeddingRetriever` cộng chú giải ngữ nghĩa, tức plan 3. Điều kiện ra ở đây là **có con số**, không phải **con số đạt ngưỡng**.

**Rủi ro lớn nhất của plan này:** Task 11 Step 7 yêu cầu viết script chuyển đổi cho ba bộ dữ liệu mà bố cục thư mục chỉ biết được sau khi tải. Đó là phần duy nhất không ước lượng được trước. Nếu một bộ hóa ra quá tốn công, hãy làm Spider trước (schema nhỏ, cấu trúc đơn giản nhất), lấy con số, rồi quyết có làm tiếp hai bộ kia không — đừng để cả plan kẹt ở đó.

**Chưa thuộc plan này, đừng làm:** `EmbeddingRetriever`, `schema.yaml`, sinh chú giải, few-shot theo khách, `refresh_profile`, kiểm `profile_stale` lúc khởi động, train lại LoRA, đóng gói on-prem. Mỗi thứ có plan riêng.
