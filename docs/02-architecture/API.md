# API & Contract — ADBA

> **Trạng thái**: ADBA v1 chưa phơi HTTP API công khai. Giao diện duy nhất là Streamlit
> (`app.py`), gọi thẳng vào Python API nội bộ. Tài liệu này mô tả (1) contract nội bộ đang
> dùng thật, (2) contract JSON giữa hệ thống và LLM, (3) đặc tả HTTP đã thiết kế cho backlog
> **B-02**, đánh dấu rõ là *kế hoạch*.

---

## 1. Contract nội bộ — Python API

### 1.1 Điểm vào duy nhất

```python
from graph.multi_agent import run_graph

result: MultiAgentState = run_graph(
    query="So sánh doanh thu theo region năm 2024",
    info_box=json.load(open("perception/info_box_all.json")),
)
```

| | |
|---|---|
| **Module** | `graph/multi_agent.py` |
| **Chữ ký** | `run_graph(query: str, info_box: dict) -> MultiAgentState` |
| **Đồng bộ** | Có — chặn cho tới khi graph chạy xong hoặc dừng |
| **Ngoại lệ** | Hầu hết lỗi được nuốt vào `state["last_error"]` và `status="failed"` thay vì ném ra; lỗi hạ tầng (không kết nối được Ollama ngay từ đầu) vẫn ném |
| **Hàm liên quan** | `build_multi_agent_graph() -> CompiledStateGraph` khi cần tự điều khiển vòng chạy |

### 1.2 `MultiAgentState` — cấu trúc trả về

`TypedDict` phẳng, JSON-serialize được. Định nghĩa tại `graph/state.py`.

<!-- AUTO:begin id=state-fields -->

| Trường | Kiểu | Bắt buộc | Nhóm |
|---|---|---|---|
| `query` | `str` | bắt buộc | Input |
| `schema_context` | `SchemaContext` | bắt buộc | Input |
| `query_id` | `str` | bắt buộc | Ngân sách |
| `deadline_ts` | `float` | bắt buộc | Ngân sách |
| `llm_calls_used` | `int` | bắt buộc | Ngân sách |
| `degradation_reason` | `list[str]` | bắt buộc | Ngân sách |
| `execution_plan` | `list[dict[str, Any]]` | bắt buộc | Supervisor |
| `dependency_graph` | `NotRequired[dict[str, list[str]]]` | tuỳ chọn | Supervisor |
| `ready_agents` | `NotRequired[list[str]]` | tuỳ chọn | Supervisor |
| `current_agent` | `str` | bắt buộc | Supervisor |
| `completed_agents` | `list[str]` | bắt buộc | Supervisor |
| `agent_outputs` | `dict[str, dict[str, Any]]` | bắt buộc | Inter-agent Communication |
| `shared_dataframe` | `NotRequired[dict[str, Any] \| None]` | tuỳ chọn | Inter-agent Communication |
| `shared_metadata` | `dict[str, Any]` | bắt buộc | Inter-agent Communication |
| `error_count` | `int` | bắt buộc | Error Handling |
| `agent_error_counts` | `dict[str, int]` | bắt buộc | Error Handling |
| `last_error` | `NotRequired[dict[str, Any] \| None]` | tuỳ chọn | Error Handling |
| `sql_result` | `NotRequired[dict[str, Any] \| None]` | tuỳ chọn | Final Outputs |
| `python_result` | `NotRequired[dict[str, Any] \| None]` | tuỳ chọn | Final Outputs |
| `chart_b64` | `NotRequired[str \| None]` | tuỳ chọn | Final Outputs |
| `chart_metadata` | `NotRequired[dict[str, Any] \| None]` | tuỳ chọn | Final Outputs |
| `insight` | `NotRequired[dict[str, Any] \| None]` | tuỳ chọn | Final Outputs |
| `action_trace` | `list[dict[str, Any]]` | bắt buộc | Trace |
| `status` | `str` | bắt buộc | Trace |

<!-- AUTO:end id=state-fields -->

**Các trường hay dùng nhất khi tích hợp:**

| Trường | Dùng để |
|---|---|
| `status` | `planning` → `running` → `success` \| `failed` |
| `insight` | Kết quả nghiệp vụ chính (validate được bằng `InsightOutput`) |
| `sql_result` | `{"sql": str, "row_count": int}` — câu SQL đã chạy, để người dùng kiểm chứng |
| `shared_dataframe` | `{"records": [...], "columns": [...], "dtypes": {...}, "shape": [n, m]}` |
| `chart_b64` | PNG base64 (chưa có tiền tố `data:`) |
| `action_trace` | Danh sách bước: `{agent, action, observation, status, timestamp}` |
| `execution_plan` | Kế hoạch đã validate, để hiển thị hoặc audit |
| `error_count`, `agent_error_counts`, `last_error` | Chẩn đoán khi thất bại |

### 1.3 Tool — hợp đồng của tầng thực thi

| Hàm | Module | Chữ ký | Ranh giới an toàn |
|---|---|---|---|
| `execute_sql` | `graph/tools/sql_tool.py` | `(sql, params=None, timeout_ms=SQL_TIMEOUT_MS) -> DataFrame` | `SET LOCAL statement_timeout`; hỗ trợ truy vấn tham số hoá |
| `explain_query_plan` | `graph/tools/sql_tool.py` | `(sql) -> dict` | Chạy `EXPLAIN` — kiểm tra tính hợp lệ trước khi tin kết quả |
| `get_table_sample` | `graph/tools/sql_tool.py` | `(table, limit=5) -> DataFrame` | Tên bảng phải nằm trong whitelist; quote bằng `psycopg2.sql.Identifier` |
| `run_pandas_safe` | `graph/tools/python_tool.py` | `(code, df, timeout=…) -> DataFrame` | Process riêng, whitelist builtins + import (`pandas`, `numpy`, `scipy`), timeout cưỡng chế |
| `run_dict_safe` | `graph/tools/python_tool.py` | `(code, local_vars, …) -> dict` | Như trên, dùng cho code viz |
| `generate_chart` | `graph/tools/viz_tool.py` | `(df, chart_type=None, title, x_col, y_col, group_col) -> {"chart_b64", "chart_type", "metadata"}` | Backend `Agg`, không đụng filesystem |
| `detect_anomaly` | `graph/tools/insight_tools.py` | `(df, …) -> dict` | Thuần tính toán |
| `compare_periods` | `graph/tools/insight_tools.py` | `(df, …) -> dict` | Thuần tính toán |

### 1.4 `ModelClient`

```python
from model.model_client import ModelClient

client = ModelClient(agent_type="sql")     # nạp temperature/max_tokens/timeout theo agent
text  = client.invoke(system_prompt=..., user_prompt=...)       # -> str
data  = client.invoke_json(system_prompt=..., user_prompt=...)  # -> dict, đã bóc ```json
```

| Hành vi | Chi tiết |
|---|---|
| Local-first | Gọi Ollama tại `OLLAMA_BASE_URL`, model `PRIMARY_MODEL` |
| Retry | `MODEL_MAX_RETRIES` (mặc định 3) với backoff |
| Fallback | OpenAI khi Ollama lỗi, `ADBA_DEPLOYMENT=hybrid`, và `ENABLE_OPENAI_FALLBACK` bật. Mặc định `onprem` — không có gì rời khỏi mạng khách |
| Parse JSON | `safe_parse_json()` bóc markdown fence; ném `ValueError` nếu không tìm được object JSON — **không** trả dict rỗng để lỗi im lặng |

## 2. Contract JSON giữa hệ thống và LLM

Đây là ranh giới hay hỏng nhất — model là thành phần duy nhất trong hệ có thể trả về bất cứ thứ gì.
Cả hai contract đều được Pydantic cưỡng chế; sai contract thì **retry**, không đi tiếp.

### 2.1 `ExecutionPlan` — output của Supervisor

`schemas/plan_schema.py`

```json
{
  "plan_summary": "So sánh doanh thu theo region năm 2024 và nêu vùng bất thường",
  "steps": [
    {"step": 1, "agent": "sql",     "task": "Lấy tổng doanh thu theo region năm 2024",
     "depends_on": [],              "skill_type": "text-to-sql"},
    {"step": 2, "agent": "python",  "task": "Tính tỷ trọng và tăng trưởng so với 2023",
     "depends_on": ["sql"],         "skill_type": "data-analysis"},
    {"step": 3, "agent": "viz",     "task": "Vẽ bar chart doanh thu theo region",
     "depends_on": ["python"],      "skill_type": "visualization"},
    {"step": 4, "agent": "insight", "task": "Sinh insight về vùng tụt doanh thu",
     "depends_on": ["python", "viz"], "skill_type": "insight-generation"}
  ]
}
```

**Bảy bất biến được kiểm tra** (vi phạm nào cũng làm plan bị từ chối):

1. Không lặp agent trong một plan.
2. `step` đánh số 1..N liên tục, không hụt.
3. `depends_on` chỉ trỏ tới agent có mặt trong plan.
4. Không phụ thuộc tiến — bước N không phụ thuộc agent ở bước > N.
5. Không có chu trình (kiểm bằng thuật toán Kahn).
6. `insight` nếu có thì luôn là bước cuối.
7. `skill_type` phải khớp agent theo `AGENT_SKILL_MAP`.

Supervisor thử tối đa `MAX_SUPERVISOR_RETRIES = 3` lần; hết lượt thì `status="failed"`.

### 2.2 `InsightOutput` — output của Insight Agent

`schemas/insight_schema.py`

```json
{
  "finding": "Doanh thu Miền Trung quý 4/2024 giảm 18,5% so với quý 3.",
  "evidence": [
    "Doanh thu Q4/2024: 12,4 tỷ so với Q3/2024: 15,2 tỷ",
    "Nhóm Electronics đóng góp 72% mức giảm, tương đương 2,0 tỷ"
  ],
  "anomaly": {"type": "negative_outlier", "detail": "Mức giảm vượt 2,3 lần độ lệch chuẩn 8 quý gần nhất"},
  "action": "Kiểm tra tồn kho và chính sách giá nhóm Electronics tại Miền Trung trước ngày 15/01.",
  "confidence": "high"
}
```

| Trường | Ràng buộc cưỡng chế |
|---|---|
| `finding` | Đúng **một câu**, phải chứa **ít nhất một con số** |
| `evidence` | **2–3 mục**, mỗi mục phải chứa ít nhất một con số |
| `anomaly.type` | `positive_outlier` \| `negative_outlier` \| `none`; `detail` bắt buộc khi khác `none`, và bắt buộc `null` khi bằng `none` |
| `action` | Đúng một câu, **bắt đầu bằng động từ hành động** (nhận cả tiếng Việt và tiếng Anh: `kiểm`, `tăng`, `giảm`, `review`, `investigate`, …) |
| `confidence` | `high` \| `medium` \| `low`; bất thường + `low` chỉ ghi cảnh báo, không từ chối |

Lý do ràng buộc gắt: **"số" là thứ ngăn insight trôi thành lời sáo rỗng**, và ép một câu ngăn
model viết đoạn văn mô tả lại bảng dữ liệu.

### 2.3 Reflector diagnosis

Không có model Pydantic riêng; contract là bốn khoá bắt buộc (eval kiểm `reflector_json_rate`):

```json
{
  "root_cause": "Cột 'month' không tồn tại trong bảng orders",
  "error_category": "schema_mismatch",
  "fix_strategy": "Dùng EXTRACT(MONTH FROM order_date) thay cho cột month",
  "corrected_context": "Lấy doanh thu theo tháng năm 2024, dùng EXTRACT(MONTH FROM order_date)"
}
```

`error_category` ∈ `sql_syntax` · `sql_logic` · `python_runtime` · `python_logic` ·
`data_quality` · `schema_mismatch` · `chart_error`.

`corrected_context` được chèn vào prompt lần thử tiếp theo dưới dạng
`[HINT from error diagnosis] …` — xem `graph/agents/sql_agent.py`.

## 3. HTTP API *(kế hoạch — backlog B-02, chưa hiện thực)*

Khi cần tích hợp ADBA vào hệ thống khác (BI portal, Slack bot), lớp HTTP dự kiến như sau.
Đặc tả này là **hợp đồng thiết kế**, chưa có code.

### 3.1 Endpoint

| Method | Path | Mục đích |
|---|---|---|
| `POST` | `/v1/query` | Chạy một truy vấn, trả kết quả đầy đủ (đồng bộ) |
| `POST` | `/v1/query/stream` | Như trên, nhưng stream từng bước agent (SSE) |
| `GET` | `/v1/runs/{run_id}` | Lấy lại kết quả một lần chạy |
| `GET` | `/v1/schema` | Trả `info_box` đang dùng (để client gợi ý câu hỏi) |
| `GET` | `/healthz` | Liveness — không chạm DB |
| `GET` | `/readyz` | Readiness — kiểm tra Postgres + Ollama |

### 3.2 `POST /v1/query`

**Request**

```json
{
  "query": "Doanh thu theo vùng năm 2024?",
  "domain": "sales",
  "options": {"include_chart": true, "max_seconds": 45}
}
```

**Response `200`**

```json
{
  "run_id": "01JB9…",
  "status": "success",
  "insight": { "...": "InsightOutput" },
  "sql": "SELECT region, SUM(amount) …",
  "row_count": 4,
  "dataframe": {"columns": ["region", "revenue"], "records": [{"region": "Miền Bắc", "revenue": 1240000000}]},
  "chart": {"mime": "image/png", "base64": "iVBORw0…", "chart_type": "bar"},
  "trace": [{"agent": "supervisor", "action": "parse_intent", "status": "ok", "duration_s": 3.1}],
  "elapsed_s": 21.4
}
```

**Mã trạng thái**

| Mã | Khi nào | Body |
|---|---|---|
| `200` | Chạy xong — kể cả `status: "partial"` | Như trên |
| `400` | `query` rỗng hoặc `options` sai kiểu | `{"error": {"code": "invalid_request", "message": …}}` |
| `401` | Thiếu / sai API key | `{"error": {"code": "unauthorized"}}` |
| `422` | Câu hỏi không lập được plan hợp lệ sau 3 lần thử | `{"error": {"code": "planning_failed", "detail": …}}` |
| `429` | Vượt rate limit | Kèm header `Retry-After` |
| `500` | Lỗi không lường trước | `{"error": {"code": "internal", "run_id": …}}` |
| `503` | Ollama hoặc PostgreSQL không sẵn sàng | `{"error": {"code": "dependency_unavailable", "dependency": "ollama"}}` |
| `504` | Vượt `max_seconds` mà không có kết quả một phần nào | `{"error": {"code": "deadline_exceeded"}}` |

**Nguyên tắc**: truy vấn chạy hết ngân sách thời gian nhưng vẫn có kết quả trung gian thì trả
`200` với `status: "partial"`, **không** trả `504`. Kết quả một phần vẫn có ích với người dùng —
đây chính là node `finalize` của M3.2.

### 3.3 Xác thực

| Cơ chế | Dùng cho |
|---|---|
| `Authorization: Bearer <api-key>` | Máy-tới-máy trong mạng nội bộ |
| mTLS | Triển khai on-prem yêu cầu cao |
| OIDC / SSO | Khi có phân quyền theo người dùng — kéo theo phân quyền theo hàng ở tầng SQL, thuộc M4 |

### 3.4 Sinh OpenAPI

FastAPI sinh `/openapi.json` và `/docs` (Swagger UI) tự động từ Pydantic model. Vì
`ExecutionPlan` và `InsightOutput` **đã là** Pydantic, phần schema của đặc tả không phải viết tay —
chỉ cần thêm model cho request/response envelope.

Khi lớp HTTP có thật, thêm vào `scripts/update_docs.py` một khối AUTO xuất bảng endpoint từ
`app.openapi()` để tài liệu này không bao giờ lệch với code.

## 4. Tương thích ngược

| Contract | Chính sách |
|---|---|
| `MultiAgentState` | Thêm trường `NotRequired` là thay đổi tương thích; đổi/xoá trường có sẵn là **MAJOR** |
| `ExecutionPlan`, `InsightOutput` | Đổi ràng buộc theo hướng chặt hơn là **MAJOR** — dữ liệu huấn luyện cũ có thể không còn hợp lệ |
| Tên agent (`sql`, `python`, `viz`, `insight`) | Cố định; xuất hiện trong dataset huấn luyện, đổi tên là huấn luyện lại |
| Đường dẫn `prompts/*.txt` | Được code đọc theo tên cố định — đổi tên file là thay đổi phá vỡ |

---

<!-- AUTO:begin id=stamp -->

| Trường | Giá trị |
|---|---|
| Commit nguồn gần nhất | `71b19c6` — feat(graph): mọi đường ra đi qua finalize, không còn END trần |
| Tác giả | Đặng Văn Vỹ |
| Ngày commit | 2026-09-03 |
| Số commit nguồn | 112 |
| Sinh bởi | `scripts/update_docs.py` (hook `post-commit`) |

<!-- AUTO:end id=stamp -->
