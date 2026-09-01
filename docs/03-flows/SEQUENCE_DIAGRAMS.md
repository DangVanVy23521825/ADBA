# User Flow & Sequence Diagrams — ADBA

> Ai gọi ai, theo thứ tự nào, và trạng thái đổi ra sao ở mỗi bước.

---

## 1. User flow tổng quát

```mermaid
flowchart TD
    A([Người dùng mở Streamlit]) --> B["Nhập câu hỏi vào chat"]
    B --> C{"info_box<br/>có sẵn?"}
    C -->|Không| C1["Cảnh báo 'No info_box found'<br/>chạy với schema rỗng"]
    C -->|Có| D["run_graph(query, info_box)"]
    C1 --> D
    D --> E["Spinner: Planning and executing agents…"]
    E --> F{"status"}
    F -->|success / running| G["Hiển thị:<br/>Execution Plan → Agent Trace →<br/>Bảng → Biểu đồ → Insight card"]
    F -->|failed| H["st.error + thông điệp lỗi"]
    G --> I["Người dùng đọc SQL đã chạy để tự kiểm"]
    I --> J{"Hỏi tiếp?"}
    J -->|Có| B
    J -->|Không| K([Kết thúc])
    H --> B
```

Chi tiết đáng chú ý: **mỗi câu hỏi là một lần chạy graph độc lập**. `st.session_state.messages`
chỉ lưu lịch sử để hiển thị, không được nạp vào prompt. Hội thoại nhiều lượt có nhớ ngữ cảnh
nằm ở backlog B-03.

## 2. Truy vấn đơn giản — chỉ SQL + Insight

Ví dụ: *"Tổng doanh thu năm 2024 là bao nhiêu?"*

```mermaid
sequenceDiagram
    autonumber
    actor U as Người dùng
    participant UI as Streamlit (app.py)
    participant G as LangGraph
    participant SUP as Supervisor
    participant MC as ModelClient
    participant OL as Ollama
    participant SQLA as SQL Agent
    participant PG as PostgreSQL
    participant INS as Insight Agent

    U->>UI: "Tổng doanh thu năm 2024?"
    UI->>G: run_graph(query, info_box)
    G->>SUP: supervisor_node(state)
    SUP->>MC: invoke_json(prompt + info_box)
    MC->>OL: POST /api/chat (temp 0.1, 800 tokens)
    OL-->>MC: JSON plan (có thể bọc ```json)
    MC-->>SUP: dict đã bóc fence
    SUP->>SUP: ExecutionPlan.model_validate()<br/>kiểm 7 bất biến DAG
    SUP-->>G: state.execution_plan = [sql, insight]

    G->>G: route_next_agent() → "sql"
    G->>SQLA: sql_agent_node(state)
    SQLA->>MC: invoke(text_to_sql + info_box + task)
    MC->>OL: POST /api/chat (temp 0.0)
    OL-->>SQLA: "```sql SELECT SUM(amount)…```"
    SQLA->>SQLA: _extract_sql() bóc fence
    SQLA->>PG: SET LOCAL statement_timeout; SELECT …
    PG-->>SQLA: rows
    SQLA->>PG: EXPLAIN <sql>
    PG-->>SQLA: cost estimate
    SQLA-->>G: shared_dataframe + sql_result + trace

    G->>G: route_next_agent() → "insight"
    G->>INS: insight_agent_node(state)
    INS->>MC: invoke_json(insight prompt + stats)
    MC->>OL: POST /api/chat (temp 0.2)
    OL-->>INS: JSON insight
    INS->>INS: InsightOutput.model_validate()
    INS-->>G: state.insight, status="success"

    G-->>UI: MultiAgentState
    UI->>U: Plan · Trace · Bảng · Insight card
```

## 3. Truy vấn phức tạp — bốn agent, có phụ thuộc

Ví dụ: *"So sánh doanh thu theo vùng 2024 với 2023, vẽ biểu đồ và cho biết vùng nào bất thường"*

```mermaid
sequenceDiagram
    autonumber
    participant G as LangGraph router
    participant SUP as Supervisor
    participant SQLA as SQL Agent
    participant PYA as Python Agent
    participant VZA as Viz Agent
    participant INS as Insight Agent
    participant ST as MultiAgentState

    SUP->>ST: plan = sql → python → viz → insight<br/>dependency_graph + ready_agents
    Note over G: route_next_agent() giải DAG sau MỖI node,<br/>không chạy theo danh sách cứng

    G->>SQLA: bước 1 (depends_on: [])
    SQLA->>ST: shared_dataframe (records + dtypes + shape)
    G->>G: resolve_ready_agents(plan, completed=[supervisor, sql])

    G->>PYA: bước 2 (depends_on: [sql])
    PYA->>ST: df_from_state() → chạy code trong sandbox → df_to_state()
    Note over PYA: sandbox: process riêng, whitelist import,<br/>timeout PANDAS_EXEC_TIMEOUT_SECONDS

    G->>VZA: bước 3 (depends_on: [python])
    VZA->>ST: chart_b64 + chart_metadata

    G->>INS: bước 4 (depends_on: [python, viz])
    INS->>ST: insight (finding · evidence · anomaly · action · confidence)
    Note over G: cạnh insight → END là cạnh cứng:<br/>insight luôn kết thúc pipeline
```

**Điểm dễ hiểu nhầm**: bộ định tuyến không "chạy tuần tự theo plan". Sau mỗi node nó giải lại
toàn bộ DAG (`resolve_ready_agents`) và tính tập agent *sẵn sàng*. Hiện tại nó nhận agent đầu tiên
trong tập đó, nhưng `state["ready_agents"]` và `shared_metadata["parallel_ready"]` đã phơi sẵn
toàn bộ tập để sau này dispatch song song mà không phải viết lại bộ định tuyến.

## 4. Vòng self-repair — SQL lỗi

Đây là luồng làm ADBA khác một wrapper text-to-SQL. Có **hai tầng sửa lỗi** lồng nhau.

```mermaid
sequenceDiagram
    autonumber
    participant G as Router
    participant SQLA as SQL Agent
    participant PG as PostgreSQL
    participant RF as Reflector
    participant MC as ModelClient

    rect rgb(245, 245, 245)
    Note over SQLA,PG: Tầng 1 — retry nội bộ trong agent (MAX_RETRIES = 3)
    SQLA->>PG: SELECT … EXTRACT(MONTH FROM month) …
    PG-->>SQLA: ERROR: column "month" does not exist
    SQLA->>SQLA: error_context = thông điệp lỗi
    SQLA->>MC: prompt + "Previous attempt failed with error: …"
    MC-->>SQLA: SQL đã sửa
    SQLA->>PG: thử lại (lần 2, 3)
    PG-->>SQLA: vẫn lỗi
    end

    SQLA->>G: agent_outputs.sql.status = "error"<br/>agent_error_counts.sql += 1<br/>last_error = {agent: sql, …}

    rect rgb(240, 246, 252)
    Note over G,RF: Tầng 2 — Reflector chẩn đoán
    G->>G: route_next_agent(): thấy status error<br/>kiểm reflect_passes_per_agent < 8<br/>và error_count > snapshot
    G->>RF: reflector_agent_node(state)
    RF->>MC: prompt chẩn đoán (failed_agent, task, error, số lần đã thử)
    MC-->>RF: {root_cause, error_category, fix_strategy, corrected_context}
    RF->>RF: ghi snapshot error_count + tăng passes
    RF->>G: shared_metadata.reflector_diagnosis
    G->>G: route_reflector_return() → "sql"
    end

    G->>SQLA: chạy lại, prompt kèm "[HINT from error diagnosis] …"
    SQLA->>PG: SELECT … EXTRACT(MONTH FROM order_date) …
    PG-->>SQLA: rows ✓
    SQLA->>G: status="running", tiếp bước sau
```

### Cơ chế chống lặp vô hạn

Điều kiện gọi Reflector không chỉ là "agent lỗi" — nếu thế thì `sql ↔ reflector` sẽ quay mãi.

```mermaid
flowchart TD
    A["Agent X lỗi"] --> B{"passes[X] < 8?"}
    B -->|Không| E["Chặn X, bỏ qua bước<br/>ghi log cảnh báo"]
    B -->|Có| C{"error_count[X] ><br/>snapshot lúc reflect trước?"}
    C -->|"Không (stale)"| E
    C -->|"Có (lỗi mới)"| D["Gọi Reflector"]
    D --> F["Reflector ghi snapshot[X] = error_count[X]<br/>passes[X] += 1"]
    F --> G["Quay lại X với corrected_context"]
    G --> A
    E --> H["route tiếp agent khác, hoặc END"]
```

Hai bộ đếm nằm trong `shared_metadata`:

| Khoá | Ý nghĩa |
|---|---|
| `reflect_passes_per_agent` | Số lần Reflector đã chẩn đoán cho agent này — trần cứng `MAX_REFLECTOR_PASSES_PER_AGENT = 8` |
| `reflect_error_snapshot` | `error_count` tại lần chẩn đoán gần nhất — nếu không tăng thêm nghĩa là thử lại **không tạo ra lỗi mới**, tức là bế tắc |

Điều kiện thứ hai tinh tế hơn vẻ ngoài: nó phân biệt *"lỗi mới, đáng chẩn đoán lại"* với
*"vẫn kẹt ở chỗ cũ"*, thay vì chỉ đếm số lần.

## 5. Vòng đời state trong một lần chạy

```mermaid
stateDiagram-v2
    [*] --> planning: make_initial_state()
    planning --> running: plan validate OK
    planning --> failed: 3 lần thử đều sai contract
    running --> running: agent xong → router chọn agent tiếp
    running --> running: agent lỗi → reflector → thử lại
    running --> success: insight_node xong
    running --> failed: không còn agent chạy được
    success --> [*]
    failed --> [*]
```

| Trạng thái | Nghĩa | UI hiển thị |
|---|---|---|
| `planning` | Supervisor đang lập kế hoạch | Spinner |
| `running` | Đang thực thi các bước | Spinner |
| `success` | Insight đã sinh và validate xong | Đầy đủ các mục |
| `failed` | Không lập được plan, hoặc không còn agent nào chạy được | `st.error` + trace để chẩn đoán |

> **Khoảng trống đã biết**: chưa có trạng thái `partial` — truy vấn hết giờ giữa chừng hiện
> không có đường trả về kết quả trung gian. Node `finalize` của M3.2 sẽ bổ sung.

## 6. Luồng chuẩn bị dữ liệu (trước khi phục vụ)

```mermaid
sequenceDiagram
    autonumber
    actor Dev as Kỹ sư
    participant DC as Docker Compose
    participant PG as PostgreSQL
    participant SEED as seed_data.py
    participant EX as extract_info_box.py

    Dev->>DC: docker compose up -d postgres
    DC->>PG: khởi động + healthcheck pg_isready
    Dev->>PG: ./scripts/apply_schemas_docker.sh<br/>(sales → inventory → hr)
    Dev->>SEED: python data/seed/seed_data.py
    SEED->>PG: ~64.000 dòng, SEED=42<br/>+ cấy 15 bất thường có chủ ý
    Dev->>EX: python perception/extract_info_box.py
    EX->>PG: information_schema + 3 dòng mẫu/bảng
    EX->>EX: bơm KNOWN_ENUMS (CHECK constraint)
    EX-->>Dev: info_box_{sales,inventory,hr,all}.json
    Note over Dev,EX: Bỏ bước này sau khi đổi schema<br/>= agent viết SQL theo schema cũ
```

---

<!-- AUTO:begin id=stamp -->

| Trường | Giá trị |
|---|---|
| Commit nguồn gần nhất | `05b24f0` — fix(sql): fail closed, statement_timeout 30s→10s, trần 50k dòng |
| Tác giả | Đặng Văn Vỹ |
| Ngày commit | 2026-09-01 |
| Số commit nguồn | 102 |
| Sinh bởi | `scripts/update_docs.py` (hook `post-commit`) |

<!-- AUTO:end id=stamp -->
