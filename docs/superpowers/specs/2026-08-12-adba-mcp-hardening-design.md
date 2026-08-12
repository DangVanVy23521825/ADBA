# ADBA — Hardening cho production + tách tool qua MCP

**Ngày:** 2026-08-12
**Trạng thái:** Design đã duyệt, chờ lập implementation plan
**Phạm vi:** `graph/`, `model/`, `eval/`, `docker-compose.yml`, thêm package `mcp_server/`

---

## 1. Bối cảnh & mục tiêu

ADBA hiện là pipeline LangGraph 5 node (Supervisor → SQL / Python / Viz / Reflector → Insight) chạy trên
Qwen2.5-Coder-7B fine-tune bằng LoRA. Mục tiêu của đợt làm này là **đưa hệ thống lên môi trường
production phục vụ người dùng nội bộ**.

Hàm mục tiêu, theo thứ tự: **độ tin cậy → latency → chi phí vận hành**. "Kiến trúc hiện đại" không
phải mục tiêu và chỉ được theo đuổi khi nó phục vụ ba thứ trên.

### Ràng buộc nền

| Ràng buộc | Giá trị | Nguồn |
|---|---|---|
| Serving | GPU server on-prem/thuê, vLLM + merged model | Quyết định của chủ dự án |
| SLO latency | 30–60s mỗi query (thiết kế nhắm **45s**) | Quyết định của chủ dự án |
| Model | `dangvanvy/adba-qwen-merged` (Qwen2.5-Coder-7B + LoRA) | `README.md` |
| Context window | `OLLAMA_NUM_CTX = 4096` | `model/model_config.py:13` |

### Số liệu xuất phát

Từ `eval/baseline_results.json` và `training/finetuned_checkpoint50_results.json` (n=98):

| Metric | Base | + LoRA ckpt-50 |
|---|---|---|
| SQL execution accuracy | 84.4% | **96.9%** |
| Python syntax rate | 96.7% | **100%** |
| Supervisor JSON rate | 94.4% | **100%** |
| Insight JSON rate | 92.9% | 92.9% |
| Overall JSON valid | 94.4% | **97.2%** |
| Avg latency / call | 75.8s (Ollama q5) | 12.4s (PEFT/CUDA) |

**Hệ quả thiết kế:** fine-tune là tài sản có giá trị đo được và nó gắn với format prompt→text theo
5 skill (`text-to-sql`, `data-analysis`, `supervisor-routing`, `insight-generation`,
`error-reflection`). Mọi thay đổi kiến trúc buộc model sinh ra định dạng khác (ví dụ tool-call
tokens) đều làm mất hiệu lực fine-tune này. Thiết kế dưới đây **không đụng tới bất kỳ prompt hay
định dạng output nào**.

Với 12.4s/call và pipeline 5 node ≈ 62s — đã sát ngân sách 45–60s ngay cả trước khi tối ưu, và vLLM
(batching + KV cache) sẽ kéo xuống thêm. **Kiến trúc 5 node hiện tại nằm trong ngân sách latency.**

---

## 2. Ba lỗ hổng production được phát hiện khi khảo sát

Đây là những gì thực sự chặn việc deploy, không phải kiến trúc.

### L1 — Không có ngân sách thời gian toàn cục

`MAX_REFLECTOR_PASSES_PER_AGENT = 8` (`graph/agents/supervisor.py:23`) nhân với 3 retry trong node
(`graph/agents/sql_agent.py:24`) cho **32 LLM call cho riêng node SQL**. Bốn specialist agent →
worst case ~128 call ≈ 25 phút. Không có cutoff ở bất kỳ đâu trong graph.

### L2 — Không có chặn mutation trên đường production

`eval/eval_runner.py:172` có `_SQL_MUTATION_RE`, nhưng `graph/tools/sql_tool.py:39` `execute_sql()`
không dùng nó. `_extract_sql()` (`graph/agents/sql_agent.py:30-41`) chỉ tìm từ khoá `SELECT`/`WITH`;
khi không tìm thấy, nó **trả về nguyên văn output của model**, đi thẳng vào `cur.execute()`.
Connection dùng `adba_user`, không phải role read-only.

### L3 — Sandbox Python dựa vào namespace thay vì cô lập process

`graph/tools/python_tool.py` dùng `mp.get_context("fork")` + restricted `__builtins__` + whitelist
import. Ba điểm yếu, xếp theo mức độ thực tế:

1. `pd` và `np` được đưa thẳng vào namespace và **vô hiệu hoá chính whitelist**:
   `pd.read_csv('/etc/passwd')`, `pd.read_csv('http://…')`, `df.to_csv(…)` không cần `import` gì.
2. Restricted `__builtins__` trong `exec()` thoát được qua introspection
   (`().__class__.__bases__[0].__subclasses__()`), không đi qua `__import__` nên whitelist không chặn.
3. `fork` khiến child kế thừa memory, file descriptor và network namespace của process cha, bao gồm
   `DATABASE_URL` trong env.

**Mức đe doạ:** code do model fine-tune sinh, không phải attacker viết — không phải rủi ro cấp cứu.
Nhưng input của model chứa câu hỏi tự do của người dùng và cả kết quả SQL, nên prompt injection là
đường đi có thật khi deploy nội bộ.

---

## 3. Kiến trúc

### Nguyên tắc chi phối

**MCP migration phải vô hình với tầng agent.** Chữ ký hàm trong `graph/tools/` giữ nguyên; chỉ
implementation phía sau đổi. Nhờ vậy 6 file trong `graph/agents/` và toàn bộ `tests/unit/` không phải
sửa, và có thể roll back bằng một env var.

### Ba process

```
┌─ adba-app (LangGraph + agents + Streamlit) ─┐
│  supervisor → sql → python → viz → insight  │
│  graph/tools/*.py  ← interface KHÔNG đổi    │
│         │ ADBA_TOOLS_BACKEND=inproc | mcp   │
└─────────┼───────────────────────────────────┘
          │ MCP (JSON-RPC over stdio)
┌─────────▼─ adba-tools (container riêng) ────┐
│  query_postgres · run_pandas · render_chart │
│  DatasetStore (giữ DataFrame trong RAM)     │
│  no network egress · non-root · mem/CPU cap │
└─────────┬───────────────────────────────────┘
          │ read-only role
     ┌────▼─────┐
     │ Postgres │
     └──────────┘
```

### Handle pattern cho DataFrame

Hiện tại `shared_dataframe` mang **toàn bộ `df.to_dict("records")`** qua LangGraph state
(`graph/utils.py:19`), copy lại ở mỗi node. Bê nguyên pattern này qua RPC sẽ serialize cả bảng qua
JSON-RPC ở mỗi bước.

| | Hiện tại | Sau thiết kế |
|---|---|---|
| SQL trả về | full records vào state | `{dataset_id, columns, dtypes, row_count, preview[5]}` |
| Python nhận | `df_from_state(...)` toàn bộ | `dataset_id` → server tự resolve |
| Python trả về | full records vào state | `dataset_id` mới |
| Viz nhận | full records | `dataset_id` |
| State chứa | cả bảng, copy mỗi node | ~200 bytes metadata |

DataFrame **không bao giờ rời container `adba-tools`**. Chỉ `dataset_id` + schema + preview 5 dòng đi
qua dây. Lợi ích kèm theo: prompt của Python/Viz/Insight agent hiện đang nhồi preview lấy từ full
records — preview trở thành nguồn duy nhất, prompt nhỏ và ổn định hơn, giảm áp lực lên `num_ctx=4096`.

### Ba tool trên MCP server

| Tool | Input | Output |
|---|---|---|
| `query_postgres` | `sql: str` | `dataset_id`, `columns`, `dtypes`, `row_count`, `preview`, `truncated` |
| `run_pandas` | `dataset_id`, `code: str` | `dataset_id` mới + cùng bộ metadata |
| `render_chart` | `dataset_id`, `code: str` | `png_b64`, `chart_metadata` |

`DatasetStore` là LRU trong RAM, key theo `query_id`, dọn sạch khi query kết thúc. Không Redis, không
persist.

### Lý do chọn MCP (nói thẳng)

Cần một ranh giới RPC **dù thế nào**, vì phải cô lập Python exec sang container khác (L3). MCP chỉ là
chọn một chuẩn có sẵn thay vì tự chế giao thức — gần như miễn phí khi đã phải làm việc đó. Nó mua
**cô lập + khả năng đổi datasource/model**. Nó **không** giảm latency; kỳ vọng ngược lại là sai.

### Cố ý để ngoài phạm vi

Không expose schema/`info_box` thành MCP resource (giữ ở `perception/`); không tách thành nhiều MCP
server; không OpenTelemetry; không đụng vào supervisor hay bất kỳ prompt nào.

---

## 4. Bảo mật SQL & sandbox

### 4.1 SQL — bốn lớp, chỉ lớp 1 là bảo đảm thật

**Lớp 1 — read-only role ở tầng Postgres.** Lớp duy nhất thực sự là bảo đảm; ba lớp còn lại chỉ phát
hiện sớm.

```sql
CREATE ROLE adba_readonly LOGIN PASSWORD :'pw';
GRANT CONNECT ON DATABASE adba_db TO adba_readonly;
GRANT USAGE ON SCHEMA public TO adba_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO adba_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO adba_readonly;
REVOKE CREATE ON SCHEMA public FROM adba_readonly;
ALTER ROLE adba_readonly SET default_transaction_read_only = on;
```

Sau bước này, dù mọi lớp trên thủng, `DELETE` báo lỗi thay vì chạy. `adba_user` chỉ còn dùng cho
`data/seed/seed_data.py`.

**Lớp 2 — chặn multi-statement.** `cur.execute("SELECT 1; DROP TABLE orders")` trong psycopg2 chạy
**cả hai câu**. Dùng `sqlparse.split()`, yêu cầu đúng một statement không rỗng.

**Lớp 3 — quét DML trên toàn bộ statement, không chỉ token đầu.** Postgres hỗ trợ data-modifying
CTE, nên:

```sql
WITH gone AS (DELETE FROM orders RETURNING *) SELECT count(*) FROM gone;
```

bắt đầu bằng `WITH`, qua được heuristic "first token ∈ {SELECT, WITH}", và xoá sạch bảng. Guard phải
quét `INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|COPY` ở **mọi vị trí** trong
statement đã parse.

**Lớp 4 — trần tài nguyên.** `statement_timeout` hạ **30s → 10s**; row cap mặc định 50 000 dòng kèm
cờ `truncated`; giới hạn connection pool.

**Sửa kèm:** `_extract_sql` phải **fail closed** — raise thay vì trả nguyên văn output model khi không
tìm thấy `SELECT`/`WITH`. `explain_query_plan` đi qua cùng validator.

### 4.2 Sandbox — ranh giới là container, không phải namespace

| Biện pháp | Chặn được gì |
|---|---|
| Container `adba-tools` riêng, non-root, `read_only` rootfs + tmpfs `/tmp` | Ghi file, sửa image |
| Egress chỉ tới host Postgres (docker network riêng) | `pd.read_csv('http://…')`, exfiltration |
| `mem_limit` + `cpus` + `pids_limit` | `df.merge` cross-join làm OOM host |
| **`fork` → `spawn`** | Child không kế thừa memory/FD của cha |
| **`spawn` + env rỗng cho child** | Dù thoát namespace cũng không đọc được `DATABASE_URL` |

Cặp cuối là mấu chốt: `spawn` vốn để sửa điểm yếu 3, nhưng nó *đồng thời* cho phép truyền env rỗng —
điều `fork` không làm được — nên thiệt hại tối đa của điểm yếu 2 giảm từ "lộ credential" xuống "đọc
được vài file trong image không chứa gì".

Namespace hạn chế hiện tại **giữ nguyên** làm defense-in-depth. **Không** đi bịt từng hàm I/O của
pandas — đó là whack-a-mole; container read-only + không egress đã khiến `pd.read_csv` chỉ đọc được
file trong image, vốn không có bí mật.

**Timeout sau thiết kế:** SQL 10s · pandas 10s · chart 10s, tất cả nằm *bên trong* ngân sách
wall-clock ở mục 5.

---

## 5. Ngân sách thời gian & xử lý lỗi

### 5.1 Deadline mang trong state

Thêm vào `MultiAgentState`: `query_id`, `deadline_ts`, `llm_calls_used`.
`make_initial_state(query, info_box, budget_s=45)` đặt `deadline_ts = time.time() + budget_s`.

Ba điểm kiểm tra:
1. **Vào node** — còn đủ thời gian chạy node này không?
2. **Trước mỗi model call** — `ModelClient` nhận `deadline`; thời gian còn lại nhỏ hơn ước lượng thì
   không khởi động.
3. **Router** — quá hạn thì đi thẳng tới `finalize`.

### 5.2 Cấp phát theo dự trữ, không chia đều

Chia đều sẽ bỏ đói bước cuối, mà bước cuối lại giá trị nhất với người dùng. Giữ lại
**`INSIGHT_RESERVE_S = 12s`** ở cuối; node nào định lấn vào phần dự trữ thì bị bỏ qua.

Thang ưu tiên: `sql` bắt buộc → `insight` được bảo vệ bằng reserve → `python` và `viz` bị cắt trước
tiên. Nếu SQL chạy được, người dùng **luôn** nhận được insight, kể cả khi biểu đồ bị bỏ.

### 5.3 Node `finalize` thay cho `END` trần

Thay đổi khiến hard timeout trở nên chấp nhận được. Hiện tại hết retry → `status: "failed"` → người
dùng nhận **không gì cả**. Một bảng dữ liệu kèm dòng "phần insight bị bỏ do hết thời gian" có giá trị
hơn hẳn một thông báo lỗi.

`finalize` luôn chạy — quá hạn, xong kế hoạch, hay lỗi chí mạng — gom mọi thứ đang có và phân loại:

| `status` | Điều kiện | Người dùng thấy |
|---|---|---|
| `success` | Kế hoạch hoàn tất | Đầy đủ |
| `partial` | Có kết quả SQL, một số bước bị bỏ | Bảng + phần nào có, kèm lý do bị cắt |
| `failed` | Không có output dùng được | Lỗi + trace |

`degradation_reason` ghi ra để UI hiển thị được, không nuốt lỗi im lặng.

### 5.4 Trần cứng thay cho đếm retry

| Tham số | Hiện tại | Sau | Lý do |
|---|---|---|---|
| `MAX_REFLECTOR_PASSES_PER_AGENT` | 8 | **1** | Reflector sinh `corrected_context`; một chẩn đoán + một lần thử lại không sửa được thì 7 lần nữa cũng vậy — model kẹt ở cùng lớp lỗi. `reflector_json_rate = 100%` cho thấy reflector không hỏng ở khâu sinh output. |
| Retry trong node SQL | 3 | **2** | Một lần sinh + một lần sửa theo error context |
| Trần LLM call toàn query | — | **12** | Trần cứng độc lập thời gian |

Worst case mới ≈ 11 call, vẫn vượt 45s — đúng ý đồ: **trần cứng chặn vòng lặp vô hạn, deadline mới ép
SLO**, `finalize` biến phần bị cắt thành kết quả partial.

### 5.5 Bug phải sửa kèm

`graph/agents/supervisor.py:274-281` — `route_next_agent()` **ghi vào state**
(`state["dependency_graph"]`, `state["ready_agents"]`, `state["shared_metadata"]`) bên trong một
conditional-edge function. LangGraph coi conditional edge là hàm thuần: nó trả về tên route, không
phải nơi cập nhật state — những gán này không đi qua reducer nên không đảm bảo được persist. Chuyển
phần tính toán vào trong các node.

### 5.6 Phân loại lỗi & trace

Nhãn lỗi: `sql_generation` · `sql_execution` · `sql_timeout` · `python_runtime` · `chart_error` ·
`schema_mismatch` · `model_timeout` · `budget_exceeded` · `tool_unavailable`. Bảy nhãn đầu trùng bộ
category reflector đang dùng nên dùng lại nguyên vẹn.

`append_trace` bổ sung `query_id`, `duration_ms`, `llm_calls`, `deadline_remaining_ms`, ghi JSONL vào
`logs/traces/`. `query_id` đồng thời là key vòng đời của `DatasetStore`.

---

## 6. Eval end-to-end

### 6.1 Vì sao eval hiện tại không dùng được

`eval/eval_runner.py` đo **độ chính xác từng prompt cô lập**. Nó không trả lời được: pipeline có ra
đúng câu trả lời không, mất bao nhiêu giây, hỏng ở đâu. Thiếu nó thì "Hướng A cải thiện được X" là
không thể kiểm chứng.

### 6.2 `eval/eval_e2e.py`

**Golden set ~50 query nội bộ.** Nguồn: `data/supervisor_routing_samples.jsonl` (200 mẫu) đã chứa câu
hỏi ngôn ngữ tự nhiên kèm plan mong đợi — chọn lọc và bổ sung `gold_sql`. Phủ 3 domain
(sales · inventory · hr) và 5 hình dạng plan: `sql` · `sql→insight` · `sql→python→insight` ·
`sql→viz→insight` · `sql→python→viz→insight`.

```json
{
  "query": "So sánh doanh thu theo region năm 2024",
  "domain": "sales",
  "gold_sql": "SELECT region, SUM(amount) ... GROUP BY region",
  "expects": {
    "needs_chart": true,
    "needs_python": false,
    "assertions": ["row_count > 0", "columns ⊇ {region}", "insight validates InsightOutput"]
  }
}
```

**Bộ đối chiếu ngoài: BEAVER.** Thêm **subset 50 query** của BEAVER (benchmark text-to-SQL doanh
nghiệp, arXiv:2409.02038) để có số liệu so sánh được với bên ngoài. Golden set nội bộ đo *pipeline*;
BEAVER đo *năng lực text-to-SQL* trên schema không do mình sinh ra, qua đó phát hiện nếu 96.9% hiện
tại chỉ là học văn phong của generator.

BEAVER chạy ở chế độ **chỉ node SQL** (không qua full pipeline) vì schema của nó khác schema ADBA —
mục đích là đo năng lực sinh SQL, không phải đo pipeline. Cần load schema BEAVER vào một database
riêng; nếu chi phí dựng vượt quá pha 0 thì hạ xuống chạy `EXPLAIN`-only, chấp nhận metric yếu hơn.
Đây là phần duy nhất của pha 0 được phép cắt giảm nếu cần.

**Cách chấm — assertion trên artifact, không phải văn xuôi.** Exact-match trên văn bản insight là vô
vọng vì output LLM dao động. Metric chính là **execution accuracy** theo cách Spider/BIRD làm cho
text-to-SQL: chạy `gold_sql`, chạy SQL model sinh, **so sánh hai result set** (bỏ qua thứ tự trừ khi
có `ORDER BY`). Hoàn toàn tất định, và DB đã có sẵn.

LLM-as-judge cho phần insight là tín hiệu phụ, thêm sau, không chặn.

### 6.3 Bộ metric

| Metric | Ý nghĩa |
|---|---|
| `slo_hit_rate` | % query dưới 45s — **con số chủ đạo** |
| `answer_accuracy` | result set khớp `gold_sql` |
| `beaver_exec_accuracy` | execution accuracy trên subset BEAVER |
| `e2e_success_rate` | `status == success` |
| `partial_rate` | phân rã theo `degradation_reason` |
| `failure_rate` | phân rã theo 9 nhãn lỗi ở 5.6 |
| `wall_clock` p50/p95 | |
| `llm_calls` p50/p95 | |

---

## 7. Thứ tự triển khai

**Nguyên tắc: MCP đi cuối cùng** — thay đổi lớn nhất, giá trị trực tiếp cho người dùng nhỏ nhất. Pha 1
và 2 gánh gần hết giá trị production. Dừng sau pha 2 vẫn có hệ thống deploy được.

| Pha | Nội dung | Quy mô | Rollback |
|---|---|---|---|
| **0** | `eval_e2e.py` + golden set + subset BEAVER + **đo baseline kiến trúc hiện tại**. Không sửa production code. | Vừa | — |
| **1** | An toàn, không cần MCP: read-only role, SQL guard 3 lớp, `_extract_sql` fail-closed, `fork`→`spawn` + env rỗng, hạ timeout. | Nhỏ | git revert |
| **2** | Ngân sách: `deadline_ts`, node `finalize`, thang partial, trần call, sửa bug 5.5, trace JSONL. | Vừa | git revert |
| **3** | MCP server + handle pattern + container cô lập, sau cờ `ADBA_TOOLS_BACKEND=inproc\|mcp`. | Lớn | đổi env var |

**Chạy lại eval sau mỗi pha** để quy trách nhiệm sạch:

- Sau pha 1 — accuracy kỳ vọng **không đổi**; tụt nghĩa là SQL guard chặn nhầm query hợp lệ.
- Sau pha 2 — pha phải làm `p95` và `slo_hit_rate` nhảy. `partial_rate` sẽ tăng từ 0 và **đó là điều
  tốt**: query trước đây thất bại im lặng sau 25 phút giờ trả kết quả một phần trong 45s.
- Sau pha 3 — accuracy phải trung tính, latency tăng nhẹ do RPC. Cờ `inproc|mcp` cho phép đo trực
  tiếp chi phí đó trên cùng golden set.

Pha 1 và 2 độc lập kỹ thuật nên song song được, nhưng làm tuần tự cho phép quy thay đổi metric về
đúng nguyên nhân.

---

## 8. Dependency & hạ tầng phát sinh

| Thứ mới | Pha | Ghi chú |
|---|---|---|
| `sqlparse` | 1 | Parse + split statement cho SQL guard lớp 2/3. Thêm vào `requirements.txt`. |
| `mcp` (Python SDK) | 3 | MCP server + client. Chỉ `adba-tools` và adapter cần. |
| Service `adba-tools` trong `docker-compose.yml` | 3 | non-root, `read_only`, network riêng chỉ tới Postgres, `mem_limit`/`cpus`/`pids_limit` |
| Role `adba_readonly` trong Postgres | 1 | Script migration kèm theo; `adba_user` giữ lại cho seed |
| Thư mục `logs/traces/` | 2 | JSONL, cần chính sách xoay vòng để không phình |

## 9. Tiêu chí thành công

| Tiêu chí | Ngưỡng |
|---|---|
| `slo_hit_rate` | ≥ 90% query dưới 45s |
| `answer_accuracy` trên golden set | không thấp hơn baseline pha 0 |
| Worst-case wall clock | ≤ 60s, không có ngoại lệ |
| Query trả về `failed` (không có output nào) | ≤ 5% |
| Mutation SQL chạm tới DB | 0, kiểm bằng test đối kháng gồm cả data-modifying CTE |
| Test hiện có | `tests/unit/` và `tests/integration/` pass không sửa đổi |

---

## 10. Phương án đã cân nhắc và loại bỏ

### 10.1 Single agent + MCP tools (thay 5 node bằng 1 agent tool-calling)

**Loại bỏ.** Ở SLO 30–60s, phương án này không mang lại lợi ích latency: 5 node ≈ 5 LLM call, còn
single-agent tool-loop cũng cần 4–5 round-trip vì mỗi tool call là một lượt gọi model. Chi phí đổi
lại rất cao:

- Viết lại 787 mẫu train sang định dạng tool-call, train lại LoRA, đánh giá lại từ đầu.
- Rủi ro mất 96.9% → 84.4% SQL accuracy (base model chưa từng thấy định dạng tool-call này).
- Phải nâng `num_ctx` từ 4096: một agent duy nhất phải giữ schema + tool definitions + kết quả tool
  tích luỹ, trong khi `perception/info_box_all.json` đã 33.5KB. Kiến trúc multi-agent hiện tại sống
  được chính nhờ mỗi node chỉ nhận đúng slice context nó cần.

Hoãn, không loại vĩnh viễn. Nếu sau này đổi sang model lớn hơn hoặc API, đánh giá lại.

### 10.2 Fast-path bỏ supervisor khỏi critical path

**Hoãn.** Tiết kiệm ~12s (~25% latency) bằng cách route bằng classifier/cache dựng từ 200 mẫu
`supervisor_routing_samples.jsonl` cho các query nhận diện được. Không đưa vào đợt này vì SLO 45s đã
đạt được mà không cần nó, và nó thêm một đường code có thể làm tệ chất lượng routing. Là ứng viên số
một cho đợt tối ưu tiếp theo nếu số liệu pha 0 cho thấy latency vẫn căng.

### 10.3 Đổi hướng sang nghiên cứu / viết báo

**Không theo đuổi trong đợt này.** Đã khảo sát hai hướng:

1. *Suy giảm có kiểm soát dưới ràng buộc thời gian* — bị kẹp giữa mảng serving-layer đã đông
   (Niyama arXiv:2503.22562, Cascade arXiv:2608.06557, EDF scheduling) và Bird-Interact
   (arXiv:2510.05318) vốn đã dựng sẵn đánh giá text-to-SQL có ràng buộc ngân sách. Cơ chế truyền
   ngân sách còn lại đã được mô tả như pattern kỹ thuật đã biết.
2. *Phân rã vai trò có thay thế được năng lực model không* — đã được nghiên cứu, và kết quả cho thấy
   quan hệ không đơn điệu: phân rã có lợi ở quy mô open-source cỡ trung, nhưng model nhỏ nhất chật
   vật tạo specialist hữu ích (arXiv:2602.03794, arXiv:2512.16698). MapCoder-Lite
   (arXiv:2509.17489) đã chưng cất multi-agent thành một small LLM cho domain coding.

Chỗ trống còn lại — so sánh ba chiều ở ngân sách LoRA cố định (N adapter theo vai trò vs một adapter
hợp nhất vs chưng cất) trong domain BI — là mở rộng MapCoder-Lite sang domain mới, tầm workshop /
applied track. Ghi lại ở đây để cân nhắc sau khi deploy xong.

Phần duy nhất được giữ lại từ nhánh này: **BEAVER vào golden set** (mục 6.2), vì nó giải quyết một
vấn đề có thật của số liệu hiện tại bất kể có viết báo hay không.
