# System Architecture — ADBA

> Hệ thống được xây thế nào: thành phần, công nghệ và lý do chọn, cách triển khai.

---

## 1. Sơ đồ tổng quan (High-Level Architecture)

```mermaid
flowchart TB
    subgraph client["Tầng giao diện"]
        UI["Streamlit Chat UI<br/><code>app.py</code> · :8501"]
    end

    subgraph orch["Tầng điều phối — LangGraph"]
        SUP["Supervisor<br/>lập ExecutionPlan"]
        ROUTE{{"route_next_agent()<br/>conditional edge"}}
        SQLA["SQL Agent"]
        PYA["Python Agent"]
        VZA["Viz Agent"]
        INA["Insight Agent"]
        RFA["Reflector Agent"]
        ST[("MultiAgentState<br/>TypedDict dùng chung")]
    end

    subgraph model["Tầng model"]
        MC["ModelClient<br/><code>model/model_client.py</code>"]
        OLL["Ollama<br/>qwen2.5-coder:7b · :11434"]
        OAI["OpenAI API<br/>(fallback, tắt được)"]
    end

    subgraph tools["Tầng tool"]
        STOOL["sql_tool<br/>psycopg2 + EXPLAIN + timeout"]
        PTOOL["python_tool<br/>sandbox process, whitelist import"]
        VTOOL["viz_tool<br/>matplotlib Agg → base64 PNG"]
        ITOOL["insight_tools<br/>phát hiện bất thường"]
    end

    subgraph data["Tầng dữ liệu"]
        PG[("PostgreSQL 15<br/>sales · inventory · hr")]
        IB["info_box JSON<br/><code>perception/*.json</code>"]
    end

    UI -->|"run_graph(query, info_box)"| SUP
    SUP --> ROUTE
    ROUTE --> SQLA & PYA & VZA & INA & RFA
    SQLA & PYA & VZA --> ROUTE
    RFA -.->|"quay lại agent lỗi"| SQLA
    INA --> UI

    SUP & SQLA & PYA & VZA & INA & RFA <--> ST

    SUP & SQLA & PYA & VZA & INA & RFA --> MC
    MC --> OLL
    MC -.->|"khi Ollama lỗi"| OAI

    SQLA --> STOOL --> PG
    PYA --> PTOOL
    VZA --> VTOOL
    INA --> ITOOL

    PG -->|"introspection<br/>extract_info_box.py"| IB
    IB --> SUP & SQLA
```

### Trách nhiệm từng tầng

| Tầng | Trách nhiệm | Không làm |
|---|---|---|
| Giao diện | Nhận câu hỏi, hiển thị plan / trace / bảng / chart / insight | Không chứa logic nghiệp vụ |
| Điều phối | Quyết định agent nào chạy, theo thứ tự nào, xử lý lỗi | Không gọi thẳng DB |
| Model | Gọi LLM, retry, parse JSON an toàn, fallback | Không biết agent nào đang gọi ngoài `agent_type` |
| Tool | Thực thi tác dụng phụ có ranh giới (DB, exec code, vẽ) | Không gọi LLM |
| Dữ liệu | Lưu trữ + mô tả bản thân (`info_box`) | — |

### Ba quyết định kiến trúc đáng chú ý

**Q1 — State là TypedDict phẳng, không phải object.**
`MultiAgentState` phải JSON-serialize được để LangGraph checkpoint và để UI đọc trace.
DataFrame vì thế đi qua `df_to_state()` / `df_from_state()` (`graph/utils.py`) thay vì
truyền tham chiếu. Chi phí: serialize hai lần cho mỗi bước. Đổi lại: mọi bước đều
kiểm tra được sau khi chạy, và pipeline có thể tạm dừng/khôi phục.

**Q2 — Tool không gọi LLM, agent không chạm hạ tầng.**
Ranh giới này khiến `graph/tools/` test được bằng unit test thuần (không mock LLM) và
`graph/agents/` test được bằng cách mock `ModelClient`. Đây là lý do thư mục tách đôi.

**Q3 — Perception layer nén schema, không nhét DDL thô.**
Với context 4096 token, đưa cả DDL ba domain vào prompt là bất khả thi. `info_box` chỉ giữ
thứ model cần để không đoán: tên cột, kiểu, nullable, khoá ngoại, **giá trị enum** (đọc từ
CHECK constraint) và 3 dòng mẫu. Không có enum thì model viết `WHERE region = 'North'`
trong khi dữ liệu là `'Miền Bắc'` — lỗi im lặng trả về 0 dòng.

## 2. Tech Stack & lý do chọn

<!-- AUTO:begin id=tech-stack -->

| Package | Ràng buộc | Vai trò | Lý do chọn |
|---|---|---|---|
| `langgraph` | `>=0.2.0` | Điều phối multi-agent dạng đồ thị có trạng thái | Cần vòng lặp agent → reflector → agent với state chia sẻ; LangGraph cho conditional edge và state reducer sẵn, còn chain tuyến tính (LCEL) hay CrewAI không diễn tả được vòng self-repair này |
| `langchain-core` | `>=0.2.0` | Kiểu dữ liệu message/runnable nền | Đi kèm LangGraph; không dùng abstraction cao hơn để giữ quyền kiểm soát prompt |
| `langchain-ollama` | `>=0.1.0` | Kết nối LangChain ↔ Ollama | Cho phép đổi sang LCEL sau này mà không viết lại adapter |
| `ollama` | `>=0.2.0` | Chạy LLM cục bộ | Yêu cầu on-prem: dữ liệu khách không rời máy; Ollama quản lý model + quantization và có HTTP API ổn định, không cần GPU cluster như vLLM |
| `psycopg2-binary` | `>=2.9.9` | Driver PostgreSQL | Cần `SET LOCAL statement_timeout`, `EXPLAIN`, và `psycopg2.sql.Identifier` để quote định danh an toàn — thứ ORM che mất |
| `python-dotenv` | `>=1.0.0` | Nạp cấu hình từ `.env` | Chuẩn 12-factor, tránh hardcode credential |
| `sqlalchemy` | `>=2.0.0` | Tiện ích kết nối/pool | Dùng ở mức thấp; agent sinh SQL thô nên ORM không phải trung tâm |
| `sqlparse` | `>=0.5.0` | — | — |
| `pandas` | `>=2.0.0` | Vật mang dữ liệu giữa các agent | SQL → DataFrame → Python → Viz dùng chung một kiểu, serialize được qua `df_to_state()` |
| `numpy` | `>=1.26.0` | Tính toán số | Phụ thuộc nền của pandas/scipy, cũng nằm trong namespace sandbox |
| `matplotlib` | `>=3.8.0` | Sinh biểu đồ PNG | Backend `Agg` chạy không cần màn hình — hợp với server; xuất base64 nhúng thẳng vào state |
| `seaborn` | `>=0.13.0` | Style biểu đồ | Chỉ dùng theme `seaborn-v0_8-whitegrid` cho đồng nhất |
| `scipy` | `>=1.11.0` | Thống kê phát hiện bất thường (z-score, IQR) | Có sẵn hàm kiểm định, không cần tự cài đặt |
| `pydantic` | `>=2.0.0` | Contract cho output LLM | Ranh giới an toàn: plan có chu trình / insight sai định dạng bị chặn tại validate thay vì nổ giữa pipeline |
| `openai` | `>=1.30.0` | Fallback khi Ollama lỗi | Chỉ dùng khi `ADBA_DEPLOYMENT=hybrid`. Mặc định (`onprem`) chặn hẳn: prompt agent `sql` mang schema, chú giải nghiệp vụ và câu hỏi thật của khách |
| `streamlit` | `>=1.35.0` | UI chat + bảng + biểu đồ | Một file Python ra được UI có state; React/FastAPI tốn công gấp nhiều lần cho cùng phạm vi demo |
| `ragas` | `>=0.1.0` | Đánh giá chất lượng sinh | Bộ metric có sẵn cho đánh giá đầu ra LLM |
| `faker` | `—` | Sinh dữ liệu seed tiếng Việt | Có locale vi_VN — tên/địa chỉ thật hợp cảnh dữ liệu doanh nghiệp Việt Nam |

<!-- AUTO:end id=tech-stack -->

### Những lựa chọn đã cân nhắc rồi loại

| Thay vì | Đã chọn | Vì |
|---|---|---|
| CrewAI / AutoGen | **LangGraph** | Cần điều khiển tường minh cạnh có điều kiện và vòng self-repair; framework role-play che mất chỗ cần can thiệp |
| Chain tuyến tính (LCEL) | **StateGraph** | Không diễn tả được vòng `agent → reflector → agent` và điều kiện dừng |
| vLLM / TGI | **Ollama** | Mục tiêu chạy trên laptop và trong mạng khách hàng; vLLM cần GPU và vận hành nặng hơn nhiều |
| GPT-4o cho mọi agent | **Qwen2.5-Coder-7B cục bộ** | Ràng buộc dữ liệu không rời hạ tầng; GPT-4o giữ lại làm fallback tuỳ chọn |
| ORM (SQLAlchemy ORM) | **psycopg2 thô** | Cần `SET LOCAL statement_timeout`, `EXPLAIN`, và quote định danh an toàn — ORM che mất |
| Nhét cả DDL vào prompt | **`info_box` nén** | Context 4096 token |
| FastAPI + React | **Streamlit** | Phạm vi v1 là chứng minh chất lượng agent, không phải UI; đổi được sau (xem [API.md](API.md)) |
| Vector DB (FAISS/Qdrant) cho schema | **Chưa dùng** | 9 bảng thì nhét hết vẫn vừa context; retrieval chỉ cần khi sang schema khách hàng hàng trăm bảng — đã thiết kế ở [plan schema context](../superpowers/plans/2026-08-15-schema-context-pipeline.md) |

### Tham số model theo agent

Đặt tại `model/model_config.py`, đọc được ở [AGENT_ARCHITECTURE.md](../03-flows/AGENT_ARCHITECTURE.md#tham-số-theo-agent).
Nguyên tắc: **temperature 0.0 cho SQL** (chỉ có một câu đúng), 0.1 cho lập kế hoạch và chẩn đoán,
0.2 cho viz và insight (cần diễn đạt tự nhiên).

## 3. Kiến trúc triển khai

### 3.1 Môi trường phát triển (máy cá nhân)

```mermaid
flowchart LR
    subgraph host["macOS / Linux — máy dev"]
        ST["streamlit run app.py<br/>:8501"]
        OL["ollama serve<br/>:11434<br/>OLLAMA_NUM_CTX=4096"]
        subgraph dk["Docker"]
            PG[("adba-postgres<br/>postgres:15 · :5432<br/>volume postgres_data")]
        end
    end
    ST -->|psycopg2| PG
    ST -->|HTTP| OL
```

```bash
docker compose up -d postgres
./scripts/apply_schemas_docker.sh
python data/seed/seed_data.py
python perception/extract_info_box.py
export OLLAMA_NUM_CTX=4096 && ollama serve
PYTHONPATH=. streamlit run app.py
```

### 3.2 Môi trường container hoá (`docker-compose.yml`)

```mermaid
flowchart LR
    subgraph compose["docker compose"]
        APP["adba-app<br/>ghcr.io/…/adba:latest<br/>:8501 · healthcheck /_stcore/health"]
        PG[("adba-postgres<br/>healthcheck pg_isready")]
        VOL[["volume: postgres_data"]]
        LOGS[["bind: ./logs → /app/logs"]]
    end
    OL["Ollama trên host<br/>host.docker.internal:11434"]

    APP -->|depends_on: service_healthy| PG
    PG --- VOL
    APP --- LOGS
    APP -->|OLLAMA_BASE_URL| OL
```

Điểm đáng lưu ý:

- `app` chỉ khởi động sau khi Postgres **healthy** (`condition: service_healthy`), không phải
  chỉ "đã start" — tránh lỗi kết nối lúc khởi động.
- Ollama nằm **ngoài** compose trên máy dev: model nhiều GB không nên nằm trong vòng đời container.
  Bản production/on-prem đưa Ollama thành service riêng có volume model.
- `./logs` bind-mount ra ngoài để trace sống sót qua `docker compose down`.

### 3.3 CI/CD

```mermaid
flowchart LR
    PR["Pull Request → main"] --> T["Job: test<br/>pytest tests/unit/<br/>service postgres:15"]
    PUSH["Push → main"] --> T
    T -->|xanh + là push main| B["Job: build-and-push<br/>docker/build-push-action"]
    B --> GHCR[("GHCR<br/>tag: latest + sha-&lt;commit&gt;")]
    GHCR -.->|"IMAGE_TAG=sha-…<br/>docker compose up -d"| DEPLOY["Môi trường đích<br/>(template: Cloud Run / ECS / VPS)"]
    PR --> D["Job: docs-sync<br/>update_docs.py --check"]
    PUSH --> D
```

| Job | File | Chạy khi | Chặn merge |
|---|---|---|---|
| `test` | `.github/workflows/ci-cd.yml` | PR + push main | Có |
| `build-and-push` | `.github/workflows/ci-cd.yml` | Chỉ push main, sau `test` | — |
| `docs-sync` | `.github/workflows/docs.yml` | PR + push main | Có |

Ảnh được gắn tag `sha-<commit>` bên cạnh `latest`. Rollback vì thế là đổi một biến:

```bash
IMAGE_TAG=sha-<commit-trước> docker compose -f docker-compose.prod.yml up -d
```

`env.example` nói rõ: **ghim `IMAGE_TAG` vào một sha cụ thể**, đừng deploy `latest`.

### 3.4 Hình thái triển khai on-prem (kế hoạch M4)

```mermaid
flowchart TB
    subgraph net["Mạng nội bộ khách hàng — không egress"]
        APP["adba-app"]
        OLS["ollama<br/>model ship kèm dạng GGUF"]
        TOOLS["adba-tools (MCP)<br/>non-root · read_only · pids_limit"]
        PGC[("PostgreSQL của khách hàng<br/>role adba_readonly")]
    end
    APP --> OLS
    APP -->|stdio/HTTP MCP| TOOLS
    TOOLS -->|chỉ SELECT| PGC
```

Ba khác biệt so với bản hiện tại: DB là của khách (không seed), tool chạy trong container
riêng qua MCP, và egress bị chặn ở tầng mạng. Chi tiết:
[spec hardening](../superpowers/specs/2026-08-12-adba-mcp-hardening-design.md) ·
[spec đa schema](../superpowers/specs/2026-08-15-adba-multi-schema-onprem-design.md).

## 4. Bản đồ mã nguồn

<!-- AUTO:begin id=repo-map -->

| Đường dẫn | File | File .py | Dòng | Vai trò |
|---|---|---|---|---|
| `ADBA_Project_Context_Prompt_v2.md` | 1 | 0 | 851 | — |
| `README.md` | 1 | 0 | 269 | — |
| `app.py` | 1 | 1 | 354 | Streamlit UI — điểm vào duy nhất cho người dùng cuối |
| `conftest.py` | 1 | 1 | 2 | — |
| `data` | 16 | 1 | 84.504 | DDL 3 domain, seed, và dataset huấn luyện/đánh giá (JSONL) |
| `docker-compose.yml` | 1 | 0 | 21 | — |
| `docs` | 19 | 0 | — | Bộ tài liệu dự án (chính file này) |
| `eval` | 14 | 10 | 3.396 | Runner đo baseline / PEFT và so sánh hai lần chạy |
| `graph` | 16 | 16 | 2.173 | LangGraph: state, các node agent, và tool thực thi |
| `model` | 3 | 3 | 374 | ModelClient (Ollama local-first, fallback OpenAI) + tham số theo agent |
| `onboard.py` | 1 | 1 | 1.012 | — |
| `pages` | 1 | 1 | 86 | — |
| `perception` | 17 | 13 | 5.551 | Perception layer — introspect PostgreSQL sinh `info_box` JSON |
| `prompts` | 5 | 0 | 514 | System prompt của từng skill, dạng file text tách khỏi code |
| `requirements.txt` | 1 | 0 | 18 | — |
| `schemas` | 3 | 3 | 735 | Pydantic contract: ExecutionPlan (Supervisor) và InsightOutput (Insight) |
| `scripts` | 7 | 3 | 1.644 | Tiện ích vận hành: áp schema, kiểm tra kết nối, sinh tài liệu |
| `tests` | 41 | 38 | 9.175 | pytest — unit theo từng agent, integration theo độ phức tạp câu hỏi |
| `training` | 13 | 5 | 3.795 | Sinh dữ liệu, LoRA/QLoRA notebook, checkpoint và kết quả |
| `.cursorrules` | 1 | 0 | 0 | — |
| `.github` | 1 | 0 | 29 | CI/CD — unit test, build & push image lên GHCR |
| `.gitignore` | 1 | 0 | 0 | — |

<!-- AUTO:end id=repo-map -->

## 5. Cấu hình

Toàn bộ cấu hình qua biến môi trường (12-factor). Không có file config nào được commit
kèm giá trị thật; `env.example` là bản mẫu.

<!-- AUTO:begin id=env-vars -->

| Biến | Mặc định trong code | Có trong `env.example` | Nơi đọc |
|---|---|---|---|
| `BACKUP_MODEL` | `"llama3.1:8b-instruct-q4_K_M"` | — | `model/model_config.py` |
| `DATABASE_URL` | `"postgresql://adba_user:adba@localhost:5432/adba_db"` | — | `data/seed/seed_data.py`, `perception/extract_info_box.py`, `training/generate_data.py` (+1) |
| `ENABLE_OPENAI_FALLBACK` | — | — | `model/model_client.py` |
| `EVAL_MODEL` | `"qwen2.5-coder:7b-instruct-q5_K_M"` | — | `eval/eval_runner.py` |
| `MODEL_MAX_RETRIES` | `"3"` | — | `model/model_config.py` |
| `OLLAMA_BASE_URL` | `"http://localhost:11434"` | — | `eval/eval_runner.py`, `model/model_config.py` |
| `OLLAMA_NUM_CTX` | `"4096"` | — | `eval/eval_runner.py`, `model/model_config.py` |
| `OPENAI_API_KEY` | `""` | — | `model/model_client.py` |
| `OPENAI_MODEL` | `"gpt-4o-mini"` | — | `model/model_client.py` |
| `PANDAS_EXEC_TIMEOUT_SECONDS` | `"10"` | — | `graph/tools/python_tool.py` |
| `POSTGRES_DB` | `"adba_db"` | — | `eval/eval_runner.py`, `scripts/test_postgres_connection.py` |
| `POSTGRES_HOST` | `"localhost"` | — | `eval/eval_runner.py`, `scripts/test_postgres_connection.py` |
| `POSTGRES_PASSWORD` | `"adba_password"` | — | `eval/eval_runner.py`, `scripts/test_postgres_connection.py` |
| `POSTGRES_PORT` | `"5432"` | — | `eval/eval_runner.py`, `scripts/test_postgres_connection.py` |
| `POSTGRES_URL` | `""` | — | `data/seed/seed_data.py`, `eval/eval_runner.py`, `scripts/test_postgres_connection.py` |
| `POSTGRES_USER` | `"adba_user"` | — | `eval/eval_runner.py`, `scripts/test_postgres_connection.py` |
| `PRIMARY_MODEL` | `"qwen2.5-coder:7b-instruct-q5_K_M"` | — | `model/model_config.py` |
| `SQL_TIMEOUT_MS` | `"30000"` | — | `graph/tools/sql_tool.py` |

<!-- AUTO:end id=env-vars -->

### Nhóm cấu hình quan trọng

| Nhóm | Biến then chốt | Lưu ý |
|---|---|---|
| Kết nối DB | `POSTGRES_URL` / `DATABASE_URL` | `sql_tool` đọc `DATABASE_URL` trước, rồi mới tới `POSTGRES_URL` |
| Model | `PRIMARY_MODEL`, `OLLAMA_BASE_URL`, `OLLAMA_NUM_CTX` | Đặt `OLLAMA_NUM_CTX=4096`; thấp hơn sẽ cắt cụt `info_box` và plan |
| Fallback | `ADBA_DEPLOYMENT`, `OPENAI_API_KEY`, `ENABLE_OPENAI_FALLBACK` | Mặc định `ADBA_DEPLOYMENT=onprem` chặn mọi lời gọi ra ngoài; đặt `hybrid` để mở, và chỉ khi khách đã đồng ý |
| An toàn | `SQL_TIMEOUT_MS`, `PANDAS_EXEC_TIMEOUT_SECONDS` | Hai trần cứng duy nhất hiện có; ngân sách toàn cục thuộc M3.2 |
| Ảnh | `IMAGE_TAG` | Ghim sha, không dùng `latest` trên production |

## 6. Hiện trạng & nợ kỹ thuật

| Vấn đề | Ảnh hưởng | Xử lý ở |
|---|---|---|
| Không có ngân sách thời gian toàn cục — một truy vấn xấu có thể chạy rất lâu | NFR-2 chưa đạt | M3.2 |
| Chưa chặn DML nhiều lớp; dựa vào việc model chỉ sinh SELECT | Rủi ro an toàn | M3.1 |
| Whitelist bảng tĩnh trong `graph/tools/sql_tool.py` | Không mở sang schema khác được | M3.1 (`ConnectionProfile`) |
| Sandbox dựa vào namespace + process, chưa phải container | Ranh giới yếu | M3.3 / B-01 |
| `docker-compose.prod.yml` tham chiếu `sandbox/Dockerfile` và `scripts/ollama-entrypoint.sh` chưa tồn tại | Compose production không dựng được | B-01, B-06 |
| `app.py` đọc `info_box_latest.json` ở thư mục gốc, còn `extract_info_box.py` ghi vào `perception/` | UI cảnh báo "No info_box found" | Cần thống nhất đường dẫn |

---

<!-- AUTO:begin id=stamp -->

| Trường | Giá trị |
|---|---|
| Commit nguồn gần nhất | `a3682b8` — fix(sql): bỏ cả phần giải thích model viết SAU câu SQL |
| Tác giả | Đặng Văn Vỹ |
| Ngày commit | 2026-09-01 |
| Số commit nguồn | 104 |
| Sinh bởi | `scripts/update_docs.py` (hook `post-commit`) |

<!-- AUTO:end id=stamp -->
