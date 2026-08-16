# ADBA — Autonomous Data & Business Intelligence Agent

## Tài liệu đặc tả dự án

**Phiên bản:** 1.0  
**Ngày cập nhật:** 14/05/2026

---

## 1. Tổng quan dự án

### 1.1 Vấn đề cần giải quyết

Các doanh nghiệp ngày nay ngập trong dữ liệu nhưng thiếu khả năng trích xuất insights kịp thời. Để trả lời một câu hỏi kinh doanh đơn giản như *"Khu vực nào tăng trưởng mạnh nhất quý vừa rồi?"*, người dùng thường phải:

1. Nhờ bộ phận IT viết câu truy vấn SQL
2. Đợi data team xử lý và làm sạch dữ liệu
3. Chờ analyst vẽ biểu đồ và viết báo cáo

Quá trình này mất từ vài giờ đến vài ngày, trong khi quyết định kinh doanh cần được đưa ra ngay lập tức.

### 1.2 Giải pháp

**ADBA** là hệ thống AI tự trị cho phép người dùng đặt câu hỏi bằng ngôn ngữ tự nhiên — tiếng Việt hoặc tiếng Anh — và nhận lại trong vòng **dưới 30 giây** (mục tiêu thiết kế; latency thực tế phụ thuộc model và phần cứng, xem mục 4.3):

- Bảng dữ liệu đã được truy vấn và phân tích  
- Biểu đồ trực quan hóa kết quả  
- Một **business insight** ngắn gọn: phát hiện chính + số liệu minh chứng + khuyến nghị hành động cụ thể

Người dùng không cần biết SQL, không cần biết lập trình.

### 1.3 Nguồn gốc học thuật

Dự án được xây dựng dựa trên bài báo *"Autonomous Data Agents"* (Fu et al., 2025, arXiv:2509.18710), đặc biệt triển khai phần kiến trúc **Multi-Agent Collaboration** mà bài báo đề xuất nhưng chưa implement. Đây là contribution chính của dự án so với nghiên cứu gốc.

---

## 2. Kiến trúc hệ thống

### 2.1 Mô hình hoạt động

Hệ thống hoạt động theo mô hình **Hierarchical Multi-Agent**: một Supervisor điều phối nhiều Specialist Agents, mỗi agent chuyên sâu vào một nhiệm vụ cụ thể.

```
Người dùng đặt câu hỏi (ngôn ngữ tự nhiên)
           │
           ▼
    ┌─────────────┐
    │  SUPERVISOR │  ← Phân tích câu hỏi, lập kế hoạch,
    │    AGENT    │    phân công cho các agent phù hợp
    └──────┬──────┘
           │ Kế hoạch thực thi (có thứ tự & phụ thuộc)
    ┌──────┼──────────────┐
    ▼      ▼              ▼
  SQL    Python          Viz
 Agent   Agent          Agent
  │        │              │
  └────────┴──────────────┘
           │ Kết quả từ tất cả agents
           ▼
    ┌─────────────┐
    │   INSIGHT   │  ← Tổng hợp → Business Insight
    │    AGENT    │
    └─────────────┘
           │
    ┌──────┴──────┐
    │  STREAMLIT  │  ← Hiển thị cho người dùng
    │     UI      │
    └─────────────┘
```

Nếu có lỗi xảy ra ở bất kỳ bước nào, **Reflector Agent** sẽ chẩn đoán lỗi và tự sửa trước khi thử lại — không cần người dùng can thiệp.

### 2.2 Các agent và vai trò


| Agent               | Vai trò                                                   | Đầu vào             | Đầu ra                    |
| ------------------- | --------------------------------------------------------- | ------------------- | ------------------------- |
| **Supervisor**      | Phân tích câu hỏi, lập kế hoạch thực thi, phân công agent | Câu hỏi + schema DB | Kế hoạch dạng JSON        |
| **SQL Agent**       | Viết và thực thi câu truy vấn PostgreSQL                  | Kế hoạch + schema   | Bảng dữ liệu thô          |
| **Python Agent**    | Xử lý, tính toán, phát hiện bất thường                    | Bảng dữ liệu        | Bảng dữ liệu đã phân tích |
| **Viz Agent**       | Vẽ biểu đồ phù hợp với dữ liệu                            | Bảng dữ liệu        | Ảnh biểu đồ (PNG)         |
| **Insight Agent**   | Tổng hợp thành business insight có thể hành động          | Tất cả kết quả trên | Insight JSON có cấu trúc  |
| **Reflector Agent** | Chẩn đoán lỗi và đề xuất cách sửa                         | Thông tin lỗi       | Hướng dẫn retry           |


### 2.3 Luồng xử lý điển hình

**Câu hỏi đơn giản** (chỉ cần lấy dữ liệu):

`Supervisor → SQL Agent → Insight Agent`

Ví dụ: *"Top 5 khách hàng doanh thu cao nhất Q4 2024 là ai?"*

**Câu hỏi phức tạp** (cần phân tích + trực quan hóa):

`Supervisor → SQL Agent → Python Agent → Viz Agent → Insight Agent`

Ví dụ: *"So sánh tăng trưởng doanh thu Q4-2024 vs Q4-2023 theo khu vực, phát hiện bất thường"*

**Khi có lỗi:**

`SQL Agent gặp lỗi → Reflector Agent chẩn đoán → SQL Agent thử lại với context đã sửa`

### 2.4 Định dạng output của Insight Agent

Mỗi câu trả lời cuối cùng luôn có cấu trúc nhất quán:

- **Finding:** *"Miền Bắc dẫn đầu tăng trưởng Q4 với +89% YoY"*  
- **Evidence:**  
  - Revenue Q4-2024: 142,000 USD vs Q4-2023: 75,000 USD  
  - Tăng trưởng trung bình toàn quốc: +23% YoY
- **Anomaly:** Miền Bắc vượt 3.2 sigma so với trung bình (positive outlier)  
- **Action:** *"Tăng tồn kho khu vực Miền Bắc ít nhất 60% cho Q1 2025"*  
- **Confidence:** High

---

## 3. Dữ liệu & phạm vi

### 3.1 Cơ sở dữ liệu

Hệ thống kết nối với **PostgreSQL**, hỗ trợ 3 domain dữ liệu:

**Domain Sales (kinh doanh)**

- **orders** — Đơn hàng: region, amount, quantity, order_date, status, payment_method  
- **products** — Sản phẩm: category, unit_price, cost  
- **customers** — Khách hàng: city, segment, region

**Domain Inventory (kho hàng)**

- **stock** — Tồn kho: quantity, min_threshold, warehouse  
- **warehouses** — Kho: city, region, capacity  
- **stock_movements** — Xuất/nhập kho: movement_type, quantity, date

**Domain HR (nhân sự)**

- **employees** — Nhân viên: department, role, level, salary, hire_date, status  
- **departments** — Phòng ban: budget, headcount  
- **payroll** — Bảng lương: base_salary, bonus, deduction, net_salary

### 3.2 Dữ liệu huấn luyện

Để model AI hiểu được ngữ cảnh kinh doanh Việt Nam, dự án tạo ra bộ training data **~1.400 mẫu** bao gồm:


| Loại                       | Số mẫu | Mục đích                                 |
| -------------------------- | ------ | ---------------------------------------- |
| Text-to-SQL                | 600    | Dạy model viết SQL từ câu hỏi tiếng Việt |
| Phân tích dữ liệu (pandas) | 400    | Dạy model xử lý và tính toán             |
| Supervisor routing         | 200    | Dạy model lập kế hoạch phân công agent   |
| Insight generation         | 150    | Dạy model viết business insight          |
| Error reflection           | 100    | Dạy model tự chẩn đoán và sửa lỗi        |


Dữ liệu được tạo bằng **GPT-4o-mini** sau đó kiểm tra tự động bằng cách chạy thực tế trên PostgreSQL — chỉ giữ lại các mẫu hợp lệ.

---

## 4. Model AI & fine-tuning

### 4.1 Model nền

- **Model chính:** `qwen2.5-coder:7b-instruct-q5_K_M` — chạy local trên Ollama  
- **Model dự phòng:** `llama3.1:8b-instruct-q4_K_M`  
- **Fallback online:** GPT-4o API (nếu Ollama không đáp ứng được)

### 4.2 Chiến lược fine-tuning

Model được fine-tune bằng kỹ thuật **LoRA (Low-Rank Adaptation)** — chỉ huấn luyện ~1% tham số của model, giúp:

- Tiết kiệm bộ nhớ (chạy được trên MacBook M2 16GB)  
- Học nhanh (~3 epochs, khoảng 3–4 giờ)  
- Giữ nguyên kiến thức nền của model gốc

**Phương án A (local):** MLX-LM trên Apple Silicon M2  
**Phương án B (fallback):** Unsloth + Google Colab Pro nếu M2 không đủ  

### 4.3 Kết quả baseline (trước fine-tune)

Đã đo lường trên **98 mẫu** test với model gốc `qwen2.5-coder:7b`:


| Chỉ số                 | Kết quả baseline | Mục tiêu |
| ---------------------- | ---------------- | -------- |
| SQL Execution Accuracy | **84,4%**        | ≥82%     |
| JSON Valid Rate (tổng) | **94,4%**        | ≥88%     |
| Supervisor JSON Rate   | **94,4%**        | ≥88%     |
| Python Syntax Rate     | **96,7%**        | ≥90%     |
| Insight JSON Rate      | **92,9%**        | ≥88%     |
| Latency trung bình     | **75,8 s**       | ≤25 s    |


**Nhận xét:** Model gốc đã vượt hầu hết mục tiêu chất lượng. Vấn đề chính cần giải quyết sau fine-tune là **giảm latency** (hiện ~76 s, mục tiêu ≤25 s).

---

## 5. Giao diện người dùng

Giao diện được xây dựng bằng **Streamlit**, gồm 4 khu vực chính:

```
┌────────────────────────────────────────────────────┐
│  ADBA — Autonomous Data & Business Intelligence    │
├────────────────────────────────────────────────────┤
│  Chat Input                                        │
│  "So sánh doanh thu Q4-2024 vs Q4-2023 theo vùng"   │
├──────────────────────┬─────────────────────────────┤
│  Bảng dữ liệu      │  Biểu đồ                      │
│  (st.dataframe)      │  (st.image)                 │
├──────────────────────┴─────────────────────────────┤
│  Business Insight Card                             │
│  Finding / Evidence / Anomaly / Action / Confidence│
├────────────────────────────────────────────────────┤
│  Agent Trace (expandable)                          │
│  > Supervisor: Lập kế hoạch 4 bước                 │
│  > SQL Agent: Truy vấn 234 rows                    │
│  > Python Agent: Tính YoY, phát hiện 1 outlier     │
│  > Viz Agent: Bar chart — grouped by region        │
│  > Insight Agent: Confidence = high                 │
└────────────────────────────────────────────────────┘
```

Người dùng có thể mở rộng phần **Agent Trace** để xem chi tiết từng agent đã làm gì — hữu ích để kiểm tra và debug.

---

## 6. Xử lý lỗi & tự phục hồi

Hệ thống có hai cơ chế tự phục hồi:

### 6.1 Retry tự động tại từng agent

- **SQL Agent:** thử lại tối đa 3 lần nếu câu SQL bị lỗi cú pháp hoặc thực thi  
- **Python Agent:** thử lại tối đa 2 lần nếu code pandas bị lỗi runtime  
- **Viz Agent:** thử lại tối đa 2 lần nếu code matplotlib lỗi  
- **Supervisor:** có thể thử lại lập kế hoạch nhiều lần (theo cấu hình trong code, ví dụ tối đa 3 lần) trước khi đánh dấu thất bại

### 6.2 Reflector Agent — phân tích lỗi chuyên sâu

Khi một agent vượt quá số lần retry hoặc cần chẩn đoán, **Reflector Agent** được kích hoạt để:

1. Phân loại lỗi (ví dụ: `sql_syntax`, `sql_logic`, `python_runtime`, `python_logic`, `data_quality`, `schema_mismatch`, `chart_error`)
2. Xác định nguyên nhân gốc rễ
3. Cung cấp **context đã sửa** cho agent thử lại

**Giới hạn an toàn:** tối đa **8 lần** reflector can thiệp theo từng agent chuyên môn (ngưỡng trong code có thể điều chỉnh). Nếu vượt ngưỡng, router có thể bỏ qua bước bị kẹt và kết thúc pipeline an toàn thay vì lặp vô hạn.

### 6.3 An toàn thực thi & timeout (engineering)

- Truy vấn SQL có thể áp dụng **statement timeout** (biến môi trường, ví dụ `SQL_TIMEOUT_MS`).  
- Thực thi code Python/Viz trong sandbox có **timeout** (ví dụ `PANDAS_EXEC_TIMEOUT_SECONDS`) để tránh treo pipeline.  
- Chi tiết triển khai: xem `graph/tools/sql_tool.py`, `graph/tools/python_tool.py`, `graph/agents/viz_agent.py`.

---

## 7. Lộ trình triển khai (38 ngày)

### Milestone 1 — Foundation & demo (ngày 1–22)


| Giai đoạn        | Thời gian  | Nội dung                                       | Checkpoint                  |
| ---------------- | ---------- | ---------------------------------------------- | --------------------------- |
| Infrastructure   | Ngày 1–2   | Docker + PostgreSQL + 3 schemas + dữ liệu seed | DB hoạt động                |
| Data & schemas   | Ngày 3–4   | Pydantic models + prompt templates             | Import không lỗi            |
| Dataset          | Ngày 4–6   | Sinh ~1.500 mẫu + validate + split             | ≥1.200 mẫu hợp lệ           |
| **Checkpoint 1** | Ngày 7     | Đo baseline model gốc                          | Ghi nhận số liệu            |
| Fine-tuning      | Ngày 8–12  | MLX-LM LoRA training ~3 epochs                 | Val loss giảm đều           |
| Model export     | Ngày 13–14 | Export GGUF + ModelClient                      | **Checkpoint 2**            |
| Agent pipeline   | Ngày 15–20 | Toàn bộ 6 agents + LangGraph wiring            | ≥80% integration tests pass |
| **Demo**         | Ngày 21–22 | Streamlit UI + 5 demo queries                  | **Checkpoint 3**            |


### Milestone 2 — Full multi-agent + production (ngày 23–38)


| Giai đoạn         | Thời gian  | Nội dung                                            |
| ----------------- | ---------- | --------------------------------------------------- |
| Refactor agents   | Ngày 23–24 | Tách biệt rõ ràng, chuẩn hóa state                  |
| Supervisor v2     | Ngày 25–26 | Dependency resolver đầy đủ + 200 mẫu routing mới    |
| Insight v2        | Ngày 27    | Anomaly tools nâng cao (sigma-based, YoY/QoQ)       |
| Reflector v2      | Ngày 28    | Domain-aware error classification + smart rerouting |
| Integration tests | Ngày 29–30 | 20 queries (10 đơn giản + 10 phức tạp)              |
| Docker sandbox    | Ngày 31–32 | Isolated code execution + Ragas evaluation          |
| So sánh           | Ngày 33–35 | Single vs multi-agent trên 120 test queries         |
| Production        | Ngày 36–37 | docker-compose hoàn chỉnh + README                  |
| **Release**       | Ngày 38    | Video demo ~5 phút + tag v1.0.0                     |


---

## 8. Mục tiêu chất lượng


| Chỉ số                  | Mục tiêu   | Ý nghĩa                                           |
| ----------------------- | ---------- | ------------------------------------------------- |
| SQL Execution Accuracy  | ≥82%       | Tỷ lệ câu SQL chạy không lỗi                      |
| JSON Plan Valid Rate    | ≥88%       | Tỷ lệ kế hoạch Supervisor hợp lệ                  |
| Agent Routing Accuracy  | ≥85%       | Supervisor chọn đúng agents cho từng loại câu hỏi |
| End-to-End Success Rate | ≥75%       | Tỷ lệ câu hỏi → insight không crash               |
| Insight Quality Score   | ≥3,8/5     | Điểm GPT-4o đánh giá chất lượng finding + action  |
| Avg Retry Count         | ≤1,2/query | Số lần thử lại trung bình mỗi câu hỏi             |
| Latency P50             | ≤25 s      | Thời gian phản hồi trung vị                       |


---

## 9. Các điểm dừng & phương án dự phòng

Dự án có **4 checkpoint** với phương án fallback rõ ràng nếu không đạt:


| Checkpoint        | Ngày | Nếu fail                                                                 |
| ----------------- | ---- | ------------------------------------------------------------------------ |
| Dataset Quality   | 7    | +1 ngày manual filter + viết tay 100 mẫu chất lượng cao                  |
| Model Quality     | 14   | Dùng GPT-4o API cho Supervisor + SQL Agent thay fine-tuned model         |
| Single-Agent Demo | 22   | Cắt bỏ Viz Agent, chỉ giữ SQL + Insight output                           |
| Multi-Agent Gate  | 30   | Submit single-agent demo + báo cáo kiến trúc multi-agent như future work |


---

## 10. Cấu trúc thư mục dự án

```
adba/
├── app.py                    # Streamlit UI
├── docker-compose.yml        # PostgreSQL + services
├── requirements.txt          # Python dependencies
│
├── data/                     # Training data + DB schemas + seed
│   ├── schemas/              # 3 SQL schemas (sales, inventory, hr)
│   └── seed/                 # Synthetic seed data generator
│
├── model/                    # Ollama client wrapper
│   ├── model_client.py       # ModelClient với retry + fallback
│   └── model_config.py       # Temperature, max_tokens per agent
│
├── perception/               # Schema introspection → info_box JSON
│
├── schemas/                  # Pydantic validation models
│   ├── plan_schema.py        # ExecutionPlan + AgentStep
│   └── insight_schema.py     # InsightOutput + AnomalyInfo
│
├── graph/                    # LangGraph multi-agent pipeline
│   ├── state.py              # MultiAgentState TypedDict
│   ├── multi_agent.py        # Compiled graph
│   ├── agents/               # 6 agent implementations
│   └── tools/                # SQL, Python, Viz, Insight tools
│
├── training/                 # End-to-end fine-tuning pipeline
│   ├── generate_data.py      # GPT-4o-mini batch generation
│   ├── validate_dataset.py   # SQL validation
│   ├── format_sharegpt.py    # ShareGPT format conversion
│   ├── train_mlx.py          # MLX-LM training runner
│   └── mlx_config.yaml       # LoRA hyperparameters
│
├── eval/                     # Evaluation framework
│   └── eval_runner.py        # Automated evaluation on test set
│
└── tests/                    # Unit + integration tests
    ├── unit/                 # Unit tests cho agents
    └── integration/          # End-to-end queries
```

---

## 11. Tech stack


| Layer            | Công nghệ               | Lý do chọn                                              |
| ---------------- | ----------------------- | ------------------------------------------------------- |
| Agent framework  | LangGraph               | State machine phù hợp cho multi-agent pipeline phức tạp |
| LLM runtime      | Ollama (Metal backend)  | Chạy local M2, không phụ thuộc cloud API                |
| Model            | Qwen2.5-Coder 7B        | Code generation tốt, nhỏ gọn cho M2                     |
| Database         | PostgreSQL 15           | Chuẩn production, hỗ trợ đầy đủ SQL features            |
| Data processing  | pandas + numpy + scipy  | Ecosystem chuẩn cho data analysis                       |
| Visualisation    | matplotlib + seaborn    | Đủ tính năng, kiểm soát được output                     |
| Validation       | Pydantic v2             | Đảm bảo output của LLM đúng schema                      |
| UI               | Streamlit               | Nhanh, không cần frontend riêng                         |
| Fine-tuning      | MLX-LM                  | Tối ưu cho Apple Silicon M2                             |
| Containerisation | Docker + docker-compose | Đảm bảo reproducibility                                 |


---

## 12. Hạn chế đã biết

Những điểm sau **không nằm trong phạm vi** của dự án này (hoặc chỉ ở mức tối thiểu):

- **Reinforcement Learning:** chỉ dùng Supervised Fine-Tuning (SFT), không có RL từ execution feedback như paper gốc đề xuất.  
- **Xử lý dữ liệu real-time:** chỉ hỗ trợ batch queries, không có streaming.  
- **Generalization:** chỉ test trên 3 schema cố định (sales, inventory, HR), chưa kiểm tra với schema hoàn toàn mới.  
- **Parallel execution:** do giới hạn RAM 16GB trên M2, các agents chạy **tuần tự**, không song song.  
- **Privacy:** không có cơ chế ẩn danh hóa dữ liệu nhạy cảm đầy đủ.  
- **Airflow integration:** không trong scope — tính năng optional cho phiên bản tương lai.

---

## Tài liệu tham chiếu trong repo

- **Đặc tả kiến trúc & quy ước coding:** `.cursorrules`  
- **Trạng thái triển khai & lệnh chạy:** `AGENTS.md`  
- **Phân tích luồng multi-agent:** `docs/multiagent_analysis.md`

---

*Tài liệu này phản ánh trạng thái dự án tính đến ngày 14/05/2026. Các con số baseline (mục 4.3) lấy từ `eval/baseline_results.json` tại thời điểm đo.*