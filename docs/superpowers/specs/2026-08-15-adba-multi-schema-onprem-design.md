# ADBA — Agent chạy nội bộ trên schema của khách hàng

**Ngày:** 2026-08-15
**Trạng thái:** Design đã duyệt, chờ implementation plan
**Spec liên quan:** [`2026-08-12-adba-mcp-hardening-design.md`](./2026-08-12-adba-mcp-hardening-design.md) — hardening + ranh giới MCP cho bản triển khai đơn schema. Spec này **không thay thế** spec đó; nó nối vào các điểm `ConnectionProfile`, deadline/budget, và BEAVER.

---

## 1. Bối cảnh & mục tiêu

Spec `2026-08-12` thiết kế cho một lần triển khai nội bộ, một schema đã biết trước, hardening để chạy thật. Spec này trả lời câu hỏi tiếp theo: **đóng gói ADBA thành thứ cài được vào hạ tầng của khách hàng, trỏ vào database của họ.**

### 1.1 Ràng buộc

| Ràng buộc | Giá trị |
|---|---|
| Hình thái triển khai | Bundle chạy trong mạng khách, kết nối DB của khách. Dữ liệu không rời mạng. |
| Quy mô schema mục tiêu | **30–200 bảng** cần truy vấn (không tính log/audit/temp) |
| Mô hình chuyên biệt hóa | **Một model chung** cho mọi khách. Không fine-tune riêng từng khách. |
| Chú giải ngữ nghĩa | Sinh tự động bằng LLM local, khách sửa những mục bị gắn cờ |
| Hàm mục tiêu | Độ tin cậy → độ trễ → chi phí vận hành (kế thừa spec trước) |
| SLO độ trễ | 30–60s, thiết kế nhắm 45s (kế thừa spec trước) |
| Egress | **Không**. Kể cả lúc onboarding. |

### 1.2 Mục tiêu

Cài được ADBA vào một Postgres lạ 30–200 bảng, đạt ngưỡng bàn giao đo được, mà không cần train lại model cho khách đó.

### 1.3 Không phải mục tiêu

- Multi-tenant trong một tiến trình. Mỗi khách một bundle, một profile.
- Hỗ trợ datasource ngoài PostgreSQL.
- SaaS ngang. Xem mục 10.1.

---

## 2. Hai quyết định nền tảng

### 2.1 "Model chung" không có nghĩa là bỏ fine-tune

LoRA hiện tại học *schema của ADBA*. Nó phải được đổi mục tiêu thành học *kỹ năng đọc bất kỳ schema nào được đưa vào context*: đọc DDL → chọn bảng → viết JOIN theo FK được khai báo. Kỹ năng đó chuyển giao được; thuộc lòng tên bảng thì không.

Dây chuyền train dùng lại được: `generate_data.py` chạy trên **nhiều** schema thay vì một, mỗi mẫu mang theo lát schema tương ứng trong prompt (mục 6.3).

Hệ quả: `beaver_exec_accuracy` từ spec trước thôi là kiểm chứng ngoại vi, nó trở thành **hàm mục tiêu** của việc train lại.

### 2.2 Thứ tự đòn bẩy khi accuracy không đạt: context trước, tham số sau

Qwen2.5-7B có GQA 4 KV head, 28 layer → KV cache ≈ **56 KB/token**. Nâng ctx 4k → 32k tốn thêm ~1,6 GB VRAM; cộng ~5,5 GB trọng số q5_K_M thì **7B ở 32k ctx vẫn vừa một card 16 GB**.

Qwen2.5-32B có 64 layer, 8 KV head → **256 KB/token**, gấp 4,6 lần. Ở 32k ctx là ~8,4 GB KV + ~19 GB trọng số q4 ≈ **28 GB** — hết cửa với card 24 GB, phải lên A100 40GB hoặc 2 card.

> 7B @ 32k = một workstation bất kỳ. 32B @ 32k = một server thật. Đó không phải chênh lệch giá, đó là chênh lệch **ai mua được**. Toàn bộ lợi thế ship-được-on-prem nằm ở phía 7B.

**Thứ tự bắt buộc khi accuracy không đạt:**

1. Cắt gọn biểu diễn schema (mục 3.4)
2. Nâng `OLLAMA_NUM_CTX` / `max_model_len`
3. Sửa dữ liệu train và chú giải (mục 6.3)
4. **Chỉ khi hết ba bước trên** mới nâng tham số model

Bước 4 phải kèm quyết định thương mại rõ ràng, không được coi là phương án dự phòng mặc định.

---

## 3. Kiến trúc

### 3.1 Hiện trạng — seam đã nằm đúng chỗ

`info_box` đi qua hệ thống theo đúng một đường:

| Vị trí | Vai trò |
|---|---|
| `app.py:226` | Đọc `info_box_latest.json` từ đĩa |
| `graph/multi_agent.py:94` | `run_graph(query, info_box)` |
| `graph/state.py:14` | `MultiAgentState.info_box` |
| `graph/agents/supervisor.py:36` | `json.dumps(info_box)` → placeholder `{info_box}` |
| `graph/agents/sql_agent.py:69` | `json.dumps(info_box)` → placeholder `{info_box}` |

Đúng **hai** nơi tiêu thụ, không nơi nào khác. Nghĩa là chỉ cần đổi **bên cung cấp**; hai bên tiêu thụ giữ nguyên chữ ký. Đây là lý do thiết kế này khả thi với chi phí vừa phải.

### 3.2 Hai trục thời gian

Tách bạch **onboarding** (một lần, mỗi khách) và **query time** (mỗi câu hỏi) là quyết định kiến trúc chính.

#### Onboarding — chạy tại chỗ khách, offline

```
Postgres của khách (role read-only)
  │
  1. extract_schema.py    → schema_raw.json   bảng, cột, kiểu, PK, FK, row_count, sample_rows
  2. annotate_schema.py   → schema.yaml       mô tả bảng/cột + cờ confidence      [LLM local]
  3. ── người ──          khách sửa các mục bị gắn cờ low
  4. build_profile.py     → profile/          schema.yaml + index.npz + examples.jsonl + profile.json
  5. verify_profile.py    → report.md         chấm golden set, ra quyết định bàn giao
```

Bước 1 mở rộng từ `perception/extract_info_box.py` đã có.

**Bước 2 gọi model local, tuyệt đối không gọi OpenAI.** Nếu không thì toàn bộ schema của khách rời khỏi mạng ngay ở thao tác đầu tiên của quá trình cài đặt — và đó chính là thứ duy nhất khiến họ chọn on-prem.

**`sample_rows` không được ghi vào `profile/`.** Nó cần cho bước 2 để đoán ý nghĩa cột, nhưng `profile/` là thứ sẽ bị copy đi khi hỗ trợ kỹ thuật. Profile chỉ chứa cấu trúc và mô tả, không chứa dữ liệu thật.

#### Query time

```
câu hỏi + danh tính user
  │
  ├─ resolve_schema_context(profile, question, user)      ◄── mới; hàm thuần, không gọi LLM, <100ms
  │     mode=full      → toàn bộ schema đã render
  │     mode=retrieval → top-K theo embedding
  │                      → mở rộng 1 bước theo cạnh FK
  │                      → lọc bảng user không có quyền
  │                      → retrieve top-2 few-shot từ examples.jsonl
  │     ⇒ SchemaContext{ tables[], rendered_text, few_shots[], allowed_tables }
  │
  ├─ run_graph(query, schema_context)
  │     supervisor  ← rendered_text
  │     sql         ← rendered_text + few_shots   (dùng lại, KHÔNG retrieve lần nữa)
  │     python / viz / insight  ← không cần schema
  │     reflector
  │
  └─ execute_sql(..., allowed_tables=schema_context.allowed_tables)
```

**Retrieval chạy một lần mỗi câu hỏi, không phải mỗi node.** Nó **không** là node LangGraph: nó deterministic, không gọi LLM, không có kiểu lỗi cần retry. Biến nó thành node là mua thêm một hop và một nhánh lỗi để đổi lấy không gì cả. Nó chạy trước `make_initial_state()`.

Chi phí runtime: index dựng một lần lúc onboarding; lúc chạy chỉ là embed một câu hỏi ngắn + cosine trên ≤200 vector. Dưới 100ms, chạy CPU thoải mái.

### 3.3 Công tắc `schema_mode` — quyết một lần lúc cài

`build_profile.py` đo kích thước schema đã render. Dưới ngưỡng thì ghi `schema_mode: full` và bỏ qua retrieval hoàn toàn; trên ngưỡng thì `schema_mode: retrieval`.

**Ngưỡng đề xuất: 6.000 token schema đã render** — ở mức ~140 token/bảng (mục 3.4) thì tương đương **khoảng 40 bảng**.

Đây **không** phải fallback động. Một nhánh code, một công tắc, quyết một lần lúc onboarding. Khách nhỏ được sự đơn giản của chế độ full; khách lớn được sự co giãn của retrieval. Không phải test tổ hợp hai đường chạy trên cùng một cài đặt.

### 3.4 `ConnectionProfile` mở rộng, và biểu diễn schema dạng DDL

Spec `2026-08-12` đã đưa ra `ConnectionProfile` gom `dsn` + `allowed_tables` + `info_box`. Spec này thay ruột chứ không đổi cấu trúc:

| Trường | Trước | Sau |
|---|---|---|
| `dsn` | biến môi trường toàn cục | giữ nguyên trong profile |
| `allowed_tables` | frozenset 9 tên hardcode (`sql_tool.py:27-31`) | **động, theo từng query** — xem dưới |
| `info_box` | file JSON tĩnh | `schema.yaml` + `index.npz` + `examples.jsonl` + fingerprint |

**Biểu diễn DDL thay cho `json.dumps`.** Hai nơi tiêu thụ hiện dump JSON thô — đó là nguồn gốc của **3,7 KB/bảng** (đo được: `info_box_all.json` = 33.497 B cho 9 bảng). Render dạng DDL:

```sql
-- Đơn hàng bán cho khách hàng
CREATE TABLE orders (
  order_id     INT PRIMARY KEY,
  customer_id  INT REFERENCES customers(customer_id),
  order_date   DATE,
  total_amount NUMERIC  -- tổng tiền sau thuế
);
```

≈ **500 B cho 12 cột ≈ 140 token**, nhỏ hơn JSON 2–3 lần. Và dễ hơn cho model: Qwen2.5-**Coder** đã đọc hàng triệu file SQL lúc pretrain; JSON mô tả schema thì không.

Con số 140 token/bảng là cơ sở cho mọi ước tính ngân sách trong spec này (mục 3.3, mục 5.1). Nó phải được đo lại trên schema thật ở pha 1 chứ không giữ nguyên như giả định.

**`allowed_tables` đổi bản chất, không chỉ đổi nguồn.** Sau thay đổi nó bằng đúng tập bảng đã nằm trong prompt của lượt đó. Guard và prompt sinh ra từ **cùng một nguồn** — chúng không thể lệch nhau, và tập cho phép hẹp lại theo từng câu hỏi thay vì mở toàn bộ schema. Đây là cải thiện an toàn thực chất so với whitelist tĩnh.

### 3.5 Tách prompt template ba đường

`prompts/text_to_sql.txt` hiện dài 71 dòng, trong đó **khoảng 45 dòng là mô tả của một database cụ thể**:

- Dòng 9–11, 13, 15–17, 21: 8 trong 13 "critical rules" gọi tên bảng/cột ADBA (`orders` không có cột month; `orders.quarter`/`year` là GENERATED; `status IN ('completed','processing')`; `stock.product_id = products.id`; `payroll` có cột month).
- Dòng 23–65: cả hai few-shot example viết trọn trên schema ADBA (`orders.region`, `orders.amount`, `products.sku`).

Trỏ nó vào DB khách thì model được dặn về những bảng không tồn tại, kèm hai ví dụ mẫu dùng cột không tồn tại. Và **LoRA đã fine-tune với chính prompt này**, nên các đặc thù đó nằm trong phân phối đầu vào mà nó đã học. Đây là cơ chế quan sát được trong source giải thích vì sao điểm trên schema lạ nhiều khả năng rơi về mức base model.

| Phần | Đi đâu |
|---|---|
| Kỹ năng chung (~12 dòng: ưu tiên CTE, dùng alias, LIMIT, định dạng output, first-token) | Ở lại template |
| Đặc thù schema (`quarter` là GENERATED, quy ước `status` cho doanh thu) | Vào `schema.yaml`, render ra lúc chạy |
| Few-shot example | **Sinh riêng cho từng khách** lúc onboarding, lưu `examples.jsonl`, retrieve top-2 theo câu hỏi |

Few-shot đi qua chính index đã có nên gần như miễn phí, và đây là cách quen thuộc để nâng accuracy text-to-SQL trên schema lạ.

**Vị trí `{info_box}` phải chuyển xuống cuối system prompt, ngay trước `{task}`.** Hiện nó ở dòng 6, giữa ROLE và RULES. Ở `schema_mode=retrieval`, khối schema thay đổi mỗi query nên 60 dòng rules cố định phía sau bị prefill lại vô ích mỗi lượt. Chuyển xuống cuối thì phần cố định thành prefix cache được, và schema nằm sát câu hỏi cũng tốt hơn cho attention. Hai lợi ích trùng hướng.

`prompts/supervisor_routing.txt` có `{info_box}` ở dòng 59/149 — áp dụng cùng quy tắc.

### 3.6 Đóng gói

Bundle hiện tại (`docker-compose.prod.yml`) là bản nháp và **không khởi động được**: nó tham chiếu `sandbox/Dockerfile` (thư mục `sandbox/` rỗng) và `scripts/ollama-entrypoint.sh` (không tồn tại).

Tách làm hai file, không dùng profiles:

| File | Dùng cho | Khác biệt |
|---|---|---|
| `docker-compose.demo.yml` | Demo/dev tự chứa | Giữ service `postgres` + `seed-data` như hiện tại |
| `docker-compose.onprem.yml` | Cài tại khách | **Bỏ hẳn** `postgres` và `seed-data`; app trỏ vào DSN của khách |

`seed-data` chạy `seed_data.py`, tức **ghi dữ liệu**. Nó tuyệt đối không được có mặt trong bundle giao khách.

Hai thay đổi bắt buộc khác cho bản on-prem:

- **`ENABLE_OPENAI_FALLBACK` mặc định `0`**, và mạng của app không có route egress. Hiện mặc định là `1` — app sẽ gọi OpenAI cho node supervisor và sql khi Ollama lỗi, âm thầm, đúng lúc hệ thống đang trục trặc. Trong một buổi security review on-prem đây là điều khoản giết deal. Phải tắt được và **chứng minh được là tắt** ở tầng mạng, không chỉ ở biến môi trường.
- **Model phải nằm sẵn trong image hoặc volume ship kèm.** Không có egress thì `ollama pull` không chạy. Dùng `ollama create` từ Modelfile + file GGUF ship kèm, hoặc volume đã nạp sẵn.

Embedding model chạy in-process trên CPU (đề xuất `multilingual-e5-small`, ~470 MB) để không phát sinh service thứ năm.

---

## 4. Bảo mật & phân quyền

Kế thừa toàn bộ bốn lớp phòng thủ SQL và cô lập sandbox từ spec `2026-08-12`. Bổ sung phân quyền theo người dùng, ba lớp:

| Lớp | Cơ chế | Ship kèm sản phẩm? |
|---|---|---|
| 1 | Retriever lọc bảng theo quyền → model **không nhìn thấy** bảng bị cấm | Có |
| 2 | `execute_sql` dùng `allowed_tables` từ cùng `SchemaContext` → model bịa tên bảng thì thực thi từ chối | Có |
| 3 | Grant cấp cột/dòng trong Postgres | Không — khách tự cấu hình |

Lớp 1 là lớp mạnh nhất và gần như miễn phí: không có cách nào sinh SQL đọc `payroll` nếu `payroll` chưa từng xuất hiện trong context.

`app.py` hiện không có xác thực (chỉ `st.session_state` phạm vi phiên trình duyệt). Bản on-prem cần tối thiểu: định danh người dùng, và ánh xạ người dùng → tập bảng được phép. Thiết kế chi tiết của lớp xác thực nằm ngoài spec này; spec này chỉ định nghĩa **điểm cắm**: `resolve_schema_context(profile, question, user)` nhận `user` và lọc theo đó.

---

## 5. Ngân sách thời gian & xử lý lỗi

### 5.1 Ngân sách — thiết kế này gần như không đụng vào

Đếm token thực tế:

| Chế độ | Thành phần | Tổng | ctx cần |
|---|---|---|---|
| `retrieval`, 12 bảng | rules ~700 (cố định) + DDL ~1.700 + few-shot ~400 + task ~100 | **~2.900** | 8k |
| `full`, 40 bảng (sát ngưỡng) | rules ~700 + DDL ~5.600 + few-shot ~400 + task ~100 | **~6.800** | 16k |

Chế độ `full` cần ctx 16k chứ không phải 8k: cộng thêm `AGENT_MAX_TOKENS["sql"] = 1024` cho phần sinh, 8k không còn biên an toàn.

Ở `full`, toàn bộ prompt cố định giữa mọi query → sau query đầu prefill gần bằng 0 nhờ prefix caching.
Ở `retrieval`, phần biến thiên ~2.300 token ≈ 0,6s prefill trên một card tầm 4090.

Chi phí thật vẫn nằm ở decode: node SQL sinh ~150 token ở ~40 tok/s ≈ 4s, nhân 5 node ≈ 20s. **Con số đó không đổi vì thiết kế này.** Cơ chế deadline + reserve + finalize node ở spec `2026-08-12` vẫn cai quản ngân sách, không cần thiết kế lại — chỉ cộng thêm ~0,7s cho retrieval và prefill biến thiên.

### 5.2 Bốn kiểu lỗi mới

**`profile_stale` — kiểu lỗi nguy hiểm nhất của on-prem.**
DBA của khách thêm cột, đổi tên bảng, và profile vẫn mô tả schema cũ. `profile.json` lưu fingerprint (hash của tập tên bảng + cột). Kiểm tra lúc khởi động và định kỳ; lệch thì **từ chối chạy** kèm chỉ dẫn chạy `refresh_profile.py` — không phải cảnh báo rồi chạy tiếp.

Kèm theo là một yêu cầu thiết kế thật: **`schema.yaml` phải merge được.** Refresh chỉ re-annotate bảng mới hoặc đã đổi, giữ nguyên mọi mục có `reviewed_by: human`. Nếu refresh xóa công sức của khách thì họ sẽ không sửa lần thứ hai, và chất lượng chú giải — tức trần accuracy — tụt vĩnh viễn.

**`retrieval_miss`.**
Reflector báo `error_category: schema_mismatch` → gọi lại `resolve_schema_context(..., must_include=[bảng thiếu])`. Mở rộng có mục tiêu, không phải fallback full-schema. Một lượt, rẻ.

**`annotation_unreviewed`.**
Câu trả lời dựa trên bảng khách chưa duyệt chú giải → không phải lỗi, nhưng hiện cảnh báo kèm câu trả lời. Đây cũng là đòn bẩy khiến họ chịu duyệt YAML.

**`permission_denied`.**
Trả lời "câu hỏi này chạm dữ liệu bạn không được cấp quyền" — **không nêu tên bảng**, vì chính tên bảng đã là thông tin.

### 5.3 Điều cố ý không làm

Không thêm cơ chế runtime phát hiện "SQL chạy trơn tru nhưng lấy nhầm bảng". Nếu retriever lấy một bảng trông hợp lý và SQL execute thành công ra số sai thì **không lớp nào bắt được** — không ở thiết kế này, không ở thiết kế nào khả thi trong ngân sách.

Nó phải được chặn ở `verify_profile.py` lúc onboarding bằng ngưỡng recall của retriever (mục 6.5). Tức là dịch một vấn đề runtime không giải được thành một tiêu chí bàn giao đo được.

---

## 6. Eval & kiểm thử

### 6.1 Ba tầng, không gộp

| Tầng | Đo gì | Cần LLM? | Cần DB? |
|---|---|---|---|
| 1 | Recall của retriever: `SchemaContext` có chứa **đủ** tập bảng đúng? | Không | Không |
| 2 | Execution accuracy trên schema khách (so tập kết quả, không so chuỗi) | Có | Có |
| 3 | End-to-end, dùng lại `eval/eval_runner.py` | Có | Có |

Tầng 1 phải là "đủ", không phải "có": thiếu một bảng JOIN là SQL sai chắc chắn.

Tập bảng đúng **parse được từ SQL mẫu** bằng `sqlparse` (đã có trong phụ thuộc của spec `2026-08-12`). Nên tầng 1 chạy hết vài giây cho hàng nghìn câu, không GPU, không dựng DB. Đây là vòng lặp sẽ chạy hàng trăm lần khi chỉnh retriever, và nó là lý do tách tầng.

### 6.2 Ba benchmark, ba câu hỏi

| Benchmark | Trả lời câu hỏi |
|---|---|
| Spider | Cơ chế có chạy không — model dùng được schema chưa từng thấy? Schema nhỏ, nhiều domain. |
| BIRD | Có trụ trên dữ liệu bẩn, giá trị thật, cần kiến thức ngoài schema? |
| BEAVER (arXiv:2409.02038) | Có trụ ở quy mô 30–200 bảng doanh nghiệp — đúng phân khúc nhắm tới? |

Cả ba đều kèm SQL mẫu nên **cả ba đo được tầng 1 ngay**. Bật tầng 1 trên cả ba từ đầu; tầng 2 chỉ trên BEAVER như spec `2026-08-12` đã chốt.

> **Cần xác minh khi dựng harness:** số database, số bảng trung bình mỗi DB, và license của từng bộ. Các con số này không được chốt từ trí nhớ; **hạng mục đầu tiên của pha 0** là tải bộ dữ liệu và ghi lại cấu hình thực tế vào `eval/README_multischema.md`.

### 6.3 Hệ quả lên dữ liệu train

LoRA đổi mục tiêu (mục 2.1) thì tập train phải đổi theo. Spider/BIRD **train split** cho sẵn bộ ba (schema, câu hỏi, SQL) — đúng định dạng cần. `generate_data.py` thêm một chế độ chuyển chúng sang layout prompt mới: rules chung + schema đã retrieve + few-shot + task. Tập train đi từ 787 mẫu lên vài nghìn.

**Cảnh báo phương ngữ:** Spider/BIRD là SQLite, ADBA là Postgres — `strftime` vs `EXTRACT`, khác biệt kiểu dữ liệu, hàm cửa sổ. Hoặc dịch phương ngữ khi chuyển, hoặc chấp nhận model học lẫn; phương án thứ hai sẽ ăn mòn đúng phần accuracy đang có. Spec này chọn **dịch**, và ghi nhận đây là hạng mục có chi phí riêng trong pha 2.

**Golden set trên schema ADBA hiện tại giữ nguyên làm bộ chống hồi quy.** Năng lực đang có (SQL 96,9%) không được phép mất khi đổi sang model chung. Đây là điều kiện chặn của pha 2.

### 6.4 Vấn đề trứng-gà lúc onboarding

Khách mới thì lấy đâu golden set để `verify_profile.py` chấm?

`build_profile.py` sinh 40–60 cặp (câu hỏi, SQL) từ schema đã chú giải, chạy thử, giữ cặp nào execute được. Chia đôi:

- một nửa → `examples.jsonl`, làm nguồn few-shot
- một nửa → **analyst của khách duyệt**, thành golden set

Ước tính nửa ngày công của họ, cộng với thời gian sửa YAML ở bước 3.

Đây là chi phí người thật của mỗi lần cài. "Cài trong một ngày" đúng với *máy*, không đúng với *người*. Phải nói trước với khách thay vì để họ phát hiện.

### 6.5 Ngưỡng bàn giao

Spec này có **hai loại cổng khác nhau**, đừng lẫn:

- **Cổng phát triển** (mục 7): quyết định một pha đã xong chưa. Áp một lần, cho đội phát triển.
- **Cổng bàn giao** (mục này): quyết định một lần cài đặt cụ thể có được giao cho khách không. Áp mỗi khách, chạy bởi `verify_profile.py`.

**Cổng bàn giao, tầng 1: recall ≥ 95%.**

Định nghĩa chính xác: tỉ lệ câu hỏi trong golden set của khách mà `SchemaContext` chứa **toàn bộ** tập bảng đúng. Một câu thiếu dù chỉ một bảng thì tính là trượt — không tính điểm từng phần, vì thiếu một bảng JOIN là SQL sai chắc chắn.

Đây là ngưỡng chặn: dưới mức này không bàn giao. Và dưới ngưỡng thì lỗi hầu như luôn nằm ở chú giải — sửa YAML, dựng lại index, chấm lại. Vòng lặp rẻ, không đụng đến model.

**Cổng bàn giao, tầng 2: không đặt ngưỡng cứng, ghi nhận và báo cáo.**

Cần nói thẳng điều này trong spec để không ai đặt sai kỳ vọng về sau: **96,9% là điểm trên schema của chính ADBA, với prompt viết riêng cho nó và LoRA train trên nó.** Trên một ERP lạ 150 bảng, model 7B đạt lại con số đó là chuyện không xảy ra. Mức thực tế cần chuẩn bị tinh thần là **50–65%**. Đặt ngưỡng bàn giao ở 90% là tự bảo đảm dự án thất bại.

**Hệ quả lên định vị sản phẩm:** không bán độ chính xác, bán **khả năng kiểm chứng**. Luôn hiện SQL đã chạy, luôn hiện số dòng trả về, luôn cho analyst xem và sửa trước khi dùng. Sản phẩm không phải "tin câu trả lời" mà là "khỏi phải tự viết SQL". Với cách đóng khung đó, một hệ thống 60% vẫn tiết kiệm thời gian thật; với cách đóng khung kia, 60% là vô dụng. Chênh lệch nằm ở UI và ở lời hứa, không ở model.

### 6.6 Kiểm thử

Gần như toàn bộ phần mới là hàm thuần không gọi LLM, nên unit-test được bình thường:

| Đơn vị | Kiểm gì |
|---|---|
| `render_schema()` | Snapshot test đầu ra DDL |
| `resolve_schema_context()` | top-K, mở rộng FK 1 bước, lọc quyền, công tắc `full`/`retrieval`, `must_include` |
| Merge `schema.yaml` khi refresh | **Giữ nguyên mục `reviewed_by: human`** — thứ hỏng âm thầm nhất |
| Fingerprint schema | Phát hiện `profile_stale` khi thêm/xóa/đổi tên cột |
| `execute_sql` guard | Chặn bảng ngoài `allowed_tables` động |
| Loại trừ `sample_rows` khỏi `profile/` | Kiểm bằng test, không bằng quy ước |

Phần phụ thuộc LLM chỉ còn chất lượng chú giải và chất lượng SQL — hai thứ đó thuộc eval, không thuộc test. **Không viết test đòi model trả lời đúng.**

---

## 7. Thứ tự triển khai

| Pha | Nội dung | Điều kiện ra |
|---|---|---|
| **0** | Harness eval tầng 1: parse SQL mẫu → tập bảng đúng; tải Spider/BIRD/BEAVER, ghi cấu hình thực tế | Đo được recall của một retriever giả (random / full) làm mốc |
| **1** | `render_schema()` DDL; tách `prompts/*.txt` ba đường; chuyển `{info_box}` xuống cuối; `SchemaContext` + `resolve_schema_context()`; `ConnectionProfile` mở rộng; `allowed_tables` động | Golden set ADBA hiện tại **không hồi quy**; recall tầng 1 đo được trên cả ba benchmark |
| **2** | Train lại LoRA đa schema: chế độ mới của `generate_data.py`, dịch phương ngữ SQLite→Postgres, train, eval | Golden set ADBA không hồi quy **và** `beaver_exec_accuracy` cải thiện so với base |
| **3** | Đường onboarding: `extract_schema` → `annotate_schema` → merge YAML → `build_profile` → `verify_profile`; `refresh_profile` | Chạy trọn trên một schema thứ hai (không phải ADBA) và ra `report.md` |
| **4** | Đóng gói on-prem: `docker-compose.onprem.yml`, tắt egress, model ship kèm, điểm cắm xác thực, lọc quyền trong retriever | Cài được trên một máy sạch không có internet |

Pha 0 và 1 độc lập với việc train lại, và pha 1 có giá trị ngay cả nếu dừng ở đó (giảm token, bỏ hardcode, guard chặt hơn).

Các pha của spec `2026-08-12` (hardening, MCP boundary) chạy song song và không xung đột. Nếu buộc phải chọn thứ tự thì **hardening trước** — không nên mang một hệ thống chưa có ngân sách thời gian và chưa chặn DML sang mạng của khách hàng.

---

## 8. Phụ thuộc & hạ tầng phát sinh

| Hạng mục | Ghi chú |
|---|---|
| `sentence-transformers` + `multilingual-e5-small` (~470 MB) | Chạy CPU in-process, không thêm service |
| `PyYAML` | Đọc/ghi/merge `schema.yaml` |
| `sqlparse` | Đã có trong spec `2026-08-12`; dùng thêm cho eval tầng 1 |
| Bộ dữ liệu Spider / BIRD / BEAVER | Cần kiểm tra license trước khi đưa vào repo hoặc CI |
| File GGUF của model + Modelfile | Artifact ship kèm bundle on-prem; cần chỗ lưu trữ và quy trình phát hành |
| `docker-compose.onprem.yml` | File mới; `docker-compose.prod.yml` hiện tại đổi tên thành `docker-compose.demo.yml` |
| `sandbox/Dockerfile`, `scripts/ollama-entrypoint.sh` | **Đang thiếu** — `docker-compose.prod.yml` tham chiếu nhưng không tồn tại |
| Thư mục `profile/` mỗi khách | Cần chính sách sao lưu; chứa chú giải do người viết, mất là mất công sức |

---

## 9. Tiêu chí thành công

1. Không còn tên bảng hardcode ở bất kỳ đâu trong `graph/`, `prompts/`, `model/`.
2. `prompts/text_to_sql.txt` không chứa tên bảng/cột của bất kỳ schema cụ thể nào.
3. `render_schema()` cho ≤ 700 B/bảng ở mức chi tiết mặc định.
4. Recall tầng 1 ≥ 95% trên golden set ADBA và trên phần BEAVER đã dựng.
5. Golden set ADBA không hồi quy sau khi đổi sang model chung.
6. `beaver_exec_accuracy` được ghi nhận và báo cáo — **là số đo để quyết định, không phải chỉ tiêu phải đạt**.
7. Chạy trọn đường onboarding trên một schema chưa từng thấy, ra được `report.md`.
8. Bundle on-prem khởi động và trả lời được câu hỏi trên máy không có internet.
9. Không có `sample_rows` trong `profile/` — kiểm bằng test.
10. `refresh_profile.py` giữ nguyên 100% mục `reviewed_by: human` — kiểm bằng test.

---

## 10. Phương án đã cân nhắc — loại bỏ hoặc hoãn

### 10.1 Fine-tune riêng cho từng khách

Giữ nguyên lợi thế đo được, nhưng mỗi lần bán là một dự án vài tuần cần truy cập schema thật và một vòng train. Đó là hình dạng **dịch vụ**, không phải sản phẩm — và nó không co giãn.

**Loại bỏ** ở spec này. Nếu về sau muốn bán bản cao cấp thì nó trở thành gói nâng cấp *sau khi* khách đã chạy được bản chung, không phải điều kiện để bắt đầu.

### 10.2 Nhét cả schema, dựa hoàn toàn vào prefix caching (không retrieval)

Không có thành phần mới, không có kiểu lỗi "chọn nhầm bảng để đưa vào context". Nhưng 7B đọc 150 bảng cùng lúc sẽ chọn sai bảng, và nó buộc bản on-prem phải ship vLLM vì prefix cache của Ollama yếu hơn nhiều. Trần của cách này là trần chú ý của model, không phải trần context — ngưỡng ở mục 3.3 đặt bảo thủ hơn (~40 bảng) chính vì lý do đó.

**Không loại bỏ — nó chính là `schema_mode: full`** ở mục 3.3, áp dụng cho khách dưới ngưỡng.

### 10.3 Retrieval kèm fallback full-schema khi lỗi

Đường fallback là đường chậm nhất và nó chạy đúng lúc ngân sách đã cạn vì vừa tiêu một lượt SQL cộng một lượt reflector. Nó cũng nhân đôi số cấu hình phải test.

**Loại bỏ**, thay bằng công tắc tĩnh (mục 3.3) cộng mở rộng có mục tiêu qua `must_include` (mục 5.2).

### 10.4 Retrieval như một node LangGraph

Thêm một hop và một nhánh lỗi để đổi lấy không gì cả — nó deterministic và không gọi LLM. **Loại bỏ**; nó là hàm gọi trước `make_initial_state()`.

### 10.5 SaaS ngang (khách trỏ DB lên cloud của bạn)

Đánh giá ở mục 10.4 spec `2026-08-12` giữ nguyên: độ phù hợp thấp. Thiết kế trong spec này đi ngược hướng đó một cách có chủ ý — không egress, một profile mỗi bundle, model ship kèm. Nếu về sau đổi ý thì phần lớn công việc ở đây vẫn dùng lại được (retrieval, chú giải, phân quyền), nhưng phần đóng gói thì không.
