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
  ├─ permitted_tables(profile, user)                      ◄── BẢO MẬT. Không phụ thuộc câu hỏi.
  │
  ├─ resolve_schema_context(profile, question, permitted) ◄── NỘI DUNG PROMPT. Hàm thuần, <100ms
  │     mode=full      → toàn bộ schema đã render (trong phạm vi permitted)
  │     mode=retrieval → top-K theo embedding
  │                      → mở rộng 1 bước theo cạnh FK
  │                      → retrieve top-2 few-shot từ examples.jsonl
  │     ⇒ SchemaContext{ retrieved_tables[], rendered_text, few_shots[] }
  │
  ├─ run_graph(query, schema_context)
  │     supervisor  ← rendered_text
  │     sql         ← rendered_text + few_shots   (dùng lại, KHÔNG retrieve lần nữa)
  │     python / viz / insight  ← không cần schema
  │     reflector
  │
  └─ execute_sql(..., permitted_tables)   ◄── KHÔNG phải retrieved_tables. Xem 3.4.1.
```

**Retrieval chạy một lần mỗi câu hỏi, không phải mỗi node.** Nó **không** là node LangGraph: nó deterministic, không gọi LLM, không có kiểu lỗi cần retry. Biến nó thành node là mua thêm một hop và một nhánh lỗi để đổi lấy không gì cả. Nó chạy trước `make_initial_state()`.

Chi phí runtime: index dựng một lần lúc onboarding; lúc chạy chỉ là embed một câu hỏi ngắn + cosine trên ≤200 vector. Dưới 100ms, chạy CPU thoải mái.

### 3.3 Công tắc `schema_mode` — quyết một lần lúc cài

`build_profile.py` đo kích thước schema đã render. Dưới ngưỡng thì ghi `schema_mode: full` và bỏ qua retrieval hoàn toàn; trên ngưỡng thì `schema_mode: retrieval`.

**Ngưỡng đề xuất: 6.000 token schema đã render.** Ngưỡng này KHÔNG đổi theo bản sửa dưới đây — nó cố ý đặt bằng đơn vị token vì token tự điều chỉnh khi bảng "béo" lên (nhiều cột, mô tả dài); một ngưỡng đặt bằng *số bảng* thì không.

> **Sửa (rà soát cuối nhánh này):** bản trước quy đổi ngưỡng trên thành "khoảng 40 bảng" ở giả định 140 token/bảng — con số đó **chưa từng đo**, chỉ ước lượng tay. Đo thật trên schema 9 bảng của chính ADBA (`perception/info_box_all.json`, không bảng nào có `description`) ra **~67 token/bảng** (mục 3.4), tức 6.000 token ≈ **khoảng 90 bảng**, gấp hơn hai lần con số cũ. Nhưng đây KHÔNG phải một hằng số thay thế 140 để dùng lại: 9 bảng đó chưa chú giải và trung bình chỉ ~9,4 cột/bảng. Đo với mô tả tiếng Việt thực tế thêm vào thì ra ~81 token/bảng (339 B/bảng, +26% so với 268 B/bảng chưa chú giải) — ngưỡng khi đó còn ~74 bảng. Bảng ERP thật có 30–60 cột, cộng chú giải do việc onboarding (kế hoạch riêng, ngoài phạm vi nhánh này) thêm vào, có thể đẩy lên ~200–350 token/bảng — ngưỡng khi đó tụt về ~17–30 bảng. Nói cách khác, "bao nhiêu bảng thì chạm ngưỡng" dao động cả chục lần tuỳ mức chú giải; **không có một con số bảng nào đáng tin để thay cho ngưỡng token**. Đừng quy đổi lại — đọc thẳng ngưỡng 6.000 token, để nó tự điều chỉnh theo mật độ chú giải thật của từng khách.

Đây **không** phải fallback động. Một nhánh code, một công tắc, quyết một lần lúc onboarding. Khách nhỏ được sự đơn giản của chế độ full; khách lớn được sự co giãn của retrieval. Không phải test tổ hợp hai đường chạy trên cùng một cài đặt.

### 3.4 `ConnectionProfile` mở rộng, và biểu diễn schema dạng DDL

Spec `2026-08-12` đã đưa ra `ConnectionProfile` gom `dsn` + `allowed_tables` + `info_box`. Spec này thay ruột chứ không đổi cấu trúc:

| Trường | Trước | Sau |
|---|---|---|
| `dsn` | biến môi trường toàn cục | giữ nguyên trong profile |
| `allowed_tables` | frozenset 9 tên hardcode (`sql_tool.py:27-31`) | **tách làm hai** — `permitted_tables(user)` và `retrieved_tables(question)`, xem 3.4.1 |
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

Nhỏ hơn JSON 2–3 lần. Và dễ hơn cho model: Qwen2.5-**Coder** đã đọc hàng triệu file SQL lúc pretrain; JSON mô tả schema thì không.

> **Sửa (rà soát cuối nhánh này) — con số 140 token/bảng ở trên chưa từng đo, chỉ ước lượng tay từ một ví dụ 12 cột.** Đo thật bằng `render_schema()` + `estimate_tokens()` (commit `6486ddc`) trên schema 9 bảng thật của ADBA, `perception/info_box_all.json` — schema này **không bảng nào có `description`** — ra **268 B/bảng, ~67 token/bảng** (603 token / 9 bảng), tức khoảng một nửa giả định cũ. Đo lại với mô tả tiếng Việt thực tế thêm vào từng bảng (task 3 review) ra **339 B/bảng, ~81 token/bảng** (+26% so với bản chưa chú giải) — vẫn thấp hơn 140 đáng kể, vì DDL của những bảng nhỏ (9–15 cột) rẻ hơn ví dụ 12-cột-với-comment-dài dùng để ước lượng ban đầu. Cả hai con số đều là **sàn dưới**: bảng ERP thật nhiều cột hơn (30–60) và mô tả dài hơn có thể đẩy lên 200–350 token/bảng (xem mục 3.3). Không có ước tính đơn nào đáng tin cho mọi schema — số thật phụ thuộc số cột và độ dài chú giải của schema cụ thể, đo bằng `render_schema()` trên schema đó chứ đừng giữ một hằng số cố định.

### 3.4.1 Hai tập bảng, khác bản chất — không được gộp

`_ALLOWED_TABLES` tĩnh hiện tại phải tách thành **hai** khái niệm, không phải một khái niệm động:

| Tập | Bản chất | Phụ thuộc | Ai dẫn ra |
|---|---|---|---|
| `permitted_tables(user)` | **Bảo mật.** Ranh giới thật. | Danh tính người dùng. Ổn định giữa các câu hỏi. | Bên thực thi SQL, tự dẫn từ profile + danh tính. **Không nhận từ bên gọi.** |
| `retrieved_tables(question)` | **Nội dung prompt.** Không phải cơ chế bảo mật. | Câu hỏi. Đổi mỗi lượt. | Retriever, phía client |

Bản duyệt đầu của spec này đặt `allowed_tables = retrieved ∩ permitted` rồi dùng cho **cả hai** việc. Trong tiến trình thì vô hại vì cùng một code, nhưng nó hỏng ngay khi tool tách qua ranh giới RPC/MCP (spec `2026-08-12` pha 3): lúc đó `allowed_tables` phải đi trong payload lời gọi, tức là **bên bị ràng buộc tự khai ràng buộc của mình**. Guard tụt xuống thành "bất cứ thứ gì client nói". Đây là lỗi thiết kế, đã sửa.

Quy tắc sau khi tách:

- `execute_sql` chỉ thực thi `permitted_tables`. Khi sang MCP, tham số này **không** nằm trong payload — server tự dẫn ra từ profile cộng danh tính phiên.
- `retrieved_tables` chỉ dùng để dựng prompt. Nó thu hẹp thứ model *nhìn thấy*, không thu hẹp thứ model *được phép chạm*.

Tách ra thì **đúng hơn và cũng tốt hơn**: trước đây nếu retrieval sót một bảng mà model vẫn đoán đúng tên, guard sẽ chặn — một thất bại giả. Sau khi tách thì câu đó chạy được, vì bảng đó vốn nằm trong quyền của người dùng. Kiểu lỗi `retrieval_miss` (mục 5.2) nhẹ đi một bậc.

**Đánh đổi được chấp nhận có ý thức:** SQL có thể chạm một bảng nằm trong quyền nhưng không được retrieve. Đó không phải leo thang đặc quyền — người dùng vốn được xem bảng đó — nên nó không phải sự cố bảo mật, chỉ là tín hiệu retrieval chưa tốt. Ghi vào trace để `verify_profile.py` đếm.

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

| Lớp | Cơ chế | Là ranh giới bảo mật? | Ship kèm? |
|---|---|---|---|
| 1 | `permitted_tables(user)` **giao vào kết quả retrieval** → model không nhìn thấy bảng ngoài quyền | Có | Có |
| 2 | `execute_sql` thực thi `permitted_tables` do chính nó dẫn ra → model bịa tên bảng thì từ chối | **Có — lớp quyết định** | Có |
| 3 | Grant cấp cột/dòng trong Postgres | Có | Không — khách tự cấu hình |
| — | `retrieved_tables(question)` thu hẹp prompt theo câu hỏi | **Không.** Chỉ là nội dung prompt. | Có |

Điểm mấu chốt sau khi tách ở mục 3.4.1: **lớp 2 là lớp chịu trách nhiệm**, vì nó là lớp duy nhất còn đứng vững khi tool tách qua RPC. Lớp 1 làm model không *nhìn thấy* bảng cấm — hữu ích và gần như miễn phí, nhưng nó là biện pháp giảm bề mặt, không phải bảo đảm. Đừng bao giờ suy luận an toàn từ "bảng đó không có trong prompt".

Dòng cuối bảng có mặt để nói rõ điều dễ nhầm nhất: retrieval **không** phải cơ chế phân quyền, dù nó trông giống.

### 4.1 Lớp 1 lọc SAU khi tìm, không phải TRƯỚC — và hệ quả

Bản duyệt đầu của spec này mô tả lớp 1 là *"giới hạn phạm vi retriever"*. Hiện thực không làm vậy, và mô tả trên đã được sửa cho khớp thực tế. Thứ tự thật trong `resolve_schema_context`:

```
hits   = retriever.search(question, k)          # xếp hạng trên TOÀN BỘ schema
chosen = expand_by_foreign_keys(hits, tables)   # mở rộng 1 bước theo FK
chosen &= permitted                              # giao với quyền — ở BƯỚC CUỐI
```

**Về bảo mật không có khác biệt.** Phép giao là thao tác tập hợp cuối cùng trên mọi nhánh, kể cả `must_include`, nên không bảng nào ngoài quyền lọt vào `SchemaContext`. Điều này đã được kiểm độc lập.

**Về chất lượng thì có, và nó lệch theo hướng đáng lo.** Retriever xếp hạng trên toàn bộ schema rồi mới cắt theo quyền, nên một bảng hợp lệ của người dùng có thể bị đẩy khỏi top-K bởi những bảng mà chính người đó **không được xem**. Với schema 150 bảng và một người dùng được cấp 5 bảng, `k=8` có thể cho ra `chosen` rỗng — model không nhận được schema nào và không trả lời được câu hỏi mà người dùng hoàn toàn có quyền hỏi.

Nghịch lý: **quyền càng hẹp thì hệ thống càng dễ vô dụng**, vì lý do nằm ở những bảng người dùng không hề thấy. Người dùng toàn quyền không bao giờ gặp.

Điều này không biểu hiện trong bản triển khai hiện tại vì `app.py` cấp `ALL_TABLES` cho đúng một người dùng. Nó sẽ biểu hiện ngay ở khách hàng đầu tiên có phân quyền thật — tức đúng phân khúc spec này nhắm tới.

**Quyết định: hoãn sang giai đoạn tích hợp embedding**, không sửa ở pha 1. Lý do là đây là quyết định về **giao thức `Retriever`**, không phải sửa lỗi cục bộ, và hai lối đi có hệ quả khác nhau:

| Lối | Cách làm | Hệ quả |
|---|---|---|
| Dựng retriever trên lát đã lọc quyền | `LexicalRetriever(permitted_slice)` | Đơn giản. Chấp nhận được với lexical (chỉ là tập token), **không** chấp nhận được với embedding — phải tính lại toàn bộ vector mỗi khi đổi người dùng |
| Thêm tham số ứng viên vào giao thức | `search(question, k, candidates=permitted)` | Index dựng một lần, lọc lúc tìm. Là hình dạng duy nhất còn dùng được khi bản embedding vào |

Chọn sai bây giờ nghĩa là phải sửa lại giao thức sau, đúng lúc đã có hai hiện thực cùng phụ thuộc vào nó. Nên quyết định này thuộc về giai đoạn embedding và phải quyết **cùng lúc** với việc chọn model — không sớm hơn.

Cho tới lúc đó, ràng buộc phải giữ: phép giao với `permitted` vẫn là thao tác tập hợp cuối cùng, bất kể phạm vi tìm kiếm được thu hẹp ở đâu.

`app.py` hiện không có xác thực (chỉ `st.session_state` phạm vi phiên trình duyệt). Bản on-prem cần tối thiểu: định danh người dùng, và ánh xạ người dùng → `permitted_tables`. Thiết kế chi tiết của lớp xác thực nằm ngoài spec này; spec này định nghĩa **hai điểm cắm**:

- `permitted_tables(profile, user)` — nguồn sự thật về quyền, được cả retriever và `execute_sql` gọi độc lập
- `resolve_schema_context(profile, question, permitted)` — nhận tập quyền làm **cận trên** cho kết quả, không tự quyết định quyền. Hiện áp cận đó ở bước cuối; xem mục 4.1 về việc chuyển nó lên trước bước tìm kiếm.

---

## 5. Ngân sách thời gian & xử lý lỗi

### 5.1 Ngân sách — thiết kế này gần như không đụng vào

> **Sửa (rà soát cuối nhánh này) — bảng dưới đây trong bản gốc dùng giả định 140 token/bảng, chưa từng đo (xem mục 3.4).** Đo thật trên schema 9 bảng ADBA ra ~67 token/bảng chưa chú giải, ~81 token/bảng có chú giải tiếng Việt thực tế — tính lại bằng mức chú giải thực tế (~81 token/bảng) bên dưới. Đây vẫn không phải trần: bảng ERP nhiều cột + mô tả dài có thể đẩy DDL/bảng cao hơn nhiều (xem mục 3.3), nên coi đây là ví dụ minh hoạ ở mật độ chú giải "vừa phải", không phải cam kết cho mọi schema khách.

Đếm token thực tế (DDL tính ở ~81 token/bảng, mức đo được với chú giải thực tế — mục 3.4):

| Chế độ | Thành phần | Tổng | ctx cần |
|---|---|---|---|
| `retrieval`, 12 bảng | rules ~700 (cố định) + DDL ~970 (12×81) + few-shot ~400 + task ~100 | **~2.200** | 8k |
| `full`, 74 bảng (sát ngưỡng 6.000-token ở mức ~81 token/bảng — KHÔNG còn là 40 bảng như bản gốc, xem mục 3.3) | rules ~700 + DDL ~6.000 (74×81) + few-shot ~400 + task ~100 | **~7.200** | 16k |

Cả hai tổng đều thấp hơn bản gốc (2.900 / 6.800) vì tỉ lệ đo được thấp hơn giả định cũ — nhưng hàng `full` vẫn cần ctx 16k chứ không phải 8k: cộng thêm `AGENT_MAX_TOKENS["sql"] = 1024` cho phần sinh, tổng ~8.224 đã vượt cửa sổ 8k (8.192), không còn biên an toàn.

Ở `full`, toàn bộ prompt cố định giữa mọi query → sau query đầu prefill gần bằng 0 nhờ prefix caching.
Ở `retrieval`, phần biến thiên ~2.300 token ≈ 0,6s prefill trên một card tầm 4090.

Chi phí thật vẫn nằm ở decode: node SQL sinh ~150 token ở ~40 tok/s ≈ 4s, nhân 5 node ≈ 20s. **Con số đó không đổi vì thiết kế này.** Cơ chế deadline + reserve + finalize node ở spec `2026-08-12` vẫn cai quản ngân sách, không cần thiết kế lại — chỉ cộng thêm ~0,7s cho retrieval và prefill biến thiên.

### 5.2 Bốn kiểu lỗi mới

**`profile_stale` — kiểu lỗi nguy hiểm nhất của on-prem.**
DBA của khách thêm cột, đổi tên bảng, và profile vẫn mô tả schema cũ. `profile.json` lưu fingerprint (hash của tập tên bảng + cột). Kiểm tra lúc khởi động và định kỳ; lệch thì **từ chối chạy** kèm chỉ dẫn chạy `refresh_profile.py` — không phải cảnh báo rồi chạy tiếp.

Kèm theo là một yêu cầu thiết kế thật: **`schema.yaml` phải merge được.** Refresh chỉ re-annotate bảng mới hoặc đã đổi, giữ nguyên mọi mục có `reviewed_by: human`. Nếu refresh xóa công sức của khách thì họ sẽ không sửa lần thứ hai, và chất lượng chú giải — tức trần accuracy — tụt vĩnh viễn.

**`retrieval_miss`.**
Nhẹ đi sau khi tách hai tập bảng (mục 3.4.1). Nếu model đoán đúng tên một bảng bị retrieval bỏ sót và bảng đó nằm trong `permitted_tables`, câu truy vấn **chạy được** — guard không chặn nữa. Chỉ ghi vào trace như tín hiệu retrieval kém.

Khi model không đoán ra và SQL báo lỗi: reflector báo `error_category: schema_mismatch` → gọi lại `resolve_schema_context(..., must_include=[bảng thiếu])`. Mở rộng có mục tiêu, không phải fallback full-schema. Một lượt, rẻ.

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

> **Cần xác minh khi dựng harness:** số database và số bảng trung bình mỗi DB. Các con số này không được chốt từ trí nhớ; **hạng mục đầu tiên của pha 0** là tải bộ dữ liệu và ghi lại cấu hình thực tế vào `eval/README_multischema.md`.

### 6.2.1 Điều kiện pháp lý khi sử dụng ba bộ

Đây là **quyết định của chủ repo**, ghi lại để tra cứu — không phải tư vấn pháp lý. Nguồn sự thật khi chạy là `eval/benchmarks.json`; mục này giải thích *vì sao* các trường trong đó được đặt như vậy.

| Bộ | License | Dùng cho mục đích thương mại | Đóng gói kèm bản giao khách |
|---|---|---|---|
| Spider | CC BY-SA 4.0 | **Được** | Không |
| BIRD | CC BY-SA 4.0 | **Được** | Không |
| BEAVER | **Chưa xác định** | **Không, cho tới khi xác định** | Không |

**Hai trục, không phải một.** "Được dùng cho mục đích thương mại" và "được đóng gói kèm sản phẩm" là hai câu hỏi khác nhau, và một giấy phép có thể cho phép cái này mà cấm cái kia. Điều khoản NonCommercial chạm vào trục thứ nhất — nó hạn chế **cách dùng**, kể cả khi không hề phân phối lại gì. Vì thế `eval/benchmarks.json` có hai trường riêng, `commercial_use` và `may_redistribute`, và `eval/fetch_dataset.py` từ chối bất kỳ mục nào chưa quyết cả hai.

**Vì sao cả ba đều `may_redistribute: false`.** Đây là **chính sách dự án, không phải hạn chế của giấy phép**. CC BY-SA 4.0 cho phép phân phối lại kèm ghi công và share-alike. Nhưng dữ liệu benchmark là công cụ đo lúc phát triển, không phải thành phần của sản phẩm — không có lý do gì để nó nằm trong bản giao khách, và điều khoản share-alike khi đi kèm một bản phân phối thương mại là thứ tốt nhất nên tránh hẳn.

**Vì sao BEAVER bị treo.** Các nguồn công khai ghi mâu thuẫn: có nơi nêu CC BY-NC-ND 4.0, có nơi nêu MIT. Cho tới khi xác định được, BEAVER không được dùng. Nếu hóa ra là BY-NC-ND thì có **hai** hệ quả, không phải một:

- **NC** cấm dùng cho mục đích thương mại — mà "đo để quyết định fine-tune có chuyển giao được sang schema khách hay không" chính là mục đích thương mại.
- **ND** cấm tạo bản phái sinh — mà bước chuyển bộ dữ liệu sang `questions.jsonl` + `schemas.json` là một bản phái sinh, kể cả khi chỉ dùng nội bộ.

Hệ quả thứ hai đáng chú ý vì nó chặn cả đường dùng nội bộ, thứ mà trực giác hay cho là an toàn.

**Hệ quả lên `beaver_exec_accuracy`.** Mục 6.5 và mục 9 đặt chỉ số này làm số quyết định fine-tune có chuyển giao được hay không. Chừng nào license BEAVER chưa xác định, **chỉ số đó không có nguồn dữ liệu hợp lệ.** Hai lối đi khi tới lúc cần:

1. BEAVER được xác nhận cho phép thương mại → giữ nguyên vai trò như spec mô tả.
2. Không được → thay bằng BIRD làm nguồn đo chuyển giao. BIRD không có quy mô doanh nghiệp như BEAVER, nên phép đo yếu hơn và spec phải hạ kỳ vọng tương ứng, chứ không được giả vờ như vẫn đo được điều cũ.

Quyết định này thuộc pha 2, cùng lúc chạy eval sau khi train lại.

**Thi hành bằng code, không bằng tài liệu.** `require_commercial()` trong `eval/fetch_dataset.py` là cổng cho những chỗ số đo dẫn tới quyết định về sản phẩm. Một bộ chỉ cho phép nghiên cứu vẫn tải được để tham khảo, nhưng gọi qua cổng đó sẽ bị từ chối. Lý do tách như vậy: một dòng trong spec thì không ai đọc lúc chạy script, còn một trường trong manifest thì script đọc được và từ chối được.

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
| `execute_sql` guard | Chặn bảng ngoài `permitted_tables`; **và** không chặn bảng nằm trong `permitted` nhưng ngoài `retrieved` (mục 3.4.1) |
| `permitted_tables()` | Dẫn ra độc lập với câu hỏi và với kết quả retrieval; không nhận tập quyền từ bên gọi |
| Loại trừ `sample_rows` khỏi `profile/` | Kiểm bằng test, không bằng quy ước |

Phần phụ thuộc LLM chỉ còn chất lượng chú giải và chất lượng SQL — hai thứ đó thuộc eval, không thuộc test. **Không viết test đòi model trả lời đúng.**

---

## 7. Thứ tự triển khai

| Pha | Nội dung | Điều kiện ra |
|---|---|---|
| **0** | Harness eval tầng 1: parse SQL mẫu → tập bảng đúng; tải Spider/BIRD/BEAVER, ghi cấu hình thực tế | Đo được recall của một retriever giả (random / full) làm mốc |
| **1** | `render_schema()` DDL; tách `prompts/*.txt` ba đường; chuyển `{info_box}` xuống cuối; `SchemaContext` + `resolve_schema_context()`; `ConnectionProfile` mở rộng; **tách `permitted_tables` / `retrieved_tables` (3.4.1)** | Golden set ADBA hiện tại **không hồi quy**; recall tầng 1 đo được trên cả ba benchmark |
| **2** | Train lại LoRA đa schema: chế độ mới của `generate_data.py`, dịch phương ngữ SQLite→Postgres, train, eval | Golden set ADBA không hồi quy **và** `beaver_exec_accuracy` cải thiện so với base |
| **3** | Đường onboarding: `extract_schema` → `annotate_schema` → merge YAML → `build_profile` → `verify_profile`; `refresh_profile`. **Kèm tích hợp retriever embedding, và cùng lúc đó chốt lại giao thức `Retriever` để lọc quyền TRƯỚC bước tìm (mục 4.1)** | Chạy trọn trên một schema thứ hai (không phải ADBA) và ra `report.md`; retriever embedding **vượt mốc lexical** trên eval tầng 1; người dùng có quyền hẹp không còn nhận context rỗng |
| **4** | Đóng gói on-prem: `docker-compose.onprem.yml`, tắt egress, model ship kèm, điểm cắm xác thực | Cài được trên một máy sạch không có internet |

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
11. `execute_sql` không có tham số nào cho phép bên gọi nới rộng tập bảng được chạm. Tập quyền chỉ dẫn ra từ profile + danh tính — kiểm bằng test, và kiểm lại khi tool tách qua MCP.

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

### 10.6 Gộp SQL + Python + Viz thành một Analyst Agent tool-calling

**Đã đo, không phải suy đoán.** Spike ngày 2026-08-15 trên `qwen2.5-coder:7b-instruct-q5_K_M` (base, chưa fine-tune), 12 câu hỏi trên schema ADBA, vòng lặp JSON 3 tool + `finish`, 3 biến thể giao thức.

| Tier | Tool cần dùng | Kết quả |
|---|---|---|
| 2 bước, tham số scalar | `query_postgres`, `render_chart` | **5/7 pass**, parse JSON 100% |
| 3 bước, có payload code | thêm `run_pandas` | **0/5 pass** qua cả ba biến thể giao thức |

Ba kiểu hỏng, đều tái lập được:

- **Không gắn được payload code.** Code trong JSON → JSON hỏng (dùng `"""` của Python). Code trong fence sau JSON → bỏ quên hẳn khối code. Thông báo lỗi có kèm ví dụ đúng, lặp ba lần, không lần nào sửa được.
- **Thử lại nhưng không thích nghi.** Một câu lặp đúng một truy vấn hỏng sáu lần, không đổi một ký tự. Một câu khác thì tự sửa lỗi SQL thành công. Tự sửa **không nhất quán**.
- **Không biết dừng đúng lúc.** Một câu không bao giờ gọi `finish`; một câu gọi `finish` sau một lỗi mà chưa làm gì — trả lời như thể đã xong.

**Kết luận:** với lớp model hiện tại thì loại bỏ. Vấn đề không phải kiến trúc mà là model — Qwen2.5-**Coder** mạnh ở sinh code đúng trong một lượt, yếu ở duy trì giao thức nhiều lượt. Kiến trúc 5 node đang dùng nó đúng chỗ nó giỏi, và 96,9% là bằng chứng.

**Điều kiện mở lại:** một spike riêng trên base agentic (`Qwen2.5-7B-Instruct` bản không-Coder, hoặc dòng Qwen3-8B), sau khi pha 2 xong. Nhớ chi phí kèm theo: **đổi base thì LoRA và bản merged hiện tại bỏ đi hoàn toàn** — câu hỏi kiến trúc và câu hỏi chọn model là một.

**Phần được tách ra và giữ lại:** bỏ node supervisor. Không gian đầu ra của nó chỉ có hai lựa chọn (`prompts/supervisor_routing.txt` quy tắc 5–6) mà đang tốn một lượt LLM đầy đủ, và router là nguyên nhân tồn tại của bug `supervisor.py:274-281`. Thay bằng luật hoặc classifier rẻ. Việc này **độc lập** với chuyện gộp và không dính rủi ro vừa đo được — nhưng nó thuộc spec `2026-08-12`, không thuộc spec này.

Giới hạn của phép đo: n=5 ở tier khó, một model, temperature 0,1, một bộ prompt. Chưa kiểm được liệu fine-tune trên định dạng tool-call có sửa được hai kiểu hỏng đầu hay không.
