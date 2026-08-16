# Lộ trình tới bản đóng gói giao khách

**Ngày:** 2026-08-16
**Trạng thái:** Lộ trình — mỗi mục dưới đây sẽ có plan chi tiết riêng khi tới lượt

**Specs:**
- `docs/superpowers/specs/2026-08-15-adba-multi-schema-onprem-design.md` — đa schema, on-prem
- `docs/superpowers/specs/2026-08-12-adba-mcp-hardening-design.md` — cứng hóa production

Tài liệu này không thay thế hai spec trên. Nó chỉ trả lời: **còn lại những gì, theo thứ tự nào, và cái gì mở khoá cái gì.**

---

## Vạch xuất phát thật (đo được, không phải ước lượng)

| | |
|---|---|
| Test | 227 (201 unit + 26 integration) |
| Model đang chạy | `qwen2.5-coder:7b-instruct-q5_K_M` — **base**, không phải bản fine-tune |
| Recall chọn bảng | 0,250 trên golden set ADBA (`lexical`, k=8) |
| Kích thước schema render | 268 B/bảng (67 token) |
| Đã xong | Đọc schema lạ, phân quyền theo người dùng, harness đo |
| Chưa xong | Onboard được một khách, cứng hóa, đóng gói |

Con số 96,9% trong tài liệu cũ thuộc về một cấu hình **không còn tồn tại**: model thuộc lòng schema ADBA, prompt viết riêng cho schema đó. Cả hai đã bị thay. Đừng dùng nó làm mốc so sánh.

---

## Ba plan, theo thứ tự

### Plan A — Đường onboarding (spec 08-15, pha 3)

**Vì sao trước:** đây là thứ duy nhất biến dự án từ một thư viện thành thứ onboard được người dùng. Không có nó thì không có gì để giao, dù mọi phần khác đã xong.

Và nó là thứ nâng recall. 0,250 thấp **vì thiếu chú giải ngữ nghĩa**, không phải vì model yếu — đã đo và biết chắc. Đặt vào ERP thật với cột tên `dt_ord_cd`, `flg_tt` thì introspect ra được cấu trúc nhưng không ra được ý nghĩa; không model nào đoán được, vì đó là thông tin không có trong đầu vào.

**Giao ra:** `extract_schema` → `annotate_schema` → người sửa YAML → `build_profile` → `verify_profile`, cộng `refresh_profile`.

**Kèm theo, không tách rời:** tích hợp retriever embedding và chốt lại giao thức `Retriever` để lọc quyền **trước** bước tìm (spec 08-15 mục 4.1). Hai việc này phải cùng lúc vì hình dạng giao thức phụ thuộc vào việc chọn model embedding — dựng index theo lát quyền thì rẻ với lexical nhưng không chấp nhận được với embedding.

**Cổng ra:**
- Chạy trọn trên một schema **không phải ADBA**, ra được `report.md`
- Retriever embedding **vượt mốc lexical** trên eval tầng 1
- Người dùng có quyền hẹp không còn nhận context rỗng
- `refresh_profile` giữ nguyên 100% mục `reviewed_by: human`

**Rủi ro lớn nhất:** chất lượng chú giải do LLM sinh. Nếu nó đoán sai ý nghĩa cột, SQL sẽ sai **mà không báo lỗi** — kiểu sai nguy hiểm nhất. Cổng recall ở `verify_profile` là chỗ duy nhất bắt được.

---

### Plan B — Cứng hóa (spec 08-12, phần còn lại của pha 1 + pha 2)

**Vì sao ở giữa:** Plan A chạy trên máy bạn, chưa chạm mạng khách hàng, nên hai lỗ hổng dưới đây chưa lộ ra ngoài. Nhưng chúng **phải đóng trước bước đóng gói**.

Hướng on-prem làm chúng cấp bách hơn, không phải bớt đi:

| Lỗ hổng | Trên máy bạn | Ở khách hàng |
|---|---|---|
| **L1** — không có ngân sách thời gian | Một lần treo gây khó chịu | Ticket hỗ trợ ngày đầu, và là ấn tượng đầu |
| **L3** — sandbox dùng `fork` | Kế thừa credential DB của bạn | Kế thừa credential DB **production của họ** |

**Trạng thái đã kiểm, không nói theo trí nhớ:**
- `MAX_REFLECTOR_PASSES_PER_AGENT = 8` — nguyên
- Không có `deadline_ts`, không có node `finalize`
- `mp.get_context("fork")` — nguyên; `sandbox/` rỗng
- Chưa có read-only role ở tầng Postgres

**L1 đã tự biểu diễn một lần trong quá trình làm plan trước:** một test đáng lẽ chạy 0,1 giây tiêu hơn 100 giây thật, vì một node lỗi và đường phục hồi chạm model. Con số 25 phút trong spec trước nay chỉ là phép nhân hai hằng số; giờ đã có bằng chứng.

**Giao ra:** `deadline_ts` mang trong state, cấp phát theo dự trữ, node `finalize` thay `END` trần, trần cứng thay đếm retry, `fork` → `spawn` với env rỗng, read-only role ở Postgres, hạ timeout.

**Cổng ra:** một câu hỏi bất kỳ không bao giờ vượt ngân sách; code trong sandbox không đọc được biến môi trường của tiến trình cha; vai trò DB không thực thi được lệnh ghi.

**Đánh giá công sức:** nhỏ nhất trong ba plan. Phần lớn là các thay đổi cục bộ đã có thiết kế sẵn trong spec.

---

### Plan C — Đóng gói on-prem (spec 08-15, pha 4)

**Giao ra:** `docker-compose.onprem.yml` (bỏ hẳn `postgres` và `seed-data` — service `seed-data` **ghi dữ liệu**, tuyệt đối không được có mặt ở bản giao khách), tắt egress ở tầng mạng, model ship kèm dạng GGUF + Modelfile, điểm cắm xác thực, loại trừ thư mục benchmark khỏi bundle.

**Hai thứ hiện đang thiếu và phải làm trong plan này:** `sandbox/Dockerfile` và `scripts/ollama-entrypoint.sh` — `docker-compose.prod.yml` tham chiếu cả hai nhưng cả hai đều không tồn tại, nên file đó hiện **không khởi động được**.

**Cổng ra:** cài được trên một máy sạch không có internet, và trả lời được một câu hỏi thật.

---

## Hai việc có điều kiện, không nằm trên đường tới bản đóng gói

### Train lại LoRA đa schema (spec 08-15, pha 2)

**Không làm trước khi đo.** Hệ thống đang chạy base model, và base Qwen2.5-**Coder** vốn đã có kỹ năng đọc DDL rồi viết SQL — đó là năng lực nền từ pretrain, không phải thứ phải dạy. Train lại là **đòn bẩy tối ưu, không phải điều kiện để hoạt động**.

**Kích hoạt khi:** sau Plan A, chạy harness tầng 1 và eval tầng 2. Chỉ train nếu số liệu nói cần, và lúc đó bạn biết cần nâng bao nhiêu, từ đâu tới đâu.

Nếu base cộng chú giải tốt đã đủ dùng cho khách đầu tiên, bạn tiết kiệm được hạng mục đắt nhất của cả kế hoạch.

**Lưu ý về nguồn dữ liệu:** `beaver_exec_accuracy` hiện **không có nguồn hợp lệ** — license BEAVER chưa xác định (mục 6.2.1 spec 08-15). Nếu không dùng được, thay bằng BIRD và hạ kỳ vọng tương ứng, chứ không giữ nguyên câu chữ rồi giả vờ vẫn đo được điều cũ.

### Ranh giới MCP (spec 08-12, pha 3)

Hoãn. Nó là refactor nội bộ, không mở khoá gì trên đường tới khách hàng đầu tiên. Nhưng khi làm, `permitted_tables` phải giữ nguyên hình dạng hiện tại — tập quyền dẫn ra bên trong bộ thực thi, không nhận từ bên gọi — vì đó chính là thứ khiến nó sống sót qua ranh giới RPC.

---

## Quyết định: dùng gì làm "schema thứ hai"

Cổng ra của Plan A đòi chạy trọn trên một schema không phải ADBA. Chưa có khách hàng thí điểm.

**Dùng một database từ BIRD.** Lý do:

- License đã xác nhận CC BY-SA 4.0, cho phép dùng thương mại, và script tải đã sẵn sàng
- Schema BIRD **bẩn và thật** — đúng thứ cần để thử đường chú giải, khác hẳn schema demo sạch sẽ
- Không phải schema tôi bịa ra. Một schema giả do tôi dựng sẽ mang đúng những giả định có sẵn trong đầu tôi, tức là nó sẽ dễ một cách không thật

Đây là **chuẩn phát triển**, không thay thế khách hàng thật. Khi có schema khách, nó trở thành cổng thứ hai và là cổng quyết định.

---

## Nếu chỉ làm được một việc

Plan A. Không có nó thì mọi thứ khác không có ai để phục vụ.

Nhưng **không đóng gói trước khi xong Plan B.** Mang một hệ thống chưa có ngân sách thời gian và chưa cô lập sandbox vào mạng khách hàng là đưa hai lỗ hổng đã biết ra khỏi tầm kiểm soát của bạn — và lần đầu chúng biểu hiện sẽ là trước mặt người dùng đầu tiên.
