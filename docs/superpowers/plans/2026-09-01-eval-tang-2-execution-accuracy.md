# Eval tầng 2 — execution accuracy

**Trạng thái:** phác thảo, chưa hiện thực.
**Vì sao bây giờ:** không có nó thì few-shot, retriever embedding, hay bất
cứ cải tiến nào về *chất lượng SQL* đều không kiểm chứng được.

---

## 1. Lỗ hổng hiện tại

Hai thước đo đang có, và cả hai đều không đo được thứ khách hàng quan tâm:

| Thước đo | Đo gì | Không đo gì |
|---|---|---|
| `eval/tier1_recall.py` | context có chứa **đủ tập bảng đúng** không | SQL viết ra có đúng không |
| `eval_runner.sql_execution_accuracy` | `EXPLAIN` có chạy trót lọt không | câu trả lời có đúng không |

`EXPLAIN` thành công chỉ chứng minh SQL đúng cú pháp và gọi đúng tên bảng,
tên cột có thật. `SELECT 1 FROM orders` qua `EXPLAIN` ngon lành.

Hệ quả trực tiếp: `few_shots` chỉ được tiêu thụ ở `graph/agents/sql_agent.py`
— nó đi vào prompt sinh SQL, **không** chạm đường retrieval. Nên tầng 1
không thể phản ánh nó, dù chỉ một điểm. Không có tầng 2 thì bật few-shot
lên là thay đổi mù.

---

## 2. Định nghĩa

Một câu **đạt** khi tập kết quả của SQL sinh ra khớp tập kết quả của SQL
vàng, chạy trên cùng một database.

Không so chuỗi SQL. Hai câu SQL khác hẳn nhau về hình thức vẫn có thể cùng
đúng, và so chuỗi sẽ phạt oan đúng những lời giải hay.

---

## 3. Sáu quyết định về ngữ nghĩa so sánh

Đây là phần khó thật sự. Mỗi quyết định đều đánh đổi giữa **phạt oan** câu
đúng và **tha nhầm** câu sai.

### D1 — Thứ tự dòng

SQL vàng không có `ORDER BY` thì Postgres không cam kết thứ tự, và thứ tự
có thể đổi giữa hai lượt chạy.

→ Mặc định so như **multiset** (bỏ qua thứ tự). Nếu SQL vàng có `ORDER BY`
ở mức ngoài cùng thì so **có thứ tự**, vì lúc đó câu hỏi thật sự hỏi về xếp
hạng ("5 khách mua nhiều nhất").

⚠️ Nhận diện `ORDER BY` ngoài cùng bằng khớp chuỗi là **sai**: nó có thể
nằm trong subquery hoặc window function. Cần parse thật, hoặc chấp nhận
giới hạn và ghi rõ. Đây là chỗ dễ đẻ lỗi âm thầm nhất trong cả thiết kế.

### D2 — Thứ tự và tên cột

`SELECT a, b` với `SELECT b, a` chứa cùng thông tin, khác thứ tự tuple.

→ Mặc định so **chặt theo vị trí**, giống BIRD, để số của ta so sánh được
với số đã công bố. Nhưng ghi **thêm một số thứ hai** ở chế độ hoán vị cột,
để biết bao nhiêu phần của khoảng cách chỉ là hình thức.

### D3 — Kiểu số

Postgres trả `Decimal` cho `numeric`, `float` cho `double precision`.
`SUM(amount)` có thể ra `Decimal('1234.00')` bên này và `1234.0` bên kia.

→ Chuẩn hoá: `Decimal` → `float`, làm tròn 6 chữ số thập phân. Với cột
tiền thì 6 chữ số là thừa an toàn.

### D4 — Dòng trùng

Vàng trả 3 dòng giống nhau, sinh ra trả 1 dòng — có tính đạt không?

→ Không. Dùng **multiset** (`Counter`), không dùng `set`. Đây là chỗ **cố ý
khác BIRD**: bộ chấm chính thức của BIRD dùng `set`, và điều đó tha nhầm
những câu thiếu `GROUP BY`. Ghi rõ khi so số với BIRD.

### D5 — Kết quả rỗng

Cả hai bên trả 0 dòng thì về kỹ thuật là khớp. Nhưng một câu SQL sai cũng
rất hay trả về rỗng.

→ Đếm riêng thành bucket `both_empty`, **không** gộp vào tỉ lệ đạt. Gộp vào
là tự thổi phồng, nhất là trên database nhỏ nơi nhiều câu hỏi vốn không có
dữ liệu trả về.

### D6 — Kết quả quá lớn

Không chèn được `LIMIT` vào SQL tuỳ ý một cách an toàn.

→ `fetchmany(N)` với N ví dụ 10.000. Nếu **bên nào** chạm trần thì phép so
không còn đáng tin → bucket `truncated` riêng, không tính đạt cũng không
tính trượt.

---

## 4. An toàn khi chạy SQL do model sinh

Bốn lớp, đều bắt buộc:

1. **Chỉ đọc** — `SET TRANSACTION READ ONLY` cho mọi giao dịch, kể cả khi
   chạy SQL vàng.
2. **Hạn giờ** — `statement_timeout` (đề xuất 30s). Một phép nối chéo trên
   29.830 dòng `orders` sẽ treo vô hạn.
3. **Trần dòng** — `fetchmany`, xem D6.
4. **Luôn rollback** — bọc trong transaction, rollback bất kể kết quả.

---

## 5. Bucket báo cáo — không gộp thành một con số

Theo đúng lối `RecallReport`:

| Bucket | Nghĩa |
|---|---|
| `match` | hai tập kết quả khớp |
| `mismatch` | cả hai chạy được, kết quả khác nhau |
| `both_empty` | cả hai trả 0 dòng (tách riêng, xem D5) |
| `pred_error` | SQL sinh ra lỗi — cú pháp, sai tên bảng, thiếu quyền |
| `pred_timeout` | SQL sinh ra quá hạn giờ |
| `truncated` | một bên chạm trần dòng |
| `gold_error` | **SQL vàng** lỗi — đây là lỗi dữ liệu |

`exec_accuracy = match / (total − gold_error)`

`gold_error` bị trừ khỏi mẫu số, đúng cách `measure_recall` bỏ qua bản ghi
mà SQL vàng không parse ra bảng nào: lỗi dữ liệu không phải lỗi hệ thống.
Nhưng phải **in ra**, vì `gold_error` cao nghĩa là golden set hỏng và mọi
con số phía trên đều đáng ngờ.

---

## 6. Hình dạng mã

Giữ nguyên lối của `tier1_recall.py`: hàm đo **thuần**, nhận callable, nên
test được mà không cần database lẫn model.

```
eval/tier2_execution.py
    @dataclass ExecReport            # các bucket ở mục 5 + as_text()
    compare_results(gold_rows, pred_rows, *, ordered) -> Verdict
    measure_execution(records, generate, execute) -> ExecReport
    main()                           # CLI

tests/unit/test_tier2_execution.py
```

`generate: (EvalRecord) -> str` và `execute: (str) -> Rows` là hai chỗ tiêm.
Phần lớn test nằm ở `compare_results` — nó thuần tuý, và toàn bộ sáu quyết
định ở mục 3 đều khoá được bằng test không cần hạ tầng gì.

---

## 7. Phương pháp đo — phần dễ tự lừa mình nhất

### 7.1 Nhiễu trước, kết luận sau

**Luôn chạy nhánh nền hai lần** trước khi tin bất kỳ khác biệt nào.

Đây là bài học trực tiếp: ở thí nghiệm chú giải song ngữ tôi đã tuyên bố
"thắng ở mọi k", rồi lượt chạy thứ ba lật ngược — 14/19 khác biệt hoá ra là
nhiễu giữa các lượt. `AGENT_TEMPERATURES["sql"]` đã là `0.0`, tốt hơn mức
`0.2` của annotate, nhưng greedy trên Ollama vẫn không đảm bảo giống hệt
từng bit. Đo sàn nhiễu rồi mới đọc kết quả.

### 7.2 So theo cặp, không so hai lượt độc lập

Chạy cả hai nhánh trên **cùng tập câu hỏi**, so kết quả **từng câu**
(McNemar). Chỉ những câu lệch nhau mới mang thông tin.

Không có thiết kế cặp: n=200, tỉ lệ nền ~50% thì khoảng tin cậy 95% cỡ
±7 điểm — **một cải thiện 5 điểm không phát hiện được**. Có thiết kế cặp
thì cần ít câu hơn nhiều cho cùng độ nhạy.

### 7.3 Cỡ mẫu và chi phí

| Golden set | Số câu | Dùng được để |
|---|---|---|
| `eval/golden_adba.jsonl` | 12 | smoke test |
| `data/demo/chinook/golden_vi.jsonl` | 20 | smoke test |
| `data/benchmarks/bird/golden_all.jsonl` | **1.534** | đo thật |

Chinook 20 câu: một câu lật là 5 điểm. Không phải phép đo.

1.534 câu × một lượt gọi model ≈ vài giờ mỗi nhánh. Nên có `--limit` với
lấy mẫu phân tầng theo `db_id` và `--seed` cố định; chạy full là lựa chọn
có chủ đích, không phải mặc định.

### 7.4 Chống rò rỉ khi đo few-shot

Ba mức, chỉ mức thứ ba trả lời được câu hỏi sản phẩm:

1. **Leave-one-out** — bỏ câu Q khỏi kho ví dụ khi chấm Q. Bắt buộc, nhưng
   chưa đủ.
2. **Khử trùng lặp gần** — LOO vẫn rò nếu golden set có câu diễn đạt lại
   ("doanh thu theo vùng" / "theo khu vực"). BIRD có sẵn các mẫu câu lặp.
3. **Giữ lại nguyên database** — con số đầu bảng phải đo trên database
   chưa hề góp ví dụ nào. Lời hứa của sản phẩm là "chạy được ở chỗ khách
   mới", nên phải giữ lại ở mức *database*, không phải mức *câu hỏi*.

---

## 8. Điều thiết kế này không giải quyết

**Khách mới không có golden set.** Ngày đầu bàn giao kho ví dụ rỗng, nên
few-shot không làm gì cả. Golden set chỉ lớn dần khi khách dùng thật — đúng
cách Vanna tích luỹ từ truy vấn thành công. Tầng 2 đo được few-shot *khi đã
có* ví dụ; nó không tạo ra ví dụ.

**So khớp tập kết quả không phải chân lý.** Một câu SQL sai vẫn có thể tình
cờ trả đúng tập kết quả trên dữ liệu nhỏ. Ngược lại, một câu đúng có thể
trượt vì khác cách làm tròn. Tầng 2 chặt hơn `EXPLAIN` rất nhiều, nhưng nó
là proxy, không phải thẩm phán.

**Không đo được thứ khách thật sự hỏi.** Golden set là câu hỏi ai đó nghĩ
ra trước, không phải câu hỏi phát sinh trong công việc.
