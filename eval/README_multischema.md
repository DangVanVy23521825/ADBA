# Eval đa schema — cấu hình thực tế

## Tầng 1 — recall chọn bảng

Không LLM, không GPU, không database. Tập bảng đúng parse từ SQL mẫu bằng
`perception/sql_tables.py`.

Chạy:

    python -m eval.tier1_recall --golden <file.jsonl> --strategy lexical --k 8

Định nghĩa recall: tỉ lệ câu mà context chứa **toàn bộ** tập bảng đúng.
Không có điểm từng phần.

### Mốc trên golden set ADBA (12 câu, 9 bảng)

| Chiến lược | recall | bảng/context TB |
|---|---|---|
| `full` | 1.000 | 9.0 |
| `lexical`, k=8 | 0.250 | 2.0 |

`full` luôn cho recall 1.0 theo định nghĩa — nó là trần trên, không phải
kết quả.

**Cảnh báo đọc bảng trên: `bảng/context TB = 2.0` của `lexical` KHÔNG phải
nén vừa phải.** Đó là trung bình của một phân phối hai cực, không phải một
con số điển hình cho từng câu. Phân theo từng câu:

| # | Câu hỏi | Bảng lấy được |
|---|---|---|
| 1 | Tổng doanh thu theo region năm 2024 | 8 |
| 2 | Top 5 sản phẩm bán chạy nhất theo doanh thu | 0 |
| 3 | Số lượng nhân viên theo từng phòng ban | 0 |
| 4 | Doanh thu theo tháng trong năm 2024 | 0 |
| 5 | Giá trị đơn hàng trung bình theo region | 8 |
| 6 | Số lượng khách hàng theo phân khúc | 0 |
| 7 | Sản phẩm có tồn kho dưới ngưỡng tối thiểu | 0 |
| 8 | Tổng doanh thu theo quý năm 2024 | 0 |
| 9 | Lương trung bình theo phòng ban | 0 |
| 10 | So sánh doanh thu Q4 2024 vs Q4 2023 theo region | 8 |
| 11 | Lượng hàng luân chuyển theo kho | 0 |
| 12 | Sản phẩm chưa bán trong 180 ngày qua | 0 |

**9 câu → 0 bảng (context rỗng, không phải context một phần). 3 câu → 8/9
bảng (gần hết schema, thiếu mỗi `payroll`).** Không có câu nào ở giữa.
`(0×9 + 8×3) / 12 = 2.0` — đúng số harness in ra, nhưng trung bình này che
mất chỗ hành vi thật sự nằm: retriever này không "nén schema xuống còn
khoảng 2 bảng", nó **hoặc trả về rỗng, hoặc trả về gần hết schema**.

Cơ chế, xác minh trực tiếp bằng cách chạy lại resolver và in điểm số
`LexicalRetriever`: ba câu "8 bảng" đều chứa nguyên từ tiếng Anh "region"
trong câu hỏi tiếng Việt. Token đó trùng tên cột ở **4 bảng cùng lúc** —
`customers.region`, `orders.region`, `warehouses.region`,
`departments.region` — nên cả 4 bảng được `LexicalRetriever` chấm điểm > 0
và lọt vào seed. `expand_by_foreign_keys` sau đó mở rộng seed đúng một
bước theo cạnh khóa ngoại (gồm cả liên kết mềm trong `cross_domain_hints`
mà `tables_from_info_box` nạp vào `foreign_keys` giống hệt FK thật, ví dụ
`warehouses.region → orders.region`), và một bước đó đã đủ kéo seed 4
bảng lan ra 8 bảng: `orders` kéo theo `products`; `departments` kéo theo
`employees`; `warehouses` kéo ngược `stock` và `stock_movements` (cả hai
đều tham chiếu `warehouses`). Bảng duy nhất còn sót lại là `payroll` — nó
cách seed đúng hai bước (`departments → employees → payroll`), ngoài tầm
một bước mở rộng. Chín câu còn lại không có token nào trùng bảng/mô tả/tên
cột (mọi `description` trong `perception/info_box_all.json` đang rỗng, và
tên bảng/cột đều là tiếng Anh trong khi câu hỏi là tiếng Việt), nên
`LexicalRetriever` không chấm điểm được bảng nào và trả về tập rỗng — không
phải một tập con nhỏ, một tập RỖNG.

Đây không phải lỗi harness — đây chính là điều tầng 1 được dựng ra để lộ
diện: baseline lexical thô, không có mô tả bảng bằng tiếng Việt, gần như
không hoạt động theo cả hai hướng. **Vạch xuất phát cho retriever kế tiếp
không phải "2.0 bảng ở recall 0.250"; đó là "9/12 câu trả về rỗng hoàn
toàn, 3/12 câu trả về gần hết schema."** Một retriever kế tiếp (embedding,
hoặc lexical có `description` tiếng Việt) phải làm khác đi ở CẢ HAI phía:
lấp khoảng trống khi không có từ trùng, và không cascade từ một token
trùng ngẫu nhiên ra gần hết schema. Việc thêm `description` tiếng Việt vào
info_box (hoặc dịch câu hỏi/embedding) là hướng cải thiện rõ ràng, và mô
tả hai cực này — không phải con số 2.0 — là mốc mọi retriever sau phải so
sánh.

Câu hỏi 4 ("Doanh thu theo tháng trong năm 2024") dùng
`EXTRACT(MONTH FROM order_date)` — đúng cú pháp mà `prompts/text_to_sql.txt`
chỉ dẫn model sinh ra. Bản đầu của golden set từng né cú pháp này (đổi
sang `DATE_PART`) để lách qua một lỗi thật trong `perception/sql_tables.py`:
`FROM` bên trong `EXTRACT(field FROM src)` từng bị bộ quét nhận nhầm là mở
đầu mệnh đề nguồn, khiến `order_date` bị thêm nhầm vào tập bảng đúng như
thể nó là một bảng. Lỗi đã được sửa tại gốc (xem `perception/sql_tables.py`
và `tests/unit/test_sql_tables.py`); golden set khôi phục lại `EXTRACT` để
phép đo phản ánh đúng cú pháp SQL mà hệ thống thực sự sinh ra. Hai con số
trong bảng trên không đổi so với bản né tránh, vì cả hai đều quy về cùng
tập bảng đúng `{orders}` cho câu 4.

## Ba bộ dữ liệu ngoài

Xem Task 11 của plan. Bảng cấu hình nằm ở đó và được `eval/describe_dataset.py`
sinh ra, không viết tay.

**Chưa tải, chưa điền số.** Việc tải Spider/BIRD/BEAVER và xác định license
có cho phép tải lẫn commit vào repo này hay không là quyết định của chủ sở
hữu repo, không phải của tác nhân triển khai — dữ liệu bên thứ ba có license
hạn chế commit nhầm vào git rất khó gỡ vì nó nằm lại trong lịch sử. Đây là
điều kiện tiên quyết còn treo trước khi bảng dưới đây có số thật.

Khi dữ liệu đã có và đã chuẩn hóa về `questions.jsonl` + `schemas.json`
(xem định dạng ở Task 11), sinh từng dòng của bảng bằng:

    python -m eval.describe_dataset --name <tên bộ> \
        --questions <dir>/questions.jsonl --schemas <dir>/schemas.json

| Bộ | Số DB | Số câu | Bảng/DB (TB) | Bảng/DB (max) | License |
|---|---|---|---|---|---|
| spider | | | | | |
| bird | | | | | |
| beaver | | | | | |

## Tầng 2 và 3

Xem spec mục 6.1. Chưa hiện thực trong plan này.
