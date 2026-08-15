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
kết quả. Con số đáng nhìn là `bảng/context TB`: mọi retriever phải giữ
recall gần 1.0 trong khi kéo con số đó xuống.

`lexical` kéo context trung bình từ 9.0 xuống 2.0 bảng (giảm 78%) nhưng
đánh đổi bằng recall rơi xuống 0.250 — chỉ 3/12 câu còn đủ bảng. Ba câu
sống sót đều là câu có từ "region" xuất hiện y nguyên trong câu hỏi tiếng
Việt, trùng thẳng vào tên cột `orders.region`. Chín câu trượt còn lại đều
mất trắng: `LexicalRetriever` chấm điểm theo token trùng giữa câu hỏi và
tên bảng/mô tả/tên cột, nhưng `perception/info_box_all.json` hiện chưa có
`description` cho bảng nào (rỗng ở cả 9 bảng), còn tên bảng và tên cột đều
là tiếng Anh. Một câu hỏi thuần tiếng Việt như "Số lượng khách hàng theo
phân khúc" vì vậy gần như không có token nào trùng với `customers`.

Đây không phải lỗi harness — đây chính là điều tầng 1 được dựng ra để lộ
diện: baseline lexical thô, không có mô tả bảng bằng tiếng Việt, thua xa
trần `full`. Việc thêm `description` tiếng Việt vào info_box (hoặc dịch
câu hỏi/embedding) là hướng cải thiện rõ ràng cho retriever kế tiếp, và
mốc 0.250 / 2.0 ở trên là vạch xuất phát nó phải vượt qua.

## Ba bộ dữ liệu ngoài

Xem Task 11 của plan. Bảng cấu hình nằm ở đó và được `eval/describe_dataset.py`
sinh ra, không viết tay.

## Tầng 2 và 3

Xem spec mục 6.1. Chưa hiện thực trong plan này.
