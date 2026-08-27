# Chinook — schema demo phát hành lại được

Cửa hàng nhạc số: nghệ sĩ, album, bài hát, khách hàng, hoá đơn, nhân viên.
11 bảng, 64 cột, 11 khoá ngoại.

## Vì sao có thư mục này

Bộ dữ liệu duy nhất trong repo mà **được phép ship kèm bản giao khách**.

| | Chinook | BIRD |
|---|---|---|
| License | **MIT** | CC BY-SA 4.0 |
| Phát hành lại | **được** | **không** |
| Định dạng | Postgres gốc | SQLite, phải chuyển đổi |

Bản on-prem **không có internet**, nên dữ liệu demo phải nằm sẵn trong
package chứ không tải lúc chạy. Đó là lý do file `.sql` được commit thẳng
vào đây thay vì đi qua `eval/fetch_dataset.py` như BIRD.

Domain của nó — hoá đơn, khách hàng, doanh thu — cũng gần với bài toán BI
của ADBA hơn hẳn các bộ benchmark khác.

## Nạp

Script tự `DROP DATABASE IF EXISTS chinook` rồi `CREATE DATABASE chinook`,
và dùng meta-command `\c` của psql — nên phải chạy qua `psql`, không chạy
được bằng psycopg2.

```bash
docker exec -i adba-postgres psql -q -U "$POSTGRES_USER" -d postgres \
  < data/demo/chinook/chinook_postgres.sql
```

Cảnh báo: nó **xoá** database `chinook` đang có, nếu có.

## Chạy trọn đường onboarding

```bash
export ADBA_DSN="postgresql://user:pass@host:5432/chinook"
python onboard.py extract  --profile data/demo/chinook/profile
python onboard.py annotate --profile data/demo/chinook/profile
python onboard.py build    --profile data/demo/chinook/profile --grant local='*'
python onboard.py verify   --profile data/demo/chinook/profile \
    --golden data/demo/chinook/golden_vi.jsonl --user local
```

Schema này render ra ~523 token, dưới ngưỡng 6000, nên `build` mặc định cho
ra chế độ `full` — và ở chế độ đó **recall tất yếu bằng 1,000** vì mọi bảng
luôn có mặt trong context. Muốn đo retriever thật thì ép chế độ `retrieval`:

```bash
python onboard.py build --profile data/demo/chinook/profile \
    --grant local='*' --threshold-tokens 200
```

## `golden_vi.jsonl`

20 câu hỏi **tiếng Việt** kèm SQL mẫu, viết cho repo này (không lấy từ
nguồn nào). Mọi câu đã được xác nhận chạy được trên database thật.

Tiếng Việt là có chủ đích: `LexicalRetriever` chấm theo trùng lặp token, và
chú giải do model sinh cũng là tiếng Việt. Golden set tiếng Anh sẽ khiến
phần mô tả (trọng số 2.0) gần như không đóng góp gì, và phép đo sẽ đánh giá
thấp giá trị thật của chú giải — đúng thứ đã xảy ra khi đo trên BIRD.

## Nguồn và giấy phép

https://github.com/lerocha/chinook-database — MIT, Copyright (c) 2008-2024
Luis Rocha. Toàn văn ở `LICENSE.md` trong thư mục này, giữ nguyên theo yêu
cầu của giấy phép.

`chinook_postgres.sql` là bản `ChinookDatabase/DataSources/Chinook_PostgreSql.sql`
lấy nguyên vẹn, không sửa.
