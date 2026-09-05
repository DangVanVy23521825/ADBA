"""Dịch SQL vàng phương ngữ SQLite sang Postgres.

BIRD và Spider phát hành SQL vàng viết cho SQLite. Nạp DỮ LIỆU sang
Postgres (`eval/load_sqlite_to_postgres.py`) không làm cho SQL chạy được:
chỉ 531/1534 câu của BIRD chạy nổi trên Postgres nếu để nguyên.

Hai khác biệt gây ra gần hết số đó, và cả hai đều dịch được:

1. Định danh bọc bằng dấu huyền (`` `x` ``) thay vì nháy kép.
2. Định danh KHÔNG nháy giữ nguyên hoa thường trong SQLite, còn Postgres
   gấp về chữ thường. Bộ nạp giữ nguyên hoa thường của cột (`CDSCode`),
   nên `T1.CDSCode` không nháy gấp thành `cdscode` và không tìm thấy.

Đo thật trên 1.534 câu BIRD, từng bước một:

    chỉ đổi dấu huyền                       531  (34.6%)
    + bọc định danh, bỏ qua tên nhập nhằng  953  (62.1%)
    + phân giải nhập nhằng qua bí danh     1194  (77.8%)

340 câu còn lại KHÔNG dịch được ở đây và cố ý để lại: hàm chỉ SQLite có
(`strftime`, `julianday`), kiểu lỏng của SQLite (`text = integer`), cú
pháp `LIMIT #,#`, và bảng chưa nạp. Chúng cần dịch ngữ nghĩa chứ không
phải dịch định danh, nên chỗ của chúng không phải module này.

⚠️ 340 câu bị loại KHÔNG phải mẫu ngẫu nhiên. Câu dùng hàm ngày tháng của
SQLite thường là câu khó hơn trung bình, nên số đo trên 1.194 câu còn lại
lệch theo hướng LẠC QUAN. Ghi con số này cạnh mọi kết quả dùng nó.
"""

from __future__ import annotations

from perception.sql_identifiers import SchemaNames, requote

__all__ = ["SchemaNames", "backticks_to_double_quotes", "translate"]


def backticks_to_double_quotes(sql: str) -> str:
    """`x` -> "x", trừ khi dấu huyền nằm trong chuỗi nháy đơn.

    Chuỗi văn bản có thể chứa dấu huyền như dữ liệu thật; đổi nó thành
    nháy kép sẽ vừa hỏng chuỗi vừa đẻ ra một định danh không tồn tại.
    """
    out: list[str] = []
    in_string = False
    for ch in sql:
        if ch == "'":
            in_string = not in_string
            out.append(ch)
        elif ch == "`" and not in_string:
            out.append('"')
        else:
            out.append(ch)
    return "".join(out)


def translate(sql: str, names: SchemaNames) -> str:
    """Đưa SQL vàng SQLite về dạng Postgres chạy được, hoặc trả gần đúng.

    Phần bọc nháy định danh KHÔNG nằm ở đây: nó ở
    `perception/sql_identifiers.py`, vì cùng thuật toán đó cũng chạy trong
    sản phẩm để sửa đầu ra của model. Giữ hai bản sao thì chúng sẽ lệch
    nhau, và bản trong eval lệch khỏi bản trong sản phẩm nghĩa là số đo
    không còn nói về thứ đang bán.

    Cái riêng của đường eval chỉ là dấu huyền — cú pháp SQLite mà model
    không bao giờ sinh ra.

    Hàm này KHÔNG hứa SQL trả về sẽ chạy: nó chỉ sửa định danh. Câu dùng
    `strftime` vẫn hỏng, và hỏng ở tầng `execute` rồi vào bucket
    `gold_error` — đúng chỗ, vì đó là lỗi dữ liệu chứ không phải lỗi hệ
    thống.
    """
    return requote(backticks_to_double_quotes(sql), names)
