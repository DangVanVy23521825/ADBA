-- Role chỉ-đọc cho đường production (spec 2026-08-12 mục 4.1, lớp 1).
--
-- Đây là lớp bảo đảm DUY NHẤT. Ba lớp phía ứng dụng (chặn multi-statement,
-- quét DML trên statement đã parse, trần tài nguyên) chỉ phát hiện sớm; nếu
-- cả ba thủng thì lớp này vẫn khiến DELETE báo lỗi thay vì chạy.
--
-- IDEMPOTENT: chạy lại nhiều lần không sao.
--
-- KHÔNG thu hồi bất cứ quyền nào của adba_user. adba_user vẫn cần quyền ghi
-- cho data/seed/seed_data.py, và container này đang phục vụ nhiều nhánh phát
-- triển song song — lấy quyền của nó đi là làm hỏng việc của người khác.

\set ON_ERROR_STOP on

-- Không dùng DO $$ ... $$ ở đây: psql KHÔNG nội suy biến ":'ro_password'"
-- bên trong một chuỗi dollar-quoted (đã kiểm chứng bằng thực nghiệm trên
-- container thật — psql 15.18 báo "syntax error at or near ':'"). Mẫu
-- SELECT ... \gexec dưới đây nội suy biến ở tầng câu lệnh phẳng (không có
-- $$), rồi thực thi chuỗi lệnh mà truy vấn trả về — vẫn giữ nguyên tính
-- idempotent: EXISTS chọn nhánh ALTER, NOT EXISTS chọn nhánh CREATE.
SELECT format('ALTER ROLE adba_readonly LOGIN PASSWORD %L', :'ro_password')
WHERE EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'adba_readonly')
UNION ALL
SELECT format('CREATE ROLE adba_readonly LOGIN PASSWORD %L', :'ro_password')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'adba_readonly')
\gexec

GRANT CONNECT ON DATABASE adba_db TO adba_readonly;
GRANT USAGE ON SCHEMA public TO adba_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO adba_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO adba_readonly;

-- Chỉ thu hồi của CHÍNH adba_readonly, không của ai khác.
REVOKE CREATE ON SCHEMA public FROM adba_readonly;

-- Cái chốt cuối: mọi transaction của role này mở ở chế độ chỉ đọc, nên cả
-- data-modifying CTE — thứ bắt đầu bằng WITH và qua được mọi heuristic
-- "token đầu tiên" — cũng bị Postgres từ chối.
ALTER ROLE adba_readonly SET default_transaction_read_only = on;
