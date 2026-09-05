"""Bốn lớp an toàn của `make_executor`, chạy trên Postgres thật.

Đây là phần của tầng 2 mà sai thì nguy hiểm, không chỉ là số đo lệch: ta
đang chạy SQL do model sinh ra trên database của khách. Test giả không
chứng minh được `SET TRANSACTION READ ONLY` thật sự chặn ghi, hay
`statement_timeout` thật sự cắt một truy vấn treo — chỉ database thật mới
chứng minh được.
"""

import os

import pytest

psycopg2 = pytest.importorskip("psycopg2")

from eval.tier2_execution import QueryError, QueryTimeout, make_executor  # noqa: E402

DSN = os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")
pytestmark = pytest.mark.skipif(not DSN, reason="cần DATABASE_URL trỏ Postgres thật")


@pytest.fixture
def conn():
    c = psycopg2.connect(DSN)
    try:
        yield c
    finally:
        c.close()


def test_a_plain_select_returns_rows(conn):
    execute = make_executor(conn)
    assert execute("SELECT 1, 'a'").rows == ((1, "a"),)


def test_write_statements_are_refused_by_the_database_itself(conn):
    """Lớp 1. Chặn ở Postgres chứ không phải bằng cách rà chuỗi SQL: rà chuỗi
    thì luôn có cách viết lách qua, còn READ ONLY thì không."""
    execute = make_executor(conn)
    with pytest.raises(QueryError) as e:
        execute("CREATE TABLE tier2_should_never_exist (x int)")
    assert "read-only" in str(e.value).lower()


def test_the_connection_still_works_after_a_refused_write(conn):
    """Rollback ở `finally` phải dọn sạch: một câu SQL hỏng không được làm
    hỏng phần còn lại của lượt chạy hàng nghìn câu."""
    execute = make_executor(conn)
    with pytest.raises(QueryError):
        execute("SELECT * FROM khong_co_bang_nay")
    assert execute("SELECT 1").rows == ((1,),)


def test_a_hanging_query_is_cut_and_reported_as_a_timeout(conn):
    """Lớp 2. `pg_sleep` thay cho phép nối chéo: cùng hiệu ứng treo, nhưng
    tất định và không phụ thuộc dữ liệu có sẵn."""
    execute = make_executor(conn, timeout_s=1)
    with pytest.raises(QueryTimeout):
        execute("SELECT pg_sleep(5)")


def test_a_timeout_is_distinguishable_from_a_syntax_error(conn):
    """`measure_execution` xếp hai thứ này vào hai bucket khác nhau, nên
    `make_executor` phải phân biệt được chúng ngay từ chỗ ném."""
    execute = make_executor(conn, timeout_s=1)
    with pytest.raises(QueryTimeout):
        execute("SELECT pg_sleep(5)")
    with pytest.raises(QueryError) as e:
        execute("SELEC 1")
    assert not isinstance(e.value, QueryTimeout)


def test_the_timeout_does_not_leak_onto_the_connection(conn):
    """`SET LOCAL` phải chết cùng transaction, không bám vào kết nối.

    Dùng HAI executor trên cùng một `conn`: cái hạn giờ 1 giây bị cắt, rồi
    cái hạn giờ 30 giây phải chạy trót lọt một truy vấn 2 giây. Nếu `SET
    LOCAL` rò ra ngoài transaction thì truy vấn thứ hai cũng chết, và trong
    một lượt chạy thật điều đó nghĩa là một câu treo sẽ kéo theo mọi câu
    chậm phía sau nó.

    (Bản test đầu của tôi dùng CHUNG một executor 1 giây cho cả hai truy
    vấn, nên nó chỉ chứng minh hạn giờ hoạt động — không chứng minh được
    điều đang cần.)
    """
    short = make_executor(conn, timeout_s=1)
    with pytest.raises(QueryTimeout):
        short("SELECT pg_sleep(5)")

    generous = make_executor(conn, timeout_s=30)
    # `pg_sleep` ở mệnh đề FROM, không ở danh sách chọn: nó trả kiểu `void`,
    # và cách psycopg2 ánh xạ `void` sang Python là chi tiết không đáng để
    # một test về hạn giờ phụ thuộc vào.
    assert generous("SELECT 1 FROM pg_sleep(2)").rows == ((1,),)


def test_a_result_over_the_cap_is_flagged_not_silently_cut(conn):
    """Lớp 3. Cắt lặng lẽ rồi đem so hai tập cụt sẽ cho ra 'khớp' vô nghĩa."""
    execute = make_executor(conn, max_rows=10)
    res = execute("SELECT generate_series(1, 100)")
    assert res.truncated is True
    assert len(res.rows) == 10


def test_a_result_exactly_at_the_cap_is_not_flagged(conn):
    """Ranh giới: lấy `max_rows + 1` chính là để phân biệt 'vừa đủ trần' với
    'vượt trần'. Lấy đúng `max_rows` thì hai ca này không phân biệt được."""
    execute = make_executor(conn, max_rows=10)
    res = execute("SELECT generate_series(1, 10)")
    assert res.truncated is False
    assert len(res.rows) == 10


def test_a_statement_returning_no_result_set_is_empty_not_an_error(conn):
    """Model đôi khi sinh ra thứ không phải SELECT. Nó chạy được nhưng không
    trả lời câu hỏi, nên `mismatch` mô tả đúng hơn `pred_error`."""
    execute = make_executor(conn)
    assert execute("SET LOCAL work_mem = '4MB'").rows == ()


def test_nothing_the_executor_ran_was_committed(conn):
    """Lớp 4. Chốt tổng: sau tất cả những gì ở trên, database không đổi."""
    execute = make_executor(conn)
    with pytest.raises(QueryError):
        execute("CREATE TABLE tier2_should_never_exist (x int)")
    rows = execute(
        "SELECT count(*) FROM information_schema.tables "
        "WHERE table_name = 'tier2_should_never_exist'"
    ).rows
    assert rows == ((0,),)
