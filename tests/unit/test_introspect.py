import os

import pytest

from perception.introspect import _IDENT, introspect_schema, sample_rows

DSN = os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")


class TestIdentifierGuard:
    """Kiểm tra thuần: validate xảy ra trước khi `sample_rows` kết nối, nên
    những test này không cần DB thật và phải luôn chạy — kể cả trong môi
    trường CI/dev không có `DATABASE_URL`. DSN giả (không bao giờ được
    dùng để kết nối) chỉ để thoả tham số bắt buộc.
    """

    _UNUSED_DSN = "postgresql://unused/db"

    def test_rejects_a_table_name_that_is_not_an_identifier(self):
        with pytest.raises(ValueError):
            sample_rows(self._UNUSED_DSN, "orders; DROP TABLE customers", n=1)

    def test_rejects_a_table_name_with_trailing_newline(self):
        """re.match + '$' khớp ngay trước một '\\n' cuối chuỗi trong Python —
        một cạm bẫy quen thuộc khiến "orders\\n" lọt qua re.match() dù nó
        không phải một identifier trần. fullmatch() đóng khe hở này."""
        with pytest.raises(ValueError):
            sample_rows(self._UNUSED_DSN, "orders\n", n=1)

    def test_rejects_a_table_name_with_a_space(self):
        with pytest.raises(ValueError):
            sample_rows(self._UNUSED_DSN, "orders drop", n=1)

    def test_rejects_a_table_name_starting_with_a_digit(self):
        with pytest.raises(ValueError):
            sample_rows(self._UNUSED_DSN, "1orders", n=1)

    # I3(b): ADBA đọc schema của khách hàng tiếng Việt — introspect_schema
    # trả nguyên văn table_name từ information_schema.columns, và Postgres
    # chấp nhận các tên này bình thường. sql.Identifier (lớp phòng thủ thứ
    # hai, xem docstring sample_rows) quote chúng đúng đắn; _IDENT không
    # được chặt hơn lớp nó bảo vệ.
    def test_accepts_a_vietnamese_table_name_with_a_dieresis(self):
        assert _IDENT.fullmatch("đơn_hàng")

    def test_accepts_another_vietnamese_table_name(self):
        assert _IDENT.fullmatch("khách_hàng")

    def test_does_not_raise_for_vietnamese_identifiers(self, monkeypatch):
        """sample_rows() đầu-đến-cuối với một tên bảng tiếng Việt, kết nối
        giả (không chạm mạng thật — mock psycopg2.connect) để chứng minh
        _IDENT không chặn nó ở lớp guard lẫn không có lỗi nào khác ở
        đường đi sau đó tới sql.Identifier."""

        class _FakeSampleCursor:
            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return False

            def execute(self, query, params=None):
                self.query = query

            def fetchall(self):
                return []

        class _FakeSampleConnection:
            def cursor(self, *args, **kwargs):
                return _FakeSampleCursor()

            def close(self):
                pass

        monkeypatch.setattr(
            "psycopg2.connect", lambda dsn: _FakeSampleConnection()
        )
        # Không ném ValueError (guard) và không ném gì khác: _IDENT chấp
        # nhận tên này, và sql.Identifier dựng câu lệnh được bình thường.
        assert sample_rows(self._UNUSED_DSN, "đơn_hàng", n=1) == []


class _FakeCursor:
    """Cursor giả tối thiểu: trả kết quả theo nội dung câu SQL vừa execute,
    đủ để introspect_schema chạy hết một vòng mà không cần Postgres thật."""

    def __init__(self, reltuples: int):
        self._reltuples = reltuples
        self._last_sql = ""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def execute(self, query, params=None):
        self._last_sql = query

    def fetchall(self):
        if "information_schema.columns" in self._last_sql:
            return [("t", "id", "integer", "NEVER")]
        if "PRIMARY KEY" in self._last_sql:
            return [("t", "id")]
        if "FOREIGN KEY" in self._last_sql:
            return []
        raise AssertionError(f"unexpected fetchall() for query: {self._last_sql!r}")

    def fetchone(self):
        if "reltuples" in self._last_sql:
            return (self._reltuples,)
        raise AssertionError(f"unexpected fetchone() for query: {self._last_sql!r}")


class _FakeConnection:
    def __init__(self, reltuples: int):
        self._reltuples = reltuples

    def cursor(self, *args, **kwargs):
        return _FakeCursor(self._reltuples)

    def close(self):
        pass


class TestRowCountNormalisation:
    """Không cần Postgres thật: mock tầng psycopg2.connect để khẳng định
    introspect_schema tự chuẩn hoá `-1` (sentinel "chưa ANALYZE" của
    pg_class.reltuples) thành None, không phải một số dòng thật.

    Không dùng bảng thật/ANALYZE trên container dùng chung để tạo điều kiện
    này — mock trực tiếp giá trị reltuples để test không phụ thuộc vào việc
    ai đó đã chạy ANALYZE trên bảng nào trước đó."""

    def test_negative_reltuples_normalises_to_none(self, monkeypatch):
        monkeypatch.setattr(
            "psycopg2.connect", lambda dsn: _FakeConnection(reltuples=-1)
        )
        tables = introspect_schema("postgresql://unused/db")
        assert len(tables) == 1
        assert tables[0].row_count is None

    def test_non_negative_reltuples_passes_through(self, monkeypatch):
        monkeypatch.setattr(
            "psycopg2.connect", lambda dsn: _FakeConnection(reltuples=42)
        )
        tables = introspect_schema("postgresql://unused/db")
        assert tables[0].row_count == 42


@pytest.mark.skipif(not DSN, reason="cần DATABASE_URL trỏ Postgres thật")
class TestAgainstLiveDatabase:
    """Những test này mở kết nối thật tới Postgres nên bị skip khi không có
    DATABASE_URL/POSTGRES_URL — không giống TestIdentifierGuard ở trên."""

    def test_finds_every_public_table(self):
        names = {t.name for t in introspect_schema(DSN)}
        assert {"orders", "customers", "products", "payroll"} <= names

    def test_columns_carry_name_and_type(self):
        orders = next(t for t in introspect_schema(DSN) if t.name == "orders")
        by_name = {c.name: c for c in orders.columns}
        assert "id" in by_name
        assert by_name["id"].data_type

    def test_primary_key_is_detected(self):
        orders = next(t for t in introspect_schema(DSN) if t.name == "orders")
        assert orders.primary_key

    def test_foreign_keys_render_as_table_paren_column(self):
        orders = next(t for t in introspect_schema(DSN) if t.name == "orders")
        assert any(ref.startswith("customers(") for ref in orders.foreign_keys.values())

    def test_generated_columns_are_flagged(self):
        orders = next(t for t in introspect_schema(DSN) if t.name == "orders")
        assert any(c.is_generated for c in orders.columns), "orders có cột GENERATED"

    def test_descriptions_start_empty(self):
        """Introspect chỉ ra cấu trúc. Ý nghĩa đến từ bước chú giải.

        Column chưa có field `description` (Task 2 sẽ thêm) nên ở đây chỉ kiểm
        tra Table.description; xem task-1-report.md.
        """
        tables = introspect_schema(DSN)
        assert all(t.description == "" for t in tables)

    def test_sample_rows_returns_dicts_capped_at_n(self):
        rows = sample_rows(DSN, "customers", n=3)
        assert len(rows) <= 3
        assert all(isinstance(r, dict) for r in rows)
