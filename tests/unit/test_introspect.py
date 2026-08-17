import os

import pytest

from perception.introspect import introspect_schema, sample_rows

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
