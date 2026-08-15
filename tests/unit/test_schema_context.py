import pytest

from perception.connection_profile import ALL_TABLES, build_profile
from perception.retrieval import LexicalRetriever
from perception.schema_context import resolve_schema_context
from tests.fixtures.mini_schema import MINI_TABLES

DSN = "postgresql://u:p@localhost:5432/db"


def _profile(threshold=6000):
    return build_profile(dsn=DSN, tables=MINI_TABLES,
                         grants={"admin": frozenset({ALL_TABLES})},
                         threshold_tokens=threshold)


ALL = frozenset(t.name for t in MINI_TABLES)


def test_full_mode_returns_every_permitted_table_regardless_of_question():
    ctx = resolve_schema_context(_profile(), "bất kỳ câu gì", permitted=ALL)
    assert set(ctx.retrieved_tables) == ALL


def test_full_mode_never_leaks_a_table_outside_permitted():
    ctx = resolve_schema_context(_profile(), "lương nhân viên",
                                 permitted=frozenset({"orders", "customers"}))
    assert set(ctx.retrieved_tables) == {"orders", "customers"}
    assert "payroll" not in ctx.rendered_text


def test_retrieval_mode_narrows_by_question():
    p = _profile(threshold=1)  # ép sang retrieval
    ctx = resolve_schema_context(p, "bảng lương nhân viên", permitted=ALL,
                                 retriever=LexicalRetriever(MINI_TABLES), k=1)
    assert "payroll" in ctx.retrieved_tables
    assert set(ctx.retrieved_tables) != ALL


def test_retrieval_mode_expands_along_foreign_keys():
    p = _profile(threshold=1)
    ctx = resolve_schema_context(p, "orders", permitted=ALL,
                                 retriever=LexicalRetriever(MINI_TABLES), k=1)
    # orders kéo theo customers và products qua FK
    assert {"orders", "customers", "products"} <= set(ctx.retrieved_tables)


def test_permitted_filter_is_applied_after_fk_expansion():
    """Mở rộng FK không được vượt qua hàng rào quyền."""
    p = _profile(threshold=1)
    ctx = resolve_schema_context(p, "orders", permitted=frozenset({"orders"}),
                                 retriever=LexicalRetriever(MINI_TABLES), k=1)
    assert set(ctx.retrieved_tables) == {"orders"}


def test_must_include_forces_a_table_in():
    p = _profile(threshold=1)
    ctx = resolve_schema_context(p, "zzzz không khớp gì", permitted=ALL,
                                 retriever=LexicalRetriever(MINI_TABLES), k=1,
                                 must_include=["payroll"])
    assert "payroll" in ctx.retrieved_tables


def test_must_include_cannot_bypass_permitted():
    """must_include là công cụ sửa lỗi retrieval, KHÔNG phải cửa hậu quyền."""
    p = _profile(threshold=1)
    ctx = resolve_schema_context(p, "gì đó", permitted=frozenset({"orders"}),
                                 retriever=LexicalRetriever(MINI_TABLES), k=1,
                                 must_include=["payroll"])
    assert "payroll" not in ctx.retrieved_tables


def test_retrieval_mode_without_a_retriever_is_an_error():
    p = _profile(threshold=1)
    with pytest.raises(ValueError, match="retriever"):
        resolve_schema_context(p, "gì đó", permitted=ALL)


def test_rendered_text_contains_only_retrieved_tables():
    p = _profile(threshold=1)
    ctx = resolve_schema_context(p, "bảng lương nhân viên", permitted=ALL,
                                 retriever=LexicalRetriever(MINI_TABLES), k=1)
    for name in ALL - set(ctx.retrieved_tables):
        assert f"CREATE TABLE {name} (" not in ctx.rendered_text
    for name in ctx.retrieved_tables:
        assert f"CREATE TABLE {name} (" in ctx.rendered_text


def test_empty_permitted_gives_empty_context():
    ctx = resolve_schema_context(_profile(), "gì đó", permitted=frozenset())
    assert ctx.retrieved_tables == ()
    assert ctx.rendered_text == ""


def test_table_order_is_stable_for_prefix_caching():
    p = _profile()
    a = resolve_schema_context(p, "câu một", permitted=ALL)
    b = resolve_schema_context(p, "câu hai khác hẳn", permitted=ALL)
    assert a.rendered_text == b.rendered_text
