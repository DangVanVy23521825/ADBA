from perception.retrieval import FullRetriever, LexicalRetriever, expand_by_foreign_keys
from tests.fixtures.mini_schema import MINI_TABLES


def test_full_retriever_returns_everything():
    r = FullRetriever(MINI_TABLES)
    assert set(r.search("bất kỳ câu gì", k=2)) == {t.name for t in MINI_TABLES}


def test_lexical_matches_on_table_name():
    r = LexicalRetriever(MINI_TABLES)
    assert "orders" in r.search("tổng doanh thu theo orders", k=2)


def test_lexical_matches_on_column_name():
    r = LexicalRetriever(MINI_TABLES)
    assert "payroll" in r.search("thống kê base_salary", k=2)


def test_lexical_matches_on_description():
    r = LexicalRetriever(MINI_TABLES)
    assert "payroll" in r.search("bảng lương nhân viên", k=1)


def test_lexical_respects_k():
    r = LexicalRetriever(MINI_TABLES)
    assert len(r.search("orders customers products payroll", k=2)) == 2


def test_lexical_returns_empty_on_no_overlap():
    r = LexicalRetriever(MINI_TABLES)
    assert r.search("zzzzz qqqqq", k=3) == []


def test_lexical_is_deterministic_on_ties():
    r = LexicalRetriever(MINI_TABLES)
    assert r.search("id", k=4) == r.search("id", k=4)


def test_fk_expansion_follows_outgoing_edges():
    # orders → customers, products
    assert expand_by_foreign_keys(["orders"], MINI_TABLES) == frozenset(
        {"orders", "customers", "products"}
    )


def test_fk_expansion_follows_incoming_edges():
    # customers ← orders
    assert "orders" in expand_by_foreign_keys(["customers"], MINI_TABLES)


def test_fk_expansion_is_one_hop_only():
    """customers → orders → products. products KHÔNG được kéo vào từ customers."""
    got = expand_by_foreign_keys(["customers"], MINI_TABLES)
    assert "products" not in got


def test_fk_expansion_of_isolated_table_returns_itself():
    assert expand_by_foreign_keys(["payroll"], MINI_TABLES) == frozenset({"payroll"})


def test_fk_expansion_ignores_unknown_names():
    assert expand_by_foreign_keys(["khong_co"], MINI_TABLES) == frozenset()
