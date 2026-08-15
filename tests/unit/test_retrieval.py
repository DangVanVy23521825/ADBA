from perception.retrieval import FullRetriever, LexicalRetriever, expand_by_foreign_keys, _tokens
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


def test_tokenizer_handles_vietnamese_diacritics():
    """Tokenizer must produce whole words, not fragments on diacritics.

    bảng lương nhân viên should tokenize to {bang, luong, nhan, vien},
    not noise fragments like {b, ng, n, l, nh, vi}.
    """
    result = _tokens("bảng lương nhân viên")
    assert result == {"bang", "luong", "nhan", "vien"}


def test_lexical_payroll_scores_higher_than_customers_on_payroll_query():
    """Payroll description should score strictly higher than customers on
    payroll-related queries. This test catches when false positives occur
    due to coincidental noise-token collisions in other tables.

    Without proper Vietnamese tokenization, both descriptions would match
    on short fragments (e.g., 'n', 'ng', 'nh') and produce false positives.
    """
    r = LexicalRetriever(MINI_TABLES)
    results = r.search("bảng lương nhân viên", k=4)
    # payroll must come before customers (higher relevance)
    payroll_idx = results.index("payroll")
    customers_idx = results.index("customers")
    assert payroll_idx < customers_idx, (
        f"payroll (index {payroll_idx}) should rank before customers (index {customers_idx})"
    )
