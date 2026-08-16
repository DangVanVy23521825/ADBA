from pathlib import Path

import pytest

from eval.datasets import EvalRecord, load_normalized
from eval.tier1_recall import _resolver, measure_recall
from tests.fixtures.mini_schema import MINI_TABLES

MINI_DATASET = Path(__file__).resolve().parents[1] / "fixtures" / "mini_dataset"


def _rec(question: str, sql: str) -> EvalRecord:
    return EvalRecord(question=question, gold_sql=sql, db_id="mini", tables=MINI_TABLES)


def test_perfect_retriever_scores_one():
    records = [_rec("q", "SELECT * FROM orders JOIN customers ON 1=1")]
    report = measure_recall(records, lambda r: frozenset({"orders", "customers"}))
    assert report.recall == 1.0
    assert report.full_hits == 1


def test_missing_one_table_counts_as_a_full_miss():
    """Không có điểm từng phần: thiếu một bảng JOIN là SQL sai chắc chắn."""
    records = [_rec("q", "SELECT * FROM orders JOIN customers ON 1=1")]
    report = measure_recall(records, lambda r: frozenset({"orders"}))
    assert report.recall == 0.0
    assert report.full_hits == 0


def test_extra_tables_do_not_hurt_recall():
    records = [_rec("q", "SELECT * FROM orders")]
    report = measure_recall(records, lambda r: frozenset({"orders", "payroll", "products"}))
    assert report.recall == 1.0


def test_avg_context_tables_is_reported_so_precision_is_visible():
    records = [_rec("q", "SELECT * FROM orders")]
    report = measure_recall(records, lambda r: frozenset({"orders", "payroll", "products"}))
    assert report.avg_context_tables == 3.0


def test_misses_record_which_tables_were_absent():
    records = [_rec("câu hỏi X", "SELECT * FROM orders JOIN payroll ON 1=1")]
    report = measure_recall(records, lambda r: frozenset({"orders"}))
    assert report.misses == [("câu hỏi X", frozenset({"payroll"}))]


def test_records_whose_gold_sql_has_no_tables_are_skipped():
    records = [_rec("q", "not sql"), _rec("q2", "SELECT * FROM orders")]
    report = measure_recall(records, lambda r: frozenset({"orders"}))
    assert report.total == 1
    assert report.recall == 1.0


def test_empty_record_list_gives_zero_without_dividing_by_zero():
    report = measure_recall([], lambda r: frozenset())
    assert report.total == 0
    assert report.recall == 0.0


class TestResolver:
    """IMPORTANT 6 (final review): _resolver had zero test coverage — only
    measure_recall itself was exercised, via hand-written lambdas. The
    threshold_tokens = 10**9 if full else 1 trick, retriever selection, and
    the `permitted = all tables` line were exercised by nothing, even though
    every recall number this project produces comes out of this function.

    Numbers below were measured by actually running _resolver over
    tests/fixtures/mini_dataset/ (3 loadable records: db_a has 2 tables/2
    questions, db_b has 1 table/1 question; the db_khong_ton_tai row is
    dropped by load_normalized because that db_id has no schema entry) —
    not guessed.
    """

    def _records(self):
        return load_normalized(
            MINI_DATASET / "questions.jsonl", MINI_DATASET / "schemas.json"
        )

    def test_full_strategy_gets_perfect_recall(self):
        """threshold_tokens=10**9 forces schema_mode='full', so
        resolve_schema_context returns every permitted table regardless of
        the question — FullRetriever.search is never even consulted for
        scoring. Measured: recall=1.0, avg_context_tables=5/3≈1.667 (db_a's
        two questions each see 2 tables, db_b's one question sees 1 table:
        (2+2+1)/3).
        """
        report = measure_recall(self._records(), _resolver("full", 8))
        assert report.recall == 1.0
        assert report.avg_context_tables == pytest.approx(5 / 3)

    def test_lexical_strategy_scores_zero_on_this_content_free_fixture(self):
        """threshold_tokens=1 forces schema_mode='retrieval', routing through
        LexicalRetriever's token-overlap scoring instead of FullRetriever.

        Measured: recall=0.0, avg_context_tables=0.0 — LexicalRetriever
        returns an empty list for every question here. This fixture's
        questions ("hỏi một", "hỏi hai", "hỏi ba" — placeholder Vietnamese
        for "question one/two/three") share no tokens with the table/column
        names ("t1", "t2", "u1", "id"), so every score is 0 and nothing
        clears LexicalRetriever's `score > 0` bar. This is the real,
        expected behavior of the lexical baseline on THIS plumbing fixture —
        it is not tuned to test retrieval quality (that's eval/tier1_recall
        run against tests/fixtures/adba_golden.jsonl or a real dataset) — but
        it is exactly the kind of "number right for an accidental reason"
        this plan already flagged once: without this test, a change that
        broke _resolver's retriever selection (e.g. always using
        FullRetriever) would silently report a perfect 1.0 here too.
        """
        report = measure_recall(self._records(), _resolver("lexical", 8))
        assert report.recall == 0.0
        assert report.avg_context_tables == 0.0
