from eval.datasets import EvalRecord
from eval.tier1_recall import measure_recall
from tests.fixtures.mini_schema import MINI_TABLES


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
