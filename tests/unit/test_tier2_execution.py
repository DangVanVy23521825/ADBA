"""Test eval tầng 2.

Phần lớn nằm ở `compare_results` và `has_top_level_order_by` — chúng thuần
tuý, nên cả sáu quyết định ngữ nghĩa của tài liệu thiết kế khoá được ở đây
mà không cần database lẫn model. `measure_execution` test bằng callable giả.
"""

from datetime import date
from decimal import Decimal

import pytest

from eval.tier2_execution import (
    ExecReport,
    Failure,
    QueryError,
    QueryResult,
    QueryTimeout,
    Verdict,
    compare_ignoring_column_order,
    compare_results,
    has_top_level_order_by,
    measure_execution,
)
from perception.schema_model import Column, Table


# --------------------------------------------------------------------------
# D1 — thứ tự dòng
# --------------------------------------------------------------------------


def test_row_order_is_ignored_when_gold_has_no_order_by():
    """Không có ORDER BY thì Postgres không cam kết thứ tự, và thứ tự có thể
    đổi giữa hai lượt chạy. So có thứ tự ở đây sẽ đẻ ra trượt giả."""
    gold = [("a", 1), ("b", 2)]
    pred = [("b", 2), ("a", 1)]
    assert compare_results(gold, pred, ordered=False) is Verdict.MATCH


def test_row_order_decides_the_verdict_when_gold_asked_for_a_ranking():
    gold = [("a", 1), ("b", 2)]
    pred = [("b", 2), ("a", 1)]
    assert compare_results(gold, pred, ordered=True) is Verdict.MISMATCH


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT x FROM t ORDER BY x",
        "select x from t order by x desc",
        "SELECT x FROM t WHERE y IN (SELECT z FROM u) ORDER BY x",
    ],
)
def test_top_level_order_by_is_detected(sql):
    assert has_top_level_order_by(sql) is True


@pytest.mark.parametrize(
    "sql",
    [
        # ORDER BY trong subquery: kết quả NGOÀI CÙNG vẫn không có thứ tự.
        "SELECT x FROM (SELECT x FROM t ORDER BY x) s",
        "SELECT x FROM t WHERE y IN (SELECT z FROM u ORDER BY z)",
        # ORDER BY của window function: sắp trong phạm vi cửa sổ, không sắp
        # kết quả trả về.
        "SELECT row_number() OVER (ORDER BY x) FROM t",
        "SELECT x FROM t",
    ],
)
def test_order_by_that_does_not_order_the_result_is_not_detected(sql):
    """Đây là lý do phải duyệt token thay vì khớp chuỗi.

    `"order by" in sql.lower()` nhận nhầm cả bốn câu này, và nhận nhầm theo
    hướng nguy hiểm: bật chế độ so chặt cho những câu vốn không cam kết thứ
    tự, biến nhiễu thành trượt."""
    assert has_top_level_order_by(sql) is False


def test_naive_string_matching_would_have_failed_these():
    """Khoá lại chính điều trên: nếu ai đó thay bằng khớp chuỗi, test này đỏ."""
    sql = "SELECT x FROM (SELECT x FROM t ORDER BY x) s"
    assert "order by" in sql.lower()
    assert has_top_level_order_by(sql) is False


# --------------------------------------------------------------------------
# D3 — kiểu số
# --------------------------------------------------------------------------


def test_decimal_and_float_of_the_same_value_match():
    """Postgres trả Decimal cho numeric và float cho double precision, nên
    cùng một phép SUM ra hai kiểu khác nhau tuỳ đường SQL."""
    assert compare_results(
        [(Decimal("1234.00"),)], [(1234.0,)], ordered=False
    ) is Verdict.MATCH


def test_float_noise_below_the_rounding_threshold_matches():
    assert compare_results(
        [(0.1 + 0.2,)], [(0.3,)], ordered=False
    ) is Verdict.MATCH


def test_a_real_numeric_difference_still_mismatches():
    assert compare_results(
        [(Decimal("1234.00"),)], [(1234.5,)], ordered=False
    ) is Verdict.MISMATCH


def test_big_integers_are_not_pushed_through_float():
    """Số nguyên lớn hơn 2^53 mất chính xác khi qua float, mà khoá chính
    bigint thì hay nằm ở vùng đó. Hai id khác nhau không được khớp."""
    a = 9007199254740993      # 2^53 + 1
    b = 9007199254740992      # 2^53   — float(a) == float(b)
    assert float(a) == float(b)
    assert compare_results([(a,)], [(b,)], ordered=False) is Verdict.MISMATCH


def test_booleans_do_not_match_the_integers_they_would_cast_to():
    """`bool` là lớp con của `int` trong Python. Chuẩn hoá bất cẩn sẽ biến
    True/False thành 1/0 và làm một cột boolean khớp nhầm một cột đếm."""
    assert compare_results([(True,), (False,)], [(1,), (0,)], ordered=True) is Verdict.MISMATCH


# --------------------------------------------------------------------------
# D4 — dòng trùng
# --------------------------------------------------------------------------


def test_duplicate_rows_are_counted_not_collapsed():
    """CỐ Ý khác bộ chấm chính thức của BIRD: họ dùng `set`, và điều đó tha
    nhầm những câu thiếu GROUP BY."""
    gold = [("a",), ("a",), ("a",)]
    pred = [("a",)]
    assert compare_results(gold, pred, ordered=False) is Verdict.MISMATCH


def test_the_same_multiset_in_a_different_order_still_matches():
    gold = [("a",), ("a",), ("b",)]
    pred = [("b",), ("a",), ("a",)]
    assert compare_results(gold, pred, ordered=False) is Verdict.MATCH


# --------------------------------------------------------------------------
# D5 — kết quả rỗng
# --------------------------------------------------------------------------


def test_both_empty_is_its_own_verdict_not_a_match():
    """Về kỹ thuật rỗng khớp rỗng, nhưng SQL sai cũng rất hay trả rỗng. Gộp
    vào tỉ lệ đạt là tự thổi phồng."""
    assert compare_results([], [], ordered=False) is Verdict.BOTH_EMPTY


@pytest.mark.parametrize("ordered", [True, False])
def test_one_side_empty_is_a_mismatch(ordered):
    assert compare_results([("a",)], [], ordered=ordered) is Verdict.MISMATCH
    assert compare_results([], [("a",)], ordered=ordered) is Verdict.MISMATCH


# --------------------------------------------------------------------------
# Kiểu hỗn hợp — NULL trộn số trộn chuỗi
# --------------------------------------------------------------------------


def test_mixed_types_including_null_do_not_raise_when_sorting():
    """So trực tiếp `None < 1` ném TypeError trong Python 3, mà một tập kết
    quả bất kỳ đều có thể trộn NULL với số với chuỗi."""
    gold = [(None, "a"), (1, None), ("z", 2)]
    pred = [("z", 2), (None, "a"), (1, None)]
    assert compare_results(gold, pred, ordered=False) is Verdict.MATCH


def test_null_does_not_match_a_missing_row():
    assert compare_results([(None,)], [], ordered=False) is Verdict.MISMATCH


def test_dates_compare_by_value():
    assert compare_results(
        [(date(2026, 1, 1),)], [(date(2026, 1, 1),)], ordered=False
    ) is Verdict.MATCH


# --------------------------------------------------------------------------
# D2 — thứ tự cột (số chẩn đoán)
# --------------------------------------------------------------------------


def test_column_permutation_is_a_mismatch_for_the_headline_number():
    """So chặt theo vị trí, giống BIRD, để số của ta so sánh được với số đã
    công bố."""
    assert compare_results([("a", 1)], [(1, "a")], ordered=False) is Verdict.MISMATCH


def test_column_permutation_is_a_match_for_the_diagnostic_number():
    assert compare_ignoring_column_order(
        [("a", 1)], [(1, "a")], ordered=False
    ) is Verdict.MATCH


def test_the_diagnostic_number_still_rejects_genuinely_different_data():
    assert compare_ignoring_column_order(
        [("a", 1)], [("b", 2)], ordered=False
    ) is Verdict.MISMATCH


# --------------------------------------------------------------------------
# measure_execution — phân loại bucket
# --------------------------------------------------------------------------

_T = (Table(name="t", columns=(Column("x", "integer"),)),)


class _Rec:
    """EvalRecord tối giản — chỉ những trường mà measure_execution chạm tới."""

    def __init__(self, question: str, gold_sql: str):
        self.question = question
        self.gold_sql = gold_sql
        self.db_id = "test"
        self.tables = _T


def _fixed(mapping):
    """execute giả: tra SQL trong dict; giá trị là QueryResult hoặc Exception."""

    def execute(sql):
        out = mapping[sql]
        if isinstance(out, Exception):
            raise out
        return out

    return execute


def test_a_broken_golden_record_is_removed_from_the_denominator():
    """Cùng lối với measure_recall, nơi bản ghi có SQL vàng không parse ra
    bảng nào bị bỏ qua chứ không tính là trượt: lỗi dữ liệu không phải lỗi
    hệ thống."""
    recs = [_Rec("q", "GOLD_BAD")]
    report = measure_execution(
        recs,
        generate=lambda r: "PRED",
        execute=_fixed({"GOLD_BAD": QueryError("relation does not exist")}),
    )
    assert report.total == 1
    assert report.gold_error == 1
    assert report.scored == 0
    assert report.exec_accuracy == 0.0


def test_the_model_is_not_called_when_the_golden_sql_is_broken():
    """Lượt gọi model là toàn bộ chi phí của tầng này (1.534 câu BIRD mất
    vài giờ mỗi nhánh), nên bản ghi vàng hỏng không được tốn lượt nào."""
    calls = []

    def generate(rec):
        calls.append(rec.question)
        return "PRED"

    measure_execution(
        [_Rec("q", "GOLD_BAD")],
        generate=generate,
        execute=_fixed({"GOLD_BAD": QueryError("boom")}),
    )
    assert calls == []


def test_timeout_is_bucketed_apart_from_a_syntax_error():
    """Quá hạn giờ thường nghĩa là thiếu điều kiện nối — lỗi sửa được, tín
    hiệu khác hẳn sai cú pháp."""
    report = measure_execution(
        [_Rec("q", "GOLD")],
        generate=lambda r: "PRED",
        execute=_fixed({
            "GOLD": QueryResult(rows=(("a",),)),
            "PRED": QueryTimeout("canceling statement"),
        }),
    )
    assert (report.pred_timeout, report.pred_error) == (1, 0)


def test_a_syntax_error_lands_in_pred_error():
    report = measure_execution(
        [_Rec("q", "GOLD")],
        generate=lambda r: "PRED",
        execute=_fixed({
            "GOLD": QueryResult(rows=(("a",),)),
            "PRED": QueryError("syntax error at or near"),
        }),
    )
    assert (report.pred_error, report.pred_timeout) == (1, 0)


def test_a_model_failure_does_not_kill_the_whole_run():
    """Một lượt chạy dài hàng giờ không được đổ vì một lượt gọi model hỏng."""
    def generate(rec):
        raise RuntimeError("ollama connection reset")

    report = measure_execution(
        [_Rec("q", "GOLD")],
        generate=generate,
        execute=_fixed({"GOLD": QueryResult(rows=(("a",),))}),
    )
    assert report.pred_error == 1
    assert report.total == 1


def test_truncation_on_either_side_counts_as_neither_pass_nor_fail():
    """Khi một bên bị cắt cụt thì phép so không còn nói lên điều gì về tính
    đúng, nên không được tính đạt mà cũng không được tính trượt."""
    report = measure_execution(
        [_Rec("q", "GOLD")],
        generate=lambda r: "PRED",
        execute=_fixed({
            "GOLD": QueryResult(rows=(("a",),), truncated=True),
            "PRED": QueryResult(rows=(("a",),)),
        }),
    )
    assert report.truncated == 1
    assert (report.match, report.mismatch) == (0, 0)


def test_a_matching_pair_is_scored_and_counted():
    report = measure_execution(
        [_Rec("q", "GOLD")],
        generate=lambda r: "PRED",
        execute=_fixed({
            "GOLD": QueryResult(rows=(("a", 1),)),
            "PRED": QueryResult(rows=(("a", 1),)),
        }),
    )
    assert (report.match, report.scored) == (1, 1)
    assert report.exec_accuracy == 1.0


def test_both_empty_is_scored_but_not_counted_as_a_match():
    report = measure_execution(
        [_Rec("q", "GOLD")],
        generate=lambda r: "PRED",
        execute=_fixed({
            "GOLD": QueryResult(rows=()),
            "PRED": QueryResult(rows=()),
        }),
    )
    assert report.both_empty == 1
    assert report.match == 0
    assert report.scored == 1
    assert report.exec_accuracy == 0.0


def test_order_sensitivity_is_taken_from_the_golden_sql():
    """measure_execution phải TỰ suy ra `ordered` từ SQL vàng; nơi gọi không
    truyền vào."""
    ranked = measure_execution(
        [_Rec("q", "SELECT x FROM t ORDER BY x")],
        generate=lambda r: "PRED",
        execute=_fixed({
            "SELECT x FROM t ORDER BY x": QueryResult(rows=((1,), (2,))),
            "PRED": QueryResult(rows=((2,), (1,))),
        }),
    )
    assert ranked.mismatch == 1

    unranked = measure_execution(
        [_Rec("q", "SELECT x FROM t")],
        generate=lambda r: "PRED",
        execute=_fixed({
            "SELECT x FROM t": QueryResult(rows=((1,), (2,))),
            "PRED": QueryResult(rows=((2,), (1,))),
        }),
    )
    assert unranked.match == 1


# --------------------------------------------------------------------------
# Báo cáo
# --------------------------------------------------------------------------


def test_report_never_hides_a_broken_golden_set():
    """gold_error cao nghĩa là golden set hỏng và mọi số phía trên đáng ngờ,
    nên nó phải hiện ra chứ không chỉ lặng lẽ rời khỏi mẫu số."""
    text = ExecReport(total=10, match=4, gold_error=5).as_text()
    assert "5" in text
    assert "golden set hỏng" in text


def test_report_shows_that_both_empty_is_excluded_from_the_pass_rate():
    text = ExecReport(total=2, match=1, both_empty=1).as_text()
    assert "KHÔNG tính là khớp" in text


def test_exec_accuracy_is_zero_rather_than_dividing_by_zero():
    assert ExecReport(total=3, gold_error=3).exec_accuracy == 0.0


# --- Chi tiết thất bại phải sống sót ra khỏi lượt chạy ---
# Lượt 50 câu đầu tiên mất 34 phút rồi chỉ in 10 dòng đầu; 18 ca pred_error
# còn lại không được lưu ở đâu, và phải chạy lại mới chẩn được.


def test_a_failure_carries_the_sql_that_actually_ran():
    """"Cột không tồn tại" mà không kèm SQL thì không chỉ ra được cột nào."""
    report = measure_execution(
        [_Rec("q", "GOLD")],
        generate=lambda r: "SELECT MailStreet FROM schools",
        execute=_fixed({
            "GOLD": QueryResult(rows=(("a",),)),
            "SELECT MailStreet FROM schools": QueryError('column "mailstreet" does not exist'),
        }),
    )
    (f,) = report.failures
    assert f.bucket == "pred_error"
    assert "mailstreet" in f.detail
    assert f.sql == "SELECT MailStreet FROM schools"


def test_a_mismatch_also_carries_its_sql():
    """Khác kết quả là ca CẦN đọc SQL nhất: nó chạy được, nên sai nằm ở
    ngữ nghĩa và chỉ nhìn câu lệnh mới thấy."""
    report = measure_execution(
        [_Rec("q", "GOLD")],
        generate=lambda r: "PRED",
        execute=_fixed({
            "GOLD": QueryResult(rows=(("a",),)),
            "PRED": QueryResult(rows=(("b",),)),
        }),
    )
    assert report.failures[0].sql == "PRED"


def test_a_broken_golden_record_carries_its_own_sql():
    report = measure_execution(
        [_Rec("q", "GOLD_BAD")],
        generate=lambda r: "PRED",
        execute=_fixed({"GOLD_BAD": QueryError("boom")}),
    )
    assert report.failures[0].sql == "GOLD_BAD"


def test_a_generate_failure_does_not_crash_on_the_unbound_sql():
    """`pred_sql` được gán trước vòng try. Không gán thì nhánh bắt lỗi ném
    NameError và giết cả lượt chạy — đúng đường code lẽ ra để cứu nó."""
    def generate(rec):
        raise RuntimeError("ollama reset")

    report = measure_execution(
        [_Rec("q", "GOLD")],
        generate=generate,
        execute=_fixed({"GOLD": QueryResult(rows=(("a",),))}),
    )
    assert report.failures[0].bucket == "generate_error"
    assert report.failures[0].sql == ""


def test_the_text_report_still_truncates_for_the_terminal():
    """Cắt ở 10 là đúng cho màn hình; cái sai là để bản ĐẦY ĐỦ không có
    đường nào ra file."""
    report = ExecReport(total=20, mismatch=20,
                        failures=[Failure(f"q{i}", "mismatch") for i in range(20)])
    assert report.as_text().count("  - [") == 10
    assert len(report.failures) == 20
