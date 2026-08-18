from perception.annotations import Annotation, SchemaAnnotations
from perception.review_state import (
    apply_edit,
    filter_rows,
    review_progress,
    review_rows,
)
from perception.schema_model import Column, Table

TABLES = (
    Table(name="orders", columns=(Column("flg_tt", "boolean"), Column("id", "integer"))),
)
ANN = SchemaAnnotations(
    tables={"orders": Annotation("Đơn hàng", confidence="low")},
    columns={"orders": {"flg_tt": Annotation("Cờ gì đó", confidence="low")}},
)


def test_rows_cover_the_table_and_each_column():
    rows = review_rows(TABLES, ANN)
    keys = {(r.table, r.column) for r in rows}
    assert ("orders", None) in keys
    assert ("orders", "flg_tt") in keys
    assert ("orders", "id") in keys, "cột chưa có chú giải vẫn phải hiện ra để sửa"


def test_rows_carry_the_current_text_and_state():
    row = next(r for r in review_rows(TABLES, ANN) if r.column == "flg_tt")
    assert row.current_text == "Cờ gì đó"
    assert row.confidence == "low"
    assert row.reviewed_by == "llm"


def test_applying_an_edit_marks_it_reviewed_by_human():
    after = apply_edit(ANN, "orders", "flg_tt", "Cờ đã thanh toán")
    entry = after.columns["orders"]["flg_tt"]
    assert entry.text == "Cờ đã thanh toán"
    assert entry.reviewed_by == "human"


def test_applying_an_edit_to_a_table_description_works_too():
    after = apply_edit(ANN, "orders", None, "Đơn bán hàng")
    assert after.tables["orders"].reviewed_by == "human"


def test_editing_a_column_that_had_no_annotation_creates_one():
    after = apply_edit(ANN, "orders", "id", "Khoá chính")
    assert after.columns["orders"]["id"].text == "Khoá chính"


def test_an_edit_never_touches_other_entries():
    after = apply_edit(ANN, "orders", "flg_tt", "mới")
    assert after.tables["orders"].text == "Đơn hàng"


def test_progress_counts_human_reviewed_over_total():
    done, total = review_progress(ANN)
    assert (done, total) == (0, 2)
    after = apply_edit(ANN, "orders", None, "x")
    assert review_progress(after)[0] == 1


def test_progress_with_tables_counts_every_reviewable_item():
    # ANN has 2 annotated entries (table + flg_tt column) but TABLES also
    # has an "id" column with no annotation at all. review_rows would show
    # 3 items total; review_progress(ann, tables=TABLES) must match that
    # denominator, not just the 2 that already have an annotation.
    done, total = review_progress(ANN, tables=TABLES)
    assert (done, total) == (0, 3)
    after = apply_edit(ANN, "orders", None, "x")
    done, total = review_progress(after, tables=TABLES)
    assert (done, total) == (1, 3)


def test_progress_without_tables_only_counts_already_annotated_entries():
    # The "id" column has no annotation, so the one-argument form must not
    # see it at all -- it only measures within entries that already exist
    # in `ann`.
    done, total = review_progress(ANN)
    assert total == 2


# ── bộ lọc của trang duyệt ──────────────────────────────────────────────────

_CONFIDENT = SchemaAnnotations(
    tables={"orders": Annotation("Đơn hàng", confidence="high")},
    columns={"orders": {"flg_tt": Annotation("Cờ đã thanh toán", confidence="high")}},
)


def test_filter_hides_confident_llm_entries_but_reports_how_many():
    """Rủi ro lớn nhất của plan: chú giải SAI mà model tự nhận là chắc chắn.

    Bộ lọc mặc định giấu chúng đi. Nếu con số không được trả về thì analyst
    không có cách nào biết tập đó tồn tại.
    """
    rows = review_rows(TABLES, _CONFIDENT)
    visible, hidden = filter_rows(rows, only_pending=True)

    assert hidden == 2, "một mục bảng + một mục cột, model tự tin cả hai"
    assert all(r.confidence == "low" for r in visible)
    assert {(r.table, r.column) for r in visible} == {("orders", "id")}


def test_filter_off_shows_everything_and_hides_nothing():
    rows = review_rows(TABLES, _CONFIDENT)
    visible, hidden = filter_rows(rows, only_pending=False)
    assert visible == list(rows)
    assert hidden == 0


def test_entries_a_human_already_reviewed_are_not_counted_as_hidden():
    """Mục người đã duyệt bị lọc ra là đúng, nhưng nó KHÔNG phải rủi ro.

    Gộp chúng vào con số "đang ẩn" sẽ thổi phồng cảnh báo, và một cảnh báo
    thổi phồng dạy người ta bỏ qua nó.
    """
    reviewed = SchemaAnnotations(
        tables={"orders": Annotation("Đơn hàng", reviewed_by="human")},
        columns={
            "orders": {"flg_tt": Annotation("Cờ đã thanh toán", reviewed_by="human")}
        },
    )
    rows = review_rows(TABLES, reviewed)
    _visible, hidden = filter_rows(rows, only_pending=True)
    assert hidden == 0
