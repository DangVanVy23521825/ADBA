from pathlib import Path

import pytest

from perception.annotations import (
    Annotation,
    SchemaAnnotations,
    apply_annotations,
    load_annotations,
    merge_annotations,
    pending_review,
    save_annotations,
)
from perception.schema_model import Column, Table


def _ann(text, reviewed_by="llm", confidence="high") -> Annotation:
    return Annotation(text=text, reviewed_by=reviewed_by, confidence=confidence)


def _sample() -> SchemaAnnotations:
    return SchemaAnnotations(
        tables={"orders": _ann("Đơn hàng", reviewed_by="human")},
        columns={"orders": {"flg_tt": _ann("Cờ thanh toán", confidence="low")}},
    )


# ── vòng đời file ───────────────────────────────────────────────────────────

def test_save_then_load_round_trips(tmp_path: Path):
    path = tmp_path / "schema.yaml"
    save_annotations(_sample(), path)
    back = load_annotations(path)
    assert back.tables["orders"].text == "Đơn hàng"
    assert back.tables["orders"].reviewed_by == "human"
    assert back.columns["orders"]["flg_tt"].confidence == "low"


def test_loading_a_missing_file_gives_empty_annotations(tmp_path: Path):
    empty = load_annotations(tmp_path / "khong-co.yaml")
    assert empty.tables == {}
    assert empty.columns == {}


def test_saved_file_is_human_readable_yaml(tmp_path: Path):
    path = tmp_path / "schema.yaml"
    save_annotations(_sample(), path)
    text = path.read_text(encoding="utf-8")
    assert "orders" in text and "Đơn hàng" in text


# ── merge: phần quan trọng nhất ─────────────────────────────────────────────

def test_human_edits_survive_a_refresh():
    existing = SchemaAnnotations(
        tables={"orders": _ann("Mô tả do NGƯỜI sửa", reviewed_by="human")}, columns={}
    )
    fresh = SchemaAnnotations(
        tables={"orders": _ann("Mô tả do LLM sinh lại", reviewed_by="llm")}, columns={}
    )
    merged = merge_annotations(existing, fresh)
    assert merged.tables["orders"].text == "Mô tả do NGƯỜI sửa"
    assert merged.tables["orders"].reviewed_by == "human"


def test_llm_entries_are_replaced_by_a_refresh():
    existing = SchemaAnnotations(tables={"orders": _ann("Bản cũ")}, columns={})
    fresh = SchemaAnnotations(tables={"orders": _ann("Bản mới")}, columns={})
    assert merge_annotations(existing, fresh).tables["orders"].text == "Bản mới"


def test_human_column_edits_survive_too():
    existing = SchemaAnnotations(
        tables={}, columns={"orders": {"flg_tt": _ann("Người sửa", reviewed_by="human")}}
    )
    fresh = SchemaAnnotations(
        tables={}, columns={"orders": {"flg_tt": _ann("LLM sinh lại")}}
    )
    merged = merge_annotations(existing, fresh)
    assert merged.columns["orders"]["flg_tt"].text == "Người sửa"


def test_new_tables_from_a_refresh_are_added():
    existing = SchemaAnnotations(tables={"orders": _ann("A")}, columns={})
    fresh = SchemaAnnotations(tables={"orders": _ann("B"), "invoices": _ann("C")}, columns={})
    merged = merge_annotations(existing, fresh)
    assert merged.tables["invoices"].text == "C"


def test_a_table_dropped_from_the_schema_is_dropped_from_annotations():
    """Giữ lại chú giải của bảng không còn tồn tại chỉ làm nhiễu bản duyệt."""
    existing = SchemaAnnotations(
        tables={"orders": _ann("A", reviewed_by="human"), "cu": _ann("B", reviewed_by="human")},
        columns={},
    )
    fresh = SchemaAnnotations(tables={"orders": _ann("A2")}, columns={})
    merged = merge_annotations(existing, fresh)
    assert "cu" not in merged.tables


def test_no_human_entry_is_ever_lost_across_a_full_refresh():
    """Ràng buộc của spec, phát biểu trực tiếp: 100% mục human sống sót."""
    existing = SchemaAnnotations(
        tables={f"t{i}": _ann(f"người {i}", reviewed_by="human") for i in range(20)},
        columns={f"t{i}": {"c": _ann(f"cột {i}", reviewed_by="human")} for i in range(20)},
    )
    fresh = SchemaAnnotations(
        tables={f"t{i}": _ann("llm ghi đè") for i in range(20)},
        columns={f"t{i}": {"c": _ann("llm ghi đè")} for i in range(20)},
    )
    merged = merge_annotations(existing, fresh)
    assert all(merged.tables[f"t{i}"].text == f"người {i}" for i in range(20))
    assert all(merged.columns[f"t{i}"]["c"].text == f"cột {i}" for i in range(20))


def test_human_column_annotation_survives_when_fresh_has_no_columns_for_that_table():
    """Model trả "columns": {} là chuyện thường; nó không được xoá công sức người."""
    existing = SchemaAnnotations(
        tables={"orders": _ann("bảng")},
        columns={"orders": {"flg_tt": _ann("Người sửa", reviewed_by="human")}},
    )
    fresh = SchemaAnnotations(tables={"orders": _ann("bảng mới")}, columns={})
    merged = merge_annotations(existing, fresh)
    assert merged.columns["orders"]["flg_tt"].text == "Người sửa"
    assert merged.columns["orders"]["flg_tt"].reviewed_by == "human"


def test_human_column_annotation_survives_when_fresh_omits_that_one_column():
    """SYSTEM_PROMPT bảo model bỏ qua cột hiển nhiên như id — nên nó sẽ bỏ qua."""
    existing = SchemaAnnotations(
        tables={"orders": _ann("bảng")},
        columns={"orders": {
            "id": _ann("Khoá chính do người viết", reviewed_by="human"),
            "flg_tt": _ann("cũ"),
        }},
    )
    fresh = SchemaAnnotations(
        tables={"orders": _ann("bảng mới")},
        columns={"orders": {"flg_tt": _ann("mới")}},
    )
    merged = merge_annotations(existing, fresh)
    assert merged.columns["orders"]["id"].text == "Khoá chính do người viết"
    assert merged.columns["orders"]["flg_tt"].text == "mới", "mục LLM vẫn phải được thay"


def test_llm_column_annotation_absent_from_fresh_is_dropped():
    """Chỉ mục người mới được giữ. Mục LLM cũ mà lần này model không nhắc thì bỏ.

    Sửa so với bản gốc trong brief: bản gốc chỉ có cột LLM trong bảng, nên
    khi bảng vắng mặt khỏi `fresh.columns`, "cu" không nằm trong kết quả dù
    code cũ (lỗi) hay code mới (đã sửa) — vì code cũ CHƯA BAO GIỜ tự thêm
    một khoá không có trong `fresh`, nó chỉ lỗi ở việc XOÁ những khoá đáng
    lẽ phải giữ. Assertion gốc xanh ngay cả trên code lỗi, không kiểm được
    gì. Thêm một cột `human` cùng bảng để test đỏ đúng lý do (mục người bị
    xoá theo cả bảng) và xanh đúng lý do (mục người sống, mục LLM cũ vẫn bị
    bỏ) sau khi sửa.
    """
    existing = SchemaAnnotations(
        tables={"orders": _ann("bảng")},
        columns={
            "orders": {
                "giu": _ann("người viết", reviewed_by="human"),
                "cu": _ann("LLM cũ"),
            }
        },
    )
    fresh = SchemaAnnotations(tables={"orders": _ann("bảng mới")}, columns={})
    merged = merge_annotations(existing, fresh)
    assert merged.columns["orders"]["giu"].text == "người viết"
    assert "cu" not in merged.columns["orders"]


def test_human_column_annotation_dies_with_its_table():
    """Bảng bị xoá khỏi schema thì chú giải cột của nó cũng đi theo."""
    existing = SchemaAnnotations(
        tables={"cu": _ann("bảng cũ")},
        columns={"cu": {"c": _ann("người viết", reviewed_by="human")}},
    )
    fresh = SchemaAnnotations(tables={"orders": _ann("còn sống")}, columns={})
    merged = merge_annotations(existing, fresh)
    assert "cu" not in merged.columns


# ── gắn vào Table ───────────────────────────────────────────────────────────

def test_apply_puts_descriptions_onto_tables_and_columns():
    tables = (Table(name="orders", columns=(Column("flg_tt", "boolean"),)),)
    out = apply_annotations(tables, _sample())
    assert out[0].description == "Đơn hàng"
    assert out[0].columns[0].description == "Cờ thanh toán"


def test_apply_leaves_unannotated_entries_empty():
    tables = (Table(name="khac", columns=(Column("x", "integer"),)),)
    out = apply_annotations(tables, _sample())
    assert out[0].description == ""
    assert out[0].columns[0].description == ""


# ── hàng đợi duyệt ──────────────────────────────────────────────────────────

def test_pending_review_lists_low_confidence_and_unreviewed():
    ann = SchemaAnnotations(
        tables={
            "a": _ann("ok", reviewed_by="human"),
            "b": _ann("chưa chắc", confidence="low"),
        },
        columns={"a": {"c1": _ann("chưa chắc", confidence="low")}},
    )
    pending = pending_review(ann)
    assert ("b", None) in pending
    assert ("a", "c1") in pending
    assert ("a", None) not in pending, "mục người đã duyệt không nằm trong hàng đợi"


# ── Amendment 1: SchemaAnnotations phải deep-freeze mapping của nó ──────────

def test_mutating_tables_mapping_raises_type_error():
    ann = _sample()
    with pytest.raises(TypeError):
        ann.tables["orders"] = _ann("bị sửa trộm")


def test_mutating_inner_columns_mapping_raises_type_error():
    ann = _sample()
    with pytest.raises(TypeError):
        ann.columns["orders"]["flg_tt"] = _ann("bị sửa trộm")


# ── Amendment 2: load_annotations phải từ chối giá trị lạ ──────────────────

def test_load_rejects_bad_reviewed_by_on_table_annotation(tmp_path: Path):
    path = tmp_path / "schema.yaml"
    path.write_text(
        "tables:\n"
        "  orders:\n"
        "    text: Đơn hàng\n"
        "    reviewed_by: Human\n"
        "    confidence: high\n"
        "columns: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as excinfo:
        load_annotations(path)
    message = str(excinfo.value)
    assert "Human" in message
    assert "orders" in message


# ── C1: ghi schema.yaml nguyên tử, giữ một bản .bak ─────────────────────────

def test_first_write_creates_no_backup(tmp_path: Path):
    """Lần ghi đầu tiên không có gì để sao lưu — không được tự bịa ra một
    file .bak thừa (phá test đếm số file trong profile/)."""
    path = tmp_path / "schema.yaml"
    save_annotations(_sample(), path)
    assert path.exists()
    assert not (tmp_path / "schema.yaml.bak").exists()


def test_second_write_backs_up_the_previous_content(tmp_path: Path):
    path = tmp_path / "schema.yaml"
    first = SchemaAnnotations(tables={"orders": _ann("bản đầu", reviewed_by="human")})
    save_annotations(first, path)
    first_content = path.read_text(encoding="utf-8")

    second = SchemaAnnotations(tables={"orders": _ann("bản hai", reviewed_by="human")})
    save_annotations(second, path)

    backup_path = tmp_path / "schema.yaml.bak"
    assert backup_path.exists()
    assert backup_path.read_text(encoding="utf-8") == first_content
    assert load_annotations(path).tables["orders"].text == "bản hai"


def test_backup_is_the_single_previous_version_not_a_history(tmp_path: Path):
    """Chỉ MỘT bản .bak — ghi đè lần lượt, không tích luỹ theo số thứ tự."""
    path = tmp_path / "schema.yaml"
    for i in range(3):
        save_annotations(
            SchemaAnnotations(tables={"orders": _ann(f"bản {i}", reviewed_by="human")}), path
        )
    backup_path = tmp_path / "schema.yaml.bak"
    assert load_annotations(backup_path).tables["orders"].text == "bản 1"
    assert load_annotations(path).tables["orders"].text == "bản 2"
    assert not (tmp_path / "schema.yaml.bak.bak").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["schema.yaml", "schema.yaml.bak"]


def test_write_does_not_leave_a_stray_tmp_file_behind(tmp_path: Path):
    path = tmp_path / "schema.yaml"
    save_annotations(_sample(), path)
    save_annotations(_sample(), path)
    names = {p.name for p in tmp_path.iterdir()}
    assert names == {"schema.yaml", "schema.yaml.bak"}


def test_load_rejects_bad_confidence_on_column_annotation(tmp_path: Path):
    path = tmp_path / "schema.yaml"
    path.write_text(
        "tables: {}\n"
        "columns:\n"
        "  orders:\n"
        "    flg_tt:\n"
        "      text: Cờ thanh toán\n"
        "      reviewed_by: llm\n"
        "      confidence: medium\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as excinfo:
        load_annotations(path)
    message = str(excinfo.value)
    assert "medium" in message
    assert "orders" in message
