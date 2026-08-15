from pathlib import Path

import pytest

PROMPTS = Path("prompts")
ADBA_TABLES = ["orders", "products", "customers", "payroll", "employees",
               "departments", "stock", "stock_movements", "warehouses"]
ADBA_COLUMNS = ["order_date", "base_salary", "min_threshold", "sku", "segment"]


@pytest.mark.parametrize("name", ["text_to_sql.txt", "supervisor_routing.txt"])
def test_template_names_no_adba_table(name):
    """Tiêu chí 2 của spec: template không được mô tả một schema cụ thể."""
    text = (PROMPTS / name).read_text().lower()
    found = [t for t in ADBA_TABLES if t in text]
    assert not found, f"{name} còn nhắc bảng ADBA: {found}"


@pytest.mark.parametrize("name", ["text_to_sql.txt", "supervisor_routing.txt"])
def test_template_names_no_adba_column(name):
    text = (PROMPTS / name).read_text().lower()
    found = [c for c in ADBA_COLUMNS if c in text]
    assert not found, f"{name} còn nhắc cột ADBA: {found}"


def test_text_to_sql_has_the_new_placeholders():
    text = (PROMPTS / "text_to_sql.txt").read_text()
    assert "{schema}" in text
    assert "{few_shots}" in text
    assert "{task}" in text
    assert "{info_box}" not in text


def test_schema_placeholder_sits_after_the_static_rules():
    """Phần cố định phải đứng trước phần biến thiên thì prefix cache mới dùng được.

    Xem spec mục 3.5.
    """
    text = (PROMPTS / "text_to_sql.txt").read_text()
    assert text.index("## RULES") < text.index("{schema}")
    assert text.index("{schema}") < text.index("{task}")


def test_supervisor_schema_placeholder_is_near_the_end():
    text = (PROMPTS / "supervisor_routing.txt").read_text()
    assert "{schema}" in text
    assert "{info_box}" not in text
    assert text.index("{schema}") > len(text) * 0.6
