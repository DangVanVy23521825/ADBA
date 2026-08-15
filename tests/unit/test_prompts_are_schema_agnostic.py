import json
import re
from pathlib import Path

import pytest

from perception.schema_model import tables_from_info_box

PROMPTS = Path("prompts")
INFO_BOX_ALL = Path("perception/info_box_all.json")

# Words that are real identifiers in the ADBA schema (perception/info_box_all.json)
# but also ordinary English/Vietnamese vocabulary or generic prompt-structure
# words that appear in the templates for reasons that have nothing to do with
# the ADBA schema. Excluded so the denylist test fails only on genuine
# schema-specific content, not on legitimate schema-agnostic prose.
_GENERIC_EXCLUSIONS = {
    "name",    # "never invent a table or column name" — refers to the concept
               # of a name, not to any table's `name` column.
    "role",    # the "## ROLE" section header in both templates, not employees.role.
    "level",   # "instruction level" (English idiom), not employees.level.
    "region",  # generic business/geography word used in supervisor_routing.txt's
               # worked examples — reviewed in task 9 and explicitly accepted as
               # out of scope (that prompt's job is agent sequencing, not SQL
               # against real columns).
    "amount",  # "SUM amount" in the same worked examples, same ruling as "region".
}

# Two-letter identifiers like "id" collide too easily with ordinary words
# even under a word-boundary regex (e.g. as a standalone word in prose).
_MIN_IDENTIFIER_LEN = 3


def _forbidden_identifiers() -> set[str]:
    """Every real table/column name in the ADBA schema, minus generic collisions.

    Derived from perception/info_box_all.json via
    perception.schema_model.tables_from_info_box instead of a hardcoded list,
    so it covers all nine tables' full column lists and self-updates if the
    schema fixture changes.
    """
    info_box = json.loads(INFO_BOX_ALL.read_text())
    tables = tables_from_info_box(info_box)
    names: set[str] = set()
    for table in tables:
        names.add(table.name)
        for column in table.columns:
            names.add(column.name)
    return {
        n.lower()
        for n in names
        if len(n) >= _MIN_IDENTIFIER_LEN and n.lower() not in _GENERIC_EXCLUSIONS
    }


FORBIDDEN_IDENTIFIERS = _forbidden_identifiers()


@pytest.mark.parametrize("name", ["text_to_sql.txt", "supervisor_routing.txt"])
def test_template_names_no_real_schema_identifier(name):
    """Tiêu chí 2 của spec: template không được mô tả một schema cụ thể.

    Checks against every table/column name in the real schema (see
    _forbidden_identifiers), not a fixed enumeration — so a future rule about
    a table/column outside any specific hand-picked list still gets caught.
    Matches on word boundaries so e.g. "orders" can't match inside a longer
    word and "id" can't match inside "consider".
    """
    text = (PROMPTS / name).read_text().lower()
    found = sorted(
        ident
        for ident in FORBIDDEN_IDENTIFIERS
        if re.search(r"\b" + re.escape(ident) + r"\b", text)
    )
    assert not found, f"{name} còn nhắc định danh có thật trong schema: {found}"


def test_forbidden_identifiers_has_teeth():
    """Guards against the derivation silently producing an empty (or tiny) set.

    A denylist test that passes because the denylist is empty is the failure
    mode this guards against.
    """
    assert len(FORBIDDEN_IDENTIFIERS) > 20
    assert "orders" in FORBIDDEN_IDENTIFIERS  # table name
    assert "order_date" in FORBIDDEN_IDENTIFIERS  # multi-word column name


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
