import json
import re
from pathlib import Path

import pytest

from perception.schema_model import tables_from_info_box

PROMPTS = Path("prompts")
INFO_BOX_ALL = Path("perception/info_box_all.json")

# Words that are real identifiers in the ADBA schema (perception/info_box_all.json)
# but are also ordinary English words or generic prompt-structure collisions in
# BOTH templates — safe to exclude everywhere.
_UNIVERSAL_EXCLUSIONS = {
    "id",      # too short/collision-prone even under a word-boundary regex
               # (dropped by _MIN_IDENTIFIER_LEN too; listed here for clarity).
    "name",    # "never invent a table or column name" — refers to the concept
               # of a name, not to any table's `name` column.
    "role",    # the "## ROLE" section header in both templates, not employees.role.
    "level",   # "instruction level" (English idiom), not employees.level.
}

# Round-1 fix (task 9 review, round 1) made `region`/`amount` global exclusions.
# Round-2 finding: that silently blinded text_to_sql.txt — the file this test
# exists to scrub — to the exact two identifiers most likely to come back,
# since they are literally the content of the removed rules ("group by
# region", "sum the amount column"). A simulated reintroduction of a
# region/amount rule into text_to_sql.txt was NOT caught under the global
# exclusion. Fix: exclude region/amount ONLY for supervisor_routing.txt,
# where they occur solely inside that file's worked examples (reviewed and
# accepted as out of scope — that prompt's job is agent sequencing, not SQL
# against real columns). DELETE this exclusion once those examples are
# genericized; do not widen it back to a global exclusion.
_SUPERVISOR_ONLY_EXCLUSIONS = {
    "region",  # generic business/geography word in supervisor_routing.txt's
               # worked examples only — see comment above.
    "amount",  # "SUM amount" in the same worked examples — see comment above.
}

# Two-letter identifiers like "id" collide too easily with ordinary words
# even under a word-boundary regex (e.g. as a standalone word in prose).
_MIN_IDENTIFIER_LEN = 3

_EXCLUSIONS_BY_FILE = {
    "text_to_sql.txt": _UNIVERSAL_EXCLUSIONS,
    "supervisor_routing.txt": _UNIVERSAL_EXCLUSIONS | _SUPERVISOR_ONLY_EXCLUSIONS,
}


def _forbidden_identifiers(exclusions: set[str]) -> set[str]:
    """Every real table/column name in the ADBA schema, minus the given exclusions.

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
        if len(n) >= _MIN_IDENTIFIER_LEN and n.lower() not in exclusions
    }


FORBIDDEN_IDENTIFIERS_BY_FILE = {
    name: _forbidden_identifiers(exclusions)
    for name, exclusions in _EXCLUSIONS_BY_FILE.items()
}


@pytest.mark.parametrize("name", ["text_to_sql.txt", "supervisor_routing.txt"])
def test_template_names_no_real_schema_identifier(name):
    """Tiêu chí 2 của spec: template không được mô tả một schema cụ thể.

    Checks against every table/column name in the real schema (see
    _forbidden_identifiers), not a fixed enumeration — so a future rule about
    a table/column outside any specific hand-picked list still gets caught.
    Exclusions are per file (see _EXCLUSIONS_BY_FILE) so a collision word
    that's legitimate in one template can't blind the check in the other.
    Matches on word boundaries so e.g. "orders" can't match inside a longer
    word and "id" can't match inside "consider".
    """
    text = (PROMPTS / name).read_text().lower()
    forbidden = FORBIDDEN_IDENTIFIERS_BY_FILE[name]
    found = sorted(
        ident
        for ident in forbidden
        if re.search(r"\b" + re.escape(ident) + r"\b", text)
    )
    assert not found, f"{name} còn nhắc định danh có thật trong schema: {found}"


def test_forbidden_identifiers_has_teeth():
    """Guards against the derivation silently producing an empty (or tiny) set.

    A denylist test that passes because the denylist is empty is the failure
    mode this guards against.
    """
    for name in ("text_to_sql.txt", "supervisor_routing.txt"):
        forbidden = FORBIDDEN_IDENTIFIERS_BY_FILE[name]
        assert len(forbidden) > 20
        assert "orders" in forbidden  # table name
        assert "order_date" in forbidden  # multi-word column name


def test_region_and_amount_stay_forbidden_for_text_to_sql():
    """Regression test for task 9 review round 2.

    Round 1 excluded `region`/`amount` globally to accommodate
    supervisor_routing.txt's worked examples, which silently blinded
    text_to_sql.txt — the file this test suite exists to scrub — to the two
    identifiers most likely to come back, since they are literally the
    content of the schema-specific rules that were removed from it
    ("group by region", "sum the amount column"). This test pins the
    exclusion to supervisor_routing.txt only, so a future edit can't quietly
    widen it back to global without this test catching it.
    """
    assert "region" in FORBIDDEN_IDENTIFIERS_BY_FILE["text_to_sql.txt"]
    assert "amount" in FORBIDDEN_IDENTIFIERS_BY_FILE["text_to_sql.txt"]


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
