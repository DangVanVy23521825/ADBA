import dataclasses
import json

from onboard import cmd_extract, cmd_refresh
from perception.annotations import Annotation, SchemaAnnotations, load_annotations, save_annotations
from perception.schema_model import Column
from tests.fixtures.mini_schema import MINI_TABLES

REPLY = '{"table": {"text": "llm mới", "confidence": "high"}, "columns": {}}'


def test_refresh_keeps_every_human_entry(tmp_path, monkeypatch):
    monkeypatch.setattr("onboard.introspect_schema", lambda dsn, **kw: MINI_TABLES)  # noqa: ARG005
    monkeypatch.setattr("onboard.sample_rows", lambda dsn, table, n=5: [])  # noqa: ARG005
    cmd_extract("postgresql://x", tmp_path)
    save_annotations(
        SchemaAnnotations(
            tables={t.name: Annotation(f"người {t.name}", reviewed_by="human")
                    for t in MINI_TABLES}
        ),
        tmp_path / "schema.yaml",
    )

    cmd_refresh(tmp_path, "postgresql://x", invoke=lambda s, u: REPLY)  # noqa: ARG005

    ann = load_annotations(tmp_path / "schema.yaml")
    assert all(ann.tables[t.name].text == f"người {t.name}" for t in MINI_TABLES)


def test_refresh_picks_up_a_new_table(tmp_path, monkeypatch):
    grown = MINI_TABLES + (
        dataclasses.replace(MINI_TABLES[0], name="invoices"),
    )
    monkeypatch.setattr("onboard.introspect_schema", lambda dsn, **kw: grown)  # noqa: ARG005
    monkeypatch.setattr("onboard.sample_rows", lambda dsn, table, n=5: [])  # noqa: ARG005
    cmd_extract("postgresql://x", tmp_path)

    cmd_refresh(tmp_path, "postgresql://x", invoke=lambda s, u: REPLY)  # noqa: ARG005
    assert "invoices" in load_annotations(tmp_path / "schema.yaml").tables


def test_refresh_updates_the_structure_file(tmp_path, monkeypatch):
    monkeypatch.setattr("onboard.introspect_schema", lambda dsn, **kw: MINI_TABLES)  # noqa: ARG005
    monkeypatch.setattr("onboard.sample_rows", lambda dsn, table, n=5: [])  # noqa: ARG005
    cmd_extract("postgresql://x", tmp_path)

    grown = list(MINI_TABLES)
    grown[0] = dataclasses.replace(
        grown[0], columns=grown[0].columns + (Column("cot_moi", "integer"),)
    )
    monkeypatch.setattr("onboard.introspect_schema", lambda dsn, **kw: tuple(grown))  # noqa: ARG005
    cmd_refresh(tmp_path, "postgresql://x", invoke=lambda s, u: REPLY)  # noqa: ARG005

    raw = json.loads((tmp_path / "structure.json").read_text(encoding="utf-8"))
    cols = {c["name"] for t in raw if t["name"] == grown[0].name for c in t["columns"]}
    assert "cot_moi" in cols


def test_refresh_reports_how_many_human_entries_it_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr("onboard.introspect_schema", lambda dsn, **kw: MINI_TABLES)  # noqa: ARG005
    monkeypatch.setattr("onboard.sample_rows", lambda dsn, table, n=5: [])  # noqa: ARG005
    cmd_extract("postgresql://x", tmp_path)
    save_annotations(
        SchemaAnnotations(tables={"orders": Annotation("người", reviewed_by="human")}),
        tmp_path / "schema.yaml",
    )
    _, preserved = cmd_refresh(tmp_path, "postgresql://x", invoke=lambda s, u: REPLY)  # noqa: ARG005
    assert preserved == 1


def test_refresh_reports_dropped_human_entry_when_table_disappears(tmp_path, monkeypatch, capsys):
    """A table dropped from the schema legitimately drops its human annotation.

    Amendment 1: `preserved` must be counted from the merged result (entries
    actually present after merge), not from `before`. A refresh that removes
    a table must NOT claim it preserved an annotation that no longer exists
    after the merge — that would overstate what survived. The dropped count
    must be reported separately, and only when nonzero.
    """
    monkeypatch.setattr("onboard.introspect_schema", lambda dsn, **kw: MINI_TABLES)  # noqa: ARG005
    monkeypatch.setattr("onboard.sample_rows", lambda dsn, table, n=5: [])  # noqa: ARG005
    cmd_extract("postgresql://x", tmp_path)
    save_annotations(
        SchemaAnnotations(
            tables={
                "orders": Annotation("người orders", reviewed_by="human"),
                "payroll": Annotation("người payroll", reviewed_by="human"),
            }
        ),
        tmp_path / "schema.yaml",
    )

    shrunk = tuple(t for t in MINI_TABLES if t.name != "payroll")
    monkeypatch.setattr("onboard.introspect_schema", lambda dsn, **kw: shrunk)  # noqa: ARG005

    capsys.readouterr()
    _, preserved = cmd_refresh(tmp_path, "postgresql://x", invoke=lambda s, u: REPLY)  # noqa: ARG005
    out = capsys.readouterr().out

    assert preserved == 1
    ann = load_annotations(tmp_path / "schema.yaml")
    assert "orders" in ann.tables
    assert "payroll" not in ann.tables
    assert "Giữ nguyên 1 mục do người duyệt." in out
    assert "1 mục do người duyệt bị loại" in out
