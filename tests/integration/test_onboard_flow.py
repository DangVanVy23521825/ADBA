import json
import os

import pytest

from onboard import cmd_build, cmd_extract, cmd_verify
from perception.connection_profile import ALL_TABLES, permitted_tables
from perception.profile_store import profile_is_stale, read_profile

DSN = os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")
pytestmark = pytest.mark.skipif(not DSN, reason="cần DATABASE_URL trỏ Postgres thật")


def test_extract_build_verify_runs_end_to_end(tmp_path):
    cmd_extract(DSN, tmp_path)
    cmd_build(tmp_path, DSN, grants={"admin": frozenset({ALL_TABLES})})

    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        json.dumps({"question": "doanh thu theo region",
                    "sql": "SELECT region, SUM(amount) FROM orders GROUP BY region"},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    report = cmd_verify(tmp_path, golden, user="admin")
    assert report.total == 1


def test_the_built_profile_matches_the_live_schema(tmp_path):
    from perception.introspect import introspect_schema

    cmd_extract(DSN, tmp_path)
    cmd_build(tmp_path, DSN, grants={})
    assert profile_is_stale(tmp_path, introspect_schema(DSN)) is False


def test_a_profile_loaded_from_disk_carries_grants(tmp_path):
    cmd_extract(DSN, tmp_path)
    cmd_build(tmp_path, DSN, grants={"sales": frozenset({"orders"})})
    profile = read_profile(tmp_path)
    assert permitted_tables(profile, "sales") == frozenset({"orders"})
