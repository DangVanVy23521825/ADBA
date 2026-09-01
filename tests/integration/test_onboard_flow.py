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

    # `total` một mình KHÔNG khẳng định chuỗi chạy thành công: nó chỉ đếm
    # số dòng golden đọc được, nên một report `total=1, passed=False,
    # recall=0.0` vẫn qua. Đây là test duy nhất chạy suốt extract → build
    # → verify, nên nó phải khẳng định KẾT QUẢ, không phải khẳng định là
    # có chạy.
    assert report.total == 1
    assert report.misses == [], f"retriever trượt bảng: {report.misses}"
    assert report.recall == 1.0, f"recall {report.recall}, kỳ vọng 1.0"
    assert report.passed
    assert report.avg_context_tables >= 1


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
