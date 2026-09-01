"""Role chỉ-đọc ở tầng Postgres — lớp bảo đảm duy nhất (spec 4.1 lớp 1).

Test này là test DUY NHẤT của Plan B chạm database thật. Nó skip khi
không có ADBA_READONLY_URL, nên bộ test vẫn chạy được trên máy không có
container.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import psycopg2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

RO_DSN = os.environ.get("ADBA_READONLY_URL")
pytestmark = pytest.mark.skipif(
    not RO_DSN, reason="cần ADBA_READONLY_URL sau khi chạy scripts/create_readonly_role.sql"
)


def _run(sql: str):
    conn = psycopg2.connect(RO_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            return cur.fetchall() if cur.description else None
    finally:
        conn.rollback()
        conn.close()


def test_select_works():
    assert _run("SELECT 1")[0][0] == 1


def test_insert_is_refused_by_the_database():
    with pytest.raises(psycopg2.Error):
        _run("INSERT INTO orders (id) VALUES (-1)")


def test_update_is_refused_by_the_database():
    with pytest.raises(psycopg2.Error):
        _run("UPDATE orders SET id = id")


def test_delete_is_refused_by_the_database():
    with pytest.raises(psycopg2.Error):
        _run("DELETE FROM orders")


def test_data_modifying_cte_is_refused_by_the_database():
    """Câu đi qua được heuristic 'first token ∈ {SELECT, WITH}' của bản cũ.

    Guard sqlparse đã chặn nó ở tầng ứng dụng. Test này kiểm điều khác:
    KỂ CẢ khi guard đó thủng, database vẫn từ chối.
    """
    with pytest.raises(psycopg2.Error):
        _run("WITH gone AS (DELETE FROM orders RETURNING *) SELECT count(*) FROM gone")


def test_create_table_is_refused_by_the_database():
    with pytest.raises(psycopg2.Error):
        _run("CREATE TABLE adba_probe_should_not_exist (id int)")


def test_the_transaction_is_read_only_by_default():
    assert _run("SHOW default_transaction_read_only")[0][0] == "on"
