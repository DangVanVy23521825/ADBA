"""
SQL execution tools — psycopg2 + pandas wrappers.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import pandas as pd
import psycopg2
import psycopg2.extras
import psycopg2.sql

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    os.getenv("POSTGRES_URL", "postgresql://adba_user:adba@localhost:5432/adba_db"),
)
SQL_TIMEOUT_MS = int(os.getenv("SQL_TIMEOUT_MS", "30000"))

# Whitelist of allowed table names (prevents injection via f-string alternative)
_TABLE_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
_ALLOWED_TABLES = frozenset({
    "customers", "products", "orders",
    "warehouses", "stock", "stock_movements",
    "departments", "employees", "payroll",
})


def get_db_connection():
    """Return a new psycopg2 connection from DATABASE_URL."""
    return psycopg2.connect(DATABASE_URL)


def execute_sql(
    sql: str,
    params: tuple | None = None,
    timeout_ms: int = SQL_TIMEOUT_MS,
) -> pd.DataFrame:
    """Execute a SELECT query and return results as a pandas DataFrame.

    Uses parameterized queries when `params` is provided.
    """
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SET LOCAL statement_timeout = %s", (timeout_ms,))
            cur.execute(sql, params)
            rows = cur.fetchall()
            if not rows:
                columns = [desc[0] for desc in cur.description] if cur.description else []
                return pd.DataFrame(columns=columns)
            return pd.DataFrame(rows)
    finally:
        conn.close()


def get_table_sample(table: str, limit: int = 5) -> pd.DataFrame:
    """Return a sample of rows from `table`. Table name is validated against
    a whitelist of known schema tables."""
    if table not in _ALLOWED_TABLES or not _TABLE_NAME_RE.match(table):
        raise ValueError(f"Table name '{table}' is not in the allowed whitelist.")
    if not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    query = psycopg2.sql.SQL("SELECT * FROM {} LIMIT %s").format(
        psycopg2.sql.Identifier(table)
    )
    return execute_sql(query, (limit,))


def explain_query_plan(sql: str) -> dict[str, Any]:
    """Run EXPLAIN on a SQL query and return the plan as a dict."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("EXPLAIN " + sql)
            rows = cur.fetchall()
            plan_lines = [row[0] for row in rows]
            return {
                "plan": plan_lines,
                "total_cost_estimate": _extract_cost(plan_lines),
            }
    except Exception as exc:
        return {"error": str(exc), "plan": []}
    finally:
        conn.close()


def _extract_cost(lines: list[str]) -> float | None:
    """Attempt to extract the top-level cost from EXPLAIN output."""
    for line in lines:
        if "cost=" in line:
            cost_part = line.split("cost=")[1].split("..")[1].split(" ")[0]
            try:
                return float(cost_part)
            except ValueError:
                return None
    return None
