"""
ADBA — Perception Layer
=======================
Connects to PostgreSQL and extracts a structured JSON info_box for each domain.

The info_box is the ONLY database context that agents receive.
It must be:
  - Compact enough to fit in a 4096-token context window alongside
    the user query, execution plan, and action history
  - Rich enough for the model to write correct SQL without guessing
    column names, types, FK relationships, or enum values

Output files:
  perception/info_box_sales.json
  perception/info_box_inventory.json
  perception/info_box_hr.json
  perception/info_box_all.json      ← merged, used for cross-domain queries
"""

from __future__ import annotations

import json
import os
import textwrap
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://adba_user:adba@localhost:5432/adba_db"
)

OUTPUT_DIR = Path("perception")

# ── Domain → table mapping ────────────────────────────────────────────────────
# Order matters: tables listed first are referenced by later tables (FK order)
DOMAIN_TABLES: dict[str, list[str]] = {
    "sales":     ["customers", "products", "orders"],
    "inventory": ["warehouses", "stock", "stock_movements"],
    "hr":        ["departments", "employees", "payroll"],
}

SAMPLE_ROWS_COUNT = 3

# ── Enum / categorical values known from schema CHECK constraints ─────────────
# Hardcoded because information_schema does not expose CHECK constraint enums.
# Agents use these to write correct WHERE / GROUP BY filters.
KNOWN_ENUMS: dict[str, dict[str, list[str]]] = {
    "customers": {
        "region":  ["Miền Bắc", "Miền Trung", "Miền Nam", "Tây Nguyên"],
        "segment": ["Enterprise", "SME", "Retail", "Government"],
    },
    "products": {
        "category": ["Electronics", "Office Supplies", "Furniture",
                     "Industrial", "Food & Beverage", "Apparel"],
    },
    "orders": {
        "region":         ["Miền Bắc", "Miền Trung", "Miền Nam", "Tây Nguyên"],
        "status":         ["pending", "processing", "completed", "cancelled", "refunded"],
        "payment_method": ["bank_transfer", "cash", "credit_card", "e_wallet"],
    },
    "warehouses": {
        "region": ["Miền Bắc", "Miền Trung", "Miền Nam", "Tây Nguyên"],
    },
    "stock_movements": {
        "movement_type": ["inbound", "outbound", "transfer",
                          "adjustment", "return", "write_off"],
    },
    "departments": {
        "region": ["Miền Bắc", "Miền Trung", "Miền Nam", "Tây Nguyên", "HQ"],
    },
    "employees": {
        "level":  ["Junior", "Mid", "Senior", "Lead", "Manager", "Director", "C-Level"],
        "status": ["active", "on_leave", "terminated", "contractor"],
    },
}

# ── Generated / computed columns that agents must NOT insert into ─────────────
GENERATED_COLUMNS: dict[str, list[str]] = {
    "orders": ["quarter", "year"],   # GENERATED ALWAYS AS (EXTRACT(...)) STORED
}

# ── Cross-domain FK hints not captured by information_schema ─────────────────
# information_schema only shows FKs within the same schema.
# These soft links are critical for cross-domain JOIN reasoning.
CROSS_DOMAIN_HINTS: list[dict] = [
    {
        "from_table":  "stock",
        "from_column": "product_id",
        "to_table":    "products",
        "to_column":   "id",
        "note":        "Cross-domain FK: inventory.stock → sales.products",
    },
    {
        "from_table":  "stock_movements",
        "from_column": "product_id",
        "to_table":    "products",
        "to_column":   "id",
        "note":        "Cross-domain FK: inventory.stock_movements → sales.products",
    },
    {
        "from_table":  "stock_movements",
        "from_column": "reference_order_id",
        "to_table":    "orders",
        "to_column":   "id",
        "note":        "Soft link (not enforced): links outbound movement to its source order",
    },
    {
        "from_table":  "departments",
        "from_column": "region",
        "to_table":    "orders",
        "to_column":   "region",
        "note":        "Soft link via region string: workforce location ↔ sales territory",
    },
    {
        "from_table":  "warehouses",
        "from_column": "region",
        "to_table":    "orders",
        "to_column":   "region",
        "note":        "Soft link via region string: warehouse location ↔ sales territory",
    },
]

# ── Important query patterns for agents ──────────────────────────────────────
QUERY_NOTES: dict[str, list[str]] = {
    "orders": [
        "IMPORTANT: orders has NO 'month' column. Use EXTRACT(MONTH FROM order_date) for month-level filtering.",
        "quarter and year are GENERATED ALWAYS columns — do NOT include them in INSERT statements.",
        "Use WHERE status IN ('completed', 'processing') for revenue rollups to exclude cancelled/refunded.",
        "Partial index idx_orders_completed is used automatically when filtering completed/processing status.",
    ],
    "stock": [
        "UNIQUE constraint on (product_id, warehouse_id) — one row per product per warehouse.",
        "Use WHERE quantity < min_threshold to find stockout-risk items (partial index idx_stock_low).",
        "last_updated is auto-refreshed by trigger trg_stock_touch on every UPDATE.",
    ],
    "stock_movements": [
        "from_warehouse_id IS NULL → inbound (purchase/production).",
        "to_warehouse_id IS NULL → outbound (sale/write-off).",
        "Both non-NULL → internal transfer between warehouses.",
        "reference_order_id links outbound movements back to their source order in the orders table.",
    ],
    "payroll": [
        "payroll HAS a 'month' column (SMALLINT 1–12) — use p.month directly.",
        "net_salary = base_salary + bonus - deduction (enforced by CHECK constraint).",
        "UNIQUE on (employee_id, month, year) — one row per employee per month.",
    ],
    "employees": [
        "Use WHERE status = 'active' for current headcount (partial index idx_employees_active).",
        "end_date IS NULL for currently active employees.",
        "departments.headcount is auto-synced by trigger trg_headcount_sync.",
    ],
    "departments": [
        "manager_id is a self-referential FK to employees.id (DEFERRABLE INITIALLY DEFERRED).",
        "budget is monthly budget in VND. Compare with SUM(payroll.net_salary) for budget analysis.",
    ],
}


# =============================================================================
# HELPERS
# =============================================================================

def _json_serializer(obj):
    """Handle non-JSON-serializable types from psycopg2."""
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, memoryview):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _get_columns(cur, table: str) -> list[dict]:
    """
    Query information_schema for column metadata.
    Returns list of dicts with: name, data_type, is_nullable,
    column_default, is_generated, ordinal_position.
    """
    cur.execute(
        """
        SELECT
            column_name,
            data_type,
            is_nullable,
            column_default,
            is_identity,
            -- is_generated is 'ALWAYS' for GENERATED ALWAYS columns
            CASE WHEN generation_expression IS NOT NULL
                 THEN 'ALWAYS' ELSE 'NEVER' END AS is_generated,
            generation_expression,
            ordinal_position
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name   = %s
        ORDER BY ordinal_position
        """,
        (table,),
    )
    rows = cur.fetchall()
    columns = []
    for r in rows:
        col: dict = {
            "name":        r[0],
            "data_type":   r[1],
            "nullable":    r[2] == "YES",
            "is_generated": r[5] == "ALWAYS",
        }
        # Include default only if meaningful (exclude sequences)
        if r[3] and "nextval" not in str(r[3]):
            col["default"] = r[3]
        if r[5] == "ALWAYS" and r[6]:
            col["generated_as"] = r[6].strip()
        # Add known enum values
        enums = KNOWN_ENUMS.get(table, {}).get(r[0])
        if enums:
            col["enum_values"] = enums
        columns.append(col)
    return columns


def _get_primary_key(cur, table: str) -> list[str]:
    cur.execute(
        """
        SELECT kcu.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.table_schema    = kcu.table_schema
        WHERE tc.constraint_type = 'PRIMARY KEY'
          AND tc.table_schema    = 'public'
          AND tc.table_name      = %s
        ORDER BY kcu.ordinal_position
        """,
        (table,),
    )
    return [r[0] for r in cur.fetchall()]


def _get_foreign_keys(cur, table: str) -> list[dict]:
    """Returns FK relationships defined in information_schema (same-schema only)."""
    cur.execute(
        """
        SELECT
            kcu.column_name,
            ccu.table_name  AS referenced_table,
            ccu.column_name AS referenced_column
        FROM information_schema.table_constraints AS tc
        JOIN information_schema.key_column_usage AS kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.table_schema    = kcu.table_schema
        JOIN information_schema.constraint_column_usage AS ccu
          ON ccu.constraint_name = tc.constraint_name
         AND ccu.table_schema    = tc.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY'
          AND tc.table_schema    = 'public'
          AND tc.table_name      = %s
        """,
        (table,),
    )
    return [
        {"column": r[0], "references": f"{r[1]}.{r[2]}"}
        for r in cur.fetchall()
    ]


def _get_row_count(cur, table: str) -> int:
    cur.execute(f"SELECT COUNT(*) FROM {table}")   # noqa: S608 — table name is internal
    return cur.fetchone()[0]


def _get_sample_rows(cur, table: str, n: int = SAMPLE_ROWS_COUNT) -> list[dict]:
    """
    Fetch n representative rows. Uses ORDER BY random() with a fixed seed
    for reproducibility, then picks rows that show diverse values.
    """
    cur.execute(
        f"SELECT * FROM {table} ORDER BY random() LIMIT %s",  # noqa: S608
        (n,),
    )
    colnames = [desc[0] for desc in cur.description]
    return [dict(zip(colnames, row)) for row in cur.fetchall()]


def _get_indexes(cur, table: str) -> list[dict]:
    """Return index names and whether they are partial (have a WHERE clause)."""
    cur.execute(
        """
        SELECT
            indexname,
            indexdef
        FROM pg_indexes
        WHERE schemaname = 'public'
          AND tablename  = %s
          AND indexname NOT LIKE %s
        ORDER BY indexname
        """,
        (table, "%_pkey"),
    )
    result = []
    for row in cur.fetchall():
        if len(row) < 2:
            continue
        name, defn = row[0], row[1]
        entry: dict = {"name": name}
        if defn and "WHERE" in str(defn).upper():
            where_part = str(defn)[str(defn).upper().index("WHERE"):]
            entry["partial_where"] = where_part
        result.append(entry)
    return result


def _build_table_info(cur, table: str) -> dict:
    """Assemble all metadata for a single table into a structured dict."""
    info: dict = {
        "table_name":    table,
        "row_count":     _get_row_count(cur, table),
        "primary_key":   _get_primary_key(cur, table),
        "columns":       _get_columns(cur, table),
        "foreign_keys":  _get_foreign_keys(cur, table),
        "indexes":       _get_indexes(cur, table),
        "sample_rows":   _get_sample_rows(cur, table, SAMPLE_ROWS_COUNT),
    }

    # Add generated columns note
    gen_cols = GENERATED_COLUMNS.get(table, [])
    if gen_cols:
        info["generated_columns_note"] = (
            f"Columns {gen_cols} are GENERATED ALWAYS — never include in INSERT."
        )

    # Add query notes
    notes = QUERY_NOTES.get(table, [])
    if notes:
        info["query_notes"] = notes

    return info


def _get_cross_domain_hints(tables_in_domain: list[str]) -> list[dict]:
    """
    Return cross-domain hints relevant to the current domain's tables.

    Include both directions so agents can reason from either side of a relation:
      - from_table in domain tables
      - to_table in domain tables
    """
    table_set = set(tables_in_domain)
    filtered = [
        h for h in CROSS_DOMAIN_HINTS
        if h["from_table"] in table_set or h["to_table"] in table_set
    ]
    # Stable order for deterministic output and easier diffs.
    return sorted(
        filtered,
        key=lambda h: (h["from_table"], h["from_column"], h["to_table"], h["to_column"]),
    )


# =============================================================================
# MAIN EXTRACTION
# =============================================================================

def extract_info_boxes() -> dict[str, dict]:
    """
    Connect to PostgreSQL and extract info_box for each domain.
    Returns dict: {domain_name: info_box_dict}
    """
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True

    # Use RealDictCursor for sample rows (returns dicts directly)
    cur = conn.cursor()

    all_info_boxes: dict[str, dict] = {}

    try:
        for domain, tables in DOMAIN_TABLES.items():
            print(f"\n[{domain.upper()}] Extracting info_box …")
            tables_info = []

            for table in tables:
                print(f"  → {table}", end=" ")
                tinfo = _build_table_info(cur, table)
                tables_info.append(tinfo)
                print(f"({tinfo['row_count']:,} rows, {len(tinfo['columns'])} columns)")

            info_box = {
                "domain":              domain,
                "tables":              tables_info,
                "cross_domain_hints":  _get_cross_domain_hints(tables),
                "extraction_note": (
                    "This info_box contains schema metadata and sample rows. "
                    "Use it to write correct SQL — column names, types, enum values, "
                    "FK relationships, and query notes are all included."
                ),
            }
            all_info_boxes[domain] = info_box

    finally:
        cur.close()
        conn.close()

    return all_info_boxes


def save_info_boxes(info_boxes: dict[str, dict]) -> None:
    """Write individual domain files and a merged all-domains file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for domain, info_box in info_boxes.items():
        path = OUTPUT_DIR / f"info_box_{domain}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(info_box, f, ensure_ascii=False, indent=2,
                      default=_json_serializer)
        size_kb = path.stat().st_size / 1024
        print(f"  ✓ {path}  ({size_kb:.1f} KB)")

    # Merged file for cross-domain queries
    merged = {
        "domain": "all",
        "tables": [
            t
            for ib in info_boxes.values()
            for t in ib["tables"]
        ],
        "cross_domain_hints": CROSS_DOMAIN_HINTS,
        "extraction_note": (
            "This merged info_box contains all 9 tables across Sales, Inventory, "
            "and HR domains. Use for cross-domain queries that JOIN across domains."
        ),
    }
    merged_path = OUTPUT_DIR / "info_box_all.json"
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2,
                  default=_json_serializer)
    size_kb = merged_path.stat().st_size / 1024
    print(f"  ✓ {merged_path}  ({size_kb:.1f} KB)  ← merged all domains")


def print_token_estimates(info_boxes: dict[str, dict]) -> None:
    """
    Rough token count estimate (1 token ≈ 4 chars).
    Critical: info_box must fit alongside query + plan + history in 4096 ctx.
    Budget: aim for < 1800 tokens per domain info_box.
    """
    print("\nToken estimates (rough: chars / 4):")
    for domain, ib in info_boxes.items():
        serialized = json.dumps(ib, ensure_ascii=False, default=_json_serializer)
        tokens = len(serialized) // 4
        status = "✓ OK" if tokens < 1800 else "⚠ LARGE — consider reducing sample_rows"
        print(f"  {domain:12s}: ~{tokens:,} tokens  {status}")

    all_path = OUTPUT_DIR / "info_box_all.json"
    if all_path.exists():
        size = all_path.stat().st_size
        tokens = size // 4
        status = "✓ OK" if tokens < 4000 else "⚠ Too large for single context"
        print(f"  {'all':12s}: ~{tokens:,} tokens  {status}")


if __name__ == "__main__":
    print("ADBA Perception Layer — extract_info_box.py")
    print("=" * 50)
    print(f"Database: {DATABASE_URL.split('@')[-1]}")

    info_boxes = extract_info_boxes()

    print("\nSaving info_box files …")
    save_info_boxes(info_boxes)

    print_token_estimates(info_boxes)

    print("\nDone. Files written to:", OUTPUT_DIR.resolve())
    print("\nQuick check — tables per domain:")
    for domain, ib in info_boxes.items():
        tables = [t["table_name"] for t in ib["tables"]]
        print(f"  {domain}: {tables}")