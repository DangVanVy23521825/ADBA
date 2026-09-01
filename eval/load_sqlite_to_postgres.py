"""Load a SQLite database into a Postgres database's `public` schema.

Why this exists: ADBA's onboarding pipeline (`onboard.py`) reads Postgres —
`perception/introspect.py` queries `pg_class`/`information_schema` and is
Postgres-only. Proving the pipeline works on a schema nobody designed it
for means running it against BIRD, a benchmark that ships SQLite. This
script is the bridge: it reads a `.sqlite` file's structure and data with
the stdlib `sqlite3` module and recreates it faithfully in Postgres.

"Faithfully" means three things a naive port drops:

1. Reserved words. BIRD's `financial` database has a table literally named
   `order`. Every identifier this module emits — table names, column
   names — goes through `psycopg2.sql.Identifier`, never through string
   interpolation, so reserved words and any exotic naming survive.
2. Scale. `trans` in `financial` has >1,000,000 rows; row-by-row INSERT
   would take far too long. `_load_table_data` batches rows through
   `psycopg2.extras.execute_values`, the same bulk-insert primitive
   `data/seed/seed_data.py` already uses elsewhere in this codebase.
3. Foreign keys. ADBA's retriever calls `expand_by_foreign_keys` and the
   rendered schema includes FK lines — a load that silently drops FKs
   produces a schema that *looks* fine and retrieves worse. FKs are read
   via `PRAGMA foreign_key_list` and added back as `ALTER TABLE ... ADD
   CONSTRAINT ... FOREIGN KEY` statements, AFTER every table is created
   and loaded (so FK checks never see a table that doesn't exist yet or
   data that hasn't landed yet).

Two phases, cleanly separated so the DDL/plan can be tested without a
live Postgres connection:

  - `read_sqlite_schema()` reads `sqlite_master` + `PRAGMA table_info` +
    `PRAGMA foreign_key_list` into plain dataclasses (`TableDef`,
    `ColumnDef`, `ForeignKeyDef`) — pure Python, no Postgres involved.
  - `build_create_table_sql()` / `build_add_foreign_key_sql()` turn those
    dataclasses into `psycopg2.sql.Composable` trees. Executing them needs
    a real connection (that's what makes `sql.Identifier` quote safely at
    all — see `render_composed`'s docstring) but *inspecting their
    structure* does not, which is what the unit tests do.

DSNs carry passwords. This module never prints one — see
`perception.profile_store.dsn_host_for_error`, reused here rather than
re-implemented, exactly as instructed.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass, field

import psycopg2
import psycopg2.extras
from psycopg2 import sql

from perception.profile_store import dsn_host_for_error

DEFAULT_BATCH_SIZE = 10_000
DEFAULT_SCHEMA = "public"


# ---------------------------------------------------------------------------
# Plain-data schema model (no Postgres, no psycopg2.sql involved yet)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnDef:
    name: str
    sqlite_type: str
    pg_type: str
    not_null: bool


@dataclass(frozen=True)
class ForeignKeyDef:
    """One FK constraint. `columns`/`ref_columns` are parallel and ordered
    by SQLite's `seq` field — required for composite foreign keys, where
    column order must match the referenced table's key order.

    As read straight off `PRAGMA foreign_key_list` (i.e. before
    `resolve_foreign_keys` runs), `ref_columns` entries may be `None`:
    SQLite permits `REFERENCES parent` with no column list, which
    implicitly targets `parent`'s primary key and reports `to = NULL` for
    every such row. `ref_table` may also disagree in case with the actual
    table name — SQLite resolves table names case-insensitively, Postgres
    does not. `resolve_foreign_keys` is what turns this raw, possibly
    ambiguous reading into something safe to hand to
    `build_add_foreign_key_sql`.
    """

    columns: tuple[str, ...]
    ref_table: str
    ref_columns: tuple[str, ...]


@dataclass(frozen=True)
class TableDef:
    name: str
    columns: tuple[ColumnDef, ...]
    primary_key: tuple[str, ...] = field(default_factory=tuple)
    foreign_keys: tuple[ForeignKeyDef, ...] = field(default_factory=tuple)
    # Column sets covered by a UNIQUE index/constraint in SQLite, read from
    # `PRAGMA index_list` + `PRAGMA index_info` (see `_read_unique_indexes`).
    # Does NOT include the primary key itself (tracked separately above) —
    # only *other* uniqueness SQLite already guarantees. Needed for bug 3:
    # a foreign key may target one of these instead of the primary key, and
    # Postgres requires a real unique constraint/index to back it.
    unique_column_sets: tuple[tuple[str, ...], ...] = field(default_factory=tuple)

    def column_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)


# ---------------------------------------------------------------------------
# Type mapping
# ---------------------------------------------------------------------------
#
# SQLite has type *affinity*, not strict column types: `PRAGMA table_info`
# reports whatever string the table's DDL declared, and any row can still
# hold any storage class regardless of it. The mapping below matches on
# substrings of that declared type, the same approach SQLite itself uses
# to derive affinity (see sqlite.org/datatype3.html #determination-of-
# column-affinity), but produces a concrete Postgres type rather than one
# of SQLite's five affinity classes:
#
#   declared type contains...   ->  Postgres type   reasoning
#   ---------------------------    --------------   ---------
#   (exact) "DATE"               ->  DATE             see below — deliberate
#   "DATETIME" / "TIMESTAMP"     ->  TIMESTAMP        same reasoning as DATE
#   "INT"                        ->  BIGINT           SQLite INTEGER is
#                                                      already 64-bit; never
#                                                      truncate to Postgres
#                                                      INTEGER's 32 bits
#   "CHAR" / "CLOB" / "TEXT"     ->  TEXT             SQLite TEXT has no
#                                                      length limit either
#   "BLOB"                       ->  BYTEA            binary, verbatim
#   "REAL" / "FLOA" / "DOUB"     ->  DOUBLE PRECISION  matches SQLite REAL
#                                                      affinity's storage
#   "NUMERIC" / "DECIMAL"        ->  NUMERIC          exact, arbitrary
#                                                      precision, as SQLite
#                                                      stores NUMERIC-affinity
#                                                      text/int/real losslessly
#   "BOOL"                       ->  BOOLEAN          not in BIRD/financial,
#                                                      but SQLite has no real
#                                                      boolean type either way
#   anything else / unrecognised ->  TEXT             SQLite's own fallback
#                                                      for untyped/unrecognised
#                                                      declarations is BLOB
#                                                      affinity, but an
#                                                      unrecognised *name*
#                                                      (typo, custom domain
#                                                      name, ...) is far more
#                                                      likely to be textual
#                                                      data than binary, so
#                                                      TEXT is the safer
#                                                      default per the spec
#
# DATE decision: verified against `financial.sqlite` directly (see
# tests/unit/test_load_sqlite_to_postgres.py and the loader report) that
# every column SQLite declares `DATE` (account.date, card.issued,
# client.birth_date, loan.date, trans.date) stores plain ISO-8601
# `YYYY-MM-DD` strings, not BIRD's older `YYMMDD` integer encoding used in
# some other benchmark databases. Postgres's `DATE` parses those directly,
# and mapping to a real DATE type (not TEXT) is what makes date
# arithmetic/comparisons in generated SQL behave the way a Vietnamese
# analyst wearing an LLM's hat would expect — landing this as TEXT would
# make every date-range query silently do string comparison, which fails
# in a way that looks exactly like a model reasoning error rather than a
# loader decision. If a future BIRD database's DATE columns turn out to
# hold non-ISO strings, THAT column should map to TEXT instead — this is a
# per-database judgment call, not a universal rule.


def map_sqlite_type(declared_type: str | None) -> str:
    """Map one SQLite declared type to a Postgres type name.

    `declared_type` is whatever `PRAGMA table_info` reports verbatim — it
    can be empty (`""` for a column declared with no type at all, which
    SQLite permits) or something no affinity rule recognises at all
    (a typo, a custom domain name a tool once used). Both fall through to
    the TEXT default below, never raise.
    """
    t = (declared_type or "").strip().upper()

    if t == "DATE":
        return "DATE"
    if "DATETIME" in t or "TIMESTAMP" in t:
        return "TIMESTAMP"
    if "INT" in t:
        return "BIGINT"
    if "CHAR" in t or "CLOB" in t or "TEXT" in t:
        return "TEXT"
    if "BLOB" in t:
        return "BYTEA"
    if "REAL" in t or "FLOA" in t or "DOUB" in t:
        return "DOUBLE PRECISION"
    if "NUMERIC" in t or "DECIMAL" in t:
        return "NUMERIC"
    if "BOOL" in t:
        return "BOOLEAN"
    return "TEXT"


# ---------------------------------------------------------------------------
# Reading the SQLite schema
# ---------------------------------------------------------------------------


def read_sqlite_schema(sqlite_path: str) -> tuple[TableDef, ...]:
    """Read every user table's structure from a SQLite file.

    Read-only: opens with `sqlite3.connect`, never writes. Internal
    bookkeeping tables (`sqlite_sequence` and anything else prefixed
    `sqlite_`) are excluded — they are SQLite implementation detail, not
    part of the schema being ported.
    """
    con = sqlite3.connect(sqlite_path)
    try:
        con.row_factory = None
        cur = con.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite\\_%' ESCAPE '\\' "
            "ORDER BY name"
        )
        table_names = [row[0] for row in cur.fetchall()]

        tables = []
        for name in table_names:
            tables.append(_read_one_table(cur, name))
        return tuple(tables)
    finally:
        con.close()


def _read_one_table(cur: sqlite3.Cursor, name: str) -> TableDef:
    cur.execute(f"PRAGMA table_info('{name}')")
    # PRAGMA table_info columns: cid, name, type, notnull, dflt_value, pk
    # `pk` is 0 when the column is not part of the primary key, else its
    # 1-based position WITHIN the primary key — required to preserve
    # column order for composite primary keys.
    raw_columns = cur.fetchall()

    columns = []
    pk_positions: list[tuple[int, str]] = []
    for _cid, col_name, col_type, notnull, _default, pk in raw_columns:
        columns.append(
            ColumnDef(
                name=col_name,
                sqlite_type=col_type or "",
                pg_type=map_sqlite_type(col_type),
                not_null=bool(notnull),
            )
        )
        if pk:
            pk_positions.append((pk, col_name))
    primary_key = tuple(n for _pos, n in sorted(pk_positions))

    cur.execute(f"PRAGMA foreign_key_list('{name}')")
    # PRAGMA foreign_key_list columns: id, seq, table, from, to,
    # on_update, on_delete, match. Rows sharing `id` belong to the same
    # (possibly composite) constraint; `seq` orders columns within it.
    fk_rows = cur.fetchall()
    foreign_keys = _group_foreign_keys(fk_rows)

    unique_column_sets = _read_unique_indexes(cur, name, primary_key)

    return TableDef(
        name=name,
        columns=tuple(columns),
        primary_key=primary_key,
        foreign_keys=foreign_keys,
        unique_column_sets=unique_column_sets,
    )


def _read_unique_indexes(
    cur: sqlite3.Cursor, name: str, primary_key: tuple[str, ...]
) -> tuple[tuple[str, ...], ...]:
    """Column sets covered by a UNIQUE index/constraint on `name`, read via
    `PRAGMA index_list` + `PRAGMA index_info`.

    Excludes:
      - the autoindex backing the primary key itself (origin `'pk'`) —
        `primary_key` above already tracks that, and duplicating it here
        would only invite confusion, never a wrong answer.
      - partial indexes — Postgres does not accept a partial unique index
        as a foreign key target, so one is no more useful here than no
        index at all.
      - expression indexes — `PRAGMA index_info` reports a column's `cid`
        as negative and its name as `None` for an indexed expression
        rather than a plain column; such an index cannot be re-expressed
        as `UNIQUE (col, ...)` in Postgres, so it is skipped rather than
        guessed at.

    This is what lets bug 3 (a foreign key targeting a non-unique-looking
    column) tell "SQLite already guarantees uniqueness here, just not via
    the primary key" apart from "SQLite never guaranteed uniqueness here
    at all" — the former can be ported faithfully, the latter genuinely
    cannot.
    """
    cur.execute(f"PRAGMA index_list('{name}')")
    # PRAGMA index_list columns: seq, name, unique, origin, partial.
    index_rows = cur.fetchall()

    unique_sets: list[tuple[str, ...]] = []
    for _seq, idx_name, is_unique, origin, partial in index_rows:
        if not is_unique or origin == "pk" or partial:
            continue
        cur.execute(f"PRAGMA index_info('{idx_name}')")
        # PRAGMA index_info columns: seqno, cid, name.
        cols = [row[2] for row in cur.fetchall()]
        if any(c is None for c in cols):
            continue  # expression index — no plain column list to port
        col_tuple = tuple(cols)
        if col_tuple == primary_key:
            continue  # a non-'pk'-origin index can still duplicate the PK
        unique_sets.append(col_tuple)
    return tuple(unique_sets)


def _group_foreign_keys(
    fk_rows: list[tuple],
) -> tuple[ForeignKeyDef, ...]:
    groups: dict[int, dict] = {}
    for fk_id, seq, ref_table, from_col, to_col, *_rest in fk_rows:
        group = groups.setdefault(
            fk_id, {"ref_table": ref_table, "pairs": []}
        )
        group["pairs"].append((seq, from_col, to_col))

    result = []
    for fk_id in sorted(groups):
        group = groups[fk_id]
        pairs = sorted(group["pairs"], key=lambda p: p[0])
        result.append(
            ForeignKeyDef(
                columns=tuple(p[1] for p in pairs),
                ref_table=group["ref_table"],
                ref_columns=tuple(p[2] for p in pairs),
            )
        )
    return tuple(result)


# ---------------------------------------------------------------------------
# Resolving foreign keys (still pure Python — no Postgres involved)
# ---------------------------------------------------------------------------
#
# `read_sqlite_schema()` reads `PRAGMA foreign_key_list` verbatim, which
# means a `TableDef` coming straight out of it can carry three things
# `build_add_foreign_key_sql` cannot cope with (each one found by running
# this loader against a real BIRD database, not hypothesised):
#
#   1. `ref_columns` containing `None` — an implicit `REFERENCES parent`
#      with no column list, which SQLite resolves against `parent`'s
#      primary key. (`debit_card_specializing`, `european_football_2`.)
#   2. `ref_table` disagreeing in case with the table's real name — SQLite
#      folds table names case-insensitively, Postgres does not.
#      (`european_football_2`: `country`/`Country`.)
#   3. `ref_columns` naming a column that is neither the referenced
#      table's primary key nor covered by any SQLite UNIQUE index —
#      something SQLite never required to be unique, which Postgres's
#      `REFERENCES` genuinely cannot express, full stop.
#      (`card_games`, `european_football_2`: `Team.team_fifa_api_id`.)
#
# `resolve_foreign_keys` is the single place all three are settled, so
# every other function in this module can assume a `ForeignKeyDef` it
# receives already has a real `ref_table` and fully-named `ref_columns`
# that Postgres can actually enforce.


@dataclass(frozen=True)
class ForeignKeyResolution:
    """Result of resolving every table's foreign keys against the full set
    of tables in one database.

    `tables` mirrors the input, in the same order, except each table's
    `foreign_keys` has been rewritten: implicit targets filled in, table
    name casing corrected, and unresolvable foreign keys dropped.

    `skipped` is one human-readable line per dropped foreign key — the
    caller is expected to print these, never swallow them (a missing FK
    degrades ADBA's retriever, which uses `expand_by_foreign_keys` and
    prints FKs via `render_schema`).

    `required_unique_indexes` lists the `(table_name, columns)` pairs that
    need a `CREATE UNIQUE INDEX` in Postgres before the foreign keys
    targeting them can be added — SQLite already guarantees uniqueness
    there (a UNIQUE index, not the primary key), Postgres just needs it
    spelled out as a real constraint/index to accept a `REFERENCES` onto
    it. Deduplicated: two foreign keys targeting the same column set only
    need the index created once.
    """

    tables: tuple[TableDef, ...]
    skipped: tuple[str, ...]
    required_unique_indexes: tuple[tuple[str, tuple[str, ...]], ...]


def resolve_foreign_keys(tables: tuple[TableDef, ...]) -> ForeignKeyResolution:
    """Resolve every foreign key in `tables` against `tables` itself.

    Must see every table in the database at once — resolving a case
    mismatch or an implicit primary-key target both require looking up
    the *referenced* table, which may not be the one `PRAGMA
    foreign_key_list` was just read for.
    """
    by_exact_name = {t.name: t for t in tables}
    by_lower_name: dict[str, list[TableDef]] = {}
    for t in tables:
        by_lower_name.setdefault(t.name.lower(), []).append(t)

    skipped: list[str] = []
    required_unique_indexes: dict[tuple[str, tuple[str, ...]], None] = {}
    resolved_tables: list[TableDef] = []

    for table in tables:
        kept_fks: list[ForeignKeyDef] = []
        for fk in table.foreign_keys:
            label = (
                f"{table.name}({', '.join(fk.columns)}) -> "
                f"{fk.ref_table}({', '.join(c if c is not None else '?' for c in fk.ref_columns)})"
            )

            ref_table_def = by_exact_name.get(fk.ref_table)
            if ref_table_def is None:
                candidates = by_lower_name.get(fk.ref_table.lower(), [])
                if len(candidates) == 1:
                    ref_table_def = candidates[0]
                elif len(candidates) > 1:
                    skipped.append(
                        f"{label}: {len(candidates)} tables match "
                        f"{fk.ref_table!r} case-insensitively — ambiguous, "
                        "cannot resolve"
                    )
                    continue
                else:
                    skipped.append(
                        f"{label}: no table named {fk.ref_table!r} "
                        "(even case-insensitively) — dangling reference, "
                        "dropping this foreign key"
                    )
                    continue

            ref_columns = fk.ref_columns
            if any(c is None for c in ref_columns):
                pk = ref_table_def.primary_key
                if len(ref_columns) == 1 and len(pk) == 1:
                    ref_columns = pk
                else:
                    reason = (
                        "a composite primary key"
                        if len(pk) > 1
                        else "no primary key"
                    )
                    skipped.append(
                        f"{label}: SQLite left the target column implicit "
                        f"(bare 'REFERENCES {ref_table_def.name}'), but "
                        f"{ref_table_def.name} has {reason} — cannot "
                        "resolve which column(s) it means, dropping this "
                        "foreign key"
                    )
                    continue

            target_set = frozenset(ref_columns)
            if ref_table_def.primary_key and frozenset(ref_table_def.primary_key) == target_set:
                pass  # primary key already backs it — nothing more needed
            else:
                matching_unique = next(
                    (
                        u
                        for u in ref_table_def.unique_column_sets
                        if frozenset(u) == target_set
                    ),
                    None,
                )
                if matching_unique is None:
                    skipped.append(
                        f"{label}: {ref_table_def.name}"
                        f"({', '.join(ref_columns)}) is neither the primary "
                        "key nor covered by a UNIQUE index in the source — "
                        "Postgres requires a unique/primary-key target for "
                        "a foreign key and none exists, dropping this "
                        "foreign key"
                    )
                    continue
                required_unique_indexes.setdefault(
                    (ref_table_def.name, matching_unique), None
                )

            kept_fks.append(
                ForeignKeyDef(
                    columns=fk.columns,
                    ref_table=ref_table_def.name,
                    ref_columns=ref_columns,
                )
            )

        # Always rebuild, even when no FK was dropped: a kept FK may still
        # have had its `ref_table` case-corrected or its `ref_columns`
        # filled in from an implicit reference, so reusing `table` as-is
        # here would silently discard those rewrites.
        resolved_tables.append(
            TableDef(
                name=table.name,
                columns=table.columns,
                primary_key=table.primary_key,
                foreign_keys=tuple(kept_fks),
                unique_column_sets=table.unique_column_sets,
            )
        )

    return ForeignKeyResolution(
        tables=tuple(resolved_tables),
        skipped=tuple(skipped),
        required_unique_indexes=tuple(required_unique_indexes.keys()),
    )


# ---------------------------------------------------------------------------
# Building DDL (psycopg2.sql.Composable — quoting deferred to psycopg2)
# ---------------------------------------------------------------------------


def build_create_table_sql(table: TableDef, schema: str = DEFAULT_SCHEMA) -> sql.Composed:
    """`CREATE TABLE <schema>.<table> (...)`, no foreign keys yet.

    Every identifier — schema, table, and each column name — is wrapped in
    `sql.Identifier`; type names are a fixed vocabulary this module
    controls (see `map_sqlite_type`), not attacker- or file-controlled
    text, so `sql.SQL` is fine for them. FKs are deliberately left out:
    they are added by `build_add_foreign_key_sql` only after every table
    in the database has been created (see module docstring, point 3).
    """
    column_parts = []
    for col in table.columns:
        parts: list[sql.Composable] = [
            sql.Identifier(col.name),
            sql.SQL(" "),
            sql.SQL(col.pg_type),
        ]
        if col.not_null:
            parts.append(sql.SQL(" NOT NULL"))
        column_parts.append(sql.Composed(parts))

    body: list[sql.Composable] = [sql.SQL(", ").join(column_parts)]
    if table.primary_key:
        pk_idents = sql.SQL(", ").join(sql.Identifier(c) for c in table.primary_key)
        body.append(sql.SQL(", PRIMARY KEY ({})").format(pk_idents))

    return sql.SQL("CREATE TABLE {}.{} ({})").format(
        sql.Identifier(schema),
        sql.Identifier(table.name),
        sql.Composed(body),
    )


def build_drop_table_sql(table_name: str, schema: str = DEFAULT_SCHEMA) -> sql.Composed:
    return sql.SQL("DROP TABLE IF EXISTS {}.{} CASCADE").format(
        sql.Identifier(schema), sql.Identifier(table_name)
    )


def _fk_constraint_name(table: TableDef, fk: ForeignKeyDef, index: int) -> str:
    # Deterministic and stable across re-runs; index disambiguates a table
    # with more than one FK to the same referenced table.
    return f"fk_{table.name}_{fk.ref_table}_{index}"


def build_add_foreign_key_sql(
    table: TableDef, schema: str = DEFAULT_SCHEMA
) -> tuple[sql.Composed, ...]:
    """One `ALTER TABLE ... ADD CONSTRAINT ... FOREIGN KEY (...) REFERENCES
    ...(...)` per foreign key on `table`. Empty tuple if `table` has none.
    """
    statements = []
    for index, fk in enumerate(table.foreign_keys):
        cols = sql.SQL(", ").join(sql.Identifier(c) for c in fk.columns)
        ref_cols = sql.SQL(", ").join(sql.Identifier(c) for c in fk.ref_columns)
        stmt = sql.SQL(
            "ALTER TABLE {schema}.{table} "
            "ADD CONSTRAINT {constraint} "
            "FOREIGN KEY ({cols}) REFERENCES {schema}.{ref_table} ({ref_cols})"
        ).format(
            schema=sql.Identifier(schema),
            table=sql.Identifier(table.name),
            constraint=sql.Identifier(_fk_constraint_name(table, fk, index)),
            cols=cols,
            ref_table=sql.Identifier(fk.ref_table),
            ref_cols=ref_cols,
        )
        statements.append(stmt)
    return tuple(statements)


def _unique_index_name(table_name: str, columns: tuple[str, ...]) -> str:
    return f"uq_{table_name}_{'_'.join(columns)}"


def build_add_unique_index_sql(
    table_name: str, columns: tuple[str, ...], schema: str = DEFAULT_SCHEMA
) -> sql.Composed:
    """`CREATE UNIQUE INDEX IF NOT EXISTS ... ON <schema>.<table> (...)`.

    Used only for `ForeignKeyResolution.required_unique_indexes` (bug 3):
    SQLite already guarantees uniqueness on `columns` via its own UNIQUE
    index, just not through `table_name`'s primary key, and Postgres needs
    that spelled out as a real index before it will accept a `REFERENCES`
    onto them. `IF NOT EXISTS` makes this idempotent when two foreign keys
    in the same database target the same column set.
    """
    cols = sql.SQL(", ").join(sql.Identifier(c) for c in columns)
    return sql.SQL("CREATE UNIQUE INDEX IF NOT EXISTS {} ON {}.{} ({})").format(
        sql.Identifier(_unique_index_name(table_name, columns)),
        sql.Identifier(schema),
        sql.Identifier(table_name),
        cols,
    )


def render_composed(obj: sql.Composable) -> str:
    """Render a `psycopg2.sql.Composable` tree to text WITHOUT a live
    Postgres connection.

    `Composable.as_string()` normally needs a real connection or cursor —
    it calls into psycopg2's C-level `quote_ident`, which type-checks its
    second argument against the actual libpq connection/cursor types, so
    nothing short of a real connection satisfies it (confirmed directly:
    passing a plain object with a matching `.encoding` attribute still
    raises `TypeError: argument 2 must be a connection or a cursor`).
    That is exactly why the unit tests in this module's test file cannot
    call `.as_string()` and use this function instead.

    This mirrors Postgres's standard identifier-quoting rule — wrap in
    double quotes, double any embedded double quote — using only the
    public `.strings`/`.string` attributes of `sql.Identifier`/`sql.SQL`.
    It is a TEST/DEBUG-ONLY rendering path: the real execution path never
    calls this. `cur.execute(composed)` hands the `Composable` straight to
    psycopg2, which does its own quoting against the live connection at
    execute time — this function never sits between a `Composable` and
    the database.
    """
    if isinstance(obj, sql.Composed):
        return "".join(render_composed(part) for part in obj.seq)
    if isinstance(obj, sql.Identifier):
        return ".".join('"' + part.replace('"', '""') + '"' for part in obj.strings)
    if isinstance(obj, sql.SQL):
        return obj.string
    raise TypeError(f"render_composed: unsupported Composable {type(obj)!r}")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


class LoadError(RuntimeError):
    """Raised for operator-fixable problems (schema already populated,
    bad path, ...). Never carries a DSN — see `dsn_host_for_error`."""


def _existing_tables(pg_cur, schema: str, names: tuple[str, ...]) -> list[str]:
    pg_cur.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = %s AND table_name = ANY(%s)",
        (schema, list(names)),
    )
    return [row[0] for row in pg_cur.fetchall()]


def load_sqlite_to_postgres(
    sqlite_path: str,
    dsn: str,
    drop: bool = False,
    schema: str = DEFAULT_SCHEMA,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, int]:
    """Load `sqlite_path` into `dsn`'s `schema` (default `public`).

    Order of operations, deliberate (see module docstring point 3):
    1. (if `drop`) DROP every target table, CASCADE.
    2. Otherwise, refuse if any target table already exists — an
       idempotent re-run only ever happens via `--drop`; a bare re-run
       against a populated schema would either fail deep in a partial
       state or silently double-insert, both worse than refusing upfront.
    3. CREATE every table (no FK constraints yet).
    4. Load every table's rows, batched.
    5. ADD every FK constraint, now that all data is in place.

    Returns `{table_name: row_count_loaded}`. Never logs `dsn` — only
    `dsn_host_for_error(dsn)` may appear in messages.
    """
    tables = read_sqlite_schema(sqlite_path)
    if not tables:
        raise LoadError(f"no tables found in {sqlite_path!r}")

    resolution = resolve_foreign_keys(tables)
    tables = resolution.tables
    if resolution.skipped:
        print(
            f"{sqlite_path}: {len(resolution.skipped)} foreign key(s) "
            "SQLite allows but Postgres cannot express — dropping them "
            "(schema and data load continue; ADBA's retriever will not see "
            "these relationships):",
            flush=True,
        )
        for msg in resolution.skipped:
            print(f"  {msg}", flush=True)

    table_names = tuple(t.name for t in tables)

    sqlite_con = sqlite3.connect(sqlite_path)
    pg_con = psycopg2.connect(dsn)
    try:
        with pg_con.cursor() as pg_cur:
            if drop:
                for table in tables:
                    pg_cur.execute(build_drop_table_sql(table.name, schema))
                pg_con.commit()
            else:
                clashing = _existing_tables(pg_cur, schema, table_names)
                if clashing:
                    host = dsn_host_for_error(dsn)
                    raise LoadError(
                        f"refusing to load into {schema!r} on {host}: "
                        f"table(s) already exist: {', '.join(sorted(clashing))} "
                        "(pass --drop to recreate)"
                    )

            print(f"creating {len(tables)} table(s) in schema {schema!r}...", flush=True)
            for table in tables:
                pg_cur.execute(build_create_table_sql(table, schema))
            pg_con.commit()

            row_counts: dict[str, int] = {}
            for table in tables:
                row_counts[table.name] = _load_table_data(
                    sqlite_con, pg_cur, table, schema, batch_size
                )
            pg_con.commit()

            if resolution.required_unique_indexes:
                print(
                    f"  adding {len(resolution.required_unique_indexes)} "
                    "supporting UNIQUE index(es) — SQLite already "
                    "guarantees uniqueness there, just not via the primary "
                    "key:",
                    flush=True,
                )
                for tbl_name, cols in resolution.required_unique_indexes:
                    print(f"    {tbl_name}({', '.join(cols)})", flush=True)
                    pg_cur.execute(build_add_unique_index_sql(tbl_name, cols, schema))
                pg_con.commit()

            print("adding foreign key constraints...", flush=True)
            unvalidated = _add_foreign_keys(pg_con, pg_cur, tables, schema)
            if unvalidated:
                print(
                    f"  {len(unvalidated)} khoá ngoại thêm dạng NOT VALID vì dữ "
                    "liệu nguồn vi phạm chính khai báo của nó:",
                    flush=True,
                )
                for name in unvalidated:
                    print(f"    {name}", flush=True)

        return row_counts
    except Exception:
        pg_con.rollback()
        raise
    finally:
        sqlite_con.close()
        pg_con.close()


def _add_foreign_keys(pg_con, pg_cur, tables, schema: str) -> list[str]:
    """Thêm khoá ngoại, lùi về `NOT VALID` khi dữ liệu nguồn vi phạm.

    SQLite KHÔNG cưỡng chế khoá ngoại trừ khi bật `PRAGMA foreign_keys`, nên
    dữ liệu benchmark thường xuyên vi phạm chính khai báo của mình — BIRD
    `formula_1` có 22 dòng `races` trỏ tới `circuitId` không tồn tại trong
    `circuits`. Đó không phải lỗi của bản nạp; đó là tính chất của dữ liệu,
    và cũng đúng thứ khiến BIRD là phép thử tốt.

    Bỏ hẳn khoá ngoại thì sai: `expand_by_foreign_keys` của retriever dùng
    chúng, và `render_schema` in chúng vào prompt — mất khoá ngoại là mất
    thông tin quan hệ mà model cần. Nhưng để nguyên thì lệnh nạp chết.

    `NOT VALID` giải đúng bài: ràng buộc được ghi vào catalog (nên
    `introspect_schema` vẫn đọc thấy) và vẫn cưỡng chế với dòng MỚI, chỉ
    không kiểm lại dòng đã có.

    Thử bản validated trước để dữ liệu sạch vẫn được kiểm thật, và trả về
    tên những ràng buộc phải lùi — im lặng chấp nhận toàn vẹn tham chiếu bị
    gãy sẽ giấu mất một tính chất thật của bộ dữ liệu.
    """
    import psycopg2
    from psycopg2 import sql

    unvalidated: list[str] = []
    for table in tables:
        for i, stmt in enumerate(build_add_foreign_key_sql(table, schema)):
            try:
                pg_cur.execute(stmt)
                pg_con.commit()
            except psycopg2.errors.ForeignKeyViolation:
                pg_con.rollback()
                # `stmt` là `sql.Composed` (định danh đã quote an toàn), không
                # phải chuỗi — ghép bằng Composed chứ không nối chuỗi.
                pg_cur.execute(sql.Composed([stmt, sql.SQL(" NOT VALID")]))
                pg_con.commit()
                fk = table.foreign_keys[i]
                unvalidated.append(
                    f"{table.name}({', '.join(fk.columns)}) → "
                    f"{fk.ref_table}({', '.join(fk.ref_columns)})"
                )
    return unvalidated


def _load_table_data(
    sqlite_con: sqlite3.Connection,
    pg_cur,
    table: TableDef,
    schema: str,
    batch_size: int,
) -> int:
    col_names = table.column_names()
    select_cols = ", ".join(f'"{c}"' for c in col_names)
    # Read-side quoting for the SQLite SELECT: standard double-quote
    # identifier form, valid SQLite syntax. This is the SOURCE read, not
    # anything sent to Postgres — the Postgres-bound INSERT below still
    # goes exclusively through sql.Identifier.
    sqlite_cur = sqlite_con.cursor()
    sqlite_cur.execute(f'SELECT {select_cols} FROM "{table.name}"')

    insert_prefix = sql.SQL("INSERT INTO {}.{} ({})").format(
        sql.Identifier(schema),
        sql.Identifier(table.name),
        sql.SQL(", ").join(sql.Identifier(c) for c in col_names),
    )
    # `execute_values` needs a plain string, not a `sql.Composed`, and its
    # `_split_sql` byte-scans that string for `%.` two-character tokens to
    # find the single `%s` VALUES placeholder — it has no idea some of
    # those `%` came from inside a quoted identifier, not a placeholder.
    # BIRD's `california_schools.frpm` has columns literally named
    # `Percent (%) Eligible Free (K-12)`; once quoted, the rendered
    # identifier contains `%)`, and `_split_sql` chokes on it with
    # `ValueError: unsupported format character: ')'` — not a psycopg2
    # bug, just what happens when a real `%` collides with `execute_values`'
    # placeholder syntax. `sql.Composed.as_string()` needs a real
    # connection/cursor to quote identifiers (see `render_composed`'s
    # docstring), which `pg_cur` is — so render here, then double every
    # literal `%` psycopg2 just emitted (`_split_sql` unescapes `%%` back
    # to one `%`, which is exactly what makes this round-trip correctly)
    # before appending the one true `%s` placeholder, which must stay
    # single.
    insert_sql = insert_prefix.as_string(pg_cur).replace("%", "%%") + " VALUES %s"

    total = 0
    print(f"loading {table.name}...", end="", flush=True)
    while True:
        batch = sqlite_cur.fetchmany(batch_size)
        if not batch:
            break
        psycopg2.extras.execute_values(
            pg_cur, insert_sql, batch, page_size=batch_size
        )
        total += len(batch)
        print(f"\rloading {table.name}... {total} row(s)", end="", flush=True)
    print(f"\rloading {table.name}... {total} row(s) done", flush=True)
    return total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load a SQLite database into a Postgres database's public schema."
        )
    )
    parser.add_argument("--sqlite", required=True, help="path to the .sqlite file")
    parser.add_argument("--dsn", required=True, help="target Postgres DSN")
    parser.add_argument(
        "--drop",
        action="store_true",
        help="drop and recreate target tables if they already exist",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"rows per insert batch (default {DEFAULT_BATCH_SIZE})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        row_counts = load_sqlite_to_postgres(
            args.sqlite, args.dsn, drop=args.drop, batch_size=args.batch_size
        )
    except LoadError as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 1

    print("done:", flush=True)
    for name, count in row_counts.items():
        print(f"  {name}: {count} row(s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
