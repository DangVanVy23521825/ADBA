import os
import sqlite3

import pytest

from eval.load_sqlite_to_postgres import (
    ColumnDef,
    ForeignKeyDef,
    LoadError,
    TableDef,
    build_add_foreign_key_sql,
    build_add_unique_index_sql,
    build_create_table_sql,
    build_drop_table_sql,
    load_sqlite_to_postgres,
    map_sqlite_type,
    read_sqlite_schema,
    render_composed,
    resolve_foreign_keys,
)

DSN = os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")


def _make_sqlite(tmp_path, ddl_statements: list[str]) -> str:
    db_path = str(tmp_path / "fixture.sqlite")
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute("PRAGMA foreign_keys = ON")
        for stmt in ddl_statements:
            cur.execute(stmt)
        con.commit()
    finally:
        con.close()
    return db_path


class TestTypeMapping:
    """Pure function, no SQLite/Postgres involved — the substring-matching
    rules documented in load_sqlite_to_postgres.map_sqlite_type."""

    def test_integer_maps_to_bigint(self):
        assert map_sqlite_type("INTEGER") == "BIGINT"

    def test_text_maps_to_text(self):
        assert map_sqlite_type("TEXT") == "TEXT"

    def test_real_maps_to_double_precision(self):
        assert map_sqlite_type("REAL") == "DOUBLE PRECISION"

    def test_blob_maps_to_bytea(self):
        assert map_sqlite_type("BLOB") == "BYTEA"

    def test_numeric_maps_to_numeric(self):
        assert map_sqlite_type("NUMERIC") == "NUMERIC"

    def test_decimal_maps_to_numeric(self):
        assert map_sqlite_type("DECIMAL(10,2)") == "NUMERIC"

    def test_varchar_maps_to_text(self):
        assert map_sqlite_type("VARCHAR(255)") == "TEXT"

    def test_date_maps_to_date_deliberately(self):
        # See the module docstring's DATE decision: financial.sqlite's
        # DATE columns hold ISO-8601 strings, so DATE (not TEXT) is
        # correct and lets date comparisons work in generated SQL.
        assert map_sqlite_type("DATE") == "DATE"

    def test_datetime_maps_to_timestamp(self):
        assert map_sqlite_type("DATETIME") == "TIMESTAMP"

    def test_unrecognised_type_falls_back_to_text(self):
        assert map_sqlite_type("FROBNICATE") == "TEXT"

    def test_empty_type_falls_back_to_text(self):
        # SQLite permits a column declared with no type at all.
        assert map_sqlite_type("") == "TEXT"
        assert map_sqlite_type(None) == "TEXT"

    def test_mapping_is_case_insensitive(self):
        assert map_sqlite_type("integer") == "BIGINT"
        assert map_sqlite_type("text") == "TEXT"


class TestReadSqliteSchema:
    """read_sqlite_schema() against small fixture databases built with the
    stdlib sqlite3 module in tmp_path — no Postgres involved."""

    def test_reads_columns_in_declared_order(self, tmp_path):
        db = _make_sqlite(
            tmp_path,
            ["CREATE TABLE widgets (id INTEGER PRIMARY KEY, name TEXT, price REAL)"],
        )
        tables = read_sqlite_schema(db)
        assert len(tables) == 1
        widgets = tables[0]
        assert widgets.name == "widgets"
        assert widgets.column_names() == ("id", "name", "price")
        assert [c.pg_type for c in widgets.columns] == ["BIGINT", "TEXT", "DOUBLE PRECISION"]

    def test_not_null_is_preserved(self, tmp_path):
        db = _make_sqlite(
            tmp_path,
            [
                "CREATE TABLE t (id INTEGER PRIMARY KEY, required TEXT NOT NULL, "
                "optional TEXT)"
            ],
        )
        (t,) = read_sqlite_schema(db)
        by_name = {c.name: c for c in t.columns}
        assert by_name["required"].not_null is True
        assert by_name["optional"].not_null is False

    def test_single_column_primary_key(self, tmp_path):
        db = _make_sqlite(
            tmp_path, ["CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)"]
        )
        (t,) = read_sqlite_schema(db)
        assert t.primary_key == ("id",)

    def test_composite_primary_key_preserves_declared_order(self, tmp_path):
        db = _make_sqlite(
            tmp_path,
            [
                "CREATE TABLE membership ("
                "team_id INTEGER, user_id INTEGER, role TEXT, "
                "PRIMARY KEY (user_id, team_id))"
            ],
        )
        (t,) = read_sqlite_schema(db)
        # PRAGMA table_info's `pk` field numbers columns by their position
        # WITHIN the primary key, not their column order in the table —
        # (user_id, team_id) was declared in that order and must survive.
        assert t.primary_key == ("user_id", "team_id")

    def test_foreign_key_is_read(self, tmp_path):
        db = _make_sqlite(
            tmp_path,
            [
                "CREATE TABLE parent (id INTEGER PRIMARY KEY)",
                "CREATE TABLE child (id INTEGER PRIMARY KEY, "
                "parent_id INTEGER REFERENCES parent(id))",
            ],
        )
        tables = {t.name: t for t in read_sqlite_schema(db)}
        child = tables["child"]
        assert len(child.foreign_keys) == 1
        fk = child.foreign_keys[0]
        assert fk.columns == ("parent_id",)
        assert fk.ref_table == "parent"
        assert fk.ref_columns == ("id",)

    def test_composite_foreign_key_columns_stay_paired(self, tmp_path):
        db = _make_sqlite(
            tmp_path,
            [
                "CREATE TABLE parent (a INTEGER, b INTEGER, PRIMARY KEY (a, b))",
                "CREATE TABLE child (id INTEGER PRIMARY KEY, "
                "pa INTEGER, pb INTEGER, "
                "FOREIGN KEY (pa, pb) REFERENCES parent(a, b))",
            ],
        )
        tables = {t.name: t for t in read_sqlite_schema(db)}
        fk = tables["child"].foreign_keys[0]
        assert fk.columns == ("pa", "pb")
        assert fk.ref_columns == ("a", "b")

    def test_table_with_no_foreign_keys_has_empty_tuple(self, tmp_path):
        db = _make_sqlite(tmp_path, ["CREATE TABLE lonely (id INTEGER PRIMARY KEY)"])
        (t,) = read_sqlite_schema(db)
        assert t.foreign_keys == ()

    def test_sqlite_internal_tables_are_excluded(self, tmp_path):
        db = _make_sqlite(
            tmp_path,
            [
                "CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT)",
            ],
        )
        # AUTOINCREMENT forces SQLite to create sqlite_sequence.
        con = sqlite3.connect(db)
        try:
            names = {
                r[0]
                for r in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        finally:
            con.close()
        assert "sqlite_sequence" in names  # sanity: the fixture really has it

        table_names = {t.name for t in read_sqlite_schema(db)}
        assert table_names == {"t"}

    def test_reserved_word_table_name_is_read_verbatim(self, tmp_path):
        db = _make_sqlite(
            tmp_path,
            ['CREATE TABLE "order" (order_id INTEGER PRIMARY KEY, amount REAL)'],
        )
        table_names = {t.name for t in read_sqlite_schema(db)}
        assert table_names == {"order"}

    def test_percent_and_parens_column_name_is_read_verbatim(self, tmp_path):
        # Bug 1's shape: california_schools.frpm's real columns, byte for
        # byte. The bug report displayed these as e.g.
        # 'Percent (%) Eligible Free (K-12)' with single quotes around
        # them, but those quotes are Python's own repr() punctuation from
        # printing a PRAGMA table_info() row tuple — not characters SQLite
        # stores. Confirmed directly against california_schools.sqlite's
        # real PRAGMA table_info('frpm') output: no quote characters
        # appear in any column name, only spaces, parentheses, '%', and
        # '/'. This fixture's column name carries no quotes either, and
        # read_sqlite_schema must hand it back exactly as declared.
        db = _make_sqlite(
            tmp_path,
            [
                'CREATE TABLE frpm (CDSCode TEXT PRIMARY KEY, '
                '"Percent (%) Eligible Free (K-12)" REAL, '
                '"Enrollment (Ages 5-17)" REAL)'
            ],
        )
        (t,) = read_sqlite_schema(db)
        assert "Percent (%) Eligible Free (K-12)" in t.column_names()
        assert "Enrollment (Ages 5-17)" in t.column_names()
        # No column name accidentally carries a literal quote character.
        assert not any("'" in c or '"' in c for c in t.column_names())


class TestGeneratedDdl:
    """Assert on the DDL/plan psycopg2.sql.Composable trees build, using
    render_composed() (see its docstring for why: Composable.as_string()
    needs a live connection, which these tests deliberately don't have)."""

    def test_reserved_word_table_name_is_quoted(self):
        table = TableDef(
            name="order",
            columns=(
                ColumnDef("order_id", "INTEGER", "BIGINT", True),
                ColumnDef("amount", "REAL", "DOUBLE PRECISION", True),
            ),
            primary_key=("order_id",),
        )
        ddl = render_composed(build_create_table_sql(table))
        assert '"order"' in ddl
        assert "CREATE TABLE" in ddl
        assert '"order_id" BIGINT NOT NULL' in ddl
        assert '"amount" DOUBLE PRECISION NOT NULL' in ddl
        assert 'PRIMARY KEY ("order_id")' in ddl
        # never bare/unquoted "order" as a token outside the identifier
        assert "TABLE order " not in ddl
        assert "TABLE order(" not in ddl

    def test_reserved_word_column_name_is_quoted(self):
        table = TableDef(
            name="t",
            columns=(ColumnDef("select", "TEXT", "TEXT", False),),
        )
        ddl = render_composed(build_create_table_sql(table))
        assert '"select"' in ddl

    def test_column_without_not_null_has_no_not_null_clause(self):
        table = TableDef(
            name="t", columns=(ColumnDef("maybe", "TEXT", "TEXT", False),)
        )
        ddl = render_composed(build_create_table_sql(table))
        assert "NOT NULL" not in ddl

    def test_composite_primary_key_ddl_preserves_order(self):
        table = TableDef(
            name="membership",
            columns=(
                ColumnDef("team_id", "INTEGER", "BIGINT", True),
                ColumnDef("user_id", "INTEGER", "BIGINT", True),
                ColumnDef("role", "TEXT", "TEXT", False),
            ),
            primary_key=("user_id", "team_id"),
        )
        ddl = render_composed(build_create_table_sql(table))
        assert 'PRIMARY KEY ("user_id", "team_id")' in ddl

    def test_table_without_primary_key_has_no_pk_clause(self):
        table = TableDef(name="t", columns=(ColumnDef("x", "TEXT", "TEXT", False),))
        ddl = render_composed(build_create_table_sql(table))
        assert "PRIMARY KEY" not in ddl

    def test_default_schema_is_public(self):
        table = TableDef(name="t", columns=(ColumnDef("x", "TEXT", "TEXT", False),))
        ddl = render_composed(build_create_table_sql(table))
        assert '"public"."t"' in ddl

    def test_foreign_key_ddl_is_emitted(self):
        table = TableDef(
            name="disp",
            columns=(
                ColumnDef("disp_id", "INTEGER", "BIGINT", True),
                ColumnDef("client_id", "INTEGER", "BIGINT", True),
            ),
            primary_key=("disp_id",),
            foreign_keys=(
                ForeignKeyDef(
                    columns=("client_id",),
                    ref_table="client",
                    ref_columns=("client_id",),
                ),
            ),
        )
        (stmt,) = build_add_foreign_key_sql(table)
        rendered = render_composed(stmt)
        assert "ALTER TABLE" in rendered
        assert '"disp"' in rendered
        assert "FOREIGN KEY" in rendered
        assert '("client_id")' in rendered
        assert 'REFERENCES "public"."client" ("client_id")' in rendered

    def test_reserved_word_table_foreign_key_is_quoted(self):
        table = TableDef(
            name="order",
            columns=(
                ColumnDef("order_id", "INTEGER", "BIGINT", True),
                ColumnDef("account_id", "INTEGER", "BIGINT", True),
            ),
            primary_key=("order_id",),
            foreign_keys=(
                ForeignKeyDef(
                    columns=("account_id",),
                    ref_table="account",
                    ref_columns=("account_id",),
                ),
            ),
        )
        (stmt,) = build_add_foreign_key_sql(table)
        rendered = render_composed(stmt)
        assert 'ALTER TABLE "public"."order"' in rendered

    def test_table_with_no_foreign_keys_produces_no_alter_statements(self):
        table = TableDef(name="t", columns=(ColumnDef("x", "TEXT", "TEXT", False),))
        assert build_add_foreign_key_sql(table) == ()

    def test_composite_foreign_key_ddl_pairs_columns_correctly(self):
        table = TableDef(
            name="child",
            columns=(
                ColumnDef("id", "INTEGER", "BIGINT", True),
                ColumnDef("pa", "INTEGER", "BIGINT", True),
                ColumnDef("pb", "INTEGER", "BIGINT", True),
            ),
            primary_key=("id",),
            foreign_keys=(
                ForeignKeyDef(
                    columns=("pa", "pb"), ref_table="parent", ref_columns=("a", "b")
                ),
            ),
        )
        (stmt,) = build_add_foreign_key_sql(table)
        rendered = render_composed(stmt)
        assert '("pa", "pb")' in rendered
        assert 'REFERENCES "public"."parent" ("a", "b")' in rendered

    def test_drop_table_ddl_is_quoted_and_cascades(self):
        rendered = render_composed(build_drop_table_sql("order"))
        assert rendered == 'DROP TABLE IF EXISTS "public"."order" CASCADE'


class TestEndToEndPlanAgainstRealBirdShapedFixture:
    """A fixture mirroring the exact trap in BIRD's financial.sqlite: a
    table named `order` (reserved word) with a foreign key to `account`.
    Exercises read_sqlite_schema -> build_create_table_sql ->
    build_add_foreign_key_sql end to end without touching Postgres."""

    def test_order_table_round_trips_with_its_foreign_key(self, tmp_path):
        db = _make_sqlite(
            tmp_path,
            [
                "CREATE TABLE account (account_id INTEGER PRIMARY KEY, "
                "district_id INTEGER NOT NULL)",
                'CREATE TABLE "order" ('
                "order_id INTEGER PRIMARY KEY, "
                "account_id INTEGER NOT NULL, "
                "amount REAL NOT NULL, "
                "FOREIGN KEY (account_id) REFERENCES account(account_id))",
            ],
        )
        tables = {t.name: t for t in read_sqlite_schema(db)}
        order = tables["order"]
        assert order.foreign_keys[0].ref_table == "account"

        create_ddl = render_composed(build_create_table_sql(order))
        assert '"public"."order"' in create_ddl
        assert '"amount" DOUBLE PRECISION NOT NULL' in create_ddl

        (fk_stmt,) = build_add_foreign_key_sql(order)
        fk_ddl = render_composed(fk_stmt)
        assert 'ALTER TABLE "public"."order"' in fk_ddl
        assert 'REFERENCES "public"."account" ("account_id")' in fk_ddl


class TestPercentInColumnName:
    """Bug 1: BIRD's california_schools.frpm has columns literally named
    'Percent (%) Eligible Free (K-12)' — no quotes in the real data (see
    TestReadSqliteSchema.test_percent_and_parens_column_name_is_read_verbatim
    below; the single quotes in the bug report are Python's own repr()
    punctuation, not characters SQLite stores).

    `_load_table_data` bulk-inserts via `psycopg2.extras.execute_values`,
    which converts its query to a string and then byte-scans it for `%.`
    two-character tokens (see `psycopg2.extras._split_sql`) to find the
    single `%s` VALUES placeholder — with no awareness that some of those
    `%` characters came from inside an already-quoted identifier rather
    than a placeholder. A column name containing `%)` renders to `..."Percent
    (%) Eligible..."...`, and `_split_sql` raises `ValueError: unsupported
    format character: ')'` on the `%)` it finds there — not a `%)s`
    placeholder, just a real character sequence colliding with
    `execute_values`'s scanning. The fix doubles every `%` in the rendered,
    already-quoted identifier text (`_split_sql` unescapes `%%` back to a
    single `%`) before appending the one genuine `%s` placeholder.
    """

    def _rendered_insert_prefix(self, table_name: str, col_names: list[str]) -> str:
        from psycopg2 import sql

        prefix = sql.SQL("INSERT INTO {}.{} ({})").format(
            sql.Identifier("public"),
            sql.Identifier(table_name),
            sql.SQL(", ").join(sql.Identifier(c) for c in col_names),
        )
        return render_composed(prefix)

    def test_unescaped_percent_column_breaks_execute_values_split(self):
        from psycopg2.extras import _split_sql

        rendered = self._rendered_insert_prefix(
            "frpm", ["Percent (%) Eligible Free (K-12)"]
        )
        unescaped = rendered + " VALUES %s"
        with pytest.raises(ValueError, match="unsupported format character"):
            _split_sql(unescaped.encode())

    def test_escaped_percent_column_survives_execute_values_split(self):
        from psycopg2.extras import _split_sql

        rendered = self._rendered_insert_prefix(
            "frpm", ["Percent (%) Eligible Free (K-12)"]
        )
        escaped = rendered.replace("%", "%%") + " VALUES %s"
        pre, post = _split_sql(escaped.encode())
        # The literal '%' from the column name must survive the escape/
        # unescape round trip intact, right where the identifier put it.
        assert b'"Percent (%) Eligible Free (K-12)"' in b"".join(pre)


class TestForeignKeyResolution:
    """resolve_foreign_keys() is where bugs 2, 3, and 4 are all settled —
    see its module-level comment for why all three showed up only when
    running this loader against real BIRD databases. Every fixture here
    is a small real SQLite database built in tmp_path, read through the
    actual read_sqlite_schema()/PRAGMA path, exactly like the loader
    itself would encounter it — nothing hand-constructed as a TableDef.
    """

    def test_implicit_target_resolves_to_single_column_primary_key(self, tmp_path):
        # Bug 2: debit_card_specializing.yearmonth -> customers, no column
        # named. PRAGMA foreign_key_list reports `to = NULL`.
        db = _make_sqlite(
            tmp_path,
            [
                "CREATE TABLE customers (CustomerID INTEGER PRIMARY KEY, "
                "name TEXT)",
                "CREATE TABLE yearmonth (CustomerID INTEGER, Date TEXT, "
                "PRIMARY KEY (CustomerID, Date), "
                "FOREIGN KEY (CustomerID) REFERENCES customers)",
            ],
        )
        tables = read_sqlite_schema(db)
        raw_fk = {t.name: t for t in tables}["yearmonth"].foreign_keys[0]
        assert raw_fk.ref_columns == (None,)  # sanity: this is really bug 2's shape

        resolution = resolve_foreign_keys(tables)
        assert resolution.skipped == ()
        by_name = {t.name: t for t in resolution.tables}
        fk = by_name["yearmonth"].foreign_keys[0]
        assert fk.ref_table == "customers"
        assert fk.ref_columns == ("CustomerID",)

    def test_implicit_target_onto_composite_primary_key_is_skipped_and_reported(
        self, tmp_path
    ):
        # Bug 2's stated edge case: the referenced table's primary key is
        # composite, so an implicit (column-less) REFERENCES cannot be
        # resolved without guessing — must skip, not guess.
        db = _make_sqlite(
            tmp_path,
            [
                "CREATE TABLE parent (a INTEGER, b INTEGER, PRIMARY KEY (a, b))",
                "CREATE TABLE child (id INTEGER PRIMARY KEY, "
                "ref INTEGER REFERENCES parent)",
            ],
        )
        tables = read_sqlite_schema(db)
        resolution = resolve_foreign_keys(tables)

        by_name = {t.name: t for t in resolution.tables}
        assert by_name["child"].foreign_keys == ()
        assert len(resolution.skipped) == 1
        assert "composite primary key" in resolution.skipped[0]
        assert "child" in resolution.skipped[0] and "parent" in resolution.skipped[0]

    def test_implicit_target_onto_table_with_no_primary_key_is_skipped_and_reported(
        self, tmp_path
    ):
        db = _make_sqlite(
            tmp_path,
            [
                "CREATE TABLE parent (a INTEGER, b INTEGER)",  # no PK at all
                "CREATE TABLE child (id INTEGER PRIMARY KEY, "
                "ref INTEGER REFERENCES parent)",
            ],
        )
        tables = read_sqlite_schema(db)
        resolution = resolve_foreign_keys(tables)

        by_name = {t.name: t for t in resolution.tables}
        assert by_name["child"].foreign_keys == ()
        assert len(resolution.skipped) == 1
        assert "no primary key" in resolution.skipped[0]

    def test_foreign_key_onto_non_unique_column_is_skipped_and_reported(self, tmp_path):
        # Bug 3: card_games.legalities -> cards(uuid) where `uuid` carries
        # no PRIMARY KEY or UNIQUE constraint in the source at all — Postgres
        # genuinely cannot express a FOREIGN KEY onto it.
        db = _make_sqlite(
            tmp_path,
            [
                "CREATE TABLE cards (id INTEGER PRIMARY KEY, uuid TEXT)",
                "CREATE TABLE legalities (id INTEGER PRIMARY KEY, "
                "card_uuid TEXT REFERENCES cards(uuid))",
            ],
        )
        tables = read_sqlite_schema(db)
        resolution = resolve_foreign_keys(tables)

        by_name = {t.name: t for t in resolution.tables}
        assert by_name["legalities"].foreign_keys == ()
        assert resolution.required_unique_indexes == ()
        assert len(resolution.skipped) == 1
        assert "cards" in resolution.skipped[0] and "uuid" in resolution.skipped[0]
        assert "neither the primary key nor covered by a UNIQUE index" in resolution.skipped[0]

    def test_foreign_key_onto_unique_indexed_non_pk_column_is_kept_with_supporting_index(
        self, tmp_path
    ):
        # The reasonable alternative this module chose for bug 3: when
        # SQLite's own schema already declares the target UNIQUE (just not
        # as the primary key), port that as a real Postgres unique index
        # instead of dropping a foreign key relationship that is, in fact,
        # perfectly resolvable. Mirrors card_games.cards.uuid exactly.
        db = _make_sqlite(
            tmp_path,
            [
                "CREATE TABLE cards (id INTEGER PRIMARY KEY, uuid TEXT UNIQUE)",
                "CREATE TABLE legalities (id INTEGER PRIMARY KEY, "
                "card_uuid TEXT REFERENCES cards(uuid))",
            ],
        )
        tables = read_sqlite_schema(db)
        resolution = resolve_foreign_keys(tables)

        assert resolution.skipped == ()
        assert resolution.required_unique_indexes == (("cards", ("uuid",)),)
        by_name = {t.name: t for t in resolution.tables}
        fk = by_name["legalities"].foreign_keys[0]
        assert fk.ref_table == "cards"
        assert fk.ref_columns == ("uuid",)

        ddl = render_composed(build_add_unique_index_sql("cards", ("uuid",)))
        assert "CREATE UNIQUE INDEX IF NOT EXISTS" in ddl
        assert '"public"."cards"' in ddl
        assert '("uuid")' in ddl

    def test_foreign_key_target_table_name_resolved_case_insensitively(self, tmp_path):
        # Bug 4: european_football_2's League.country_id -> country(id)
        # while the real table is "Country" (PascalCase). SQLite folds
        # table names case-insensitively; Postgres does not, and the
        # loader must preserve the real table's case rather than the
        # lowercase spelling the foreign key happens to use.
        db = _make_sqlite(
            tmp_path,
            [
                'CREATE TABLE "Country" (id INTEGER PRIMARY KEY, name TEXT)',
                'CREATE TABLE league (id INTEGER PRIMARY KEY, '
                "country_id INTEGER REFERENCES country(id))",
            ],
        )
        tables = read_sqlite_schema(db)
        raw_fk = {t.name: t for t in tables}["league"].foreign_keys[0]
        assert raw_fk.ref_table == "country"  # sanity: lowercase, as SQLite reports it

        resolution = resolve_foreign_keys(tables)
        assert resolution.skipped == ()
        by_name = {t.name: t for t in resolution.tables}
        fk = by_name["league"].foreign_keys[0]
        assert fk.ref_table == "Country"  # resolved to the real, case-preserved name

        ddl = render_composed(build_add_foreign_key_sql(by_name["league"])[0])
        assert 'REFERENCES "public"."Country" ("id")' in ddl

    def test_dangling_reference_even_case_insensitively_is_skipped_and_reported(
        self, tmp_path
    ):
        db = _make_sqlite(
            tmp_path,
            ["CREATE TABLE t (id INTEGER PRIMARY KEY)"],
        )
        # Hand-construct a TableDef with a FK to a table that doesn't
        # exist at all, even case-insensitively (SQLite itself would
        # refuse to declare this with foreign_keys pragma enforcement at
        # DDL time in some configurations, but PRAGMA foreign_key_list
        # only reads whatever the DDL says — a genuinely dangling
        # reference is exactly what resolve_foreign_keys must also catch,
        # not just case/None-target issues).
        tables = read_sqlite_schema(db)
        (t,) = tables
        broken = TableDef(
            name=t.name,
            columns=t.columns,
            primary_key=t.primary_key,
            foreign_keys=(
                ForeignKeyDef(columns=("id",), ref_table="nonexistent", ref_columns=("id",)),
            ),
        )
        resolution = resolve_foreign_keys((broken,))
        assert resolution.tables[0].foreign_keys == ()
        assert len(resolution.skipped) == 1
        assert "nonexistent" in resolution.skipped[0]


class TestLoadErrorGuards:
    """load_sqlite_to_postgres()'s pre-flight checks that don't need a
    live Postgres connection to verify: an empty source database."""

    def test_empty_sqlite_database_raises_load_error(self, tmp_path):
        db_path = str(tmp_path / "empty.sqlite")
        sqlite3.connect(db_path).close()
        with pytest.raises(LoadError):
            load_sqlite_to_postgres(db_path, "postgresql://unused/db")


@pytest.mark.skipif(not DSN, reason="cần DATABASE_URL trỏ Postgres thật")
class TestAgainstLiveDatabase:
    """End-to-end load against a real, throwaway Postgres schema. Skipped
    without DATABASE_URL/POSTGRES_URL, matching tests/unit/test_introspect.py's
    TestAgainstLiveDatabase pattern: guard-only tests above run always,
    live tests are isolated in a class that skips.

    Uses its own dedicated schema (`_bird_loader_test`) inside whatever
    database DATABASE_URL points at, and drops it in teardown — never
    touches `public` or any pre-existing table.
    """

    @pytest.fixture(autouse=True)
    def _schema(self):
        import psycopg2 as pg

        con = pg.connect(DSN)
        con.autocommit = True
        cur = con.cursor()
        cur.execute("DROP SCHEMA IF EXISTS _bird_loader_test CASCADE")
        cur.execute("CREATE SCHEMA _bird_loader_test")
        try:
            yield
        finally:
            cur.execute("DROP SCHEMA IF EXISTS _bird_loader_test CASCADE")
            con.close()

    def test_loads_reserved_word_table_with_foreign_key_and_data(self, tmp_path):
        import psycopg2 as pg

        db = _make_sqlite(
            tmp_path,
            [
                "CREATE TABLE account (account_id INTEGER PRIMARY KEY, "
                "district_id INTEGER NOT NULL)",
                'CREATE TABLE "order" ('
                "order_id INTEGER PRIMARY KEY, "
                "account_id INTEGER NOT NULL, "
                "amount REAL NOT NULL, "
                "FOREIGN KEY (account_id) REFERENCES account(account_id))",
            ],
        )
        con = sqlite3.connect(db)
        con.execute("INSERT INTO account VALUES (1, 10)")
        con.execute('INSERT INTO "order" VALUES (100, 1, 250.5)')
        con.commit()
        con.close()

        counts = load_sqlite_to_postgres(
            db, DSN, drop=True, schema="_bird_loader_test"
        )
        assert counts == {"account": 1, "order": 1}

        pg_con = pg.connect(DSN)
        try:
            cur = pg_con.cursor()
            cur.execute('SELECT amount FROM _bird_loader_test."order"')
            assert cur.fetchone() == (250.5,)
            cur.execute(
                "SELECT constraint_type FROM information_schema.table_constraints "
                "WHERE table_schema = %s AND table_name = %s "
                "AND constraint_type = 'FOREIGN KEY'",
                ("_bird_loader_test", "order"),
            )
            assert cur.fetchone() is not None
        finally:
            pg_con.close()

    def test_refuses_to_clobber_without_drop(self, tmp_path):
        db = _make_sqlite(tmp_path, ["CREATE TABLE t (id INTEGER PRIMARY KEY)"])
        load_sqlite_to_postgres(db, DSN, drop=True, schema="_bird_loader_test")
        with pytest.raises(LoadError):
            load_sqlite_to_postgres(db, DSN, drop=False, schema="_bird_loader_test")
