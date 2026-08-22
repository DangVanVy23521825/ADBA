import os
import sqlite3

import pytest

from eval.load_sqlite_to_postgres import (
    ColumnDef,
    ForeignKeyDef,
    LoadError,
    TableDef,
    build_add_foreign_key_sql,
    build_create_table_sql,
    build_drop_table_sql,
    load_sqlite_to_postgres,
    map_sqlite_type,
    read_sqlite_schema,
    render_composed,
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
