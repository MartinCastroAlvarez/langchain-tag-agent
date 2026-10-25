"""The read-only connection: writes impossible, timeouts, row caps, custom allowlists."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tag_agent.catalog import MOVIES_CATALOG
from tag_agent.catalog import Catalog
from tag_agent.catalog import ColumnSpec
from tag_agent.catalog import TableSpec
from tag_agent.db import DEFAULT_DB
from tag_agent.db import QueryError
from tag_agent.db import QueryTimeout
from tag_agent.db import ReadOnlyDatabase
from tag_agent.guard import SafeQuery
from tag_agent.guard import SqlGuard


@pytest.fixture
def scratch(tmp_path: Path) -> Path:
    path = tmp_path / "shop.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, ssn TEXT, notes BLOB);
        CREATE TABLE secrets (id INTEGER PRIMARY KEY, token TEXT);
        INSERT INTO customers VALUES (1, 'Ana', '123-45-6789', X'00FF'), (2, 'Bo', '987', NULL);
        INSERT INTO secrets VALUES (1, 'hunter2');
        """
    )
    conn.commit()
    conn.close()
    return path


def test_file_is_opened_read_only(scratch: Path) -> None:
    db = ReadOnlyDatabase(scratch)
    for sql in (
        "DELETE FROM customers",
        "UPDATE customers SET name = 'x'",
        "PRAGMA query_only = OFF",
        "CREATE TABLE t (x)",
        "ATTACH DATABASE ':memory:' AS m",
    ):
        with pytest.raises(QueryError):
            db.execute(SafeQuery(sql))
    db.close()
    raw = sqlite3.connect(scratch)
    assert raw.execute("SELECT count(*) FROM customers").fetchone() == (2,)
    raw.close()


def test_the_connection_is_read_only_at_the_sqlite_level(scratch: Path) -> None:
    """Even with the authorizer removed, mode=ro + query_only refuse writes."""
    db = ReadOnlyDatabase(scratch)
    db._conn.set_authorizer(None)
    with pytest.raises(sqlite3.OperationalError, match="readonly|read-only|query_only"):
        db._conn.execute("DELETE FROM customers")
    db.close()


def test_execute_takes_only_safe_queries(db: ReadOnlyDatabase) -> None:
    with pytest.raises(TypeError):
        db.execute("SELECT 1")  # type: ignore[arg-type]


def test_timeout_interrupts_a_long_query() -> None:
    db = ReadOnlyDatabase(DEFAULT_DB, MOVIES_CATALOG, timeout=0.001)
    heavy = SqlGuard(db.catalog, max_tables=4).check(
        "SELECT count(*) FROM movies a, movies b, movies c, movies d WHERE a.id + b.id + c.id + d.id > 1"
    )
    with pytest.raises(QueryTimeout, match="timeout"):
        db.execute(heavy)
    db.timeout = 5.0
    assert db.query("SELECT count(*) FROM movies").rows == [(28,)]


def test_row_cap_marks_truncation(db: ReadOnlyDatabase) -> None:
    table = db.execute(db.guard(max_rows=5).check("SELECT title FROM movies"))
    assert len(table) == 5
    assert (
        db.execute(db.guard(max_rows=5).check("SELECT title FROM movies LIMIT 3")).truncated
        is False
    )
    query = SafeQuery("SELECT title FROM movies", max_rows=5)
    assert db.execute(query).truncated is True


def test_custom_database_introspection_and_hidden_columns(scratch: Path) -> None:
    conn = sqlite3.connect(f"{scratch.as_uri()}?mode=ro", uri=True)
    catalog = Catalog.introspect(conn, ["customers"], ["customers.ssn"])
    conn.close()
    db = ReadOnlyDatabase(scratch, catalog, max_cell_chars=2)
    assert db.catalog.hidden == {"customers": frozenset({"ssn"})}
    rows = db.query("SELECT name, notes FROM customers ORDER BY id").rows
    assert rows == [("An...", "<blob 2 bytes>"), ("Bo", None)]
    for sql in (
        "SELECT ssn FROM customers",
        "SELECT token FROM secrets",
        "SELECT * FROM customers",
    ):
        with pytest.raises(Exception, match="not allowlisted|not allowed"):
            db.query(sql)
    with pytest.raises(QueryError, match="prohibited"):
        db.execute(SafeQuery("SELECT token FROM secrets"))
    with db:
        pass


def test_default_catalog_is_every_table(scratch: Path) -> None:
    db = ReadOnlyDatabase(scratch)
    assert sorted(db.catalog.tables) == ["customers", "secrets"]
    db.close()


def test_catalog_must_match_the_database(scratch: Path) -> None:
    missing_table = Catalog.of(TableSpec("orders", (ColumnSpec("id"),)))
    with pytest.raises(ValueError, match="does not exist"):
        ReadOnlyDatabase(scratch, missing_table)
    missing_col = Catalog.of(TableSpec("customers", (ColumnSpec("email"),)))
    with pytest.raises(ValueError, match="missing"):
        ReadOnlyDatabase(scratch, missing_col)
    conn = sqlite3.connect(scratch)
    with pytest.raises(ValueError, match="no such table"):
        Catalog.introspect(conn, ["nope"])
    conn.close()


def test_paths_must_be_regular_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ReadOnlyDatabase(tmp_path / "absent.sqlite")
    with pytest.raises(ValueError, match="regular file"):
        ReadOnlyDatabase(tmp_path)


def test_bundled_database_hides_reviewer_email(db: ReadOnlyDatabase) -> None:
    assert db.catalog.hidden["reviews"] == frozenset({"reviewer_email"})
    assert "reviewer_email" not in db.catalog.columns("reviews")
