"""Read-only SQLite access with an engine-level allowlist, a row cap and a timeout.

Layers, from the outside in:

1. The file is opened through a ``file:...?mode=ro`` URI, so SQLite itself refuses writes.
2. ``PRAGMA query_only = ON`` and ``PRAGMA trusted_schema = OFF`` are set before anything
   else runs; extension loading stays disabled; ``SQLITE_LIMIT_ATTACHED`` is 0, so
   ``ATTACH`` can't work even if a statement got through.
3. An authorizer callback allows only ``SELECT``, reads of allowlisted ``table.column``
   pairs and calls of allowlisted SQL functions. Everything else, ``PRAGMA``, ``ATTACH``,
   recursive CTEs, writes and reads of hidden columns, is denied by SQLite at prepare time.
4. A progress handler aborts any statement that runs longer than ``timeout`` seconds, and
   at most ``max_rows`` rows are fetched.

Only :class:`~tag_agent.guard.SafeQuery` objects are executed, and values travel as bound
parameters, never as SQL text.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

from tag_agent.catalog import Catalog
from tag_agent.guard import SafeQuery
from tag_agent.guard import SqlGuard
from tag_agent.table import Cell
from tag_agent.table import Table

__all__ = [
    "DEFAULT_DB",
    "SQL_FUNCTIONS",
    "QueryError",
    "QueryTimeout",
    "ReadOnlyDatabase",
]

DEFAULT_DB = Path(__file__).parent / "data" / "movies.sqlite"

# Function names as SQLite reports them to the authorizer (LIKE and GLOB are functions).
SQL_FUNCTIONS: frozenset[str] = frozenset(
    {
        "count",
        "sum",
        "avg",
        "min",
        "max",
        "total",
        "rank",
        "dense_rank",
        "row_number",
        "substr",
        "substring",
        "coalesce",
        "ifnull",
        "round",
        "lower",
        "upper",
        "length",
        "instr",
        "group_concat",
        "abs",
        "trim",
        "replace",
        "nullif",
        "iif",
        "like",
        "glob",
    }
)

_LIMITS = {
    "SQLITE_LIMIT_ATTACHED": 0,
    "SQLITE_LIMIT_LENGTH": 1_000_000,
    "SQLITE_LIMIT_SQL_LENGTH": 20_000,
    "SQLITE_LIMIT_COMPOUND_SELECT": 4,
    "SQLITE_LIMIT_EXPR_DEPTH": 100,
    "SQLITE_LIMIT_FUNCTION_ARG": 8,
    "SQLITE_LIMIT_VARIABLE_NUMBER": 32,
    "SQLITE_LIMIT_LIKE_PATTERN_LENGTH": 200,
}


class QueryError(RuntimeError):
    """The database refused or failed the query. The message is safe to show the model."""


class QueryTimeout(QueryError):
    """The query ran longer than the timeout and was interrupted."""


class ReadOnlyDatabase:
    """A read-only connection to one SQLite file, restricted to ``catalog``."""

    def __init__(
        self,
        path: str | Path = DEFAULT_DB,
        catalog: Catalog | None = None,
        *,
        timeout: float = 2.0,
        max_cell_chars: int = 2000,
    ) -> None:
        resolved = Path(path).expanduser().resolve(strict=True)
        if not resolved.is_file():
            raise ValueError(f"not a regular file: {resolved}")
        self.path = resolved
        self.timeout = timeout
        self.max_cell_chars = max_cell_chars
        self._lock = threading.Lock()
        self._deadline = 0.0
        conn = sqlite3.connect(
            f"{resolved.as_uri()}?mode=ro", uri=True, check_same_thread=False, isolation_level=None
        )
        if hasattr(conn, "enable_load_extension"):
            conn.enable_load_extension(False)
        conn.execute("PRAGMA query_only = ON")
        conn.execute("PRAGMA trusted_schema = OFF")
        for name, value in _LIMITS.items():
            conn.setlimit(getattr(sqlite3, name), value)
        base = catalog if catalog is not None else Catalog.introspect(conn)
        self.catalog = base.with_hidden(self._hidden_columns(conn, base))
        conn.set_authorizer(self._authorize)
        conn.set_progress_handler(self._progress, 1000)
        self._conn = conn

    @staticmethod
    def _hidden_columns(conn: sqlite3.Connection, catalog: Catalog) -> dict[str, frozenset[str]]:
        hidden: dict[str, frozenset[str]] = {}
        for name, spec in catalog.tables.items():
            info = conn.execute("SELECT name FROM pragma_table_info(?)", (name,)).fetchall()
            actual = {str(r[0]).lower() for r in info}
            if not actual:
                raise ValueError(f"catalog table {name!r} does not exist in the database")
            missing = spec.column_names - actual
            if missing:
                raise ValueError(f"catalog columns missing from {name}: {sorted(missing)}")
            hidden[name] = frozenset(actual - spec.column_names)
        return hidden

    def guard(self, max_rows: int = 100) -> SqlGuard:
        return SqlGuard(self.catalog, max_rows=max_rows)

    # ------------------------------------------------------------------ callbacks

    def _authorize(
        self,
        action: int,
        arg1: str | None,
        arg2: str | None,
        db_name: str | None,
        trigger: str | None,
    ) -> int:
        if action == sqlite3.SQLITE_SELECT:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ:
            table = (arg1 or "").lower()
            column = (arg2 or "").lower()
            # An empty column name is SQLite reading the rowid for count(*) and friends.
            if (
                db_name in ("main", None)
                and table in self.catalog.tables
                and (column == "" or column in self.catalog.columns(table))
            ):
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION:
            return (
                sqlite3.SQLITE_OK if (arg2 or "").lower() in SQL_FUNCTIONS else sqlite3.SQLITE_DENY
            )
        return sqlite3.SQLITE_DENY

    def _progress(self) -> int:
        return 1 if time.monotonic() > self._deadline else 0

    # ------------------------------------------------------------------ execution

    def execute(self, query: SafeQuery) -> Table:
        if not isinstance(query, SafeQuery):
            raise TypeError("execute() takes a SafeQuery from SqlGuard.check()")
        with self._lock:
            self._deadline = time.monotonic() + self.timeout
            try:
                cursor = self._conn.execute(query.sql, dict(query.params))
                columns = [d[0] for d in cursor.description or ()]
                raw = cursor.fetchmany(query.max_rows + 1)
            except sqlite3.OperationalError as exc:
                if "interrupt" in str(exc).lower():
                    raise QueryTimeout(f"query exceeded the {self.timeout:g}s timeout") from exc
                raise QueryError(f"database error: {exc}") from exc
            except sqlite3.Error as exc:
                raise QueryError(f"database error: {exc}") from exc
            finally:
                self._deadline = 0.0
        rows = [tuple(self._cell(v) for v in r) for r in raw[: query.max_rows]]
        return Table(columns, rows, truncated=len(raw) > query.max_rows)

    def query(
        self, sql: str, params: dict[str, object] | None = None, max_rows: int = 100
    ) -> Table:
        """Guard + execute in one call."""
        return self.execute(self.guard(max_rows).check(sql, params))

    def _cell(self, value: object) -> Cell:
        if value is None:
            return None
        if isinstance(value, int | float):
            return value
        if isinstance(value, bytes):
            return f"<blob {len(value)} bytes>"
        text = str(value)
        return text if len(text) <= self.max_cell_chars else text[: self.max_cell_chars] + "..."

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> ReadOnlyDatabase:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
