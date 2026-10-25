"""The allowlist: which tables and columns the model may see and query.

The catalog is the single source of truth for both the SQL guard (static check) and the
SQLite authorizer (engine-level check). A column that is not in the catalog does not
exist as far as the model is concerned: it is not in the schema prompt, the guard
rejects any reference to it, and the authorizer denies reading it.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field

__all__ = ["MOVIES_CATALOG", "Catalog", "ColumnSpec", "TableSpec", "identifier"]

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


def identifier(name: str) -> str:
    """Validate a plain SQL identifier (letters, digits, underscore) and lower-case it."""
    if not _IDENT.fullmatch(name):
        raise ValueError(f"not a plain identifier: {name!r}")
    return name.lower()


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    name: str
    type: str = ""
    description: str = ""


@dataclass(frozen=True, slots=True)
class TableSpec:
    name: str
    columns: tuple[ColumnSpec, ...]
    description: str = ""

    @property
    def column_names(self) -> frozenset[str]:
        return frozenset(c.name for c in self.columns)


@dataclass(frozen=True)
class Catalog:
    """Allowlisted tables and columns. Names are stored lower-case."""

    tables: Mapping[str, TableSpec]
    hidden: Mapping[str, frozenset[str]] = field(default_factory=dict)
    """Columns that exist in the database but are not allowlisted (filled by the database)."""

    @classmethod
    def of(cls, *tables: TableSpec) -> Catalog:
        normalised = {}
        for t in tables:
            cols = tuple(ColumnSpec(identifier(c.name), c.type, c.description) for c in t.columns)
            if not cols:
                raise ValueError(f"table {t.name!r} has no allowlisted columns")
            normalised[identifier(t.name)] = TableSpec(identifier(t.name), cols, t.description)
        return cls(normalised)

    @classmethod
    def introspect(
        cls,
        conn: sqlite3.Connection,
        tables: Iterable[str] | None = None,
        exclude_columns: Iterable[str] = (),
    ) -> Catalog:
        """Build a catalog from a live database: the given tables (default: every user table)
        minus ``table.column`` entries in ``exclude_columns``."""
        names = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        ).fetchall()
        existing = {str(n[0]).lower() for n in names}
        wanted = [identifier(t) for t in tables] if tables else sorted(existing)
        excluded = {e.lower() for e in exclude_columns}
        specs = []
        for table in wanted:
            if table not in existing:
                raise ValueError(f"no such table: {table}")
            # ``table`` is a validated identifier that exists in sqlite_master.
            info = conn.execute("SELECT name, type FROM pragma_table_info(?)", (table,)).fetchall()
            cols = tuple(
                ColumnSpec(identifier(str(n)), str(t))
                for n, t in info
                if f"{table}.{str(n).lower()}" not in excluded
            )
            specs.append(TableSpec(table, cols))
        return cls.of(*specs)

    def with_hidden(self, hidden: Mapping[str, frozenset[str]]) -> Catalog:
        return Catalog(self.tables, dict(hidden))

    def columns(self, table: str) -> frozenset[str]:
        return self.tables[table].column_names

    def has_hidden(self, table: str) -> bool:
        return bool(self.hidden.get(table))

    def schema_text(self, samples: Mapping[str, list[dict[str, object]]] | None = None) -> str:
        """The schema as shown to the model: tables, columns, types, descriptions.

        No data unless ``samples`` is given (an explicit opt-in)."""
        lines: list[str] = []
        for t in self.tables.values():
            head = f"TABLE {t.name}"
            lines.append(f"{head} -- {t.description}" if t.description else head)
            for c in t.columns:
                col = f"  {c.name} {c.type}".rstrip()
                lines.append(f"{col} -- {c.description}" if c.description else col)
            for row in (samples or {}).get(t.name, []):
                lines.append(f"  sample: {row}")
        return "\n".join(lines)


MOVIES_CATALOG = Catalog.of(
    TableSpec(
        "movies",
        (
            ColumnSpec("id", "INTEGER", "primary key"),
            ColumnSpec("title", "TEXT"),
            ColumnSpec("year", "INTEGER", "release year"),
            ColumnSpec(
                "genre", "TEXT", "one of Drama, Comedy, Sci-Fi, Thriller, Documentary, Animation"
            ),
            ColumnSpec("director", "TEXT"),
            ColumnSpec("setting_city", "TEXT", "the real-world city the story is set in"),
            ColumnSpec("runtime_min", "INTEGER", "runtime in minutes"),
            ColumnSpec("budget_musd", "REAL", "budget in millions of USD"),
            ColumnSpec("gross_musd", "REAL", "worldwide box office gross in millions of USD"),
            ColumnSpec("rating", "REAL", "average audience rating, 0-10"),
            ColumnSpec("plot", "TEXT", "one-sentence plot summary"),
        ),
        "fictional films",
    ),
    TableSpec(
        "reviews",
        (
            ColumnSpec("id", "INTEGER", "primary key"),
            ColumnSpec("movie_id", "INTEGER", "references movies.id"),
            ColumnSpec("reviewer", "TEXT", "public handle"),
            ColumnSpec("stars", "INTEGER", "1-5"),
            ColumnSpec("text", "TEXT", "free-text review written by the public"),
            ColumnSpec("posted", "TEXT", "ISO date"),
        ),
        "audience reviews (reviewer_email exists in the file but is not allowlisted)",
    ),
)
"""The bundled database's allowlist. ``reviews.reviewer_email`` is deliberately absent."""
