"""Static SQL guard: one read-only SELECT over allowlisted tables and columns.

Every query the model writes goes through :meth:`SqlGuard.check` before it reaches the
database. The guard parses with sqlglot and refuses anything that is not exactly one
``SELECT`` (or a set operation of ``SELECT``s): DDL, DML, ``PRAGMA``, ``ATTACH``,
``VACUUM INTO``, ``SELECT ... INTO``, recursive CTEs, unknown or file-reading functions
(``read_csv``, ``readfile``, ``load_extension``, ``pragma_*`` table functions...),
non-allowlisted tables or columns (``sqlite_master``, hidden columns), schema-qualified
names, positional ``?`` placeholders and over-long or over-complex queries. It then
**regenerates** the SQL from the parsed tree without comments, so the text the database
runs is the text that was checked (no comment or parser-differential tricks), and it
forces a ``LIMIT``.

The guard is the first of three layers; the read-only connection and the SQLite
authorizer in :mod:`tag_agent.db` enforce the same allowlist again at the engine.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from tag_agent.catalog import Catalog

__all__ = ["ALLOWED_FUNCTIONS", "GuardError", "SafeQuery", "Scalar", "SqlGuard"]

Scalar = str | int | float | bool | None

# sqlglot expression classes that may appear (functions and function-like operators).
ALLOWED_FUNCTIONS: frozenset[str] = frozenset(
    {
        "And",
        "Or",
        "Xor",
        "Exists",
        "Count",
        "Sum",
        "Avg",
        "Min",
        "Max",
        "Rank",
        "DenseRank",
        "RowNumber",
        "Substring",
        "Coalesce",
        "Cast",
        "Round",
        "Lower",
        "Upper",
        "Length",
        "StrPosition",
        "GroupConcat",
        "Abs",
        "Trim",
        "Replace",
        "Nullif",
        "Case",
        "If",
    }
)
_ALLOWED_ANONYMOUS = frozenset({"total"})

# Statement / clause types that must never appear anywhere in the tree.
_FORBIDDEN = tuple(
    t
    for t in (
        getattr(exp, name, None)
        for name in (
            "Insert",
            "Update",
            "Delete",
            "Merge",
            "Drop",
            "Create",
            "Alter",
            "Command",
            "Pragma",
            "Attach",
            "Detach",
            "Into",
            "Lock",
            "Transaction",
            "Commit",
            "Rollback",
            "Set",
            "Use",
            "Copy",
            "LoadData",
            "Analyze",
            "Parameter",
            "SessionParameter",
            "Semicolon",
        )
    )
    if isinstance(t, type)
)
_PARAM_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


class GuardError(ValueError):
    """The query is not allowed. The message is safe to show to the model."""


@dataclass(frozen=True, slots=True)
class SafeQuery:
    """A checked query: regenerated SQL, bound parameters and the enforced row cap."""

    sql: str
    params: Mapping[str, Scalar] = field(default_factory=dict)
    max_rows: int = 100
    tables: frozenset[str] = frozenset()


@dataclass(frozen=True)
class SqlGuard:
    catalog: Catalog
    max_rows: int = 100
    max_length: int = 4000
    max_nodes: int = 800
    max_tables: int = 4
    max_offset: int = 10_000
    max_params: int = 32
    max_param_chars: int = 1000

    def check(
        self, sql: str, params: Mapping[str, object] | None = None, *, max_rows: int | None = None
    ) -> SafeQuery:
        cap = self.max_rows if max_rows is None else max_rows
        if not isinstance(sql, str) or not sql.strip():
            raise GuardError("empty query")
        if len(sql) > self.max_length:
            raise GuardError(f"query longer than {self.max_length} characters")
        if "\x00" in sql:
            raise GuardError("NUL byte in query")
        tree = self._parse_single(sql)
        tables = self._check_tree(tree)
        self._enforce_limit(tree, cap)
        names = self._placeholders(tree)
        bound = self._check_params(names, params or {})
        text = tree.sql(dialect="sqlite", comments=False)
        # Belt and braces: what we will run must parse back to the same single SELECT.
        self._check_tree(self._parse_single(text))
        return SafeQuery(sql=text, params=bound, max_rows=cap, tables=tables)

    # ------------------------------------------------------------------ parsing

    def _parse_single(self, sql: str) -> exp.Expr:
        try:
            statements = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
        except SqlglotError as exc:
            raise GuardError(f"could not parse the query: {str(exc)[:200]}") from exc
        if len(statements) != 1:
            raise GuardError("exactly one statement is allowed")
        tree = statements[0]
        if not _is_select_tree(tree):
            raise GuardError(f"only SELECT is allowed, got {type(tree).__name__.upper()}")
        return tree

    # ------------------------------------------------------------------ tree checks

    def _check_tree(self, tree: exp.Expr) -> frozenset[str]:
        nodes = list(tree.walk())
        if len(nodes) > self.max_nodes:
            raise GuardError(f"query too complex ({len(nodes)} > {self.max_nodes} nodes)")
        cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
        aliases = _defined_aliases(tree)
        shadowing = aliases & {c for cols in self.catalog.hidden.values() for c in cols}
        if shadowing:
            raise GuardError(f"alias {sorted(shadowing)[0]!r} shadows a hidden column")
        referenced: list[str] = []
        qualifiers = set(cte_names) | aliases
        for node in nodes:
            if isinstance(node, _FORBIDDEN):
                raise GuardError(f"{type(node).__name__.upper()} is not allowed")
            if isinstance(node, exp.With) and node.args.get("recursive"):
                raise GuardError("recursive CTEs are not allowed")
            if isinstance(node, exp.Func):
                _check_function(node)
            if isinstance(node, exp.Table):
                name = self._check_table(node, cte_names)
                if name not in cte_names:
                    referenced.append(name)
                qualifiers.add(name)
        if len(referenced) > self.max_tables:
            raise GuardError(f"at most {self.max_tables} table references are allowed")
        used = frozenset(referenced)
        allowed_cols = set(aliases)
        for t in used:
            allowed_cols |= self.catalog.columns(t)
        for node in nodes:
            if isinstance(node, exp.Column):
                self._check_column(node, qualifiers, allowed_cols)
            if isinstance(node, exp.Star) and not isinstance(node.parent, exp.Count):
                hidden = sorted(t for t in used if self.catalog.has_hidden(t))
                if hidden:
                    raise GuardError(
                        f"'SELECT *' is not allowed on {', '.join(hidden)}; list the columns"
                    )
        return used

    def _check_table(self, node: exp.Table, cte_names: set[str]) -> str:
        if not isinstance(node.this, exp.Identifier):
            raise GuardError("table functions are not allowed")
        if node.args.get("db") or node.args.get("catalog"):
            raise GuardError("schema-qualified table names are not allowed")
        name = node.name.lower()
        if name in cte_names:
            return name
        if name not in self.catalog.tables:
            raise GuardError(f"table {name!r} is not allowlisted")
        return name

    def _check_column(self, node: exp.Column, qualifiers: set[str], allowed: set[str]) -> None:
        if node.args.get("db") or node.args.get("catalog"):
            raise GuardError("schema-qualified column names are not allowed")
        table = node.table.lower()
        if table and table not in qualifiers:
            raise GuardError(f"unknown table or alias {table!r}")
        if isinstance(node.this, exp.Star):
            if table and self.catalog.has_hidden(table):
                raise GuardError(f"{table}.* is not allowed; list the columns")
            return
        name = node.name.lower()
        if name not in allowed:
            raise GuardError(f"column {name!r} is not allowlisted")

    # ------------------------------------------------------------------ params + limit

    @staticmethod
    def _placeholders(tree: exp.Expr) -> set[str]:
        names = set()
        for p in tree.find_all(exp.Placeholder):
            if not p.name:
                raise GuardError("use named parameters (:name), not '?'")
            names.add(p.name)
        return names

    def _check_params(self, names: set[str], params: Mapping[str, object]) -> dict[str, Scalar]:
        if len(params) > self.max_params:
            raise GuardError(f"at most {self.max_params} parameters")
        bound: dict[str, Scalar] = {}
        for key, value in params.items():
            if not isinstance(key, str) or not _PARAM_NAME.fullmatch(key):
                raise GuardError(f"bad parameter name {key!r}")
            if value is not None and not isinstance(value, str | int | float | bool):
                raise GuardError(f"parameter {key!r} must be a string, number, bool or null")
            if isinstance(value, str) and len(value) > self.max_param_chars:
                raise GuardError(f"parameter {key!r} is too long")
            bound[key] = value
        missing = names - bound.keys()
        if missing:
            raise GuardError(f"missing parameters: {', '.join(sorted(missing))}")
        unused = bound.keys() - names
        if unused:
            raise GuardError(f"unused parameters: {', '.join(sorted(unused))}")
        return bound

    def _enforce_limit(self, tree: exp.Expr, cap: int) -> None:
        offset = tree.args.get("offset")
        if offset is not None:
            value = _int_literal(offset.expression)
            if value is None or value > self.max_offset:
                raise GuardError(f"OFFSET must be a number <= {self.max_offset}")
        limit = tree.args.get("limit")
        current = None
        if limit is not None:
            current = _int_literal(limit.expression) if isinstance(limit, exp.Limit) else None
            if current is None:
                raise GuardError("LIMIT must be a plain number")
        n = cap if current is None else min(current, cap)
        tree.set("limit", exp.Limit(expression=exp.Literal.number(n)))


def _is_select_tree(node: exp.Expr) -> bool:
    if isinstance(node, exp.Select):
        return True
    if isinstance(node, exp.SetOperation):
        return _is_select_tree(node.this) and _is_select_tree(node.expression)
    if isinstance(node, exp.Subquery):
        return _is_select_tree(node.this)
    return False


def _check_function(node: exp.Func) -> None:
    if isinstance(node, exp.Anonymous):
        if node.name.lower() not in _ALLOWED_ANONYMOUS:
            raise GuardError(f"function {node.name.lower()}() is not allowed")
        return
    if type(node).__name__ not in ALLOWED_FUNCTIONS:
        raise GuardError(f"function {node.sql_name().lower()}() is not allowed")


def _defined_aliases(tree: exp.Expr) -> set[str]:
    names: set[str] = set()
    for a in tree.find_all(exp.Alias):
        names.add(a.alias.lower())
    for t in tree.find_all(exp.TableAlias):
        if t.name:
            names.add(t.name.lower())
        for col in t.columns:
            names.add(col.name.lower())
    return names


def _int_literal(node: exp.Expr | None) -> int | None:
    if isinstance(node, exp.Literal) and not node.is_string:
        try:
            value = int(node.this)
        except ValueError:
            return None
        return value if value >= 0 else None
    return None
