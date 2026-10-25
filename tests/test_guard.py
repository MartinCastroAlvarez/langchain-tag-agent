"""The SQL guard (static layer) and the database (engine layer), attacked separately."""

from __future__ import annotations

import json

import pytest

from tag_agent.bench import adversarial_queries
from tag_agent.benchmark import BenchItem
from tag_agent.catalog import MOVIES_CATALOG
from tag_agent.catalog import Catalog
from tag_agent.catalog import ColumnSpec
from tag_agent.catalog import TableSpec
from tag_agent.db import QueryError
from tag_agent.db import ReadOnlyDatabase
from tag_agent.guard import GuardError
from tag_agent.guard import SafeQuery
from tag_agent.guard import SqlGuard
from tests.conftest import ITEMS

ADVERSARIAL = adversarial_queries()


def test_the_corpus_is_large_and_covers_the_brief() -> None:
    names = " ".join(a["name"] for a in ADVERSARIAL)
    assert len(ADVERSARIAL) >= 45
    for word in (
        "drop",
        "multi-statement",
        "attach",
        "read_csv",
        "readfile",
        "recursive",
        "comment",
    ):
        assert word in names


@pytest.mark.parametrize("case", ADVERSARIAL, ids=[a["name"] for a in ADVERSARIAL])
def test_guard_rejects_every_adversarial_query(db: ReadOnlyDatabase, case: dict[str, str]) -> None:
    with pytest.raises(GuardError):
        db.guard().check(case["sql"])


ENGINE_MUST_DENY = [
    a
    for a in ADVERSARIAL
    if a["name"] not in {"schema-qualified table", "cartesian join bomb", "positional placeholder"}
]


@pytest.mark.parametrize("case", ENGINE_MUST_DENY, ids=[a["name"] for a in ENGINE_MUST_DENY])
def test_engine_denies_them_even_without_the_guard(
    db: ReadOnlyDatabase, case: dict[str, str]
) -> None:
    """Defense in depth: hand the raw text straight to the database, skipping the guard."""
    with pytest.raises(QueryError):
        db.execute(SafeQuery(case["sql"]))


def test_engine_still_answers_after_attacks(db: ReadOnlyDatabase) -> None:
    for case in ENGINE_MUST_DENY:
        with pytest.raises(QueryError):
            db.execute(SafeQuery(case["sql"]))
    assert db.query("SELECT count(*) FROM movies").rows == [(28,)]
    assert db.query("SELECT count(*) FROM reviews").rows == [(23,)]


@pytest.mark.parametrize("bench", ITEMS, ids=[i.id for i in ITEMS])
def test_every_gold_query_is_accepted_and_runs(db: ReadOnlyDatabase, bench: BenchItem) -> None:
    plan = bench.plan
    table = db.execute(db.guard().check(plan.sql, plan.params))
    assert table.rows


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM movies",
        "SELECT m.title, r.stars FROM movies AS m JOIN reviews AS r ON r.movie_id = m.id",
        "SELECT genre, count(*) AS n FROM movies GROUP BY genre HAVING n > 1 ORDER BY n DESC",
        "WITH top AS (SELECT title, rating FROM movies ORDER BY rating DESC LIMIT 3) SELECT title FROM top",
        "SELECT title FROM movies WHERE id IN (SELECT movie_id FROM reviews WHERE stars = 1)",
        "SELECT title FROM movies UNION SELECT reviewer FROM reviews",
        "SELECT title, RANK() OVER (PARTITION BY genre ORDER BY rating DESC) AS r FROM movies",
        "SELECT x.title FROM (SELECT title FROM movies) AS x",
        "SELECT lower(title), length(plot), round(avg(rating), 1), total(gross_musd) FROM movies",
        "SELECT title FROM movies WHERE title LIKE 'The %' AND genre GLOB 'D*'",
        "SELECT CASE WHEN rating > 8 THEN 'great' ELSE 'ok' END AS verdict FROM movies",
        "SELECT title FROM movies -- trailing comment",
        "SELECT count(*) FROM reviews",
        "SELECT title FROM movies;",
    ],
)
def test_accepts_ordinary_selects_and_they_run(db: ReadOnlyDatabase, sql: str) -> None:
    query = db.guard().check(sql)
    assert query.sql.upper().startswith(("SELECT", "WITH"))
    assert "--" not in query.sql and "/*" not in query.sql
    db.execute(query)


def test_comments_are_stripped_from_what_runs(db: ReadOnlyDatabase) -> None:
    q = db.guard().check("SELECT title /* ; DROP TABLE movies */ FROM movies -- x")
    assert q.sql == "SELECT title FROM movies LIMIT 100"


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT title FROM movies", 100),
        ("SELECT title FROM movies LIMIT 5", 5),
        ("SELECT title FROM movies LIMIT 100000", 100),
        ("SELECT title FROM movies UNION SELECT reviewer FROM reviews LIMIT 500", 100),
    ],
)
def test_limit_is_always_enforced(db: ReadOnlyDatabase, sql: str, expected: int) -> None:
    q = db.guard().check(sql)
    assert q.sql.endswith(f"LIMIT {expected}")
    assert q.max_rows == 100


def test_offset_must_be_a_small_literal(db: ReadOnlyDatabase) -> None:
    assert "OFFSET 2" in db.guard().check("SELECT title FROM movies LIMIT 3 OFFSET 2").sql
    with pytest.raises(GuardError, match="OFFSET"):
        db.guard().check("SELECT title FROM movies LIMIT 3 OFFSET 99999999")


def test_parameters_are_bound_not_interpolated(db: ReadOnlyDatabase) -> None:
    evil = "x' OR '1'='1"
    q = db.guard().check("SELECT title FROM movies WHERE title = :t", {"t": evil})
    assert evil not in q.sql and ":t" in q.sql
    assert db.execute(q).rows == []
    q = db.guard().check("SELECT title FROM movies WHERE title = :t", {"t": "Salt Lines"})
    assert db.execute(q).rows == [("Salt Lines",)]


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({}, "missing"),
        ({"t": "a", "u": "b"}, "unused"),
        ({"t": ["a"]}, "must be"),
        ({"t": "x" * 1001}, "too long"),
        ({"bad name": "x"}, "bad parameter"),
        ({f"p{i}": i for i in range(40)}, "at most"),
    ],
)
def test_parameter_validation(
    db: ReadOnlyDatabase, params: dict[str, object], message: str
) -> None:
    with pytest.raises(GuardError, match=message):
        db.guard().check("SELECT title FROM movies WHERE title = :t", params)


@pytest.mark.parametrize(
    ("sql", "message"),
    [
        ("", "empty"),
        ("   ", "empty"),
        ("SELECT 1\x00", "NUL"),
        ("SELECT title FROM movies LIMIT :n", "LIMIT must be"),
        ("SELECT title FROM movies LIMIT 1 + 1", "LIMIT must be"),
        ("SELECT nope FROM movies", "not allowlisted"),
        ("SELECT z.title FROM movies AS m", "unknown table or alias"),
        ("SELECT main.movies.title FROM movies", "schema-qualified"),
        ("SELECT title FROM movies m1, movies m2, movies m3, movies m4, movies m5", "at most 4"),
        ("SELECT " + " + ".join(["1"] * 500), "too complex"),
    ],
)
def test_other_rejections(db: ReadOnlyDatabase, sql: str, message: str) -> None:
    with pytest.raises(GuardError, match=message):
        db.guard().check(sql)


def test_guard_errors_are_safe_to_show() -> None:
    guard = SqlGuard(MOVIES_CATALOG)
    with pytest.raises(GuardError) as info:
        guard.check("SELECT secret FROM vault")
    assert "vault" in str(info.value) and "Traceback" not in str(info.value)


def test_star_is_fine_where_nothing_is_hidden(db: ReadOnlyDatabase) -> None:
    assert len(db.execute(db.guard().check("SELECT * FROM movies")).columns) == 11
    with pytest.raises(GuardError, match="list the columns"):
        db.guard().check("SELECT * FROM movies JOIN reviews ON reviews.movie_id = movies.id")


def test_catalog_identifiers_are_validated() -> None:
    with pytest.raises(ValueError, match="identifier"):
        Catalog.of(TableSpec("movies; DROP", (ColumnSpec("id"),)))
    with pytest.raises(ValueError, match="no allowlisted columns"):
        Catalog.of(TableSpec("movies", ()))


def test_schema_text_shows_no_data_and_no_hidden_column(db: ReadOnlyDatabase) -> None:
    text = db.catalog.schema_text()
    assert "reviewer_email TEXT" not in text and "@example.com" not in text
    assert "Glass Orchard" not in text
    assert "TABLE movies" in text and "setting_city TEXT" in text
    with_samples = db.catalog.schema_text({"movies": [{"title": "X"}]})
    assert "sample: {'title': 'X'}" in with_samples


def test_adversarial_file_is_valid_json() -> None:
    from tag_agent.bench import ADVERSARIAL_FILE

    data = json.loads(ADVERSARIAL_FILE.read_text(encoding="utf-8"))
    assert all(set(d) == {"name", "sql"} for d in data)
