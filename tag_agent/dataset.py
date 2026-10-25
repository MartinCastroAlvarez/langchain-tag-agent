"""Build the bundled SQLite file from ``data/movies.json`` (the human-readable source).

``python -m tag_agent.dataset`` rebuilds ``data/movies.sqlite``; ``--check`` verifies the
committed file still matches the JSON (CI runs it). The data is fictional: invented films,
directors and reviewers, set in real cities so that questions can need world knowledge.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
from collections.abc import Iterator
from pathlib import Path

__all__ = ["DATA_DIR", "build", "dump", "main"]

DATA_DIR = Path(__file__).parent / "data"
SOURCE = DATA_DIR / "movies.json"
TARGET = DATA_DIR / "movies.sqlite"

_SCHEMA = """
CREATE TABLE movies (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    year INTEGER NOT NULL,
    genre TEXT NOT NULL,
    director TEXT NOT NULL,
    setting_city TEXT NOT NULL,
    runtime_min INTEGER NOT NULL,
    budget_musd REAL NOT NULL,
    gross_musd REAL NOT NULL,
    rating REAL NOT NULL,
    plot TEXT NOT NULL
);
CREATE TABLE reviews (
    id INTEGER PRIMARY KEY,
    movie_id INTEGER NOT NULL REFERENCES movies(id),
    reviewer TEXT NOT NULL,
    reviewer_email TEXT NOT NULL,
    stars INTEGER NOT NULL CHECK (stars BETWEEN 1 AND 5),
    text TEXT NOT NULL,
    posted TEXT NOT NULL
);
"""
_MOVIE_COLS = (
    "id",
    "title",
    "year",
    "genre",
    "director",
    "setting_city",
    "runtime_min",
    "budget_musd",
    "gross_musd",
    "rating",
    "plot",
)
_REVIEW_COLS = ("id", "movie_id", "reviewer", "reviewer_email", "stars", "text", "posted")


def build(target: Path = TARGET, source: Path = SOURCE) -> Path:
    data = json.loads(source.read_text(encoding="utf-8"))
    target.unlink(missing_ok=True)
    conn = sqlite3.connect(target)
    try:
        conn.executescript(_SCHEMA)
        conn.executemany(
            "INSERT INTO movies VALUES (:id, :title, :year, :genre, :director, :setting_city, "
            ":runtime_min, :budget_musd, :gross_musd, :rating, :plot)",
            [{k: m[k] for k in _MOVIE_COLS} for m in data["movies"]],
        )
        conn.executemany(
            "INSERT INTO reviews VALUES (:id, :movie_id, :reviewer, :reviewer_email, :stars, "
            ":text, :posted)",
            [{k: r[k] for k in _REVIEW_COLS} for r in data["reviews"]],
        )
        conn.commit()
        conn.execute("VACUUM")
    finally:
        conn.close()
    return target


def dump(path: Path) -> list[str]:
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        lines: Iterator[str] = conn.iterdump()
        return list(lines)
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tag_agent.dataset", description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify, don't write")
    args = parser.parse_args(argv)
    if not args.check:
        print(f"wrote {build()}")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        fresh = build(Path(tmp) / "movies.sqlite")
        same = dump(fresh) == dump(TARGET)
    print("movies.sqlite matches movies.json" if same else "movies.sqlite is stale: rebuild it")
    return 0 if same else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
