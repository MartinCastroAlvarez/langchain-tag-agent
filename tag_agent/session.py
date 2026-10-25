"""Wiring: settings -> database (+ allowlist) -> model -> pipeline."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from tag_agent.catalog import MOVIES_CATALOG
from tag_agent.catalog import Catalog
from tag_agent.config import Settings
from tag_agent.db import DEFAULT_DB
from tag_agent.db import ReadOnlyDatabase
from tag_agent.llm import make_model
from tag_agent.pipeline import TagPipeline

__all__ = ["build_pipeline", "open_database"]


def open_database(settings: Settings) -> ReadOnlyDatabase:
    """The bundled database with its curated allowlist, or ``TAG_DB`` introspected and
    narrowed by ``TAG_TABLES`` / ``TAG_HIDDEN_COLUMNS``."""
    if settings.db_path is None:
        return ReadOnlyDatabase(DEFAULT_DB, MOVIES_CATALOG, timeout=settings.timeout)
    path = Path(settings.db_path).expanduser().resolve(strict=True)
    conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    try:
        catalog = Catalog.introspect(conn, settings.tables or None, settings.hidden_columns)
    finally:
        conn.close()
    return ReadOnlyDatabase(path, catalog, timeout=settings.timeout)


def build_pipeline(settings: Settings) -> TagPipeline:
    model = make_model(settings)
    return TagPipeline(model, open_database(settings), settings)
