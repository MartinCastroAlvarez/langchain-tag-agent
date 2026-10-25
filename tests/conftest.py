"""Shared fixtures: the bundled read-only database, a fake-model pipeline, the benchmark."""

from __future__ import annotations

import logging

import pytest

from tag_agent.benchmark import BenchItem
from tag_agent.benchmark import load_items
from tag_agent.catalog import MOVIES_CATALOG
from tag_agent.config import Settings
from tag_agent.db import DEFAULT_DB
from tag_agent.db import ReadOnlyDatabase
from tag_agent.fake import FakeTagModel
from tag_agent.pipeline import TagPipeline

logging.getLogger("sqlglot").setLevel(logging.ERROR)

ITEMS: tuple[BenchItem, ...] = load_items()


@pytest.fixture(scope="session")
def db() -> ReadOnlyDatabase:
    return ReadOnlyDatabase(DEFAULT_DB, MOVIES_CATALOG)


@pytest.fixture
def settings() -> Settings:
    return Settings(provider="fake")


@pytest.fixture
def fake() -> FakeTagModel:
    return FakeTagModel()


@pytest.fixture
def pipeline(fake: FakeTagModel, db: ReadOnlyDatabase, settings: Settings) -> TagPipeline:
    return TagPipeline(fake, db, settings)


def item(item_id: str) -> BenchItem:
    return next(i for i in ITEMS if i.id == item_id)
