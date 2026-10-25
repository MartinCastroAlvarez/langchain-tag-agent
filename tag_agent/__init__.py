"""Table-Augmented Generation (TAG) with LangChain over a read-only SQLite database."""

from tag_agent.agent import build_agent
from tag_agent.agent import run_agent
from tag_agent.catalog import MOVIES_CATALOG
from tag_agent.catalog import Catalog
from tag_agent.config import Settings
from tag_agent.db import ReadOnlyDatabase
from tag_agent.fake import FakeTagModel
from tag_agent.guard import GuardError
from tag_agent.guard import SafeQuery
from tag_agent.guard import SqlGuard
from tag_agent.pipeline import TagPipeline
from tag_agent.pipeline import TagResult
from tag_agent.session import build_pipeline
from tag_agent.session import open_database

__all__ = [
    "MOVIES_CATALOG",
    "Catalog",
    "FakeTagModel",
    "GuardError",
    "ReadOnlyDatabase",
    "SafeQuery",
    "Settings",
    "SqlGuard",
    "TagPipeline",
    "TagResult",
    "build_agent",
    "build_pipeline",
    "open_database",
    "run_agent",
]
__version__ = "0.1.0"
