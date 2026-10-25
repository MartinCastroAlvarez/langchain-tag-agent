"""Settings, read from ``TAG_*`` environment variables."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

__all__ = ["DEFAULT_MODELS", "MODES", "PROVIDERS", "Mode", "Provider", "Settings"]

Provider = Literal["ollama", "anthropic", "fake"]
PROVIDERS: tuple[Provider, ...] = ("ollama", "anthropic", "fake")
Mode = Literal["tag", "agent", "text2sql", "text2sql_lm", "rag"]
MODES: tuple[Mode, ...] = ("text2sql", "text2sql_lm", "rag", "tag", "agent")

DEFAULT_MODELS: dict[Provider, str] = {
    "ollama": "qwen3:4b",
    "anthropic": "claude-sonnet-5-5",
    "fake": "fake-tag",
}


def _int(env: Mapping[str, str], key: str, default: int, minimum: int = 0) -> int:
    raw = env.get(key, "")
    value = int(raw) if raw else default
    if value < minimum:
        raise ValueError(f"{key} must be >= {minimum}, got {value}")
    return value


def _float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key, "")
    value = float(raw) if raw else default
    if value < 0:
        raise ValueError(f"{key} must be >= 0, got {value}")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    provider: Provider = "ollama"
    model: str = DEFAULT_MODELS["ollama"]
    ollama_host: str = "http://127.0.0.1:11434"
    allow_remote_ollama: bool = False
    temperature: float = 0.0
    db_path: str | None = None
    """A SQLite file; ``None`` = the bundled movies database with its curated allowlist."""
    tables: tuple[str, ...] = ()
    """Allowlisted tables for a custom database (default: all user tables)."""
    hidden_columns: tuple[str, ...] = ()
    """``table.column`` entries a custom database must never expose."""
    max_rows: int = 100
    """Row cap enforced on every query (LIMIT rewrite + fetch cap)."""
    timeout: float = 2.0
    """Seconds before a running query is interrupted."""
    max_lm_rows: int = 60
    """Rows one semantic operator may judge (one LM call per row)."""
    max_lm_calls: int = 120
    """LM calls a question may spend on semantic operators."""
    max_concurrency: int = 4
    """Parallel LM calls inside one semantic operator."""
    max_tool_calls: int = 12
    """Tool calls the agent may make per question."""
    answer_rows: int = 30
    """Rows shown verbatim to answer generation; larger tables also get a summary."""
    schema_samples: int = 0
    """Sample rows per table in the schema prompt. 0 (default) = no data in the prompt."""
    rag_k: int = 10
    """Rows the RAG baseline retrieves (the TAG paper uses 10)."""

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        e = os.environ if env is None else env
        provider = (e.get("TAG_PROVIDER") or "ollama").strip().lower()
        if provider not in PROVIDERS:
            raise ValueError(f"TAG_PROVIDER must be one of {PROVIDERS}")
        p: Provider = provider
        return cls(
            provider=p,
            model=e.get("TAG_MODEL") or DEFAULT_MODELS[p],
            ollama_host=e.get("OLLAMA_HOST") or "http://127.0.0.1:11434",
            allow_remote_ollama=e.get("TAG_ALLOW_REMOTE_OLLAMA", "") == "1",
            temperature=_float(e, "TAG_TEMPERATURE", 0.0),
            db_path=e.get("TAG_DB") or None,
            tables=_list(e.get("TAG_TABLES", "")),
            hidden_columns=_list(e.get("TAG_HIDDEN_COLUMNS", "")),
            max_rows=_int(e, "TAG_MAX_ROWS", 100, 1),
            timeout=_float(e, "TAG_TIMEOUT", 2.0),
            max_lm_rows=_int(e, "TAG_MAX_LM_ROWS", 60, 1),
            max_lm_calls=_int(e, "TAG_MAX_LM_CALLS", 120, 1),
            max_concurrency=_int(e, "TAG_MAX_CONCURRENCY", 4, 1),
            max_tool_calls=_int(e, "TAG_MAX_TOOL_CALLS", 12, 1),
            answer_rows=_int(e, "TAG_ANSWER_ROWS", 30, 1),
            schema_samples=_int(e, "TAG_SCHEMA_SAMPLES", 0),
            rag_k=_int(e, "TAG_RAG_K", 10, 1),
        )


def _list(raw: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in raw.split(",") if x.strip())
