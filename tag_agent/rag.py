"""The naive RAG-over-rows baseline: every row becomes a text chunk, BM25 picks the top k.

This mirrors the TAG paper's RAG baseline (row-level retrieval, top 10 rows) with a
lexical retriever instead of embeddings, so it runs offline with no model. Hidden
columns are never indexed: rows are read through the same guard and allowlist.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from sqlglot import exp

from tag_agent.db import ReadOnlyDatabase
from tag_agent.table import Table
from tag_agent.table import format_value

__all__ = ["RowIndex", "tokenize"]

_STOP = frozenset(
    {"a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from"}
    | {"has", "have", "how", "in", "is", "it", "many", "more", "most", "of", "on", "or"}
    | {"set", "that", "the", "their", "this", "to", "was", "were", "what", "which", "who"}
    | {"with"}
)


def _stem(token: str) -> str:
    """A deliberately tiny plural folder (comedies -> comedy, reviews -> review)."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


@dataclass
class RowIndex:
    keys: list[str]
    docs: list[str]
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        self._tokens = [tokenize(d) for d in self.docs]
        self._tf = [Counter(t) for t in self._tokens]
        n = len(self.docs)
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self._avg = sum(len(t) for t in self._tokens) / n if n else 0.0

    @classmethod
    def build(cls, db: ReadOnlyDatabase, max_rows_per_table: int = 5000) -> RowIndex:
        keys: list[str] = []
        docs: list[str] = []
        guard = db.guard(max_rows_per_table)
        for name, spec in db.catalog.tables.items():
            cols = [c.name for c in spec.columns]
            sql = exp.select(*cols).from_(name).sql(dialect="sqlite")
            table = db.execute(guard.check(sql))
            for n, record in enumerate(table.records()):
                key = f"{name}:{record.get('id', n)}"
                body = "; ".join(f"{c}={format_value(v)}" for c, v in record.items())
                keys.append(key)
                docs.append(f"{name}: {body}")
        return cls(keys, docs)

    def search(self, query: str, k: int = 10) -> list[tuple[str, str, float]]:
        terms = tokenize(query)
        scored = []
        for i, tf in enumerate(self._tf):
            length = len(self._tokens[i])
            score = 0.0
            for t in terms:
                f = tf.get(t, 0)
                if f:
                    norm = f + self.k1 * (1 - self.b + self.b * length / (self._avg or 1))
                    score += self._idf.get(t, 0.0) * f * (self.k1 + 1) / norm
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda s: (-s[0], s[1]))
        return [(self.keys[i], self.docs[i], s) for s, i in scored[:k]]

    def table(self, query: str, k: int = 10) -> Table:
        return Table(["row"], [(doc,) for _, doc, _ in self.search(query, k)])
