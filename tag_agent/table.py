"""A small result table: what query execution returns and what the LM operators consume."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import Any

__all__ = ["Cell", "Table", "format_value"]

Cell = str | int | float | None


def format_value(value: Cell) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, float):
        if math.isfinite(value) and value == int(value):
            return str(int(value))
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


@dataclass
class Table:
    columns: list[str]
    rows: list[tuple[Cell, ...]] = field(default_factory=list)
    truncated: bool = False
    """True when the query had more rows than the cap (only the first ``max_rows`` kept)."""

    def __post_init__(self) -> None:
        width = len(self.columns)
        for row in self.rows:
            if len(row) != width:
                raise ValueError("row width does not match the columns")

    def __len__(self) -> int:
        return len(self.rows)

    def index(self, column: str) -> int:
        lowered = [c.lower() for c in self.columns]
        try:
            return lowered.index(column.lower())
        except ValueError:
            raise KeyError(f"no column {column!r}; columns are {', '.join(self.columns)}") from None

    def column(self, column: str) -> list[Cell]:
        i = self.index(column)
        return [r[i] for r in self.rows]

    def records(self) -> list[dict[str, Cell]]:
        return [dict(zip(self.columns, r, strict=True)) for r in self.rows]

    def take(self, rows: Sequence[tuple[Cell, ...]]) -> Table:
        return Table(list(self.columns), list(rows), self.truncated)

    def to_json(self, max_rows: int | None = None) -> str:
        rows = self.rows if max_rows is None else self.rows[:max_rows]
        payload: dict[str, Any] = {
            "columns": self.columns,
            "rows": [list(r) for r in rows],
            "row_count": len(self.rows),
        }
        if self.truncated:
            payload["truncated"] = True
        if max_rows is not None and len(self.rows) > max_rows:
            payload["shown"] = max_rows
        return json.dumps(payload, ensure_ascii=False, default=str)

    @classmethod
    def from_json(cls, text: str) -> Table:
        data = json.loads(text)
        return cls(
            [str(c) for c in data["columns"]],
            [tuple(r) for r in data["rows"]],
            bool(data.get("truncated", False)),
        )

    def summary(self) -> dict[str, Any]:
        """Deterministic column statistics, for answers over tables too big to show whole."""
        out: dict[str, Any] = {"row_count": len(self.rows), "columns": {}}
        for i, name in enumerate(self.columns):
            values = [r[i] for r in self.rows if r[i] is not None]
            nums = [v for v in values if isinstance(v, int | float) and not isinstance(v, bool)]
            stats: dict[str, Any] = {"non_null": len(values)}
            if nums and len(nums) == len(values):
                stats |= {
                    "min": min(nums),
                    "max": max(nums),
                    "mean": round(sum(nums) / len(nums), 4),
                    "sum": round(sum(nums), 4),
                }
            else:
                counts: dict[str, int] = {}
                for v in values:
                    key = str(v)[:80]
                    counts[key] = counts.get(key, 0) + 1
                top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
                stats |= {"distinct": len(counts), "top": top}
            out["columns"][name] = stats
        return out

    def answer_text(self) -> str:
        """The plain text2SQL answer: the single cell, or the first column as a list."""
        if not self.rows:
            return "no rows"
        if len(self.rows) == 1 and len(self.columns) == 1:
            return format_value(self.rows[0][0])
        return ", ".join(format_value(r[0]) for r in self.rows)
