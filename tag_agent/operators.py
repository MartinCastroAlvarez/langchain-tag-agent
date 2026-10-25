"""LM-UDF-style semantic operators, run in Python over a bounded result table.

These are the TAG paper's hand-written-pipeline building blocks (LOTUS ``sem_filter``,
``sem_topk``, ``sem_map``), plus deterministic ``limit`` and ``aggregate``. Every LM call
judges **one row**, and its reply is parsed into a constrained value: ``yes``/``no``,
an integer 0-10, or one short line. A row whose text tries to give instructions can at
worst change its own label; it can't change the instructions for other rows, the SQL, or
the shape of the result. Rows above ``max_rows`` are refused rather than silently
truncated, and every call is charged to a per-question :class:`LMBudget`.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig

from tag_agent import prompts
from tag_agent.plan import FilterOp
from tag_agent.plan import LimitOp
from tag_agent.plan import MapOp
from tag_agent.plan import Op
from tag_agent.plan import RankOp
from tag_agent.table import Cell
from tag_agent.table import Table
from tag_agent.table import format_value

__all__ = [
    "LMBudget",
    "OperatorError",
    "SemanticEngine",
    "aggregate",
    "limit",
    "clean_reply",
    "parse_score",
    "parse_yes_no",
]

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_INT = re.compile(r"-?\d+")


class OperatorError(RuntimeError):
    """An operator could not run (bad column, too many rows, budget). Safe to show."""


@dataclass
class LMBudget:
    """LM calls a question may spend on semantic operators."""

    max_calls: int = 120
    used: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def take(self, n: int) -> None:
        with self._lock:
            if self.used + n > self.max_calls:
                raise OperatorError(
                    f"LM call budget exceeded ({self.used} used + {n} > {self.max_calls})"
                )
            self.used += n


def clean_reply(reply: object) -> str:
    text = reply if isinstance(reply, str) else str(reply)
    return _THINK.sub("", text).strip()


def parse_yes_no(reply: str) -> bool | None:
    words = re.findall(r"[a-z]+", clean_reply(reply).lower())
    if not words:
        return None
    if words[0] in {"yes", "true"}:
        return True
    if words[0] in {"no", "false"}:
        return False
    return None


def parse_score(reply: str) -> int | None:
    match = _INT.search(clean_reply(reply))
    if not match:
        return None
    value = int(match.group())
    return value if 0 <= value <= 10 else None


def _map_value(reply: str) -> str:
    line = next((ln.strip() for ln in clean_reply(reply).splitlines() if ln.strip()), "")
    return re.sub(r"[<>`]", "", line)[:100]


@dataclass
class SemanticEngine:
    model: BaseChatModel
    budget: LMBudget = field(default_factory=LMBudget)
    max_rows: int = 60
    max_concurrency: int = 4
    callbacks: list[Any] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------ LM plumbing

    def _ask(self, system: str, table: Table, column: str, task: dict[str, str]) -> list[str]:
        i = table.index(column)
        if len(table) > self.max_rows:
            raise OperatorError(
                f"{len(table)} rows is more than the semantic operator cap ({self.max_rows}); "
                "narrow the SQL first"
            )
        if not table.rows:
            return []
        self.budget.take(len(table))
        batch: list[LanguageModelInput] = []
        for row in table.rows:
            parts = [prompts.data_block(k, v) for k, v in task.items()]
            parts.append(prompts.data_block("value", format_value(row[i])))
            record = dict(zip(table.columns, row, strict=True))
            parts.append(prompts.data_block("row", json.dumps(record, ensure_ascii=False)))
            batch.append([SystemMessage(system), HumanMessage("\n".join(parts))])
        config: RunnableConfig = {"max_concurrency": self.max_concurrency}
        if self.callbacks:
            config["callbacks"] = self.callbacks
        replies = self.model.batch(batch, config=config, return_exceptions=True)
        return [r.text if isinstance(r, BaseMessage) else "" for r in replies]

    # ------------------------------------------------------------------ operators

    def filter(self, table: Table, column: str, criterion: str) -> Table:
        replies = self._ask(prompts.FILTER, table, column, {"criterion": criterion})
        verdicts = [parse_yes_no(r) for r in replies]
        kept = [row for row, ok in zip(table.rows, verdicts, strict=True) if ok]
        self._log("filter", table, len(kept), invalid=sum(v is None for v in verdicts))
        return table.take(kept)

    def rank(self, table: Table, column: str, criterion: str, k: int) -> Table:
        replies = self._ask(prompts.RANK, table, column, {"criterion": criterion})
        scores = [parse_score(r) for r in replies]
        order = sorted(range(len(table.rows)), key=lambda j: -(scores[j] or 0))
        kept = [table.rows[j] for j in order[:k]]
        self._log("rank", table, len(kept), invalid=sum(s is None for s in scores))
        return table.take(kept)

    def map(self, table: Table, column: str, instruction: str, into: str) -> Table:
        if into.lower() in (c.lower() for c in table.columns):
            raise OperatorError(f"column {into!r} already exists")
        replies = self._ask(prompts.MAP, table, column, {"instruction": instruction})
        values = [_map_value(r) for r in replies]
        rows = [(*row, v) for row, v in zip(table.rows, values, strict=True)]
        self._log("map", table, len(rows), invalid=sum(not v for v in values))
        return Table([*table.columns, into], rows, table.truncated)

    def apply(self, table: Table, op: Op) -> Table:
        try:
            if isinstance(op, FilterOp):
                return self.filter(table, op.column, op.criterion)
            if isinstance(op, RankOp):
                return self.rank(table, op.column, op.criterion, op.k)
            if isinstance(op, MapOp):
                return self.map(table, op.column, op.instruction, op.into)
            if isinstance(op, LimitOp):
                return limit(table, op.k)
            return aggregate(table, op.fn, op.column, op.group_by)
        except KeyError as exc:
            raise OperatorError(str(exc.args[0])) from exc

    def run(self, table: Table, ops: Sequence[Op]) -> Table:
        for op in ops:
            table = self.apply(table, op)
        return table

    def _log(self, op: str, table: Table, kept: int, *, invalid: int) -> None:
        self.trace.append({"op": op, "rows_in": len(table), "rows_out": kept, "invalid": invalid})


def limit(table: Table, k: int) -> Table:
    return table.take(table.rows[:k])


def _number(value: Cell) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(value)
    except ValueError:
        return None


def aggregate(table: Table, fn: str, column: str | None, group_by: str | None) -> Table:
    """count / sum / avg / min / max, optionally grouped (groups keep first-seen order)."""
    if fn != "count" and column is None:
        raise OperatorError(f"{fn} needs a column")
    vi = table.index(column) if column else None
    gi = table.index(group_by) if group_by else None
    groups: dict[Cell, list[tuple[Cell, ...]]] = {}
    for row in table.rows:
        groups.setdefault(row[gi] if gi is not None else None, []).append(row)
    if gi is None and not groups:
        groups[None] = []
    name = f"{fn}_{column}" if column else fn
    out: list[tuple[Cell, ...]] = []
    for key, rows in groups.items():
        value = _reduce(fn, [r[vi] for r in rows] if vi is not None else [1] * len(rows))
        out.append((key, value) if gi is not None else (value,))
    columns = [table.columns[gi], name] if gi is not None else [name]
    return Table(columns, out)


def _reduce(fn: str, values: Sequence[Cell]) -> Cell:
    if fn == "count":
        return sum(v is not None for v in values)
    nums = [n for n in (_number(v) for v in values) if n is not None]
    if not nums:
        return None
    if fn == "sum":
        return round(sum(nums), 4)
    if fn == "avg":
        return round(sum(nums) / len(nums), 4)
    return min(nums) if fn == "min" else max(nums)
