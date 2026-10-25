"""The bundled TAG-Bench-style benchmark: questions, gold plans, gold answers, grading."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from dataclasses import field
from functools import cache
from pathlib import Path
from typing import Any
from typing import Literal

from tag_agent.plan import TagPlan

__all__ = ["BENCH_FILE", "BenchItem", "grade", "load_items", "normalise"]

BENCH_FILE = Path(__file__).parent / "data" / "bench.json"

Render = Literal["first", "list", "scalar", "max_label", "scripted"]


@dataclass(frozen=True)
class BenchItem:
    id: str
    category: Literal["match", "comparison", "ranking", "aggregation"]
    needs: Literal["none", "knowledge", "reasoning"]
    question: str
    plan: TagPlan
    render: Render
    answer: str | float | list[str]
    exclude: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    evidence: str = ""
    evidence_table: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@cache
def _load(path: Path) -> tuple[BenchItem, ...]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = []
    for raw in data["items"]:
        raw = dict(raw)
        plan = TagPlan.model_validate(raw.pop("plan"))
        items.append(
            BenchItem(
                plan=plan,
                exclude=tuple(raw.pop("exclude", ())),
                keywords=tuple(raw.pop("keywords", ())),
                **raw,
            )
        )
    return tuple(items)


def load_items(path: Path = BENCH_FILE) -> tuple[BenchItem, ...]:
    return _load(path)


def normalise(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return " ".join(re.findall(r"[a-z0-9@]+(?:\.[a-z0-9]+)*", folded.lower().replace("'", "")))


_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def grade(answer: str, item: BenchItem) -> bool:
    """Exact-match style grading, as in TAG-Bench: every gold value present, no distractor.

    Numbers must be the only number in the answer (within 0.01). Summaries need at least
    two of the item's keywords and none of its excluded strings."""
    text = normalise(answer)
    padded = f" {text} "
    for excluded in item.exclude:
        x = normalise(excluded)
        if (x in text) if x.startswith("@") else (f" {x} " in padded):
            return False
    if item.render == "scripted":
        hits = sum(normalise(k) in text for k in item.keywords)
        return hits >= 2
    gold = item.answer
    if isinstance(gold, int | float) and not isinstance(gold, bool):
        nums = [float(n) for n in _NUMBER.findall(text.replace(",", ""))]
        return len(nums) == 1 and abs(nums[0] - float(gold)) <= 0.01
    golds = gold if isinstance(gold, list) else [str(gold)]
    return all(f" {normalise(g)} " in padded for g in golds)
