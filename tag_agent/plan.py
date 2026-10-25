"""The query plan the synthesizer writes: SQL + bound params + semantic operators."""

from __future__ import annotations

import json
import re
from typing import Annotated
from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import ValidationError

from tag_agent.guard import Scalar

__all__ = [
    "AggregateOp",
    "FilterOp",
    "LimitOp",
    "MapOp",
    "Op",
    "PlanError",
    "RankOp",
    "TagPlan",
    "extract_json",
    "parse_plan",
]

_Name = Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]
_Text = Annotated[str, Field(min_length=1, max_length=300)]


class _Op(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FilterOp(_Op):
    op: Literal["filter"] = "filter"
    column: _Name
    criterion: _Text


class RankOp(_Op):
    op: Literal["rank"] = "rank"
    column: _Name
    criterion: _Text
    k: int = Field(default=3, ge=1, le=50)


class MapOp(_Op):
    op: Literal["map"] = "map"
    column: _Name
    instruction: _Text
    into: _Name


class LimitOp(_Op):
    op: Literal["limit"] = "limit"
    k: int = Field(ge=1, le=1000)


class AggregateOp(_Op):
    op: Literal["aggregate"] = "aggregate"
    fn: Literal["count", "sum", "avg", "min", "max"]
    column: _Name | None = None
    group_by: _Name | None = None


Op = Annotated[FilterOp | RankOp | MapOp | LimitOp | AggregateOp, Field(discriminator="op")]


class TagPlan(BaseModel):
    """syn(R) -> Q: one guarded SELECT, then up to six row-wise / table operators."""

    model_config = ConfigDict(extra="forbid")

    sql: str = Field(min_length=1, max_length=4000)
    params: dict[str, Scalar] = Field(default_factory=dict, max_length=32)
    ops: list[Op] = Field(default_factory=list, max_length=6)
    answer: Literal["generate", "table"] = "generate"


class PlanError(ValueError):
    """The model's reply is not a valid plan."""


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> Any:
    """The first JSON object in a model reply (tolerates <think>, code fences, prose)."""
    cleaned = _THINK.sub("", text or "")
    decoder = json.JSONDecoder()
    for i, ch in enumerate(cleaned):
        if ch == "{":
            try:
                value, _ = decoder.raw_decode(cleaned, i)
            except json.JSONDecodeError:
                continue
            return value
    raise PlanError("no JSON object in the reply")


def parse_plan(text: str, *, allow_ops: bool = True) -> TagPlan:
    data = extract_json(text)
    try:
        plan = TagPlan.model_validate(data)
    except ValidationError as exc:
        errors = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:5]
        )
        raise PlanError(f"invalid plan: {errors}") from exc
    if plan.ops and not allow_ops:
        plan = plan.model_copy(update={"ops": []})
    return plan
