"""Plan parsing and the result table."""

from __future__ import annotations

import json

import pytest

from tag_agent.plan import PlanError
from tag_agent.plan import extract_json
from tag_agent.plan import parse_plan
from tag_agent.table import Table
from tag_agent.table import format_value


@pytest.mark.parametrize(
    "reply",
    [
        '{"sql": "SELECT 1"}',
        '```json\n{"sql": "SELECT 1"}\n```',
        'Sure! Here is the plan: {"sql": "SELECT 1"} hope it helps',
        '<think>{"sql": "DROP"}</think>{"sql": "SELECT 1"}',
        '{broken {"sql": "SELECT 1"}',
    ],
)
def test_extract_json_tolerates_wrappers(reply: str) -> None:
    assert extract_json(reply) == {"sql": "SELECT 1"}


def test_parse_plan_validates_ops() -> None:
    plan = parse_plan(
        json.dumps(
            {
                "sql": "SELECT title FROM movies",
                "params": {"g": "Drama"},
                "ops": [
                    {"op": "filter", "column": "title", "criterion": "x"},
                    {"op": "rank", "column": "title", "criterion": "y", "k": 2},
                    {"op": "aggregate", "fn": "count"},
                ],
            }
        )
    )
    assert [o.op for o in plan.ops] == ["filter", "rank", "aggregate"]
    assert parse_plan(plan.model_dump_json(), allow_ops=False).ops == []


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        ("no json here", "no JSON"),
        ('{"sql": ""}', "invalid plan"),
        ('{"sql": "SELECT 1", "ops": [{"op": "exec", "code": "x"}]}', "invalid plan"),
        (
            '{"sql": "SELECT 1", "ops": [{"op": "filter", "column": "a;b", "criterion": "x"}]}',
            "invalid",
        ),
        ('{"sql": "SELECT 1", "extra": 1}', "invalid plan"),
        ('{"sql": "SELECT 1", "params": {"a": [1]}}', "invalid plan"),
        (
            '{"sql": "SELECT 1", "ops": [' + ",".join(['{"op": "limit", "k": 1}'] * 7) + "]}",
            "invalid",
        ),
    ],
)
def test_parse_plan_rejects(reply: str, message: str) -> None:
    with pytest.raises(PlanError, match=message):
        parse_plan(reply)


def test_extract_json_needs_an_object() -> None:
    with pytest.raises(PlanError, match="no JSON object"):
        parse_plan("[1, 2]")


def test_table_basics() -> None:
    t = Table(["a", "b"], [(1, "x"), (2.5, None)])
    assert t.records() == [{"a": 1, "b": "x"}, {"a": 2.5, "b": None}]
    assert t.column("B") == ["x", None]
    assert Table.from_json(t.to_json()).rows == t.rows
    shown = json.loads(t.to_json(1))
    assert shown["shown"] == 1 and shown["row_count"] == 2
    assert json.loads(Table(["a"], [], truncated=True).to_json())["truncated"] is True
    with pytest.raises(ValueError, match="width"):
        Table(["a"], [(1, 2)])
    with pytest.raises(KeyError):
        t.index("zzz")


def test_table_summary_and_answer_text() -> None:
    t = Table(["n", "s"], [(1, "x"), (3, "x"), (2, "y")])
    s = t.summary()
    assert s["row_count"] == 3
    assert s["columns"]["n"] == {"non_null": 3, "min": 1, "max": 3, "mean": 2.0, "sum": 6}
    assert s["columns"]["s"]["top"][0] == ("x", 2)
    assert t.answer_text() == "1, 3, 2"
    assert Table(["n"], [(7.0,)]).answer_text() == "7"
    assert Table(["n"], []).answer_text() == "no rows"


@pytest.mark.parametrize(
    ("value", "text"),
    [(None, "NULL"), (2.0, "2"), (2.3333333, "2.3333"), (2.5, "2.5"), ("x", "x"), (3, "3")],
)
def test_format_value(value: object, text: str) -> None:
    assert format_value(value) == text  # type: ignore[arg-type]
