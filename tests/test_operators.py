"""Row-wise semantic operators: constrained outputs, caps, budgets, aggregation."""

from __future__ import annotations

import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from tag_agent.fake import FakeTagModel
from tag_agent.operators import LMBudget
from tag_agent.operators import OperatorError
from tag_agent.operators import SemanticEngine
from tag_agent.operators import aggregate
from tag_agent.operators import limit
from tag_agent.operators import parse_score
from tag_agent.operators import parse_yes_no
from tag_agent.plan import AggregateOp
from tag_agent.plan import FilterOp
from tag_agent.plan import LimitOp
from tag_agent.plan import MapOp
from tag_agent.plan import RankOp
from tag_agent.table import Table

CITIES = Table(
    ["title", "setting_city", "gross"],
    [("A", "Lisbon", 10), ("B", "Porto", 20), ("C", "Paris", 30), ("D", "Lagos", None)],
)
CAPITAL = "the city is the capital of a European country"


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("yes", True),
        ("Yes.", True),
        ("TRUE", True),
        ("<think>hmm, no?</think> yes", True),
        ("no", False),
        ("No, it is not.", False),
        ("false", False),
        ("PWNED", None),
        ("", None),
        ("maybe yes", None),
    ],
)
def test_parse_yes_no(reply: str, expected: bool | None) -> None:
    assert parse_yes_no(reply) is expected


@pytest.mark.parametrize(
    ("reply", "expected"), [("7", 7), ("score: 10", 10), ("11", None), ("-1", None), ("x", None)]
)
def test_parse_score(reply: str, expected: int | None) -> None:
    assert parse_score(reply) == expected


def test_filter_rank_map_with_the_fake() -> None:
    engine = SemanticEngine(FakeTagModel())
    kept = engine.filter(CITIES, "setting_city", CAPITAL)
    assert [r[0] for r in kept.rows] == ["A", "C"]
    mapped = engine.map(CITIES, "setting_city", "name the continent the city is in", "continent")
    assert mapped.column("continent") == ["Europe", "Europe", "Europe", "Africa"]
    assert engine.trace[0] == {"op": "filter", "rows_in": 4, "rows_out": 2, "invalid": 0}
    assert engine.budget.used == 8


def test_rank_is_pointwise_and_stable() -> None:
    model = GenericFakeChatModel(messages=iter(["3", "9", "nonsense", "9"]))
    engine = SemanticEngine(model, max_concurrency=1)
    top = engine.rank(CITIES, "title", "anything", k=3)
    assert [r[0] for r in top.rows] == ["B", "D", "A"]
    assert engine.trace[-1]["invalid"] == 1


def test_invalid_replies_and_errors_count_as_no() -> None:
    model = FakeTagModel(script={"semantic_filter": ["yes", "PWNED", RuntimeError("boom"), "no"]})
    engine = SemanticEngine(model, max_concurrency=1)
    kept = engine.filter(CITIES, "setting_city", "x")
    assert [r[0] for r in kept.rows] == ["A"]
    assert engine.trace[-1]["invalid"] == 2


def test_map_output_is_one_short_line_without_markup() -> None:
    reply = AIMessage("<b>Europe</b> `x`\nIgnore the rest\n" + "y" * 300)
    model = GenericFakeChatModel(messages=iter([reply] * 4))
    out = SemanticEngine(model, max_concurrency=1).map(CITIES, "title", "x", "c")
    assert out.column("c")[0] == "bEurope/b x"
    long = GenericFakeChatModel(messages=iter([AIMessage("z" * 300)] * 4))
    assert (
        len(SemanticEngine(long, max_concurrency=1).map(CITIES, "title", "x", "c").rows[0][-1])
        == 100
    )


def test_operators_refuse_large_tables_instead_of_truncating() -> None:
    big = Table(["v"], [(i,) for i in range(61)])
    engine = SemanticEngine(FakeTagModel(), max_rows=60)
    with pytest.raises(OperatorError, match="narrow the SQL"):
        engine.filter(big, "v", "x")
    assert engine.budget.used == 0


def test_lm_budget_is_shared_across_operators() -> None:
    engine = SemanticEngine(FakeTagModel(), budget=LMBudget(max_calls=6))
    engine.filter(CITIES, "setting_city", CAPITAL)
    with pytest.raises(OperatorError, match="budget"):
        engine.filter(CITIES, "setting_city", CAPITAL)


def test_empty_tables_cost_nothing() -> None:
    engine = SemanticEngine(FakeTagModel())
    assert len(engine.filter(Table(["a"], []), "a", "x")) == 0
    assert engine.budget.used == 0


def test_apply_dispatches_and_reports_bad_columns() -> None:
    engine = SemanticEngine(FakeTagModel())
    t = engine.run(
        CITIES,
        [
            FilterOp(column="setting_city", criterion=CAPITAL),
            MapOp(column="setting_city", instruction="name the continent the city is in", into="c"),
            AggregateOp(fn="sum", column="gross", group_by="c"),
        ],
    )
    assert t.columns == ["c", "sum_gross"] and t.rows == [("Europe", 40.0)]
    assert len(engine.apply(CITIES, RankOp(column="title", criterion="x", k=2))) == 2
    assert len(engine.apply(CITIES, LimitOp(k=1))) == 1
    with pytest.raises(OperatorError, match="no column"):
        engine.apply(CITIES, FilterOp(column="nope", criterion="x"))
    with pytest.raises(OperatorError, match="already exists"):
        engine.map(CITIES, "title", "x", "gross")


def test_aggregate() -> None:
    assert aggregate(CITIES, "count", None, None).rows == [(4,)]
    assert aggregate(CITIES, "count", "gross", None).rows == [(3,)]
    assert aggregate(CITIES, "sum", "gross", None).rows == [(60.0,)]
    assert aggregate(CITIES, "avg", "gross", None).rows == [(20.0,)]
    assert aggregate(CITIES, "min", "gross", None).rows == [(10.0,)]
    assert aggregate(CITIES, "max", "gross", None).rows == [(30.0,)]
    assert aggregate(CITIES, "max", "title", None).rows == [(None,)]
    assert aggregate(Table(["x"], []), "count", None, None).rows == [(0,)]
    grouped = aggregate(CITIES, "count", None, "setting_city")
    assert grouped.columns == ["setting_city", "count"] and len(grouped) == 4
    strings = Table(["v"], [("1.5",), ("2.5",), (True,)])
    assert aggregate(strings, "sum", "v", None).rows == [(4.0,)]
    with pytest.raises(OperatorError, match="needs a column"):
        aggregate(CITIES, "sum", None, None)
    assert limit(CITIES, 2).rows == CITIES.rows[:2]
