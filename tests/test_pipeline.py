"""The TAG pipeline and agent on the bundled benchmark, driven by the scripted fake."""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage

from tag_agent.agent import TagSession
from tag_agent.agent import build_agent
from tag_agent.agent import run_agent
from tag_agent.benchmark import BenchItem
from tag_agent.benchmark import grade
from tag_agent.config import Settings
from tag_agent.db import ReadOnlyDatabase
from tag_agent.fake import FakeTagModel
from tag_agent.pipeline import CallCounter
from tag_agent.pipeline import TagPipeline
from tag_agent.pipeline import schema_for
from tests.conftest import ITEMS
from tests.conftest import item

IDS = [i.id for i in ITEMS]


@pytest.mark.parametrize("bench", ITEMS, ids=IDS)
def test_tag_pipeline_answers_every_benchmark_question(
    pipeline: TagPipeline, bench: BenchItem
) -> None:
    result = pipeline.run(bench.question, "tag")
    assert result.status == "ok", result.answer
    assert grade(result.answer, bench), (bench.id, result.answer)
    # syn + gen, plus one LM call per row each semantic operator judged.
    judged = sum(e["rows_in"] for e in result.trace if e.get("op") in {"filter", "rank", "map"})
    assert result.lm_calls == 2 + judged


@pytest.mark.parametrize("bench", ITEMS, ids=IDS)
def test_tag_agent_answers_every_benchmark_question(
    pipeline: TagPipeline, bench: BenchItem
) -> None:
    result = run_agent(pipeline, bench.question)
    assert result.status == "ok"
    assert grade(result.answer, bench), (bench.id, result.answer)
    calls = result.trace[0]["calls"]
    assert calls[0] == "sql_query" and len(calls) == 1 + len(bench.plan.ops)


@pytest.mark.parametrize("bench", ITEMS, ids=IDS)
def test_plain_sql_answers_only_the_questions_that_need_no_lm(
    pipeline: TagPipeline, bench: BenchItem
) -> None:
    result = pipeline.run(bench.question, "text2sql")
    assert result.lm_calls == 1
    assert grade(result.answer, bench) is (bench.needs == "none")


def test_the_semantic_step_is_what_makes_the_difference(pipeline: TagPipeline) -> None:
    q = item("q01")
    tag = pipeline.run(q.question, "tag")
    sql = pipeline.run(q.question, "text2sql")
    assert tag.sql == sql.sql  # same query ...
    assert tag.answer == "The Last Tram"  # ... but only TAG filters by world knowledge
    assert sql.answer.startswith("Salt Lines")  # Barcelona is not a capital


def test_trace_records_each_tag_step(pipeline: TagPipeline) -> None:
    result = pipeline.run(item("q06").question, "tag")
    steps = [e.get("step") or e.get("op") for e in result.trace]
    assert steps == ["synthesize", "execute", "filter", "map"]
    assert result.table is not None and result.table.columns == ["continent", "sum_gross_musd"]
    assert dict(result.table.rows) == {"South America": 427.0, "Africa": 358.0}


def test_synthesis_gets_one_repair_round(db: ReadOnlyDatabase) -> None:
    fake = FakeTagModel(script={"synthesizer": ["not json", '{"sql": "DROP TABLE movies"}']})
    result = TagPipeline(fake, db).run(item("q04").question, "tag")
    assert result.status == "failed" and "only SELECT" in result.answer
    assert fake.calls == ["synthesizer", "synthesizer"]
    fake = FakeTagModel(script={"synthesizer": ['{"sql": "SELECT reviewer_email FROM reviews"}']})
    result = TagPipeline(fake, db).run(item("q04").question, "tag")
    assert result.status == "ok" and result.answer == "Glass Horizon"
    assert "not allowlisted" in fake.prompts[1]


def test_operator_errors_fail_the_question_cleanly(db: ReadOnlyDatabase) -> None:
    plan = {
        "sql": "SELECT title FROM movies",
        "ops": [{"op": "filter", "column": "x", "criterion": "y"}],
    }
    fake = FakeTagModel(script={"synthesizer": [json.dumps(plan)]})
    result = TagPipeline(fake, db).run("anything", "tag")
    assert result.status == "failed" and "no column" in result.answer
    small = TagPipeline(FakeTagModel(), db, Settings(provider="fake", max_lm_rows=5))
    assert "cap" in small.run(item("q01").question, "tag").answer


def test_answer_table_mode_skips_generation(db: ReadOnlyDatabase) -> None:
    plan = {"sql": "SELECT count(*) FROM movies", "answer": "table"}
    fake = FakeTagModel(script={"synthesizer": [json.dumps(plan)]})
    result = TagPipeline(fake, db).run("How many movies?", "tag")
    assert result.answer == "28" and result.lm_calls == 1


def test_large_tables_reach_generation_as_a_summary(db: ReadOnlyDatabase) -> None:
    fake = FakeTagModel(script={"synthesizer": ['{"sql": "SELECT title, rating FROM movies"}']})
    pipe = TagPipeline(fake, db, Settings(provider="fake", answer_rows=5))
    pipe.run("Which movies are there?", "tag")
    answer_prompt = fake.prompts[-1]
    assert '<data name="summary">' in answer_prompt and '"row_count": 28' in answer_prompt


def test_text2sql_lm_and_rag_modes_run(pipeline: TagPipeline) -> None:
    lm = pipeline.run(item("q02").question, "text2sql_lm")
    assert lm.lm_calls == 2 and lm.table is not None and len(lm.table) == 5
    rag = pipeline.run(item("q02").question, "rag")
    assert rag.lm_calls == 1 and rag.table is not None and len(rag.table) == 10
    assert any("genre=Comedy" in r[0] for r in rag.table.rows)
    assert rag.trace[0]["step"] == "retrieve"


def test_rag_index_never_contains_hidden_columns(pipeline: TagPipeline) -> None:
    index = pipeline.rag_index()
    assert len(index.docs) == 28 + 23
    assert not any("@example.com" in d or "reviewer_email" in d for d in index.docs)
    assert index.search("zzzz") == []


def test_unknown_mode_and_bad_questions(pipeline: TagPipeline) -> None:
    with pytest.raises(ValueError, match="unknown pipeline mode"):
        pipeline.run("q", "agent")
    with pytest.raises(ValueError, match="characters"):
        pipeline.run("   ", "tag")
    with pytest.raises(ValueError, match="characters"):
        run_agent(pipeline, "x" * 2001)


def test_unknown_questions_fall_back_to_a_generic_plan(pipeline: TagPipeline) -> None:
    result = pipeline.run("Tell me something about these movies", "tag")
    assert result.status == "ok" and result.answer.startswith("Salt Lines")
    agent = run_agent(pipeline, "Tell me something about these movies")
    assert agent.answer.startswith("Salt Lines")


def test_agent_tool_errors_go_back_to_the_model(pipeline: TagPipeline) -> None:
    bad = AIMessage(
        content="",
        tool_calls=[{"name": "sql_query", "args": {"sql": "DROP TABLE movies"}, "id": "c1"}],
    )
    fake = FakeTagModel(script={"agent": [bad]})
    pipe = TagPipeline(fake, pipeline.db)
    result = run_agent(pipe, item("q04").question)
    assert result.answer.startswith("I could not answer: error: only SELECT")


def test_agent_tools_validate_result_ids_and_columns(pipeline: TagPipeline) -> None:
    session = TagSession(pipeline, pipeline.engine(CallCounter(), []))
    tools = {t.name: t for t in session.tools()}
    assert set(tools) == {
        "sql_query",
        "semantic_filter",
        "semantic_rank",
        "semantic_map",
        "aggregate",
        "head",
    }
    first = tools["sql_query"].invoke({"sql": "SELECT title, setting_city FROM movies LIMIT 3"})
    assert first.startswith("result r1: 3 rows")
    assert tools["head"].invoke({"result_id": "r9", "k": 1}).startswith("error: unknown result")
    assert "no column" in tools["aggregate"].invoke({"result_id": "r1", "fn": "sum", "column": "x"})
    out = tools["semantic_map"].invoke(
        {
            "result_id": "r1",
            "column": "setting_city",
            "instruction": "name the continent the city is in",
            "into": "c",
        }
    )
    assert out.startswith("result r2: 3 rows, columns title, setting_city, c")
    assert (
        tools["semantic_rank"]
        .invoke({"result_id": "r2", "column": "title", "criterion": "x", "k": 1})
        .startswith("result r3: 1 rows")
    )
    capped = tools["sql_query"].invoke({"sql": "SELECT title FROM movies", "params": {}})
    assert "r4: 28 rows" in capped
    session.max_results = 4
    assert "too many results" in tools["head"].invoke({"result_id": "r1", "k": 1})


def test_agent_stops_at_the_tool_call_limit(db: ReadOnlyDatabase) -> None:
    loop = AIMessage(
        content="",
        tool_calls=[{"name": "sql_query", "args": {"sql": "SELECT title FROM movies"}, "id": "x"}],
    )
    fake = FakeTagModel(script={"agent": [loop] * 10})
    pipe = TagPipeline(fake, db, Settings(provider="fake", max_tool_calls=2))
    result = run_agent(pipe, "loop forever")
    assert result.trace[0]["calls"].count("sql_query") <= 3
    assert result.lm_calls <= 4


def test_agent_graph_shape(pipeline: TagPipeline) -> None:
    session = TagSession(pipeline, pipeline.engine(CallCounter(), []))
    nodes = set(build_agent(pipeline.model, session).get_graph().nodes)
    assert {"model", "tools"} <= nodes


def test_schema_samples_are_opt_in(db: ReadOnlyDatabase) -> None:
    assert "sample:" not in schema_for(db)
    with_samples = schema_for(db, samples=2)
    assert with_samples.count("sample:") == 4
    assert "@example.com" not in with_samples
