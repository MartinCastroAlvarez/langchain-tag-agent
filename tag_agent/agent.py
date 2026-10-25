"""TAG as agent context: a LangChain ``create_agent`` ReAct loop over TAG tools.

The agent gets the schema (no data) in its system prompt and six tools that are the TAG
steps: ``sql_query`` (query synthesis is the agent's own tool call, execution goes through
the guard), ``semantic_filter`` / ``semantic_rank`` / ``semantic_map`` (row-wise LM
operators), and ``aggregate`` / ``head`` (deterministic table operators). Every result is
stored server-side under an id (``r1``, ``r2``, ...), so the model refers to rows by id
and never re-types, or forges, the data. ``ToolCallLimitMiddleware`` bounds the loop.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Literal

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from langchain_core.tools import ToolException
from pydantic import BaseModel
from pydantic import Field

from tag_agent import prompts
from tag_agent.config import Settings
from tag_agent.db import QueryError
from tag_agent.guard import GuardError
from tag_agent.guard import Scalar
from tag_agent.operators import OperatorError
from tag_agent.operators import SemanticEngine
from tag_agent.operators import aggregate
from tag_agent.operators import clean_reply
from tag_agent.operators import limit
from tag_agent.pipeline import CallCounter
from tag_agent.pipeline import TagPipeline
from tag_agent.pipeline import TagResult
from tag_agent.table import Table

__all__ = ["TagSession", "build_agent", "run_agent"]

_Id = Field(pattern=r"^r\d{1,4}$", description="a result id such as r1")
_Col = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
_Crit = Field(min_length=1, max_length=300)


class SqlArgs(BaseModel):
    sql: str = Field(min_length=1, max_length=4000, description="one SQLite SELECT")
    params: dict[str, Scalar] | None = Field(
        default=None, description="values for :name placeholders in the SQL"
    )


class FilterArgs(BaseModel):
    result_id: str = _Id
    column: str = _Col
    criterion: str = _Crit


class RankArgs(BaseModel):
    result_id: str = _Id
    column: str = _Col
    criterion: str = _Crit
    k: int = Field(default=3, ge=1, le=50)


class MapArgs(BaseModel):
    result_id: str = _Id
    column: str = _Col
    instruction: str = _Crit
    into: str = _Col


class AggregateArgs(BaseModel):
    result_id: str = _Id
    fn: Literal["count", "sum", "avg", "min", "max"]
    column: str | None = Field(default=None, max_length=64)
    group_by: str | None = Field(default=None, max_length=64)


class HeadArgs(BaseModel):
    result_id: str = _Id
    k: int = Field(ge=1, le=1000)


@dataclass
class TagSession:
    """Per-question tool state: the result store and the LM-operator engine."""

    pipeline: TagPipeline
    engine: SemanticEngine
    results: dict[str, Table] = field(default_factory=dict)
    max_results: int = 50

    def _store(self, table: Table) -> str:
        if len(self.results) >= self.max_results:
            raise ToolException("too many results in this session")
        rid = f"r{len(self.results) + 1}"
        self.results[rid] = table
        return self._show(rid, table)

    def _show(self, rid: str, table: Table) -> str:
        n = self.pipeline.settings.answer_rows
        head = f"result {rid}: {len(table)} rows, columns {', '.join(table.columns)}"
        if table.truncated:
            head += " (truncated at the row cap)"
        return head + "\n" + prompts.data_block(rid, table.to_json(n))

    def _get(self, rid: str) -> Table:
        if rid not in self.results:
            raise ToolException(f"error: unknown result id {rid!r}")
        return self.results[rid]

    def _run(self, fn: Callable[[], Table]) -> str:
        try:
            return self._store(fn())
        except (GuardError, QueryError, OperatorError) as exc:
            raise ToolException(f"error: {exc}") from exc
        except KeyError as exc:
            raise ToolException(f"error: {exc.args[0]}") from exc

    # ------------------------------------------------------------------ tools

    def sql_query(self, sql: str, params: dict[str, Scalar] | None = None) -> str:
        """Run one read-only SQLite SELECT over the schema. Pass literal values as :name
        params. Returns a result id and the first rows."""
        pipe = self.pipeline
        return self._run(lambda: pipe.db.execute(pipe.guard.check(sql, params or {})))

    def semantic_filter(self, result_id: str, column: str, criterion: str) -> str:
        """Keep the rows of a result whose `column` meets a natural-language criterion
        (world knowledge or judgment, one LM call per row)."""
        return self._run(lambda: self.engine.filter(self._get(result_id), column, criterion))

    def semantic_rank(self, result_id: str, column: str, criterion: str, k: int = 3) -> str:
        """Keep the k rows of a result whose `column` best matches a criterion."""
        return self._run(lambda: self.engine.rank(self._get(result_id), column, criterion, k))

    def semantic_map(self, result_id: str, column: str, instruction: str, into: str) -> str:
        """Add column `into` computed by the LM from `column`, row by row."""
        return self._run(lambda: self.engine.map(self._get(result_id), column, instruction, into))

    def aggregate(
        self,
        result_id: str,
        fn: Literal["count", "sum", "avg", "min", "max"],
        column: str | None = None,
        group_by: str | None = None,
    ) -> str:
        """count/sum/avg/min/max over a result, optionally grouped by a column."""
        return self._run(lambda: aggregate(self._get(result_id), fn, column, group_by))

    def head(self, result_id: str, k: int) -> str:
        """Keep the first k rows of a result (keeps the SQL ORDER BY)."""
        return self._run(lambda: limit(self._get(result_id), k))

    def tools(self) -> list[StructuredTool]:
        specs: list[tuple[Callable[..., str], type[BaseModel]]] = [
            (self.sql_query, SqlArgs),
            (self.semantic_filter, FilterArgs),
            (self.semantic_rank, RankArgs),
            (self.semantic_map, MapArgs),
            (self.aggregate, AggregateArgs),
            (self.head, HeadArgs),
        ]
        return [
            StructuredTool.from_function(
                func=fn,
                name=fn.__name__,
                description=(fn.__doc__ or "").strip(),
                args_schema=schema,
                handle_tool_error=_tool_error,
            )
            for fn, schema in specs
        ]


def _tool_error(exc: ToolException) -> str:
    text = str(exc)
    return text if text.startswith("error") else f"error: {text}"


def build_agent(model: BaseChatModel, session: TagSession) -> Any:
    """The compiled ``create_agent`` graph (model <-> tools, bounded by the tool limit)."""
    settings = session.pipeline.settings
    return create_agent(
        model,
        tools=session.tools(),
        system_prompt=prompts.AGENT + session.pipeline._schema,
        middleware=[
            ToolCallLimitMiddleware(run_limit=settings.max_tool_calls, exit_behavior="end")
        ],
        name="tag_agent",
    )


def run_agent(pipeline: TagPipeline, question: str) -> TagResult:
    """Answer one question with the TAG agent; returns the same ``TagResult`` shape."""
    question = question.strip()
    if not question or len(question) > 2000:
        raise ValueError("the question must be 1-2000 characters")
    counter = CallCounter()
    trace: list[dict[str, Any]] = []
    session = TagSession(pipeline, pipeline.engine(counter, trace))
    graph = build_agent(pipeline.model, session)
    settings: Settings = pipeline.settings
    state = graph.invoke(
        {"messages": [HumanMessage(prompts.data_block("question", question))]},
        config={"callbacks": [counter], "recursion_limit": 4 * settings.max_tool_calls + 10},
    )
    messages = state["messages"]
    final = next(
        (m for m in reversed(messages) if isinstance(m, AIMessage) and not m.tool_calls), None
    )
    calls = [c["name"] for m in messages if isinstance(m, AIMessage) for c in m.tool_calls]
    trace.insert(0, {"step": "tools", "calls": calls})
    last = list(session.results.values())[-1] if session.results else None
    answer = clean_reply(final.text) if final is not None else "Stopped: tool-call limit reached."
    return TagResult(
        question=question,
        mode="agent",
        answer=answer,
        status="ok" if final is not None else "failed",
        table=last,
        lm_calls=counter.total,
        trace=trace,
    )
