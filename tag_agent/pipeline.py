"""Table-Augmented Generation: query synthesis -> query execution -> answer generation.

``TagPipeline.run(question, mode)`` runs one of five strategies over the same database,
guard and model, so they can be compared on equal terms:

* ``tag``: the full TAG model. syn(R) -> plan (SQL + semantic operators), exec(Q) -> T
  through the guard and the read-only database, the operators run row-wise in Python on
  the bounded result, then gen(R, T) answers from the final table.
* ``text2sql``: plain Text2SQL. The query result *is* the answer; no LM after SQL.
* ``text2sql_lm``: Text2SQL + LM. The model writes SQL, then answers from the raw rows.
* ``rag``: naive RAG over rows. BM25 top-k rows, then the model answers.
* ``agent`` lives in :mod:`tag_agent.agent` (a ``create_agent`` ReAct loop over the same
  steps exposed as tools); :func:`answer` dispatches to it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from sqlglot import exp

from tag_agent import prompts
from tag_agent.config import Mode
from tag_agent.config import Settings
from tag_agent.db import QueryError
from tag_agent.db import ReadOnlyDatabase
from tag_agent.guard import GuardError
from tag_agent.guard import SafeQuery
from tag_agent.operators import LMBudget
from tag_agent.operators import OperatorError
from tag_agent.operators import SemanticEngine
from tag_agent.operators import clean_reply
from tag_agent.plan import PlanError
from tag_agent.plan import TagPlan
from tag_agent.plan import parse_plan
from tag_agent.rag import RowIndex
from tag_agent.table import Table

__all__ = ["CallCounter", "TagPipeline", "TagResult", "schema_for"]

MAX_QUESTION_CHARS = 2000


class CallCounter(BaseCallbackHandler):
    """Counts chat-model calls by the role in their system prompt."""

    def __init__(self) -> None:
        self.roles: list[str] = []

    def on_chat_model_start(
        self, serialized: dict[str, Any], messages: list[list[BaseMessage]], **kwargs: Any
    ) -> None:
        for batch in messages:
            system = next((m.text for m in batch if isinstance(m, SystemMessage)), "")
            self.roles.append(prompts.role_of(system))

    @property
    def total(self) -> int:
        return len(self.roles)


@dataclass
class TagResult:
    question: str
    mode: str
    answer: str
    status: str = "ok"
    """``ok`` or ``failed`` (the answer then says why)."""
    plan: TagPlan | None = None
    sql: str | None = None
    """The guarded SQL that actually ran."""
    table: Table | None = None
    """The final table answer generation saw."""
    lm_calls: int = 0
    trace: list[dict[str, Any]] = field(default_factory=list)


def schema_for(db: ReadOnlyDatabase, samples: int = 0) -> str:
    """The schema prompt. Data rows are included only when ``samples > 0`` (opt-in)."""
    if samples <= 0:
        return db.catalog.schema_text()
    rows: dict[str, list[dict[str, object]]] = {}
    for name, spec in db.catalog.tables.items():
        sql = exp.select(*(c.name for c in spec.columns)).from_(name).sql(dialect="sqlite")
        query = db.guard(samples).check(sql)
        rows[name] = [dict(r) for r in db.execute(query).records()]
    return db.catalog.schema_text(rows)


class TagPipeline:
    def __init__(
        self, model: BaseChatModel, db: ReadOnlyDatabase, settings: Settings | None = None
    ) -> None:
        self.model = model
        self.db = db
        self.settings = settings or Settings(provider="fake")
        self.guard = db.guard(self.settings.max_rows)
        self._schema = schema_for(db, self.settings.schema_samples)
        self._rag: RowIndex | None = None

    # ------------------------------------------------------------------ public

    def run(self, question: str, mode: Mode = "tag") -> TagResult:
        question = question.strip()
        if not question or len(question) > MAX_QUESTION_CHARS:
            raise ValueError(f"the question must be 1-{MAX_QUESTION_CHARS} characters")
        counter = CallCounter()
        result = TagResult(question=question, mode=mode, answer="")
        try:
            if mode == "rag":
                self._run_rag(question, result, counter)
            elif mode in ("tag", "text2sql", "text2sql_lm"):
                self._run_sql(question, mode, result, counter)
            else:
                raise ValueError(f"unknown pipeline mode {mode!r}; the agent is agent.run_agent")
        except (PlanError, GuardError, QueryError, OperatorError) as exc:
            result.status = "failed"
            result.answer = f"Could not answer: {exc}"
            result.trace.append({"step": "error", "error": str(exc)})
        result.lm_calls = counter.total
        return result

    # ------------------------------------------------------------------ steps

    def synthesize(
        self, question: str, *, allow_ops: bool, counter: CallCounter, trace: list[dict[str, Any]]
    ) -> tuple[TagPlan, SafeQuery]:
        """syn(R) -> Q, with one repair round when the plan or the guard says no."""
        system = prompts.SYNTHESIZER if allow_ops else prompts.TEXT2SQL
        messages: list[BaseMessage] = [
            SystemMessage(system),
            HumanMessage(
                prompts.data_block("question", question)
                + "\n"
                + prompts.data_block("schema", self._schema)
            ),
        ]
        last_error: PlanError | GuardError = PlanError("no plan")
        for attempt in range(2):
            reply = self.model.invoke(messages, config=self._config(counter))
            try:
                plan = parse_plan(reply.text, allow_ops=allow_ops)
                query = self.guard.check(plan.sql, plan.params)
            except (PlanError, GuardError) as exc:
                trace.append({"step": "synthesize", "attempt": attempt + 1, "error": str(exc)})
                last_error = exc
                messages += [
                    AIMessage(reply.text),
                    HumanMessage(
                        "Your plan was rejected. Fix it and reply with JSON only.\n"
                        + prompts.data_block("error", str(exc))
                    ),
                ]
                continue
            trace.append({"step": "synthesize", "attempt": attempt + 1, "sql": query.sql})
            return plan, query
        raise last_error

    def execute(self, query: SafeQuery, trace: list[dict[str, Any]]) -> Table:
        """exec(Q) -> T on the read-only connection."""
        table = self.db.execute(query)
        trace.append({"step": "execute", "rows": len(table), "truncated": table.truncated})
        return table

    def generate(self, question: str, table: Table, counter: CallCounter) -> str:
        """gen(R, T) -> A. Big tables are shown as a summary plus the first rows."""
        n = self.settings.answer_rows
        parts = [
            prompts.data_block("question", question),
            prompts.data_block("table", table.to_json(n)),
        ]
        if len(table) > n or table.truncated:
            parts.append(prompts.data_block("summary", json.dumps(table.summary(), default=str)))
        messages = [SystemMessage(prompts.ANSWER), HumanMessage("\n".join(parts))]
        reply = self.model.invoke(messages, config=self._config(counter))
        return clean_reply(reply.text)

    # ------------------------------------------------------------------ modes

    def _run_sql(self, question: str, mode: str, result: TagResult, counter: CallCounter) -> None:
        plan, query = self.synthesize(
            question, allow_ops=mode == "tag", counter=counter, trace=result.trace
        )
        result.plan, result.sql = plan, query.sql
        table = self.execute(query, result.trace)
        if mode == "tag" and plan.ops:
            engine = self.engine(counter, result.trace)
            table = engine.run(table, plan.ops)
        result.table = table
        if mode == "text2sql" or (mode == "tag" and plan.answer == "table"):
            result.answer = table.answer_text()
        else:
            result.answer = self.generate(question, table, counter)

    def _run_rag(self, question: str, result: TagResult, counter: CallCounter) -> None:
        hits = self.rag_index().search(question, self.settings.rag_k)
        result.trace.append({"step": "retrieve", "keys": [k for k, _, _ in hits]})
        table = Table(["row"], [(doc,) for _, doc, _ in hits])
        result.table = table
        result.answer = self.generate(question, table, counter)

    def engine(self, counter: CallCounter, trace: list[dict[str, Any]]) -> SemanticEngine:
        return SemanticEngine(
            self.model,
            budget=LMBudget(self.settings.max_lm_calls),
            max_rows=self.settings.max_lm_rows,
            max_concurrency=self.settings.max_concurrency,
            callbacks=[counter],
            trace=trace,
        )

    def rag_index(self) -> RowIndex:
        if self._rag is None:
            self._rag = RowIndex.build(self.db)
        return self._rag

    @staticmethod
    def _config(counter: CallCounter) -> RunnableConfig:
        return {"callbacks": [counter]}
