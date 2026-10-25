# Architecture

```
tag_agent/
  config.py      Settings from TAG_* env vars; the five modes
  catalog.py     Catalog / TableSpec / ColumnSpec: the allowlist; schema_text() (no data by default)
  guard.py       SqlGuard.check(sql, params) -> SafeQuery: sqlglot parse, one SELECT, allowlists,
                 named params, forced LIMIT, SQL regenerated without comments
  db.py          ReadOnlyDatabase: mode=ro URI, query_only, limits, authorizer, timeout, row cap
  table.py       Table (columns, rows, truncated): JSON, summary(), answer_text()
  plan.py        TagPlan + operator models (filter, rank, map, limit, aggregate); parse_plan
  prompts.py     constant system prompts ("Role: <name>." first) and the escaped <data> envelope
  operators.py   SemanticEngine: row-wise LM filter/rank/map with caps and an LM budget;
                 deterministic limit/aggregate
  pipeline.py    TagPipeline: synthesize (1 repair) -> execute -> operators -> generate;
                 text2sql, text2sql_lm and rag baselines; CallCounter
  agent.py       TagSession (result store + six StructuredTools), build_agent (create_agent +
                 ToolCallLimitMiddleware), run_agent
  rag.py         RowIndex: rows -> text chunks -> BM25 (the RAG baseline)
  llm.py         make_model: init_chat_model for ollama (loopback-only) / anthropic; the fake
  fake.py        FakeTagModel: role-aware scripted fake (gold plans, oracle labels, agent loop)
  benchmark.py   BenchItem, load_items, grade (exact match, distractors, numbers, keywords)
  bench.py       python -m tag_agent.bench: guard counts, per-question metrics, accuracy by mode
  session.py     settings -> database (+ allowlist) -> model -> pipeline
  dataset.py     movies.json -> movies.sqlite, --check
  cli.py         python -m tag_agent "<question>" [--mode ...] [--schema] [--mermaid]
  data/          movies.json/.sqlite, bench.json, oracle.json, adversarial.json
```

## One question, end to end (`mode="tag"`)

```
question ──► synthesize ──► SqlGuard.check ──► ReadOnlyDatabase.execute ──► SemanticEngine.run ──► generate
  (data)    Role: synthesizer   sqlglot,         mode=ro, authorizer,         filter/rank/map:       Role: answer
            schema, no rows     allowlist,       progress-handler timeout,    1 LM call per row,     table JSON
            JSON plan           LIMIT, :params   fetchmany(max_rows + 1)      limit/aggregate in Py  (+ summary)
                 ▲                   │
                 └── error, once ────┘
```

- `synthesize` validates the reply with `parse_plan` (Pydantic, discriminated operator
  union, `extra="forbid"`) and the guard. A refusal is fed back once, inside a `<data>`
  block; a second refusal fails the question with `status="failed"` and nothing has run.
- `execute` only accepts a `SafeQuery`. The guard is the only thing that produces one
  from model text (tests build them by hand to attack the engine directly).
- `SemanticEngine` sends one `[SystemMessage, HumanMessage]` per row through
  `model.batch(..., max_concurrency=n, return_exceptions=True)`. Replies are parsed into
  `bool | None`, `int | None` or a one-line string; `None` means invalid and counts as
  no / 0. The trace records `rows_in`, `rows_out` and `invalid` per operator.
- `generate` sees the table as JSON (first `answer_rows` rows) and, for bigger tables, a
  deterministic `summary()`.

## The agent

`TagSession` wraps one pipeline and one `SemanticEngine` (the same LM budget for the
whole question) and exposes six `StructuredTool`s with Pydantic argument schemas
(identifier patterns, length limits, `k` ranges). Every tool returns
`result rN: <rows> rows, columns ...` plus the rows in a `<data name="rN">` block; tool
errors come back as `error: ...` strings (`handle_tool_error`), so the model can correct
itself. `build_agent` is `create_agent(model, tools, system_prompt=AGENT + schema,
middleware=[ToolCallLimitMiddleware(run_limit=max_tool_calls, exit_behavior="end")])`.

## The fake model

`FakeTagModel` dispatches on the `Role:` line: the benchmark's gold plan (synthesizer,
text2sql), `data/oracle.json` labels (semantic_filter/rank/map), the question's render
rule over the table in the prompt (answer), or a tool-calling loop that walks the gold
plan through the real tools (agent). It records every call's role, system prompt and full
prompt so tests can assert what each step saw.

## Security layering

```
model text ─► Pydantic plan ─► sqlglot guard ─► mode=ro + query_only ─► authorizer ─► timeout/row cap
                                (static)          (SQLite)               (per column)    (runtime)
```

Each layer is tested alone: guard tests never touch the database, engine tests bypass the
guard, and the read-only test removes the authorizer to show SQLite itself refuses writes.
