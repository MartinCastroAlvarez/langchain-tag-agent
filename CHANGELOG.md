# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-25

### Added

- `TagPipeline`: Table-Augmented Generation in three steps: query synthesis to a JSON
  plan (SQL, bound params, semantic operators) with one repair round, guarded read-only
  execution, row-wise LM operators, and answer generation over the table (plus a summary
  for large tables).
- Semantic operators run in Python on bounded results: `filter` (yes/no), `rank`
  (0-10 pointwise, top k), `map` (one line), plus deterministic `limit` and `aggregate`;
  row cap, LM-call budget, batched calls with bounded concurrency.
- A `create_agent` ReAct agent with six TAG tools, a server-side result store and
  `ToolCallLimitMiddleware`.
- Baselines: Text2SQL, Text2SQL + LM and BM25 RAG over rows.
- `SqlGuard` (sqlglot): one SELECT, allowlisted tables/columns/functions, no recursive
  CTEs or file/extension functions, named params, forced LIMIT, comment-free regenerated
  SQL. `ReadOnlyDatabase`: `mode=ro`, `query_only`, SQLite limits, an allowlist
  authorizer, a timeout and a row cap.
- A fictional movies database with a hidden `reviewer_email` column and a prompt-injection
  review; 13 TAG-Bench-style questions; a 51-query adversarial corpus.
- `FakeTagModel`, a role-aware scripted fake; `python -m tag_agent` CLI and
  `python -m tag_agent.bench`.
