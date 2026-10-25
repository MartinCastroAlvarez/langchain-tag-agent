# CLAUDE.md

Contract between this repository and any AI agent contributing to it.

## Required reading

1. This file. 2. [`README.md`](README.md). 3. [`ARCHITECTURE.md`](ARCHITECTURE.md).
4. [`SECURITY.md`](SECURITY.md).

## Mission

`langchain-tag-agent` shows Table-Augmented Generation (syn → exec → gen, with row-wise
LM operators) in current LangChain (`init_chat_model`, `create_agent`, tools,
middleware) as small, correct, tested code, and compares it honestly with Text2SQL,
Text2SQL + LM and RAG.

## The rules

1. **Local by default.** ollama on loopback; hosted providers stay optional. No
   telemetry, no LangSmith upload, no network in tests.
2. **Model SQL is hostile.** Every query goes through `SqlGuard.check` and only a
   `SafeQuery` reaches `ReadOnlyDatabase.execute`. Never weaken the guard, the authorizer,
   the read-only connection or the limits; a new allowed function must be added to both
   `ALLOWED_FUNCTIONS` and `SQL_FUNCTIONS`, with a test.
3. **No SQL from strings.** Values are bound parameters; generated SQL uses sqlglot's
   builder.
4. **Rows are data.** Untrusted text enters prompts only through `prompts.data_block`.
   System prompts stay constants. Schema samples stay opt-in.
5. **Bounded everything.** Rows, LM calls, tool calls, timeouts. Operators refuse
   oversized input instead of truncating it.
6. **Offline tests with fakes** (`FakeTagModel`, LangChain fake chat models). Every new
   attack gets a case in `data/adversarial.json`.
7. **Honest numbers.** README numbers come from `python -m tag_agent.bench --json
   docs/benchmark/results.json`. Fake-model results are plumbing checks, never LM accuracy.
8. Never add `# noqa`, `# nosec` or `# type: ignore` to the package, skip a test, or
   loosen lint/type/coverage config.

## Done means

`poetry run pytest` (coverage >= 85%), `ruff check`, `ruff format --check`, `mypy`
(strict), `bandit`, `pip-audit`, `python -m tag_agent.dataset --check` and
`python -m examples.quickstart` all green; README and CHANGELOG updated.

## Git

Small commits, imperative subjects, never force-push `main`, never commit secrets or `.env`.
