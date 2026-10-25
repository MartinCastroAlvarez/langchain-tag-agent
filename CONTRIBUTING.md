# Contributing to langchain-tag-agent

## TL;DR

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install poetry
poetry install                       # add -E anthropic for the optional adapter
poetry run pytest
poetry run ruff check tag_agent tests examples && poetry run ruff format --check tag_agent tests examples
poetry run mypy tag_agent
poetry run bandit -c pyproject.toml -q -r tag_agent
poetry run pip-audit
poetry run python -m tag_agent.dataset --check
poetry run python -m examples.quickstart
```

Open a PR against `main` from a `feat/...`, `fix/...`, `docs/...` or `chore/...` branch.

## What this project will not accept

- Tests that call a real model or touch the network.
- Anything that lets model text reach the database without `SqlGuard`, or that builds SQL
  from strings.
- Write access, extensions, file functions, or allowlist entries without tests.
- Hosted providers or tracing turned on by default.
- Hand-edited benchmark numbers, or fake-model numbers presented as LM accuracy.

## What it does accept

- New semantic operators (e.g. semantic join, batched filters), with caps, constrained
  outputs and tests.
- New benchmark questions in `tag_agent/data/bench.json` (with gold plan, answer,
  distractors and evidence), plus oracle labels for the fake.
- New attacks in `tag_agent/data/adversarial.json`.
- A DuckDB backend with the same layers (read-only, allowlist, timeout).

## Changing the data

Edit `tag_agent/data/movies.json`, then `python -m tag_agent.dataset` to rebuild the
SQLite file. Keep it fictional and self-made.

## Updating the benchmark table

```bash
poetry run python -m tag_agent.bench --json docs/benchmark/results.json
```

Paste the printed tables into the README together with the versions and the date.

## House rules

1. Tests alongside features; security-relevant changes get a hostile case.
2. README, ARCHITECTURE/SECURITY (when relevant) and CHANGELOG change with the code.
3. No secrets in commits; `.env.example` holds defaults only.
