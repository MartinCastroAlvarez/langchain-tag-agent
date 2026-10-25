"""Offline quickstart: one question, five strategies, the same database and model.

    python -m examples.quickstart

Uses the scripted fake model, so it needs no ollama and no network. Swap in a real model
with ``build_pipeline(Settings.from_env())`` (ollama by default).
"""

from __future__ import annotations

import logging

from tag_agent import FakeTagModel
from tag_agent import Settings
from tag_agent import TagPipeline
from tag_agent import open_database
from tag_agent import run_agent
from tag_agent.guard import GuardError

logging.getLogger("sqlglot").setLevel(logging.ERROR)

QUESTION = "Which drama set in a European capital has the highest rating?"


def main() -> None:
    settings = Settings(provider="fake")
    with open_database(settings) as db:
        run(TagPipeline(FakeTagModel(), db, settings))


def run(pipeline: TagPipeline) -> None:

    print(f"Q: {QUESTION}\n")
    sql = pipeline.run(QUESTION, "text2sql")
    print(f"{'text2sql':>9}: {sql.answer}  (the query result is the answer)")
    rag = pipeline.run(QUESTION, "rag")
    keys = ", ".join(rag.trace[0]["keys"][:5])
    print(f"{'rag':>9}: retrieved {keys}, ...  (answering needs a real LM)")
    tag = pipeline.run(QUESTION, "tag")
    print(f"{'tag':>9}: {tag.answer}  (LM calls: {tag.lm_calls})")
    agent = run_agent(pipeline, QUESTION)
    print(f"{'agent':>9}: {agent.answer}  (tools: {' -> '.join(agent.trace[0]['calls'])})")
    print(f"\nTAG plan SQL : {tag.sql}")
    print(f"TAG plan ops : {[op.model_dump() for op in tag.plan.ops] if tag.plan else []}")

    print("\nThe guard in action:")
    for sql in (
        "SELECT reviewer, reviewer_email FROM reviews",
        "SELECT title FROM movies; DROP TABLE movies",
        "SELECT * FROM read_csv('/etc/passwd')",
    ):
        try:
            pipeline.guard.check(sql)
        except GuardError as exc:
            print(f"  refused: {sql!r}\n           -> {exc}")


if __name__ == "__main__":
    main()
