"""``python -m tag_agent "<question>"``: answer a question over the database with TAG."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace

from tag_agent.agent import build_agent
from tag_agent.agent import run_agent
from tag_agent.config import DEFAULT_MODELS
from tag_agent.config import MODES
from tag_agent.config import Mode
from tag_agent.config import Settings
from tag_agent.fake import FakeTagModel
from tag_agent.llm import ModelConfigError
from tag_agent.pipeline import CallCounter
from tag_agent.pipeline import TagPipeline
from tag_agent.pipeline import TagResult
from tag_agent.pipeline import schema_for
from tag_agent.session import build_pipeline
from tag_agent.session import open_database

__all__ = ["main"]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tag_agent",
        description="Table-Augmented Generation over a read-only SQLite database.",
    )
    p.add_argument("question", nargs="?", help="the question to answer")
    p.add_argument("--mode", choices=[*MODES, "all"], default="tag")
    p.add_argument("--provider", choices=["ollama", "anthropic", "fake"], default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--db", default=None, help="a SQLite file (default: the bundled movies db)")
    p.add_argument("--samples", type=int, default=None, help="sample rows in the schema prompt")
    p.add_argument("--schema", action="store_true", help="print the schema the model sees")
    p.add_argument("--mermaid", action="store_true", help="print the agent graph as Mermaid")
    p.add_argument("--json", action="store_true", help="print results as JSON")
    p.add_argument("-v", "--verbose", action="store_true", help="print the trace")
    return p


def _settings(args: argparse.Namespace) -> Settings:
    s = Settings.from_env()
    if args.provider:
        s = replace(s, provider=args.provider, model=args.model or DEFAULT_MODELS[args.provider])
    elif args.model:
        s = replace(s, model=args.model)
    if args.db:
        s = replace(s, db_path=args.db)
    if args.samples is not None:
        s = replace(s, schema_samples=args.samples)
    return s


def _print(result: TagResult, verbose: bool) -> None:
    print(f"[{result.mode}] {result.answer}")
    detail = f"  status={result.status} lm_calls={result.lm_calls}"
    if result.sql:
        detail += f"\n  sql: {result.sql}"
        if result.plan and result.plan.params:
            detail += f"\n  params: {json.dumps(result.plan.params, ensure_ascii=False)}"
    if result.plan and result.plan.ops:
        detail += "\n  ops: " + " -> ".join(op.op for op in result.plan.ops)
    print(detail)
    if verbose:
        for event in result.trace:
            print("  trace:", json.dumps(event, ensure_ascii=False, default=str))


def _as_json(result: TagResult) -> dict[str, object]:
    return {
        "mode": result.mode,
        "answer": result.answer,
        "status": result.status,
        "sql": result.sql,
        "plan": result.plan.model_dump() if result.plan else None,
        "lm_calls": result.lm_calls,
        "trace": result.trace,
    }


def main(argv: list[str] | None = None) -> int:
    logging.getLogger("sqlglot").setLevel(logging.ERROR)
    args = _parser().parse_args(argv)
    settings = _settings(args)
    if args.schema:
        print(schema_for(open_database(settings), settings.schema_samples))
        return 0
    if args.mermaid:
        pipe = TagPipeline(FakeTagModel(), open_database(settings), settings)
        from tag_agent.agent import TagSession

        session = TagSession(pipe, pipe.engine(CallCounter(), []))
        print(build_agent(pipe.model, session).get_graph().draw_mermaid())
        return 0
    if not args.question:
        _parser().error("a question is required")
    try:
        pipeline = build_pipeline(settings)
    except ModelConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    modes: tuple[Mode, ...] = MODES if args.mode == "all" else (args.mode,)
    results = []
    for mode in modes:
        if mode == "agent":
            result = run_agent(pipeline, args.question)
        else:
            result = pipeline.run(args.question, mode)
        results.append(result)
        if not args.json:
            _print(result, args.verbose)
    if args.json:
        print(json.dumps([_as_json(r) for r in results], indent=1, ensure_ascii=False))
    return 0 if all(r.status == "ok" for r in results) else 1
