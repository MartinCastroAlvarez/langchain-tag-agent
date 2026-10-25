"""The comparison benchmark: TAG vs Text2SQL vs Text2SQL + LM vs RAG vs the TAG agent.

``python -m tag_agent.bench`` (fake provider, offline) measures what is real without an
LM: the guard (adversarial queries rejected, gold queries accepted), whether plain SQL
alone can answer each question, the TAG pipeline and agent plumbing with the oracle fake,
LM calls per question and the RAG baseline's retrieval recall. LM answer accuracy needs a
real model: ``python -m tag_agent.bench --provider ollama`` grades every mode.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from pathlib import Path
from typing import Any

from tag_agent.agent import run_agent
from tag_agent.benchmark import BenchItem
from tag_agent.benchmark import grade
from tag_agent.benchmark import load_items
from tag_agent.config import MODES
from tag_agent.config import Mode
from tag_agent.config import Settings
from tag_agent.guard import GuardError
from tag_agent.llm import make_model
from tag_agent.pipeline import TagPipeline
from tag_agent.pipeline import TagResult
from tag_agent.session import open_database

__all__ = ["ADVERSARIAL_FILE", "Report", "main", "run_bench"]

ADVERSARIAL_FILE = Path(__file__).parent / "data" / "adversarial.json"
CATEGORIES = ("match", "comparison", "ranking", "aggregation")


@dataclass
class Report:
    provider: str
    model: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    guard: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "python": platform.python_version(),
            "seconds": round(self.seconds, 3),
            "guard": self.guard,
            "rows": self.rows,
        }


def adversarial_queries() -> list[dict[str, str]]:
    data: list[dict[str, str]] = json.loads(ADVERSARIAL_FILE.read_text(encoding="utf-8"))
    return data


def _answer(pipeline: TagPipeline, item: BenchItem, mode: Mode) -> TagResult:
    if mode == "agent":
        return run_agent(pipeline, item.question)
    return pipeline.run(item.question, mode)


def run_bench(settings: Settings, modes: tuple[Mode, ...] | None = None) -> Report:
    started = time.perf_counter()
    db = open_database(settings)
    model = make_model(settings)
    pipeline = TagPipeline(model, db, settings)
    items = load_items()
    report = Report(settings.provider, settings.model)

    rejected = 0
    for case in adversarial_queries():
        try:
            pipeline.guard.check(case["sql"])
        except GuardError:
            rejected += 1
    accepted = 0
    for item in items:
        try:
            pipeline.guard.check(item.plan.sql, item.plan.params)
            accepted += 1
        except GuardError:
            pass
    report.guard = {
        "adversarial_rejected": rejected,
        "adversarial_total": len(adversarial_queries()),
        "gold_accepted": accepted,
        "gold_total": len(items),
    }

    index = pipeline.rag_index()
    fake_modes: tuple[Mode, ...] = ("text2sql", "tag", "agent")
    chosen = modes or (fake_modes if settings.provider == "fake" else MODES)
    for item in items:
        row: dict[str, Any] = {"id": item.id, "category": item.category, "needs": item.needs}
        evidence = db.query(item.evidence, max_rows=1000)
        wanted = {f"{item.evidence_table}:{r[0]}" for r in evidence.rows}
        got = {k for k, _, _ in index.search(item.question, settings.rag_k)}
        row["rag_recall"] = round(len(wanted & got) / len(wanted), 3) if wanted else 1.0
        row["evidence_rows"] = len(wanted)
        for mode in chosen:
            result = _answer(pipeline, item, mode)
            row[mode] = grade(result.answer, item)
            row[f"{mode}_lm_calls"] = result.lm_calls
        report.rows.append(row)
    report.seconds = time.perf_counter() - started
    return report


def _yes(v: bool) -> str:
    return "yes" if v else "no"


def format_report(report: Report) -> str:
    g = report.guard
    out = [
        f"guard: {g['adversarial_rejected']}/{g['adversarial_total']} adversarial queries "
        f"rejected, {g['gold_accepted']}/{g['gold_total']} gold queries accepted",
        "",
    ]
    modes = [m for m in MODES if m in report.rows[0]] if report.rows else []
    fake = report.provider == "fake"
    labels = {
        "text2sql": "SQL alone" if fake else "Text2SQL",
        "text2sql_lm": "Text2SQL + LM",
        "rag": "RAG",
        "tag": "TAG pipeline" if not fake else "TAG pipeline (fake)",
        "agent": "TAG agent" if not fake else "TAG agent (fake)",
    }
    head = ["id", "category", "needs", *(labels[m] for m in modes)]
    head += ["TAG LM calls", "RAG recall@k"]
    out.append("| " + " | ".join(head) + " |")
    out.append("| " + " | ".join(["---"] * 3 + ["---"] * len(modes) + ["---:", "---:"]) + " |")
    for r in report.rows:
        cells = [r["id"], r["category"], r["needs"], *(_yes(r[m]) for m in modes)]
        cells += [str(r.get("tag_lm_calls", "")), f"{r['rag_recall']:.2f}"]
        out.append("| " + " | ".join(cells) + " |")
    out.append("")
    out.append("| category | " + " | ".join(labels[m] for m in modes) + " | RAG recall@k |")
    out.append("| --- | " + " | ".join("---:" for _ in modes) + " | ---: |")
    for cat in (*CATEGORIES, "all"):
        rows = [r for r in report.rows if cat in ("all", r["category"])]
        if not rows:
            continue
        accs = [f"{sum(r[m] for r in rows)}/{len(rows)}" for m in modes]
        recall = sum(r["rag_recall"] for r in rows) / len(rows)
        out.append(f"| {cat} | " + " | ".join(accs) + f" | {recall:.2f} |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    logging.getLogger("sqlglot").setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(prog="python -m tag_agent.bench", description=__doc__)
    parser.add_argument("--provider", choices=["fake", "ollama", "anthropic"], default="fake")
    parser.add_argument("--model", default=None)
    parser.add_argument("--json", type=Path, default=None, help="write the raw report here")
    args = parser.parse_args(argv)
    base = Settings.from_env()
    settings = replace(base, provider=args.provider)
    if args.model or args.provider != base.provider:
        from tag_agent.config import DEFAULT_MODELS

        settings = replace(settings, model=args.model or DEFAULT_MODELS[args.provider])
    report = run_bench(settings)
    print(format_report(report))
    print(f"\n{report.provider}:{report.model}  {report.seconds:.2f}s")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report.to_json(), indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
