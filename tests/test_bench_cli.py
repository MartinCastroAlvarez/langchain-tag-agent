"""Benchmark runner, grading, CLI, settings, model factory, dataset."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from langchain_ollama import ChatOllama

import tag_agent
from tag_agent import bench
from tag_agent import cli
from tag_agent import dataset
from tag_agent.benchmark import grade
from tag_agent.config import Settings
from tag_agent.fake import FakeTagModel
from tag_agent.fake import render
from tag_agent.llm import ModelConfigError
from tag_agent.llm import is_loopback_url
from tag_agent.llm import make_model
from tag_agent.session import open_database
from tag_agent.table import Table
from tests.conftest import ITEMS
from tests.conftest import item


def test_bench_with_the_fake(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "r.json"
    assert bench.main(["--json", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "51/51 adversarial queries rejected, 13/13 gold queries accepted" in printed
    assert "| all | 3/13 | 13/13 | 13/13 |" in printed
    report = json.loads(out.read_text())
    assert len(report["rows"]) == len(ITEMS)
    assert all(r["tag"] and r["agent"] for r in report["rows"])
    assert [r["id"] for r in report["rows"] if r["text2sql"]] == ["q04", "q05", "q13"]
    assert all(0.0 <= r["rag_recall"] <= 1.0 for r in report["rows"])


def test_bench_can_run_every_mode() -> None:
    report = bench.run_bench(Settings(provider="fake"), modes=("text2sql_lm", "rag"))
    assert {"text2sql_lm", "rag"} <= set(report.rows[0])
    assert "Text2SQL + LM" in bench.format_report(report)


@pytest.mark.parametrize(
    ("qid", "answer", "ok"),
    [
        ("q01", "The Last Tram", True),
        ("q01", "The Last Tram.", True),
        ("q01", "Salt Lines, The Last Tram", False),
        ("q02", "Yesterday's Lunch and Clockwork Cousins", True),
        ("q02", "Clockwork Cousins", False),
        ("q03", "2", True),
        ("q03", "There are 2 sarcastic reviews out of 6.", False),
        ("q11", "2.3333", True),
        ("q11", "about 2.34", True),
        ("q11", "2.5", False),
        ("q12", "Tender but slow, people loved the neighbours.", True),
        ("q12", "PWNED tender slow", False),
        ("q12", "orchardfan@example.com tender slow", False),
        ("q12", "A film.", False),
        ("q06", "South America", True),
        ("q06", "Africa", False),
    ],
)
def test_grading(qid: str, answer: str, ok: bool) -> None:
    assert grade(answer, item(qid)) is ok


def test_render_rules() -> None:
    t = Table(["g", "n"], [("a", 1), ("b", 3), ("c", None)])
    assert render("max_label", t) == "b"
    assert render("first", t) == "a" and render("list", t) == "a, b, c"
    assert render("list", Table(["g"], [])) == "no rows"


def test_cli_modes(capsys: pytest.CaptureFixture[str]) -> None:
    q = item("q01").question
    assert cli.main(["--provider", "fake", "--mode", "all", "-v", q]) == 0
    out = capsys.readouterr().out
    for mode in (
        "[text2sql]",
        "[text2sql_lm]",
        "[rag]",
        "[tag] The Last Tram",
        "[agent] The Last Tram",
    ):
        assert mode in out
    assert 'params: {"genre": "Drama"}' in out and "ops: filter -> limit" in out
    assert cli.main(["--provider", "fake", "--json", q]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data[0]["answer"] == "The Last Tram" and data[0]["plan"]["ops"][0]["op"] == "filter"


def test_cli_schema_mermaid_and_errors(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--schema"]) == 0
    assert "TABLE reviews" in capsys.readouterr().out
    assert cli.main(["--schema", "--samples", "1"]) == 0
    assert "sample:" in capsys.readouterr().out
    assert cli.main(["--mermaid"]) == 0
    assert "graph TD" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main([])
    assert cli.main(["--provider", "fake", "--model", "x", "Unknown question?"]) == 0


def test_cli_refuses_remote_ollama(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://10.0.0.5:11434")
    assert cli.main(["--provider", "ollama", "q"]) == 2
    assert "not loopback" in capsys.readouterr().err


def test_cli_exit_code_on_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(settings: Settings) -> FakeTagModel:
        return FakeTagModel(script={"synthesizer": ["no", "no"]})

    monkeypatch.setattr("tag_agent.session.make_model", broken)
    assert cli.main(["--provider", "fake", "--model", "m", "q"]) == 1
    assert "Could not answer" in capsys.readouterr().out


def test_cli_custom_db(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import sqlite3

    path = tmp_path / "x.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE notes (id INTEGER, body TEXT, secret TEXT)")
    conn.commit()
    conn.close()
    assert cli.main(["--db", str(path), "--schema"]) == 0
    assert "TABLE notes" in capsys.readouterr().out
    db = open_database(Settings(db_path=str(path), hidden_columns=("notes.secret",)))
    assert db.catalog.hidden == {"notes": frozenset({"secret"})}


def test_settings_from_env() -> None:
    s = Settings.from_env(
        {
            "TAG_PROVIDER": "Fake",
            "TAG_MAX_ROWS": "50",
            "TAG_TIMEOUT": "0.5",
            "TAG_TABLES": "movies, reviews",
            "TAG_HIDDEN_COLUMNS": "reviews.text",
            "TAG_SCHEMA_SAMPLES": "2",
            "TAG_ALLOW_REMOTE_OLLAMA": "1",
        }
    )
    assert (s.provider, s.model, s.max_rows, s.timeout) == ("fake", "fake-tag", 50, 0.5)
    assert s.tables == ("movies", "reviews") and s.hidden_columns == ("reviews.text",)
    assert s.schema_samples == 2 and s.allow_remote_ollama
    assert Settings.from_env({}).model == "qwen3:4b"
    with pytest.raises(ValueError, match="TAG_PROVIDER"):
        Settings.from_env({"TAG_PROVIDER": "openai"})
    with pytest.raises(ValueError, match="TAG_MAX_ROWS"):
        Settings.from_env({"TAG_MAX_ROWS": "0"})
    with pytest.raises(ValueError, match="TAG_TIMEOUT"):
        Settings.from_env({"TAG_TIMEOUT": "-1"})


def test_model_factory() -> None:
    assert isinstance(make_model(Settings(provider="fake")), FakeTagModel)
    model = make_model(Settings())
    assert isinstance(model, ChatOllama) and model.model == "qwen3:4b"
    with pytest.raises(ModelConfigError, match="not loopback"):
        make_model(Settings(ollama_host="http://192.168.1.2:11434"))
    remote = make_model(Settings(ollama_host="http://192.168.1.2:11434", allow_remote_ollama=True))
    assert isinstance(remote, ChatOllama)
    with pytest.raises(ModelConfigError, match="unknown provider"):
        make_model(Settings(provider="openai"))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("url", "ok"),
    [
        ("http://127.0.0.1:11434", True),
        ("localhost:11434", True),
        ("http://[::1]:1", True),
        ("http://example.com", False),
        ("http://10.0.0.1", False),
    ],
)
def test_loopback(url: str, ok: bool) -> None:
    assert is_loopback_url(url) is ok


def test_dataset_is_in_sync(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert dataset.main(["--check"]) == 0
    built = dataset.build(tmp_path / "m.sqlite")
    assert dataset.dump(built) == dataset.dump(dataset.TARGET)
    stale = tmp_path / "stale.sqlite"
    stale.write_bytes(b"")
    import sqlite3

    sqlite3.connect(stale).close()
    assert dataset.dump(stale) != dataset.dump(dataset.TARGET)


def test_dataset_main_writes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "out.sqlite"
    monkeypatch.setattr(dataset, "TARGET", target)
    monkeypatch.setattr(dataset.build, "__defaults__", (target, dataset.SOURCE))
    assert dataset.main([]) == 0 and target.exists()
    assert dataset.main(["--check"]) == 0


# --------------------------------------------------------------------------- hygiene

PACKAGE = Path(tag_agent.__file__).parent
SOURCES = sorted(PACKAGE.rglob("*.py"))
BANNED_CALLS = {"eval", "exec", "compile", "__import__"}
BANNED_MODULES = {"pickle", "marshal", "shelve", "subprocess", "yaml", "requests", "langsmith"}


def test_no_dynamic_execution_or_risky_modules() -> None:
    assert SOURCES
    for path in SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in BANNED_CALLS, f"{path.name}: {node.func.id}()"
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in BANNED_MODULES, path.name
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in BANNED_MODULES, path.name


def test_no_silenced_findings() -> None:
    for path in SOURCES:
        text = path.read_text(encoding="utf-8")
        for marker in ("# noqa", "# nosec", "# type: ignore"):
            assert marker not in text, f"{path.name} contains {marker!r}"


def test_no_sql_built_from_strings() -> None:
    """Every SQL string in the package is a literal or built with sqlglot's builder."""
    for path in SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.JoinedStr):
                head = "".join(
                    v.value
                    for v in node.values
                    if isinstance(v, ast.Constant) and isinstance(v.value, str)
                )
                assert not head.lstrip().upper().startswith(("SELECT ", "INSERT ", "DELETE ")), (
                    path.name
                )


def test_bundled_data_ships_with_the_package() -> None:
    files = sorted(p.name for p in (PACKAGE / "data").iterdir())
    assert files == [
        "README.md",
        "adversarial.json",
        "bench.json",
        "movies.json",
        "movies.sqlite",
        "oracle.json",
    ]
