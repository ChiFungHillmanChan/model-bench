import yaml

from model_bench.repro import format_repro, repro
from test_report import build_db, grid


def refs_file(tmp_path, entries):
    p = tmp_path / "refs.yaml"
    p.write_text(yaml.safe_dump({"references": entries}))
    return p


def ref(model="m", task="gpqa", score=0.5, kind="provider", explanation=None):
    return {"model_id": model, "task": task, "score": score, "source_url": "https://example.org",
            "source_kind": kind, "settings": "t=0.6", "retrieved": "2026-11-22", "explanation": explanation}


VALUES = [1, 0, 1, 0, 1, 0, 1, 0, 1, 1]  # mean 0.6, CI roughly [0.28, 0.92]


def test_match_when_published_inside_ci(tmp_path):
    db = build_db(tmp_path / "r.duckdb", rows=grid("m", "api", VALUES))
    [row] = repro(db, refs_file(tmp_path, [ref(score=0.55)]))
    assert row.verdict == "match"


def test_gap_when_outside_ci(tmp_path):
    db = build_db(tmp_path / "r.duckdb", rows=grid("m", "api", VALUES))
    [row] = repro(db, refs_file(tmp_path, [ref(score=0.99)]))
    assert row.verdict == "gap"


def test_no_data_when_model_task_not_run(tmp_path):
    db = build_db(tmp_path / "r.duckdb", rows=grid("m", "api", VALUES))
    [row] = repro(db, refs_file(tmp_path, [ref(task="ifeval")]))
    assert row.verdict == "no_data"


def test_cli_models_excluded(tmp_path):
    db = build_db(tmp_path / "r.duckdb", rows=grid("claude-cli-opus", "cli", VALUES))
    refs = refs_file(tmp_path, [ref(model="claude-cli-opus")])
    assert repro(db, refs) == []
    assert "claude-cli-opus: excluded (via CLI, not a bare model)" in format_repro(db, refs)


def test_provider_source_preferred(tmp_path):
    db = build_db(tmp_path / "r.duckdb", rows=grid("m", "api", VALUES))
    [row] = repro(db, refs_file(tmp_path, [ref(score=0.99, kind="third_party"), ref(score=0.55)]))
    assert row.published == 0.55 and row.verdict == "match"


def test_gap_without_explanation_marked_open(tmp_path):
    db = build_db(tmp_path / "r.duckdb", rows=grid("m", "api", VALUES))
    text = format_repro(db, refs_file(tmp_path, [ref(score=0.99)]))
    assert "OPEN: needs explanation" in text
    text = format_repro(db, refs_file(tmp_path, [ref(score=0.99, explanation="different prompt")]))
    assert "OPEN" not in text and "different prompt" in text


def test_repo_reference_file_loads(tmp_path):
    from pathlib import Path

    db = build_db(tmp_path / "r.duckdb", rows=grid("m", "api", VALUES))
    rows = repro(db, Path(__file__).resolve().parents[1] / "reference_scores.yaml")
    assert all(r.verdict == "no_data" for r in rows)
