import re
from datetime import datetime, timezone

import duckdb
import pytest

from model_bench.ingest import SCHEMA
from model_bench.report import compare, report

CI_RE = re.compile(r"\d\.\d{3} \[-?\d\.\d{3}, -?\d\.\d{3}\]")


def build_db(path, run_id="r1", dirty=False, rows=(), created=None):
    """rows: (task, model_id, access, sample_id, epoch, value or None, error, cost, time, out_tokens)"""
    con = duckdb.connect(str(path))
    con.execute(SCHEMA)
    con.execute(
        "insert into runs values (?,?,?,?,?,?,?,?,?,?)",
        [run_id, created or datetime(2026, 11, 22, tzinfo=timezone.utc), "smoke", "abc123", dirty,
         "0.3.291", "0.29.0", "complete", 0.5, "{}"],
    )
    for task, model, access, sid, epoch, value, error, cost, t, out in rows:
        con.execute("insert into samples values (?,?,?,?,?,?,?,?,?,?,?,?)",
                    [run_id, task, model, sid, epoch, model, access, 100, out, cost, t, error])
        if value is not None:
            con.execute("insert into scores values (?,?,?,?,?,?,?,?)",
                        [run_id, task, model, sid, epoch, "choice", value, "x"])
    con.close()
    return path


def grid(model, access, values, task="gpqa", epochs=1, cost=0.01):
    rows = []
    for i, v in enumerate(values):
        for e in range(1, epochs + 1):
            rows.append((task, model, access, f"s{i}", e, v, None, cost, 1.0 + i, 50))
    return rows


@pytest.fixture
def db(tmp_path):
    rows = grid("deepseek-chat", "api", [1, 1, 0, 1, 0, 1]) + grid("claude-cli-opus", "cli", [1, 1, 1, 1, 0, 1], cost=0.0)
    return build_db(tmp_path / "r.duckdb", rows=rows)


def score_lines(text):
    return [l for l in text.splitlines() if l.startswith("| gpqa")]


def test_report_every_score_has_ci(db):
    lines = score_lines(report(db))
    assert len(lines) == 2
    for line in lines:
        assert CI_RE.search(line) or "n/a" in line


def test_cli_rows_labelled(db):
    [cli_line] = [l for l in score_lines(report(db)) if "claude-cli-opus" in l]
    assert "via CLI" in cli_line


def test_dirty_tree_flagged(tmp_path):
    db = build_db(tmp_path / "d.duckdb", dirty=True, rows=grid("m", "api", [1, 0]))
    assert "WARNING: run from a dirty working tree" in report(db)


def test_errors_counted_not_scored(tmp_path):
    rows = grid("m", "api", [1, 1, 0]) + [("gpqa", "m", "api", "s9", 1, None, "CLIBadOutput: x", 0.0, 1.0, 0)]
    text = report(build_db(tmp_path / "e.duckdb", rows=rows))
    assert "errors: 1" in text
    [line] = score_lines(text)
    assert line.split("|")[4].strip().startswith("0.667")


def test_report_includes_cost_latency_tokens(db):
    header = [l for l in report(db).splitlines() if l.startswith("| task")][0]
    for col in ["cost_usd", "p50_s", "p95_s", "out_tokens"]:
        assert col in header


def test_pass_k_column_na_when_epochs_1(db):
    for line in score_lines(report(db)):
        assert line.split("|")[7].strip() == "n/a"


def test_pass_k_value_when_epochs_3(tmp_path):
    text = report(build_db(tmp_path / "k.duckdb", rows=grid("m", "api", [1, 0], epochs=3)))
    [line] = score_lines(text)
    assert line.split("|")[7].strip() == "0.500"


def test_quality_vs_cost_section(tmp_path):
    rows = (grid("cheap", "api", [1, 0, 1, 0], cost=0.001) + grid("pricey", "api", [1, 1, 1, 0], cost=0.1)
            + grid("bad", "api", [0, 0, 1, 0], cost=0.2) + grid("claude-cli-opus", "cli", [1, 1, 1, 1], cost=0.0))
    text = report(build_db(tmp_path / "q.duckdb", rows=rows))
    assert "## Quality vs cost" in text
    section = text.split("## Quality vs cost")[1]
    line = {name: l for l in section.splitlines() for name in ["cheap", "pricey", "bad", "claude-cli-opus"] if f"| {name}" in l}
    assert "frontier" in line["cheap"] and "frontier" in line["pricey"]
    assert "frontier" not in line["bad"]
    assert "quota" in line["claude-cli-opus"]


def test_compare_paired_diff_and_mcnemar(db):
    text = compare(db, "claude-cli-opus", "deepseek-chat")
    header = [l for l in text.splitlines() if l.startswith("| task")][0]
    assert "diff" in header and "McNemar p" in header
    [line] = [l for l in text.splitlines() if l.startswith("| gpqa")]
    assert CI_RE.search(line)
    assert "0.167" in line  # 5/6 - 4/6


def test_compare_unknown_model_message(db):
    assert "no runs found for nobody" in compare(db, "nobody", "deepseek-chat")
