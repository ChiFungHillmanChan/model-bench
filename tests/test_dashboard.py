import json
import re
from datetime import datetime, timezone

import duckdb

from model_bench.dashboard import build_data, write_dashboard
from test_report import build_db, grid


def manifest(run_id, created, finished, models, tier="smoke"):
    return {
        "run_id": run_id, "created": created, "finished": finished, "tier": tier,
        "git_sha": "abc", "git_dirty": False, "inspect_ai": "0.3.291", "inspect_evals": "0.29.0",
        "cli_versions": {}, "budget_usd": 1.0, "spend_usd": 0.02, "status": "complete",
        "models": [{"id": m, "access": a, "inspect_model": f"x/{m}", "caps": ["text"], "params": p}
                   for m, a, p in models],
        "tasks": [{"name": "gpqa", "source": "inspect_evals/gpqa", "category": "reasoning", "needs": ["text"],
                   "tiers": {tier: {"limit": 10, "epochs": 1}}, "split": "test", "version": "1",
                   "scorer_kind": "deterministic", "est_tokens_per_sample": [1, 1]}],
    }


def two_run_db(tmp_path):
    db = build_db(tmp_path / "d.duckdb", run_id="r1",
                  created=datetime(2026, 11, 20, tzinfo=timezone.utc),
                  rows=grid("opus", "cli", [1, 1, 0, 1]) + grid("qwen", "local", [1, 0, 0, 0], cost=0.0))
    con = duckdb.connect(str(db))
    con.execute("update runs set manifest_json=? where run_id='r1'", [json.dumps(manifest(
        "r1", "2026-11-20T10:00:00Z", "2026-11-20T10:05:00Z",
        [("opus", "cli", {"effort": "high"}), ("qwen", "local", {})]))])
    con.execute("insert into runs values (?,?,?,?,?,?,?,?,?,?)",
                ["r2", datetime(2026, 11, 22, tzinfo=timezone.utc), "smoke", "abc", False, "0.3.291", "0.29.0",
                 "complete", 0.0, json.dumps(manifest("r2", "2026-11-22T10:00:00Z", "2026-11-22T10:02:00Z",
                                                      [("opus", "cli", {"effort": "high"})]))])
    for i, v in enumerate([1, 1, 1, 1]):
        con.execute("insert into samples (run_id, task, model_id, sample_id, epoch, reported_model, access,"
                    " input_tokens, output_tokens, total_cost, total_time, error, input_text, target, answer)"
                    " values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    ["r2", "gpqa", "opus", f"s{i}", 1, "opus", "cli", 1, 1, 0.0, 2.0, None, f"Question s{i}?", "A", "A"])
        con.execute("insert into scores values (?,?,?,?,?,?,?,?)", ["r2", "gpqa", "opus", f"s{i}", 1, "choice", v, "x"])
    con.execute("insert into samples (run_id, task, model_id, sample_id, epoch, reported_model, access,"
                " input_tokens, output_tokens, total_cost, total_time, error, input_text, target, answer)"
                " values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                ["r2", "gpqa", "opus", "s9", 1, "opus", "cli", 0, 0, 0.0, 1.0, "CLIUsageLimit: x", "Question s9?", "A", None])
    con.close()
    return db


def test_build_data_results_per_run_task_model(tmp_path):
    data = build_data(two_run_db(tmp_path))
    r1 = {(r["run_id"], r["model"]): r for r in data["results"]}
    opus1 = r1[("r1", "opus")]
    assert opus1["field"] == "reasoning" and opus1["task"] == "gpqa"
    assert opus1["score_pct"] == 75.0 and opus1["correct"] == 3 and opus1["answered"] == 4
    assert opus1["thinking"] == "high" and opus1["access"] == "cli"
    assert opus1["date"].startswith("2026-11-20")
    opus2 = r1[("r2", "opus")]
    assert opus2["errors"] == 1 and opus2["error_rate_pct"] == 20.0  # 1 of 5 samples errored
    assert opus2["score_pct"] == 100.0 and opus2["ci_low_pct"] is None  # perfect score: no interval


def test_build_data_runs_and_models(tmp_path):
    data = build_data(two_run_db(tmp_path))
    assert [r["run_id"] for r in data["runs"]] == ["r2", "r1"]  # newest first
    run1 = [r for r in data["runs"] if r["run_id"] == "r1"][0]
    assert run1["duration_min"] == 5.0 and run1["models"] == ["opus", "qwen"]
    assert data["models"] == ["opus", "qwen"]  # fixed order -> fixed colours


def test_dashboard_carries_no_questions_or_answers(tmp_path):
    # Question text, answer keys and model answers stay in results.duckdb, never in the page.
    data = build_data(two_run_db(tmp_path))
    assert "questions" not in data
    out = tmp_path / "d.html"
    (tmp_path / "b").mkdir()
    write_dashboard(two_run_db(tmp_path / "b"), out)
    html = out.read_text()
    assert "s-q" not in html and "questionsCard" not in html and "Mistakes" not in html


def test_write_dashboard_is_self_contained(tmp_path):
    out = tmp_path / "dashboard.html"
    write_dashboard(two_run_db(tmp_path), out)
    html = out.read_text()
    assert html.startswith("<!doctype html>")
    assert not re.search(r"<script[^>]+src=", html)  # no external scripts: file works offline
    for heading in ["Bench score", "Time per question", "Cost per question", "Score by task", "Score vs cost",
                    "Best score per benchmark", "Thinking levels",
                    "Degradation watch", "Leaderboard", "Runs"]:
        assert heading in html
    data = json.loads(re.search(r'<script id="bench-data" type="application/json">(.*?)</script>', html, re.S).group(1))
    assert data["models"] == ["opus", "qwen"]


def test_dashboard_escapes_script_breakout(tmp_path):
    db = two_run_db(tmp_path)
    con = duckdb.connect(str(db))
    con.execute("update samples set input_text='</script><script>alert(1)</script>' where sample_id='s0'")
    con.close()
    out = tmp_path / "dashboard.html"
    write_dashboard(db, out)
    assert "</script><script>alert(1)" not in out.read_text()


def test_build_data_model_info_uses_registry_label_and_latest_run(tmp_path):
    reg = tmp_path / "models.yaml"
    reg.write_text("models:\n  - id: opus\n    label: Claude Opus\n    access: cli\n"
                   "    inspect_model: claude-cli/x\n    caps: [text]\n")
    data = build_data(two_run_db(tmp_path), models_path=reg)
    opus, qwen = data["model_info"]["opus"], data["model_info"]["qwen"]
    assert opus["label"] == "Claude Opus" and qwen["label"] == "qwen"  # unknown to registry -> id
    assert opus["access"] == "cli" and opus["thinking"] == "high" and opus["last_run"] == "r2"
    assert opus["reported_model"] == "opus" and opus["runs"] == 2
    assert qwen["last_run"] == "r1" and qwen["runs"] == 1


def test_build_data_per_question_cost_and_tokens(tmp_path):
    data = build_data(two_run_db(tmp_path))
    rows = {(r["run_id"], r["model"]): r for r in data["results"]}
    assert rows[("r1", "opus")]["cost_per_q"] is None  # subscription: no per-call price
    assert rows[("r1", "qwen")]["cost_per_q"] == 0.0
    q = rows[("r1", "qwen")]
    assert q["out_tokens_per_q"] == round(q["out_tokens"] / q["samples"], 1)
    assert data["tasks"]["gpqa"] == {"title": "gpqa", "field": "reasoning", "about": ""}


def test_dashboard_pools_error_rates(tmp_path):
    db = two_run_db(tmp_path)
    con = duckdb.connect(str(db))
    rows = [("s0", {"values_bad": 1, "values_n": 10, "missed_bad": 1, "missed_n": 1, "false_alarm_bad": 0, "false_alarm_n": 0}),
            ("s1", {"values_bad": 3, "values_n": 10, "missed_bad": 0, "missed_n": 1, "false_alarm_bad": 1, "false_alarm_n": 1}),
            ("s2", {"values_bad": 0, "values_n": 0, "missed_bad": 0, "missed_n": 0, "false_alarm_bad": 0, "false_alarm_n": 1})]
    for sid, rates in rows:
        for k, v in rates.items():
            con.execute("insert into scores values (?,?,?,?,?,?,?,?)", ["r2", "gpqa", "opus", sid, 1, f"rates.{k}", v, str(v)])
    con.close()
    data = build_data(db)
    r = next(x for x in data["results"] if x["run_id"] == "r2" and x["model"] == "opus")
    assert r["rates_pct"]["values"] == 20.0
    assert r["rates_pct"]["missed"] == 50.0 and r["rates_pct"]["false_alarm"] == 50.0
    assert r["rates_pct"]["balanced_accuracy"] == 50.0
    other = next(x for x in data["results"] if x["run_id"] == "r1")
    assert other["rates_pct"] == {}


def test_dashboard_task_titles_come_from_the_registry(tmp_path):
    reg = tmp_path / "registry.yaml"
    reg.write_text("""
tasks:
  - name: gpqa
    source: local:tasks/x.py@x
    category: reasoning
    title: "Hidden task"
    about: "Described privately."
    needs: [text]
    tiers: {smoke: {limit: 1, epochs: 1}}
    split: test
    version: "1"
    scorer_kind: deterministic
    est_tokens_per_sample: [1, 1]
""")
    data = build_data(two_run_db(tmp_path), tasks_path=reg)
    assert data["tasks"]["gpqa"]["title"] == "Hidden task" and data["tasks"]["gpqa"]["about"] == "Described privately."


def test_rate_labels_cover_the_pooled_metrics():
    from model_bench.dashboard import RATE_LABELS

    for m in ["values", "missed", "false_alarm", "facts", "overflow", "invented", "balanced_accuracy", "scorer_error"]:
        assert m in RATE_LABELS


def test_model_info_groups_effort_variants_under_one_base(tmp_path):
    # The thinking-level chart joins one model's effort settings into one line.
    reg = tmp_path / "models.yaml"
    reg.write_text("models:\n"
                   "  - {id: opus, label: Claude Opus, access: cli, inspect_model: claude-cli/x, caps: [text], params: {effort: high}}\n"
                   "  - {id: qwen, label: 'Claude Opus [low]', access: cli, inspect_model: claude-cli/x, caps: [text], params: {effort: low}}\n")
    data = build_data(two_run_db(tmp_path), models_path=reg)
    opus, low = data["model_info"]["opus"], data["model_info"]["qwen"]
    assert opus["base"] == low["base"] == "Claude Opus"
    assert opus["effort"] == "high" and low["effort"] == "low"


def test_registry_category_overrides_the_one_saved_in_old_manifests(tmp_path):
    # Re-categorising a task must relabel past runs too, not only runs made after the change.
    reg = tmp_path / "registry.yaml"
    reg.write_text("""
tasks:
  - name: gpqa
    source: local:tasks/x.py@x
    category: hard_reasoning
    needs: [text]
    tiers: {smoke: {limit: 1, epochs: 1}}
    split: test
    version: "1"
    scorer_kind: deterministic
    est_tokens_per_sample: [1, 1]
""")
    data = build_data(two_run_db(tmp_path), tasks_path=reg)
    assert data["tasks"]["gpqa"]["field"] == "hard_reasoning"
    assert {r["field"] for r in data["results"] if r["task"] == "gpqa"} == {"hard_reasoning"}
