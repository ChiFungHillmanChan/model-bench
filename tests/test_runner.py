import json

import pytest

from model_bench import runner
from model_bench.planner import build_plan
from model_bench.providers import claude_cli
from model_bench.runner import BudgetExceeded, run

from conftest import FIXTURES_LIMIT, cli_model, mock_model, tiny_task


def make(models, tasks, tier="smoke"):
    plan = build_plan(models, tasks, tier, {t.name: 3 for t in tasks})
    return plan, {m.id: m for m in models}, {t.name: t for t in tasks}


def go(plan, models, tasks, tmp_path, **kw):
    return run(plan, kw.pop("budget_usd", 1.0), models, tasks, log_root=tmp_path / "logs", display="none", **kw)


def eval_files(path):
    return sorted(path.rglob("*.eval"))


def test_run_writes_logs_and_manifest(tmp_path):
    plan, models, tasks = make([mock_model()], [tiny_task()])
    m = go(plan, models, tasks, tmp_path)
    run_dir = tmp_path / "logs" / m.run_id
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert eval_files(run_dir)


def test_manifest_records_versions_and_git(tmp_path):
    plan, models, tasks = make([mock_model()], [tiny_task()])
    m = go(plan, models, tasks, tmp_path)
    assert m.inspect_ai == "0.3.291"
    assert m.inspect_evals == "0.29.0"
    assert m.git_sha
    assert isinstance(m.git_dirty, bool)


def test_refuses_when_estimate_exceeds_budget(tmp_path):
    plan, models, tasks = make([mock_model()], [tiny_task()])
    plan.est_cost_usd = 5.0
    with pytest.raises(BudgetExceeded):
        go(plan, models, tasks, tmp_path, budget_usd=1.0)
    assert not (tmp_path / "logs").exists() or not eval_files(tmp_path / "logs")


def test_stops_between_chunks_when_spend_crosses_budget(tmp_path, monkeypatch):
    plan, models, tasks = make([mock_model()], [tiny_task("t1"), tiny_task("t2")])
    monkeypatch.setattr(runner, "chunk_spend", lambda logs, model: 2.0)
    m = go(plan, models, tasks, tmp_path, budget_usd=1.0)
    assert m.status == "stopped_budget"
    assert m.spend_usd == 2.0
    run_dir = tmp_path / "logs" / m.run_id
    assert (run_dir / "mock__t1").exists()
    assert not (run_dir / "mock__t2").exists()


def test_same_items_every_run(tmp_path):
    from inspect_ai.log import read_eval_log

    plan, models, tasks = make([mock_model()], [tiny_task(limit=2)])
    ids = []
    for _ in range(2):
        m = go(plan, models, tasks, tmp_path)
        [f] = eval_files(tmp_path / "logs" / m.run_id)
        ids.append(sorted(s.id for s in read_eval_log(str(f)).samples))
    assert ids[0] == ids[1] and len(ids[0]) == 2


def test_resume_reuses_run_id(tmp_path, monkeypatch):
    plan, models, tasks = make([mock_model()], [tiny_task("t1"), tiny_task("t2")])
    real = runner.execute_chunk
    calls = {"n": 0}

    def flaky(task, model, log_dir, display):
        calls["n"] += 1
        if "t2" in str(log_dir) and calls["n"] == 2:
            raise RuntimeError("boom")
        return real(task, model, log_dir, display)

    monkeypatch.setattr(runner, "execute_chunk", flaky)
    first = go(plan, models, tasks, tmp_path)
    assert first.status == "failed"
    run_dir = tmp_path / "logs" / first.run_id
    [t1_log] = eval_files(run_dir / "mock__t1")
    mtime = t1_log.stat().st_mtime_ns
    second = go(plan, models, tasks, tmp_path, run_id=first.run_id)
    assert second.status == "complete"
    assert t1_log.stat().st_mtime_ns == mtime
    assert eval_files(run_dir / "mock__t2")


def test_quota_error_stops_model(tmp_path, monkeypatch):
    async def limited(argv, stdin, timeout_s):
        return 1, FIXTURES_LIMIT.read_text(), ""

    monkeypatch.setattr(claude_cli, "run_cli", limited)
    plan, models, tasks = make([cli_model(), mock_model()], [tiny_task("t1"), tiny_task("t2")])
    m = go(plan, models, tasks, tmp_path)
    run_dir = tmp_path / "logs" / m.run_id
    assert m.status == "stopped_quota"
    assert (run_dir / "claude-cli-haiku__t1").exists()
    assert not (run_dir / "claude-cli-haiku__t2").exists()
    assert eval_files(run_dir / "mock__t1") and eval_files(run_dir / "mock__t2")


def test_ledger_appended(tmp_path):
    plan, models, tasks = make([mock_model()], [tiny_task()])
    m = go(plan, models, tasks, tmp_path)
    lines = (tmp_path / "logs" / "ledger.jsonl").read_text().splitlines()
    assert json.loads(lines[-1])["run_id"] == m.run_id


def test_zero_budget_runs_free_models(tmp_path):
    from conftest import ROOT
    from model_bench.registry import ModelEntry

    local = ModelEntry(id="local", access="local", inspect_model="mockllm/model", caps=["text"])
    plan, models, tasks = make([local], [tiny_task("t1"), tiny_task("t2")])
    m = go(plan, models, tasks, tmp_path, budget_usd=0.0)
    assert m.status == "complete"
    assert m.completed_chunks == ["local__t1", "local__t2"]


def test_same_items_when_task_shuffles_without_seed():
    from model_bench.runner import build_task

    entry = tiny_task("u", limit=5, func="tiny_unseeded")
    plan, _, _ = make([mock_model()], [entry])
    first = [s.id for s in build_task(entry, plan.items[0]).dataset]
    second = [s.id for s in build_task(entry, plan.items[0]).dataset]
    assert first == second and len(first) == 5


def test_failed_api_chunk_not_marked_complete_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setenv("MB_TEST_FAIL", "1")
    plan, models, tasks = make([mock_model()], [tiny_task("t1", func="tiny_flaky")])
    first = go(plan, models, tasks, tmp_path)
    assert first.status == "failed"
    assert first.completed_chunks == []
    assert "simulated provider error" in (first.error or "")
    monkeypatch.setenv("MB_TEST_FAIL", "0")
    second = go(plan, models, tasks, tmp_path, run_id=first.run_id)
    assert second.status == "complete"
    assert second.completed_chunks == ["mock__t1"]
