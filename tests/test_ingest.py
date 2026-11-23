import duckdb
import pytest

from conftest import FIX_DIR, cli_model, mock_model, tiny_task
from model_bench import runner
from model_bench.ingest import ingest
from model_bench.planner import build_plan
from model_bench.providers import claude_cli


def do_run(tmp_path, models, tasks, **kw):
    plan = build_plan(models, tasks, "smoke", {t.name: 3 for t in tasks})
    return runner.run(plan, 1.0, {m.id: m for m in models}, {t.name: t for t in tasks},
                      log_root=tmp_path / "logs", display="none", **kw)


def q(db, sql):
    return duckdb.connect(str(db), read_only=True).execute(sql).fetchall()


@pytest.fixture
def two_epoch_run(tmp_path):
    do_run(tmp_path, [mock_model()], [tiny_task(epochs=2)])
    return tmp_path


def test_ingest_row_counts(two_epoch_run):
    db = two_epoch_run / "r.duckdb"
    ingest(two_epoch_run / "logs", db)
    assert q(db, "select count(*) from samples")[0][0] == 6
    assert q(db, "select count(*) from scores")[0][0] == 6
    assert q(db, "select count(*) from runs")[0][0] == 1


def test_ingest_idempotent(two_epoch_run):
    db = two_epoch_run / "r.duckdb"
    ingest(two_epoch_run / "logs", db)
    ingest(two_epoch_run / "logs", db)
    assert q(db, "select count(*) from samples")[0][0] == 6
    assert q(db, "select count(*) from scores")[0][0] == 6


def test_ingest_after_resume_no_duplicates(tmp_path, monkeypatch):
    real = runner.execute_chunk
    state = {"n": 0}

    def flaky(task, model, log_dir, display, roles=None):
        state["n"] += 1
        if state["n"] == 2:
            raise RuntimeError("boom")
        return real(task, model, log_dir, display, roles)

    monkeypatch.setattr(runner, "execute_chunk", flaky)
    first = do_run(tmp_path, [mock_model()], [tiny_task("t1"), tiny_task("t2")])
    do_run(tmp_path, [mock_model()], [tiny_task("t1"), tiny_task("t2")], run_id=first.run_id)
    db = tmp_path / "r.duckdb"
    ingest(tmp_path / "logs", db)
    dupes = q(db, "select count(*) from (select run_id, task, model_id, sample_id, epoch, count(*) c "
                  "from samples group by all having c > 1)")[0][0]
    assert dupes == 0
    assert q(db, "select count(*) from samples")[0][0] == 6


def test_errored_sample_has_error_and_no_score(tmp_path, monkeypatch):
    ok = (FIX_DIR / "claude_ok.json").read_text()

    async def fake(argv, stdin, timeout_s):
        return (0, "not json", "") if "Say B" in stdin else (0, ok, "")

    monkeypatch.setattr(claude_cli, "run_cli", fake)
    do_run(tmp_path, [cli_model()], [tiny_task()])
    db = tmp_path / "r.duckdb"
    ingest(tmp_path / "logs", db)
    [(err,)] = q(db, "select error from samples where sample_id = 'b'")
    assert "CLIBadOutput" in err
    assert q(db, "select count(*) from scores where sample_id = 'b'")[0][0] == 0
    assert q(db, "select count(*) from scores")[0][0] == 2


def test_rebuild_recreates_from_logs(two_epoch_run):
    db = two_epoch_run / "r.duckdb"
    ingest(two_epoch_run / "logs", db)
    db.unlink()
    ingest(two_epoch_run / "logs", db, rebuild=True)
    assert q(db, "select count(*) from samples")[0][0] == 6


def test_model_id_maps_back_to_registry(two_epoch_run):
    db = two_epoch_run / "r.duckdb"
    ingest(two_epoch_run / "logs", db)
    assert q(db, "select distinct model_id, access from samples") == [("mock", "api")]


def test_value_num_from_inspect_letters(two_epoch_run):
    db = two_epoch_run / "r.duckdb"
    ingest(two_epoch_run / "logs", db)
    values = {v for (v,) in q(db, "select distinct value_num from scores")}
    assert values <= {0.0, 1.0}


def test_ingest_keeps_question_target_and_answer(two_epoch_run):
    db = two_epoch_run / "r.duckdb"
    ingest(two_epoch_run / "logs", db)
    rows = q(db, "select sample_id, input_text, target, answer from samples where epoch = 1 order by sample_id")
    assert rows[0][0] == "a" and rows[0][1] == "Say A" and rows[0][2] == "A"
    assert rows[0][3]  # mockllm's completion text
