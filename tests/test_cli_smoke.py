from typer.testing import CliRunner

from model_bench.cli import app


def test_help_lists_commands():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in ["models", "plan", "run", "ingest", "report", "compare", "repro"]:
        assert name in result.output


def test_bench_models_lists_registry():
    result = CliRunner().invoke(app, ["models"])
    assert result.exit_code == 0, result.output
    assert "claude-cli-opus" in result.output
    assert "quota" in result.output  # cli models have no price
    assert "deepseek-chat" in result.output


def test_bench_plan_shows_items_skips_and_estimate(monkeypatch):
    import model_bench.cli as cli

    monkeypatch.setattr(cli, "dataset_size", lambda t: 30)
    result = CliRunner().invoke(app, ["plan", "--models", "deepseek-chat,claude-cli-opus", "--tier", "smoke"])
    assert result.exit_code == 0, result.output
    assert "aime2025" in result.output
    assert "missing capability: reasoning" in result.output  # deepseek-chat lacks reasoning
    assert "Estimated cost: $" in result.output
    assert "Quota: claude-cli-opus: 40 samples" in result.output  # 4 tasks x 10


def test_bench_plan_unknown_model_errors():
    result = CliRunner().invoke(app, ["plan", "--models", "nope", "--tier", "smoke"])
    assert result.exit_code != 0
    assert "unknown model: nope" in result.output


def _tmp_registries(tmp_path, monkeypatch):
    import model_bench.cli as cli
    from conftest import TINY

    (tmp_path / "models.yaml").write_text(
        "models:\n  - {id: mock, access: api, inspect_model: mockllm/model, caps: [text],"
        " price: {input: 1.0, output: 1.0, as_of: 2026-11-22}}\n"
    )
    (tmp_path / "tasks.yaml").write_text(
        "tasks:\n  - {name: tiny, source: '" + TINY + "', category: t, needs: [text],"
        " tiers: {smoke: {limit: null, epochs: 1}}, split: test, version: '1',"
        " scorer_kind: deterministic, est_tokens_per_sample: [10, 10]}\n"
    )
    monkeypatch.setattr(cli, "MODELS_PATH", tmp_path / "models.yaml")
    monkeypatch.setattr(cli, "TASKS_PATH", tmp_path / "tasks.yaml")
    monkeypatch.setattr(cli, "LOG_ROOT", tmp_path / "logs")
    monkeypatch.setattr(cli, "dataset_size", lambda t: 3)


def test_bench_run_asks_confirmation_for_paid_models(tmp_path, monkeypatch):
    _tmp_registries(tmp_path, monkeypatch)
    result = CliRunner().invoke(app, ["run", "--models", "mock", "--tier", "smoke", "--budget", "1"], input="n\n")
    assert "Aborted" in result.output
    assert not (tmp_path / "logs").exists()


def test_bench_run_yes_runs_and_reports_status(tmp_path, monkeypatch):
    _tmp_registries(tmp_path, monkeypatch)
    result = CliRunner().invoke(app, ["run", "--models", "mock", "--tier", "smoke", "--budget", "1", "--yes"])
    assert result.exit_code == 0, result.output
    assert "status: complete" in result.output
    assert list((tmp_path / "logs").glob("*/manifest.json"))
