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
