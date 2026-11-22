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
