from typer.testing import CliRunner

from model_bench.cli import app


def test_help_lists_commands():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in ["models", "plan", "run", "ingest", "report", "compare", "repro"]:
        assert name in result.output
