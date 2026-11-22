"""bench CLI."""

import typer
from dotenv import load_dotenv

app = typer.Typer(no_args_is_help=True, help="Private model benchmark harness.")


@app.callback()
def _main() -> None:
    load_dotenv()


def _todo() -> None:
    typer.echo("not implemented")
    raise typer.Exit(2)


@app.command()
def models() -> None:
    """List registered models and their capabilities."""
    _todo()


@app.command()
def plan() -> None:
    """Dry run: show tasks, items and estimated cost."""
    _todo()


@app.command()
def run() -> None:
    """Run a plan."""
    _todo()


@app.command()
def ingest() -> None:
    """Load Inspect logs into DuckDB."""
    _todo()


@app.command()
def report() -> None:
    """Write a report for a run."""
    _todo()


@app.command()
def compare() -> None:
    """Paired comparison of two models."""
    _todo()


@app.command()
def repro() -> None:
    """Compare harness scores with published scores."""
    _todo()
