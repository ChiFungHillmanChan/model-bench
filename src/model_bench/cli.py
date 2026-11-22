"""bench CLI."""

from pathlib import Path

import typer
from dotenv import load_dotenv

from model_bench.registry import load_models, load_tasks

MODELS_PATH = Path("models.yaml")
TASKS_PATH = Path("tasks/registry.yaml")

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
    for m in load_models(MODELS_PATH).values():
        price = f"${m.price.input}/{m.price.output} per M" if m.price else "quota"
        caps = ",".join(sorted(m.caps))
        typer.echo(f"{m.id:22} {m.access:6} sealed_ok={str(m.sealed_ok):5} {price:22} {caps}")


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
