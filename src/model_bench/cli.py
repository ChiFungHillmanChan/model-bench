"""bench CLI."""

from pathlib import Path

import typer
from dotenv import load_dotenv

from model_bench.planner import Plan, build_plan, dataset_size
from model_bench.registry import ModelEntry, TaskEntry, load_models, load_tasks

MODELS_PATH = Path("models.yaml")
TASKS_PATH = Path("tasks/registry.yaml")
LOG_ROOT = Path("logs")
DB_PATH = Path("results.duckdb")
REPORTS_DIR = Path("reports")

app = typer.Typer(no_args_is_help=True, help="Private model benchmark harness.")


@app.callback()
def _main() -> None:
    load_dotenv()


def _select_models(spec: str) -> list[ModelEntry]:
    registry = load_models(MODELS_PATH)
    if spec == "all":
        return list(registry.values())
    out = []
    for mid in spec.split(","):
        if mid not in registry:
            typer.echo(f"unknown model: {mid}")
            raise typer.Exit(1)
        out.append(registry[mid])
    return out


def _select_tasks(spec: str | None) -> list[TaskEntry]:
    registry = load_tasks(TASKS_PATH)
    if not spec:
        return list(registry.values())
    out = []
    for name in spec.split(","):
        if name not in registry:
            typer.echo(f"unknown task: {name}")
            raise typer.Exit(1)
        out.append(registry[name])
    return out


def make_plan(models: str, tier: str, tasks: str | None) -> Plan:
    selected_models = _select_models(models)
    selected_tasks = [t for t in _select_tasks(tasks) if tier in t.tiers]
    sizes = {t.name: dataset_size(t) for t in selected_tasks}
    return build_plan(selected_models, selected_tasks, tier, sizes)


def print_plan(p: Plan) -> None:
    typer.echo(f"Tier: {p.tier}")
    typer.echo(f"{'model':22} {'task':16} {'items':>6} {'epochs':>6} {'samples':>8} {'est $':>9}")
    for i in p.items:
        cost = "quota" if i.est_cost_usd is None else f"{i.est_cost_usd:.4f}"
        typer.echo(f"{i.model_id:22} {i.task_name:16} {i.n_items:>6} {i.epochs:>6} {i.est_samples:>8} {cost:>9}")
    if p.skips:
        typer.echo("Skipped:")
        for s in p.skips:
            typer.echo(f"  {s.model_id:22} {s.task_name:16} {s.reason}")
    typer.echo(f"Estimated cost: ${p.est_cost_usd:.4f} (API models only)")
    for mid, n in p.quota_samples.items():
        typer.echo(f"Quota: {mid}: {n} samples")


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
def plan(
    models: str = typer.Option(..., help="Comma-separated model ids, or 'all'"),
    tier: str = typer.Option("smoke"),
    tasks: str | None = typer.Option(None, help="Comma-separated task names"),
) -> None:
    """Dry run: show tasks, items and estimated cost."""
    print_plan(make_plan(models, tier, tasks))


@app.command()
def run(
    models: str = typer.Option(..., help="Comma-separated model ids, or 'all'"),
    tier: str = typer.Option("smoke"),
    tasks: str | None = typer.Option(None, help="Comma-separated task names"),
    budget: float = typer.Option(..., help="Hard cap on real API spend in USD"),
    run_id: str | None = typer.Option(None, help="Resume an existing run"),
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation prompt"),
) -> None:
    """Run a plan (resumable with --run-id)."""
    from model_bench.runner import BudgetExceeded
    from model_bench.runner import run as run_plan

    p = make_plan(models, tier, tasks)
    print_plan(p)
    registry = load_models(MODELS_PATH)
    needs_ok = any(registry[i.model_id].access in ("api", "cli") for i in p.items)
    if needs_ok and not yes:
        typer.confirm("This spends API money or subscription quota. Continue?", abort=True)
    try:
        m = run_plan(p, budget, registry, load_tasks(TASKS_PATH), run_id=run_id, log_root=LOG_ROOT)
    except BudgetExceeded as e:
        typer.echo(str(e))
        raise typer.Exit(1)
    typer.echo(f"run_id: {m.run_id}  status: {m.status}  spend: ${m.spend_usd:.4f}")
    if m.error:
        typer.echo(f"error: {m.error}  (resume with --run-id {m.run_id})")
    if m.quota_stopped:
        typer.echo(f"quota stopped: {', '.join(m.quota_stopped)}  (resume later with --run-id {m.run_id})")


@app.command()
def ingest(rebuild: bool = typer.Option(False, help="Recreate the DB from logs/")) -> None:
    """Load Inspect logs into DuckDB."""
    from model_bench.ingest import ingest as do_ingest

    n = do_ingest(LOG_ROOT, DB_PATH, rebuild=rebuild)
    typer.echo(f"ingested {n} samples into {DB_PATH}")


@app.command()
def report(run_id: str | None = typer.Option(None, help="Defaults to the latest run")) -> None:
    """Write a markdown report for a run to reports/ and print it."""
    import duckdb

    from model_bench.report import report as make_report

    text = make_report(DB_PATH, run_id)
    if run_id is None:
        con = duckdb.connect(str(DB_PATH), read_only=True)
        row = con.execute("select run_id from runs order by created desc limit 1").fetchone()
        con.close()
        run_id = row[0] if row else "none"
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / f"{run_id}.md").write_text(text)
    typer.echo(text)


@app.command()
def compare(a: str, b: str, tier: str | None = typer.Option(None)) -> None:
    """Paired comparison of two models on shared items."""
    from model_bench.report import compare as make_compare

    typer.echo(make_compare(DB_PATH, a, b, tier))


@app.command()
def repro() -> None:
    """Compare harness scores with published scores."""
    _todo()
