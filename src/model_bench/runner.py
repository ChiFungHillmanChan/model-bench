"""Run a Plan through Inspect eval_set with a manifest, budget guard, resume and ledger."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Literal

from inspect_ai import Task, eval_set, task_with
from inspect_ai.log import EvalLog, read_eval_log
from inspect_ai.model import ModelUsage
from pydantic import BaseModel

from model_bench.planner import Plan, PlanItem, load_inspect_task
from model_bench.registry import ModelEntry, TaskEntry

SHUFFLE_SEED = 20261122
CLI_BINARIES = {"claude-cli": "claude", "codex-cli": "codex", "gemini-cli": "gemini"}
Status = Literal["running", "complete", "stopped_budget", "stopped_quota", "failed"]


class BudgetExceeded(RuntimeError):
    pass


class Manifest(BaseModel):
    run_id: str
    created: datetime
    finished: datetime | None = None
    tier: str
    git_sha: str
    git_dirty: bool
    inspect_ai: str
    inspect_evals: str
    cli_versions: dict[str, str]
    models: list[ModelEntry]
    tasks: list[TaskEntry]
    budget_usd: float
    spend_usd: float = 0.0
    status: Status = "running"
    completed_chunks: list[str] = []
    quota_stopped: list[str] = []
    error: str | None = None


def build_task(entry: TaskEntry, item: PlanItem) -> Task:
    """Fresh task, same items for every model and run of a tier, tier epochs applied."""
    task = load_inspect_task(entry)
    dataset = task.dataset
    dataset.sort(key=lambda s: str(s.id))  # undo any unseeded shuffle inside the task itself
    dataset.shuffle(seed=SHUFFLE_SEED)
    if item.limit is not None:
        dataset = dataset[: item.limit]
    return task_with(task, dataset=dataset, epochs=item.epochs, name=entry.name)


# Spec: per-CLI concurrency stays low (1-2). Must be explicit: Inspect's default adaptive
# concurrency ignores ModelAPI.max_connections() and ran 40-80 CLI processes at once.
CLI_MAX_CONNECTIONS = 2


def execute_chunk(task: Task, model: ModelEntry, log_dir: Path, display: str | None) -> list[EvalLog]:
    kwargs: dict = {}
    if model.access == "cli":
        kwargs["model_args"] = dict(model.params)
        kwargs["fail_on_error"] = False
        kwargs["max_connections"] = CLI_MAX_CONNECTIONS
    else:
        kwargs.update(model.params)  # GenerateConfig fields (temperature, max_tokens, ...)
    _, logs = eval_set(
        tasks=[task],
        model=model.inspect_model,
        log_dir=str(log_dir),
        retry_attempts=0,
        display=display,
        **kwargs,
    )
    # eval_set returns header-only logs; sample errors are needed for the quota check
    return [read_eval_log(log.location) for log in logs]


def usage_cost(usage: ModelUsage, model: ModelEntry) -> float:
    """Real API spend for one usage record: tokens x registry price. CLI and local cost 0."""
    if model.access != "api" or model.price is None:
        return 0.0
    p = model.price
    cache_read = usage.input_tokens_cache_read or 0
    cache_write = usage.input_tokens_cache_write or 0
    # Inspect convention: input_tokens already excludes cache reads and writes
    return (
        usage.input_tokens * p.input
        + cache_read * (p.input_cache_read or p.input)
        + cache_write * (p.input_cache_write or p.input)
        + usage.output_tokens * p.output
    ) / 1e6


def chunk_spend(logs: list[EvalLog], model: ModelEntry) -> float:
    return sum(usage_cost(u, model) for log in logs for u in (log.stats.model_usage or {}).values())


def _quota_hit(logs: list[EvalLog]) -> bool:
    return any(
        s.error is not None and "CLIUsageLimit" in s.error.message
        for log in logs
        for s in (log.samples or [])
    )


def _git() -> tuple[str, bool]:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", True
    return sha, dirty


def _cli_versions(models: list[ModelEntry]) -> dict[str, str]:
    out = {}
    for m in models:
        provider = m.inspect_model.split("/", 1)[0]
        binary = CLI_BINARIES.get(provider)
        if binary and binary not in out:
            try:
                out[binary] = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=30).stdout.strip()
            except (FileNotFoundError, subprocess.TimeoutExpired):
                out[binary] = "unavailable"
    return out


def _save(manifest: Manifest, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2))


def run(
    plan: Plan,
    budget_usd: float,
    models: dict[str, ModelEntry],
    tasks: dict[str, TaskEntry],
    run_id: str | None = None,
    log_root: Path = Path("logs"),
    display: str | None = None,
) -> Manifest:
    existing = log_root / run_id / "manifest.json" if run_id else None
    done = (
        set(Manifest.model_validate_json(existing.read_text()).completed_chunks)
        if existing and existing.exists() else set()
    )
    spent = Manifest.model_validate_json(existing.read_text()).spend_usd if done else 0.0
    remaining = plan.est_cost_usd if not done else sum(
        i.est_cost_usd or 0 for i in plan.items if f"{i.model_id}__{i.task_name}" not in done
    )
    if spent + remaining > budget_usd:
        raise BudgetExceeded(
            f"estimated ${remaining:.4f} (plus ${spent:.4f} already spent) exceeds budget ${budget_usd:.4f}"
        )

    if existing and existing.exists():
        manifest = Manifest.model_validate_json(existing.read_text())
        manifest.status, manifest.error, manifest.quota_stopped = "running", None, []
        manifest.budget_usd = budget_usd
    else:
        sha, dirty = _git()
        used_models = [models[mid] for mid in dict.fromkeys(i.model_id for i in plan.items)]
        manifest = Manifest(
            run_id=run_id or f"{datetime.now():%Y%m%d-%H%M%S}-{plan.tier}",
            created=datetime.now(timezone.utc),
            tier=plan.tier,
            git_sha=sha,
            git_dirty=dirty,
            inspect_ai=version("inspect-ai"),
            inspect_evals=version("inspect-evals"),
            cli_versions=_cli_versions(used_models),
            models=used_models,
            tasks=[tasks[n] for n in dict.fromkeys(i.task_name for i in plan.items)],
            budget_usd=budget_usd,
        )
    run_dir = log_root / manifest.run_id
    _save(manifest, run_dir)

    failures: list[str] = []
    try:
        for item in plan.items:
            chunk = f"{item.model_id}__{item.task_name}"
            if chunk in manifest.completed_chunks or item.model_id in manifest.quota_stopped:
                continue
            next_cost = item.est_cost_usd or 0
            if manifest.spend_usd + next_cost > budget_usd or (0 < manifest.spend_usd >= budget_usd):
                manifest.status = "stopped_budget"
                break
            model = models[item.model_id]
            logs = execute_chunk(build_task(tasks[item.task_name], item), model, run_dir / chunk, display)
            manifest.spend_usd += chunk_spend(logs, model)
            bad = [log for log in logs if log.status != "success"]
            if _quota_hit(logs):
                manifest.quota_stopped.append(item.model_id)
            elif bad or not logs:
                detail = bad[0].error.message if bad and bad[0].error else "no log written"
                failures.append(f"{chunk}: {' '.join(detail.split())[:300]}")
            else:
                manifest.completed_chunks.append(chunk)
            _save(manifest, run_dir)
        else:
            if failures:
                manifest.status, manifest.error = "failed", "; ".join(failures)
            else:
                manifest.status = "stopped_quota" if manifest.quota_stopped else "complete"
        if manifest.status == "running":
            manifest.status = "stopped_budget"
    except Exception as e:  # recorded, resumable with the same run_id
        manifest.status = "failed"
        manifest.error = f"{type(e).__name__}: {e}"
    finally:
        manifest.finished = datetime.now(timezone.utc)
        _save(manifest, run_dir)
        with (log_root / "ledger.jsonl").open("a") as f:
            f.write(json.dumps({
                "run_id": manifest.run_id,
                "date": datetime.now(timezone.utc).isoformat(),
                "tier": manifest.tier,
                "status": manifest.status,
                "spend_usd": round(manifest.spend_usd, 6),
                "quota_samples": plan.quota_samples,
            }) + "\n")
    return manifest
