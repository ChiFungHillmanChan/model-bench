"""Capability matching, tiers and cost/quota estimates."""

from __future__ import annotations

import importlib
import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path

from inspect_ai import Task

from model_bench.registry import ModelEntry, TaskEntry, ctx_value

SIZE_CACHE = Path.home() / ".cache" / "model-bench" / "sizes.json"


@dataclass(frozen=True)
class PlanItem:
    model_id: str
    task_name: str
    limit: int | None
    epochs: int
    n_items: int
    est_cost_usd: float | None
    est_samples: int


@dataclass(frozen=True)
class Skip:
    model_id: str
    task_name: str
    reason: str


@dataclass
class Plan:
    tier: str
    items: list[PlanItem] = field(default_factory=list)
    skips: list[Skip] = field(default_factory=list)
    est_cost_usd: float = 0.0
    quota_samples: dict[str, int] = field(default_factory=dict)


def capable(model: ModelEntry, task: TaskEntry) -> str | None:
    """None if the model can run the task, else the reason it cannot."""
    model_ctx = max((ctx_value(c) or 0 for c in model.caps), default=0)
    missing = []
    for need in sorted(task.needs):
        need_ctx = ctx_value(need)
        if need_ctx is not None:
            if model_ctx < need_ctx:
                missing.append(need)
        elif need not in model.caps:
            missing.append(need)
    if missing:
        return "missing capability: " + ", ".join(missing)
    if task.sealed and not model.sealed_ok:
        return "sealed task, model not sealed_ok"
    return None


def build_plan(
    models: list[ModelEntry], tasks: list[TaskEntry], tier: str, dataset_sizes: dict[str, int]
) -> Plan:
    plan = Plan(tier=tier)
    for task in tasks:
        spec = task.tiers.get(tier)
        if spec is None:
            continue
        size = dataset_sizes[task.name]
        n_items = size if spec.limit is None else min(spec.limit, size)
        est_samples = n_items * spec.epochs
        for model in models:
            reason = capable(model, task)
            if reason:
                plan.skips.append(Skip(model.id, task.name, reason))
                continue
            cost = None
            if model.price is not None and model.access == "api":
                tin, tout = task.est_tokens_per_sample
                cost = est_samples * (tin * model.price.input + tout * model.price.output) / 1e6
                plan.est_cost_usd += cost
            if model.access == "cli":
                plan.quota_samples[model.id] = plan.quota_samples.get(model.id, 0) + est_samples
            plan.items.append(
                PlanItem(model.id, task.name, spec.limit, spec.epochs, n_items, cost, est_samples)
            )
    return plan


def load_inspect_task(entry: TaskEntry) -> Task:
    """Construct a fresh Inspect Task object for a registry entry."""
    if entry.source.startswith("inspect_evals/"):
        registry = importlib.import_module("inspect_evals._registry")
        factory = getattr(registry, entry.source.removeprefix("inspect_evals/"))
    elif entry.source.startswith(("local:", "private:")):
        ref = entry.source.split(":", 1)[1]
        file, _, func = ref.partition("@")
        spec = importlib.util.spec_from_file_location(f"mb_task_{entry.name}", file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        factory = getattr(module, func)
    else:  # registry validation makes this unreachable
        raise ValueError(entry.source)
    return factory(**entry.task_args)


def dataset_size(task: TaskEntry) -> int:
    key = f"{task.source}|{json.dumps(task.task_args, sort_keys=True)}|{task.version}"
    cache = json.loads(SIZE_CACHE.read_text()) if SIZE_CACHE.exists() else {}
    if key not in cache:
        cache[key] = len(load_inspect_task(task).dataset)
        SIZE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        SIZE_CACHE.write_text(json.dumps(cache, indent=1))
    return cache[key]
