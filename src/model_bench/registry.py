"""Model and task registries (YAML) with validation."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

CAPS: frozenset[str] = frozenset(
    {"text", "json", "tools", "vision", "reasoning", "sandbox_agent", "lang:yue", "lang:zh-hant"}
)
_CTX = re.compile(r"^ctx:(\d+)$")
TIERS = ("smoke", "standard", "full", "canary")


class RegistryError(ValueError):
    pass


def _check_caps(values: Any) -> frozenset[str]:
    caps = frozenset(values or [])
    for cap in caps:
        if cap in CAPS or _CTX.match(cap):
            continue
        raise ValueError(f"unknown capability: {cap}")
    return caps


def ctx_value(cap: str) -> int | None:
    m = _CTX.match(cap)
    return int(m.group(1)) if m else None


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Price(_Strict):
    input: float
    output: float
    input_cache_read: float = 0
    input_cache_write: float = 0
    as_of: date


class ModelEntry(_Strict):
    id: str
    access: Literal["cli", "api", "local"]
    inspect_model: str
    pinned: str | None = None
    caps: frozenset[str]
    params: dict[str, Any] = {}
    price: Price | None = None
    sealed_ok: bool = False
    role: Literal["reference"] | None = None
    label: str | None = None  # display name in reports, e.g. "Claude Opus 5.5"

    _v_caps = field_validator("caps", mode="before")(lambda cls, v: _check_caps(v))

    @model_validator(mode="after")
    def _api_needs_price(self) -> "ModelEntry":
        if self.access == "api" and self.price is None:
            raise ValueError(f"api model {self.id} requires a price")
        return self


class TierSpec(_Strict):
    limit: int | None
    epochs: int


class TaskEntry(_Strict):
    name: str
    source: str
    category: str
    needs: frozenset[str]
    tiers: dict[Literal["smoke", "standard", "full", "canary"], TierSpec]
    split: Literal["dev", "test"]
    version: str
    scorer_kind: Literal["deterministic", "judge", "human"]
    sealed: bool = False
    est_tokens_per_sample: tuple[int, int]
    task_args: dict[str, Any] = {}
    model_roles: dict[str, str] = {}  # judge role -> registered model id (e.g. HLE's "grader")
    title: str | None = None  # dashboard name; tasks without one show their name
    about: str | None = None  # one-line dashboard description

    _v_needs = field_validator("needs", mode="before")(lambda cls, v: _check_caps(v))

    @field_validator("source")
    @classmethod
    def _v_source(cls, v: str) -> str:
        if not v.startswith(("inspect_evals/", "private:", "local:")):
            raise ValueError(f"source must start with inspect_evals/, private: or local:, got {v}")
        return v


def _load(path: Path, key: str, cls: type[BaseModel]) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    out: dict[str, Any] = {}
    for item in raw.get(key) or []:
        try:
            entry = cls.model_validate(item)
        except ValidationError as e:
            raise RegistryError(f"{path}: {e}") from e
        ident = getattr(entry, "id", None) or getattr(entry, "name")
        if ident in out:
            raise RegistryError(f"{path}: duplicate entry {ident}")
        out[ident] = entry
    return out


def load_models(path: Path) -> dict[str, ModelEntry]:
    return _load(path, "models", ModelEntry)


def load_tasks(path: Path) -> dict[str, TaskEntry]:
    """The registry plus, when present, the untracked private/registry.yaml next to it."""
    tasks = _load(path, "tasks", TaskEntry)
    private = Path(path).parent / "private" / "registry.yaml"
    if private.exists():
        for name, entry in _load(private, "tasks", TaskEntry).items():
            if name in tasks:
                raise RegistryError(f"{private}: duplicate entry {name}")
            tasks[name] = entry
    return tasks


def append_model(path: Path, entry: dict[str, Any]) -> ModelEntry:
    """Validate one model entry and append it to the YAML file, keeping comments. Nothing is written on error."""
    try:
        model = ModelEntry.model_validate(entry)
    except ValidationError as e:
        raise RegistryError("; ".join(err["msg"] for err in e.errors())) from e
    if model.id in load_models(path):
        raise RegistryError(f"model {model.id} already exists in {path}")
    text = Path(path).read_text()
    block = yaml.safe_dump([entry], sort_keys=False, default_flow_style=None, allow_unicode=True)
    new = text.rstrip("\n") + "\n\n" + "".join("  " + line for line in block.splitlines(keepends=True))
    Path(path).write_text(new)
    try:
        load_models(path)
    except RegistryError:
        Path(path).write_text(text)
        raise
    return model
