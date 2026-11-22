from datetime import date
from pathlib import Path

import pytest

from model_bench.registry import ModelEntry, Price, TaskEntry, TierSpec

ROOT = Path(__file__).resolve().parents[1]
TINY = f"local:{ROOT / 'tasks/fixtures/tiny_task.py'}@tiny"


def tiny_task(name="tiny", epochs=1, limit=None) -> TaskEntry:
    return TaskEntry(
        name=name,
        source=TINY,
        category="test",
        needs=["text"],
        tiers={"smoke": TierSpec(limit=limit, epochs=epochs)},
        split="test",
        version="1",
        scorer_kind="deterministic",
        est_tokens_per_sample=(10, 10),
    )


def mock_model(id="mock") -> ModelEntry:
    return ModelEntry(
        id=id,
        access="api",
        inspect_model="mockllm/model",
        caps=["text"],
        price=Price(input=1.0, output=1.0, as_of=date(2026, 11, 22)),
    )


def cli_model(id="claude-cli-haiku") -> ModelEntry:
    return ModelEntry(id=id, access="cli", inspect_model="claude-cli/haiku", caps=["text"], sealed_ok=True)


@pytest.fixture
def tiny():
    return tiny_task

FIXTURES_LIMIT = ROOT / "tests/fixtures/claude_limit.txt"
