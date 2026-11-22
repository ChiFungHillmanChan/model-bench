from datetime import date

import pytest

from model_bench.planner import build_plan, capable
from model_bench.registry import ModelEntry, Price, TaskEntry, TierSpec


def model(id="m", access="api", caps=("text",), sealed_ok=False, price=(1.0, 2.0)):
    return ModelEntry(
        id=id,
        access=access,
        inspect_model=f"x/{id}",
        caps=list(caps),
        sealed_ok=sealed_ok,
        price=Price(input=price[0], output=price[1], as_of=date(2026, 11, 22)) if price else None,
    )


def task(name="t", needs=("text",), tiers=None, sealed=False, tokens=(1000, 500)):
    return TaskEntry(
        name=name,
        source=f"inspect_evals/{name}",
        category="c",
        needs=list(needs),
        tiers=tiers or {"smoke": TierSpec(limit=10, epochs=1)},
        split="test",
        version="1",
        scorer_kind="deterministic",
        sealed=sealed,
        est_tokens_per_sample=tokens,
    )


def test_missing_capability_skipped_with_reason():
    p = build_plan([model()], [task(needs=("text", "reasoning"))], "smoke", {"t": 50})
    assert p.items == []
    assert len(p.skips) == 1
    assert p.skips[0].reason == "missing capability: reasoning"


def test_ctx_compared_numerically():
    t = task(needs=("text", "ctx:128000"))
    assert capable(model(caps=("text", "ctx:32000")), t) == "missing capability: ctx:128000"
    assert capable(model(caps=("text", "ctx:200000")), t) is None


def test_sealed_task_only_for_sealed_ok():
    t = task(sealed=True)
    assert capable(model(sealed_ok=False), t) == "sealed task, model not sealed_ok"
    assert capable(model(sealed_ok=True), t) is None


def test_task_not_in_tier_is_not_listed():
    t = task(tiers={"full": TierSpec(limit=None, epochs=3)})
    p = build_plan([model()], [t], "smoke", {"t": 50})
    assert p.items == [] and p.skips == []


def test_limit_capped_by_dataset_size():
    t = task(tiers={"smoke": TierSpec(limit=100, epochs=2)})
    p = build_plan([model()], [t], "smoke", {"t": 40})
    assert p.items[0].n_items == 40
    assert p.items[0].est_samples == 80


def test_cost_estimate():
    p = build_plan([model()], [task()], "smoke", {"t": 10})
    assert p.items[0].est_cost_usd == pytest.approx(0.02)
    assert p.est_cost_usd == pytest.approx(0.02)


def test_cli_counts_toward_quota_not_cost():
    m = model(id="claude-cli-opus", access="cli", price=None)
    p = build_plan([m], [task()], "smoke", {"t": 10})
    assert p.items[0].est_cost_usd is None
    assert p.quota_samples["claude-cli-opus"] == p.items[0].est_samples == 10
    assert p.est_cost_usd == 0


def test_task_load_failure_is_skip_not_crash():
    p = build_plan([model(), model(id="m2")], [task(), task(name="broken")], "smoke", {"t": 10},
                   load_errors={"broken": "AssertionError: install instruction_following_eval"})
    assert [i.task_name for i in p.items] == ["t", "t"]
    assert {(s.model_id, s.reason) for s in p.skips} == {
        ("m", "task failed to load: AssertionError: install instruction_following_eval"),
        ("m2", "task failed to load: AssertionError: install instruction_following_eval"),
    }
