"""Three-sample local task used by tests (no network, no cost)."""

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import match
from inspect_ai.solver import generate, solver


@task
def tiny() -> Task:
    return Task(
        dataset=MemoryDataset(
            [
                Sample(id="a", input="Say A", target="A"),
                Sample(id="b", input="Say B", target="B"),
                Sample(id="c", input="Say C", target="C"),
            ]
        ),
        solver=generate(),
        scorer=match(),
    )


@task
def tiny_unseeded() -> Task:
    """Like tiny, but shuffles its dataset without a seed (as mmlu_pro does)."""
    import random

    samples = [Sample(id=f"s{i}", input=f"Say {i}", target=str(i)) for i in range(20)]
    random.shuffle(samples)
    return Task(dataset=MemoryDataset(samples), solver=generate(), scorer=match())


@solver
def fail_when_env():
    """Raises for sample b while MB_TEST_FAIL=1 (simulates a mid-chunk API error)."""
    import os

    async def solve(state, generate):
        if os.environ.get("MB_TEST_FAIL") == "1" and state.sample_id == "b":
            raise RuntimeError("simulated provider error")
        return await generate(state)

    return solve


@task
def tiny_flaky() -> Task:
    t = tiny()
    return Task(dataset=t.dataset, solver=fail_when_env(), scorer=match())
