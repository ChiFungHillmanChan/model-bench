"""Three-sample local task used by tests (no network, no cost)."""

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import match
from inspect_ai.solver import generate


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
