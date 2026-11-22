"""Inspect entry point: registers model-bench model providers."""

from inspect_ai.model import ModelAPI, modelapi


@modelapi(name="claude-cli")
def claude_cli() -> type[ModelAPI]:
    from model_bench.providers.claude_cli import ClaudeCLIAPI

    return ClaudeCLIAPI


@modelapi(name="codex-cli")
def codex_cli() -> type[ModelAPI]:
    from model_bench.providers.codex_cli import CodexCLIAPI

    return CodexCLIAPI
