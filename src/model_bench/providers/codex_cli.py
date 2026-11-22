"""ChatGPT via the owner's subscription: `codex exec` (Codex headless).

Switched off: user config.toml (--ignore-user-config), execpolicy rules (--ignore-rules),
AGENTS.md project docs (-c project_doc_max_bytes=0), session persistence (--ephemeral); read-only
sandbox; runs in an empty temp dir.

Cannot be switched off: Codex's own agent system prompt and shell-tool definitions (~16k input
tokens observed on codex-cli 0.166.0), installed Codex plugins/skills listings (~1.8k tokens), and
the model may run read-only shell commands in the empty dir. The task's system prompt is prepended
to the prompt as a [system] block rather than replacing Codex's. Results are "via CLI".
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from inspect_ai.model import GenerateConfig, ModelAPI, ModelOutput, ModelUsage

from model_bench.providers.cli_common import (
    DEFAULT_TIMEOUT_S,
    CLIBadOutput,
    CLIError,
    classify_failure,
    run_cli,
    serialise_messages,
)


def codex_argv(model: str, out_file: Path, effort: str | None) -> list[str]:
    argv = [
        "codex", "exec",
        "--json",
        "--ephemeral",
        "--skip-git-repo-check",
        "--ignore-user-config",
        "--ignore-rules",
        "-c", "project_doc_max_bytes=0",
        "-s", "read-only",
        "-m", model,
        "-o", str(out_file),
    ]
    if effort:
        argv += ["-c", f"model_reasoning_effort={effort}"]
    return argv + ["-"]


def parse_codex(stdout: str, stderr: str, code: int, last_message: str) -> tuple[str, ModelUsage]:
    events = []
    for line in stdout.splitlines():
        try:
            events.append(json.loads(line))
        except (json.JSONDecodeError, ValueError):
            continue
    failures = [
        (e.get("error") or {}).get("message") or e.get("message") or ""
        for e in events
        if e.get("type") in ("turn.failed", "error")
    ]
    if failures or code != 0:
        raise classify_failure("\n".join(failures) + "\n" + stderr + ("" if events else stdout))
    messages = [
        e["item"].get("text", "")
        for e in events
        if e.get("type") == "item.completed" and (e.get("item") or {}).get("type") == "agent_message"
    ]
    text = last_message.strip() or (messages[-1] if messages else "")
    if not text:
        raise CLIBadOutput("no agent message in codex output")
    u: dict[str, int] = {}
    for e in events:
        if e.get("type") == "turn.completed":
            for k, v in (e.get("usage") or {}).items():
                u[k] = u.get(k, 0) + int(v)
    usage = ModelUsage(
        input_tokens=u.get("input_tokens", 0),
        output_tokens=u.get("output_tokens", 0),
        total_tokens=u.get("input_tokens", 0) + u.get("output_tokens", 0),
        input_tokens_cache_read=u.get("cached_input_tokens", 0),
        reasoning_tokens=u.get("reasoning_output_tokens", 0),
        total_cost=None,  # subscription: no real spend
    )
    return text, usage


class CodexCLIAPI(ModelAPI):
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        effort: str | None = None,
        timeout_s: int = DEFAULT_TIMEOUT_S,
        **model_args: Any,
    ) -> None:
        super().__init__(model_name=model_name, base_url=base_url, api_key=api_key, config=config)
        self.effort = effort
        self.timeout_s = timeout_s

    def max_connections(self) -> int:
        return 1

    async def generate(self, input, tools, tool_choice, config) -> ModelOutput:
        if tools:
            raise CLIError("tool tasks unsupported on CLI providers")
        system, prompt = serialise_messages(input)
        if system:
            prompt = f"[system]\n{system}\n\n{prompt}"
        with tempfile.TemporaryDirectory(prefix="mb-codex-out-") as d:
            out_file = Path(d) / "last.txt"
            code, out, err = await run_cli(codex_argv(self.model_name, out_file, self.effort), prompt, self.timeout_s)
            last = out_file.read_text() if out_file.exists() else ""
        text, usage = parse_codex(out, err, code, last)
        output = ModelOutput.from_content(model=self.model_name, content=text)
        output.usage = usage
        return output
