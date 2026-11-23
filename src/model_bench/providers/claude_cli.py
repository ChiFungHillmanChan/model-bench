"""Claude via the owner's subscription: `claude -p` (Claude Code headless).

Switched off: built-in tools (--tools ""), MCP servers (--strict-mcp-config), user/project/local
settings (--setting-sources ""), session persistence; runs in an empty temp dir; the task's system
prompt replaces Claude Code's (--system-prompt).

Cannot be switched off: whatever Claude Code itself still sends with every request (observed
~900 cache-creation tokens for a 4-word system prompt on 2.1.330), and the subscription's model
routing (no dated snapshot, no temperature control). Results are "via CLI", not bare-model scores.
"""

from __future__ import annotations

import base64
import json
import tempfile
from pathlib import Path
from typing import Any

from inspect_ai.model import GenerateConfig, ModelAPI, ModelOutput, ModelUsage

from model_bench.providers.cli_common import (
    DEFAULT_TIMEOUT_S,
    CLIBadOutput,
    CLIError,
    check_isolation,
    classify_failure,
    extract_images,
    run_cli,
    serialise_messages,
)


def claude_argv(model: str, system_prompt: str | None, effort: str | None, with_images: bool = False) -> list[str]:
    # Images only travel in stream-json input, which in turn needs stream-json output.
    io_flags = (
        ["--input-format", "stream-json", "--output-format", "stream-json", "--verbose"]
        if with_images
        else ["--output-format", "json"]
    )
    argv = [
        "claude", "-p",
        *io_flags,
        "--model", model,
        "--tools", "",
        "--strict-mcp-config",
        "--setting-sources", "",
        "--no-session-persistence",
        "--system-prompt", system_prompt or "",
    ]
    if effort:
        argv += ["--effort", effort]
    return argv


def parse_claude(stdout: str, stderr: str, code: int) -> tuple[str, ModelUsage, str]:
    try:
        data = json.loads(stdout)
    except (json.JSONDecodeError, ValueError):
        raise classify_failure(stdout + "\n" + stderr) from None
    if not isinstance(data, dict) or "result" not in data:
        raise CLIBadOutput(stdout[:500])
    if data.get("is_error") or code != 0:
        raise classify_failure(str(data.get("result")) + "\n" + stderr)
    u = data.get("usage") or {}
    cache_write = u.get("cache_creation_input_tokens", 0)
    cache_read = u.get("cache_read_input_tokens", 0)
    fresh = u.get("input_tokens", 0)  # Inspect convention: input_tokens excludes cache
    usage = ModelUsage(
        input_tokens=fresh,
        output_tokens=u.get("output_tokens", 0),
        total_tokens=fresh + cache_write + cache_read + u.get("output_tokens", 0),
        input_tokens_cache_write=cache_write,
        input_tokens_cache_read=cache_read,
        total_cost=None,  # subscription: no real spend
    )
    model_id = next(iter(data.get("modelUsage") or {}), "")
    return data["result"], usage, model_id


def claude_stdin(prompt: str, images: list[Path]) -> str:
    """One stream-json user message: the images, then the prompt text."""
    content = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                     "data": base64.b64encode(p.read_bytes()).decode()}}
        for p in images
    ]
    content.append({"type": "text", "text": prompt})
    return json.dumps({"type": "user", "message": {"role": "user", "content": content}}) + "\n"


def parse_claude_stream(stdout: str, stderr: str, code: int) -> tuple[str, ModelUsage, str]:
    """The final {"type":"result"} line carries the same fields as --output-format json."""
    for line in reversed(stdout.splitlines()):
        try:
            data = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict) and data.get("type") == "result":
            return parse_claude(line, stderr, code)
    raise classify_failure(stdout[-500:] + "\n" + stderr)


class ClaudeCLIAPI(ModelAPI):
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
        check_isolation()
        self.effort = effort
        self.timeout_s = timeout_s

    def max_connections(self) -> int:
        return 1

    async def generate(self, input, tools, tool_choice, config) -> ModelOutput:
        if tools:
            raise CLIError("tool tasks unsupported on CLI providers")
        system, prompt = serialise_messages(input)
        with tempfile.TemporaryDirectory(prefix="mb-claude-img-") as d:
            images = extract_images(input, Path(d))
            if images:
                argv = claude_argv(self.model_name, system, self.effort, with_images=True)
                code, out, err = await run_cli(argv, claude_stdin(prompt, images), self.timeout_s)
                text, usage, reported = parse_claude_stream(out, err, code)
            else:
                argv = claude_argv(self.model_name, system, self.effort)
                code, out, err = await run_cli(argv, prompt, self.timeout_s)
                text, usage, reported = parse_claude(out, err, code)
        output = ModelOutput.from_content(model=reported or self.model_name, content=text)
        output.usage = usage
        return output
