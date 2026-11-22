"""Shared pieces for subscription-CLI model providers."""

from __future__ import annotations

import asyncio
import os
import re
import tempfile

from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageSystem

STRIPPED_ENV = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
DEFAULT_TIMEOUT_S = 600

_LIMIT = re.compile(r"usage limit|hit your limit|limit reached|rate limit|quota", re.I)
_AUTH = re.compile(r"/login|not logged in|log in|invalid api key|authenticat|unauthori[sz]ed", re.I)


class CLIError(RuntimeError):
    """Base error. The class name is part of the message so it survives into Inspect logs."""

    def __init__(self, detail: str) -> None:
        super().__init__(f"{type(self).__name__}: {detail}")


class CLIUsageLimit(CLIError):
    pass


class CLIAuthError(CLIError):
    pass


class CLIBadOutput(CLIError):
    pass


class CLITimeout(CLIError):
    pass


def classify_failure(text: str) -> CLIError:
    """Map a CLI failure message to the matching error type."""
    snippet = text.strip()[:500]
    if _LIMIT.search(text):
        return CLIUsageLimit(snippet)
    if _AUTH.search(text):
        return CLIAuthError(snippet)
    return CLIBadOutput(snippet or "empty output")


def serialise_messages(messages: list[ChatMessage]) -> tuple[str | None, str]:
    """(system_prompt, prompt). Multi-turn chats become one prompt with [role] markers."""
    system = [m.text for m in messages if isinstance(m, ChatMessageSystem)]
    turns = [m for m in messages if not isinstance(m, ChatMessageSystem)]
    if len(turns) == 1 and not isinstance(turns[0], ChatMessageAssistant):
        prompt = turns[0].text
    else:
        prompt = "\n\n".join(f"[{m.role}]\n{m.text}" for m in turns)
    return ("\n\n".join(system) if system else None), prompt


async def run_cli(argv: list[str], stdin: str | None, timeout_s: int) -> tuple[int, str, str]:
    """Run a CLI in a fresh empty temp dir with provider API keys removed from the env."""
    env = {k: v for k, v in os.environ.items() if k not in STRIPPED_ENV}
    with tempfile.TemporaryDirectory(prefix="mb-cli-") as cwd:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(
                proc.communicate(stdin.encode() if stdin is not None else None), timeout_s
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise CLITimeout(f"{argv[0]} exceeded {timeout_s}s")
    return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")
