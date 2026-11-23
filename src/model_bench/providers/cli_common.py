"""Shared pieces for subscription-CLI model providers."""

from __future__ import annotations

import asyncio
import base64
import io
import os
import re
import tempfile
from pathlib import Path

from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageSystem, ContentImage

# Only these variables reach a CLI: no API keys, base URLs, provider switches or parent-session vars.
ENV_ALLOWLIST = ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "TERM", "SHELL")
DEFAULT_TIMEOUT_S = 600
MAX_IMAGE_SIDE = 1600  # long side in px; an A4 render at print DPI is far larger than a CLI needs

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


def check_isolation() -> None:
    """Refuse to run if global instruction files the CLIs cannot switch off are present."""
    home = Path(os.environ.get("HOME", Path.home()))
    claude_md = home / ".claude" / "CLAUDE.md"
    if claude_md.exists():
        raise CLIError(f"{claude_md} exists; Claude Code would load it into every benchmark call")
    agents_md = home / ".codex" / "AGENTS.md"
    if agents_md.exists() and agents_md.read_text().strip():
        raise CLIError(f"{agents_md} is not empty; Codex would load it into every benchmark call")


def serialise_messages(messages: list[ChatMessage]) -> tuple[str | None, str]:
    """(system_prompt, prompt). Multi-turn chats become one prompt with [role] markers."""
    system = [m.text for m in messages if isinstance(m, ChatMessageSystem)]
    turns = [m for m in messages if not isinstance(m, ChatMessageSystem)]
    if len(turns) == 1 and not isinstance(turns[0], ChatMessageAssistant):
        prompt = turns[0].text
    else:
        prompt = "\n\n".join(f"[{m.role}]\n{m.text}" for m in turns)
    return ("\n\n".join(system) if system else None), prompt


def extract_images(messages: list[ChatMessage], dest: Path) -> list[Path]:
    """Write every image in the messages to dest as PNG (long side capped), in message order."""
    from PIL import Image

    dest.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for m in messages:
        if isinstance(m.content, str):
            continue
        for c in m.content:
            if not isinstance(c, ContentImage):
                continue
            src = c.image
            if src.startswith("data:"):
                img = Image.open(io.BytesIO(base64.b64decode(src.split(",", 1)[1])))
            else:
                img = Image.open(src)
            img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
            out = dest / f"image-{len(paths):02d}.png"
            img.save(out, "PNG")
            paths.append(out)
    return paths


async def run_cli(argv: list[str], stdin: str | None, timeout_s: int) -> tuple[int, str, str]:
    """Run a CLI in a fresh empty temp dir with an allowlisted environment."""
    env = {k: v for k, v in os.environ.items() if k in ENV_ALLOWLIST}
    with tempfile.TemporaryDirectory(prefix="mb-cli-") as cwd, tempfile.TemporaryFile() as prompt:
        # The prompt is in a file before the process starts: a pipe written after spawn arrived
        # late under load and the CLI quit with "no stdin data received in 3s".
        prompt.write((stdin or "").encode())
        prompt.seek(0)
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            env=env,
            stdin=prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise CLITimeout(f"{argv[0]} exceeded {timeout_s}s")
    return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")
