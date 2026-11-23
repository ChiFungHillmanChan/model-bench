import os
import sys
import tempfile
from pathlib import Path

import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser
from inspect_ai.scorer import match
from inspect_ai.solver import generate

from model_bench.providers import claude_cli
from model_bench.providers.cli_common import (
    CLIAuthError,
    CLIBadOutput,
    CLITimeout,
    CLIUsageLimit,
    run_cli,
    serialise_messages,
)
from model_bench.providers.claude_cli import claude_argv, parse_claude
from model_bench.providers.codex_cli import codex_argv, parse_codex

FIX = Path(__file__).parent / "fixtures"


def test_serialise_single_user_message_passthrough():
    assert serialise_messages([ChatMessageUser(content="What is 2+2?")]) == (None, "What is 2+2?")


def test_serialise_system_and_multiturn():
    system, prompt = serialise_messages(
        [
            ChatMessageSystem(content="Be brief."),
            ChatMessageUser(content="Hi"),
            ChatMessageAssistant(content="Hello"),
            ChatMessageUser(content="Bye"),
        ]
    )
    assert system == "Be brief."
    lines = [l for l in prompt.splitlines() if l.startswith("[")]
    assert lines == ["[user]", "[assistant]", "[user]"]
    assert prompt.index("Hi") < prompt.index("Hello") < prompt.index("Bye")


def test_claude_argv_isolation():
    argv = claude_argv("sonnet", "sys", None)
    assert argv[:2] == ["claude", "-p"]
    i = argv.index("--tools")
    assert argv[i + 1] == ""
    i = argv.index("--setting-sources")
    assert argv[i + 1] == ""
    for flag in ["--strict-mcp-config", "--no-session-persistence"]:
        assert flag in argv
    assert "--continue" not in argv and "--resume" not in argv
    assert argv[argv.index("--system-prompt") + 1] == "sys"
    assert "--effort" not in argv
    assert claude_argv("opus", None, "high")[-2:] == ["--effort", "high"]


def test_codex_argv_isolation(tmp_path):
    argv = codex_argv("gpt-x", tmp_path / "last.txt", None)
    for flag in ["--json", "--ephemeral", "--skip-git-repo-check", "--ignore-user-config", "--ignore-rules"]:
        assert flag in argv
    assert argv[argv.index("-s") + 1] == "read-only"
    assert "project_doc_max_bytes=0" in argv
    assert argv[-1] == "-"
    assert "model_reasoning_effort=high" in codex_argv("gpt-x", tmp_path / "l", "high")


async def test_run_cli_uses_empty_temp_cwd():
    code, out, _ = await run_cli(
        [sys.executable, "-c", "import os;print(os.getcwd());print(len(os.listdir()))"], None, 30
    )
    cwd, count = out.split()
    assert code == 0
    assert os.path.realpath(cwd).startswith(os.path.realpath(tempfile.gettempdir()))
    assert count == "0"


async def test_run_cli_strips_api_keys(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "y")
    _, out, _ = await run_cli(
        [sys.executable, "-c", "import os;print(os.environ.get('ANTHROPIC_API_KEY'), os.environ.get('OPENAI_API_KEY'))"],
        None,
        30,
    )
    assert out.strip() == "None None"


async def test_run_cli_passes_stdin():
    _, out, _ = await run_cli([sys.executable, "-c", "import sys;print(sys.stdin.read().upper())"], "abc", 30)
    assert out.strip() == "ABC"


async def test_timeout_raises_cli_timeout():
    with pytest.raises(CLITimeout):
        await run_cli([sys.executable, "-c", "import time;time.sleep(5)"], None, 1)


def test_parse_claude_ok():
    text, usage, model_id = parse_claude((FIX / "claude_ok.json").read_text(), "", 0)
    assert text == "OK"
    assert usage.output_tokens > 0
    assert usage.total_cost is None  # subscription: no real spend
    assert model_id == "claude-haiku-5-5"


def test_parse_claude_usage_limit_raises():
    with pytest.raises(CLIUsageLimit):
        parse_claude((FIX / "claude_limit.txt").read_text(), "", 1)


def test_parse_claude_auth_error_raises():
    with pytest.raises(CLIAuthError):
        parse_claude("", "Invalid API key · Please run /login", 1)


def test_parse_claude_garbage_raises_bad_output():
    with pytest.raises(CLIBadOutput):
        parse_claude("not json", "", 0)


def test_parse_codex_ok():
    text, usage = parse_codex((FIX / "codex_ok.jsonl").read_text(), "", 0, "OK")
    assert text == "OK"
    assert usage.input_tokens > 0 and usage.output_tokens > 0
    assert usage.total_cost is None


def test_parse_codex_usage_limit_raises():
    with pytest.raises(CLIUsageLimit):
        parse_codex((FIX / "codex_limit.txt").read_text(), "", 1, "")


def test_parse_codex_empty_answer_is_bad_output():
    with pytest.raises(CLIBadOutput):
        parse_codex('{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":0}}', "", 0, "")


def test_generate_error_is_sample_error_not_empty_answer(monkeypatch, tmp_path):
    async def fake_run_cli(argv, stdin, timeout_s):
        return 1, (FIX / "claude_limit.txt").read_text(), ""

    monkeypatch.setattr(claude_cli, "run_cli", fake_run_cli)
    task = Task(dataset=[Sample(input="Say OK", target="OK")], solver=generate(), scorer=match())
    [log] = eval(task, model="claude-cli/haiku", log_dir=str(tmp_path), display="none", fail_on_error=False)
    sample = log.samples[0]
    assert sample.error is not None
    assert "CLIUsageLimit" in sample.error.message
    assert not sample.scores


@pytest.mark.live
def test_live_claude_and_codex_one_sample(tmp_path):
    task = Task(dataset=[Sample(input="Reply with exactly: OK", target="OK")], solver=generate(), scorer=match())
    for model in ["claude-cli/haiku", "codex-cli/gpt-6.2-astra"]:
        [log] = eval(task, model=model, log_dir=str(tmp_path), display="none")
        assert log.status == "success", model
        assert log.samples[0].scores["match"].value == "C", model


def test_parse_claude_input_tokens_exclude_cache():
    _, usage, _ = parse_claude((FIX / "claude_ok.json").read_text(), "", 0)
    assert usage.input_tokens == 2  # fresh only, Inspect convention
    assert usage.input_tokens_cache_write == 927
    assert usage.total_tokens == 2 + 927 + 0 + usage.output_tokens


def test_parse_codex_input_tokens_exclude_cache():
    _, usage = parse_codex((FIX / "codex_reconnect.jsonl").read_text(), "", 0, "")
    assert usage.input_tokens == 200 and usage.input_tokens_cache_read == 800


def test_parse_codex_transient_error_then_success_is_ok():
    text, _ = parse_codex((FIX / "codex_reconnect.jsonl").read_text(), "", 0, "")
    assert text == "42"


def test_parse_codex_command_execution_is_error():
    with pytest.raises(CLIBadOutput, match="shell command"):
        parse_codex((FIX / "codex_command.jsonl").read_text(), "", 0, "42")


async def test_run_cli_env_is_allowlisted(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://evil")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("CLAUDECODE", "1")
    _, out, _ = await run_cli(
        [sys.executable, "-c", "import os,json;print(json.dumps(sorted(os.environ)))"], None, 30
    )
    import json as _json

    names = set(_json.loads(out))
    assert not names & {"ANTHROPIC_BASE_URL", "DEEPSEEK_API_KEY", "CLAUDECODE"}
    assert "HOME" in names and "PATH" in names


def test_isolation_guard_blocks_global_instruction_files(tmp_path, monkeypatch):
    from model_bench.providers.cli_common import CLIError, check_isolation

    monkeypatch.setenv("HOME", str(tmp_path))
    check_isolation()  # nothing present: ok
    (tmp_path / ".codex").mkdir()
    (tmp_path / ".codex" / "AGENTS.md").write_text("")
    check_isolation()  # empty AGENTS.md is fine
    (tmp_path / ".codex" / "AGENTS.md").write_text("Always answer in French.")
    with pytest.raises(CLIError, match="AGENTS.md"):
        check_isolation()
    (tmp_path / ".codex" / "AGENTS.md").write_text("")
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "CLAUDE.md").write_text("x")
    with pytest.raises(CLIError, match="CLAUDE.md"):
        check_isolation()


async def test_run_cli_feeds_stdin_from_a_ready_file(monkeypatch):
    # Under load the CLI gave up after 3s waiting on a pipe we had not written yet
    # ("no stdin data received in 3s"); a file holds the prompt before the process starts.
    import asyncio

    import model_bench.providers.cli_common as cc

    seen = {}
    real = asyncio.create_subprocess_exec

    async def spy(*argv, **kw):
        stdin = kw["stdin"]
        seen["is_file"] = hasattr(stdin, "read") and hasattr(stdin, "fileno")
        seen["content"] = stdin.read() if seen["is_file"] else None
        if seen["is_file"]:
            stdin.seek(0)
        return await real(*argv, **kw)

    monkeypatch.setattr(cc.asyncio, "create_subprocess_exec", spy)
    code, out, _ = await cc.run_cli(["cat"], "hello prompt", 10)
    assert seen["is_file"] and seen["content"] in ("hello prompt", b"hello prompt")
    assert code == 0 and out == "hello prompt"


def test_codex_argv_switches_off_tools_web_search_and_skills(tmp_path, monkeypatch):
    # Live run 2026-11-22: GPT used web_search (found a GPQA results page), ran curl, and read
    # ~/.agents/skills. Every tool surface is switched off; every skill on disk is disabled.
    home = tmp_path / "home"
    for d in ["~/.agents/skills/my-skill", "~/.codex/skills/other", "~/.codex/skills/.system/imagegen"]:
        p = home / d[2:]
        p.mkdir(parents=True)
        (p / "SKILL.md").write_text("x")
    monkeypatch.setenv("HOME", str(home))
    argv = codex_argv("gpt-x", tmp_path / "l", None)
    disabled = {argv[i + 1] for i, a in enumerate(argv) if a == "--disable"}
    assert {"shell_tool", "unified_exec", "plugins", "apps", "browser_use", "computer_use", "in_app_browser"} <= disabled
    # Standard run 2026-11-23: 13 IFEval answers still made mcp_tool_call; built-in tool features
    # (code-mode exec with tools.*, multi-agent, goals, clock, image gen/view, tool suggest, hooks).
    assert {"code_mode_host", "multi_agent", "goals", "sleep_tool", "image_generation", "view_image",
            "tool_suggest", "hooks"} <= disabled
    assert 'web_search="disabled"' in argv
    cfg = next(a for a in argv if a.startswith("skills.config="))
    for d in ["my-skill", "other", ".system/imagegen"]:
        assert f'/{d}/SKILL.md", enabled=false}}' in cfg
    assert argv[-1] == "-"


def test_parse_codex_web_search_is_error():
    with pytest.raises(CLIBadOutput, match="web_search"):
        parse_codex((FIX / "codex_websearch.jsonl").read_text(), "", 0, "B")
