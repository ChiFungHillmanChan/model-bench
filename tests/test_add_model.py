from datetime import date

from typer.testing import CliRunner

from model_bench.cli import app
from model_bench.registry import load_models

BASE = """# keep this comment
models:
  - id: claude-cli-opus-5-5
    access: cli
    inspect_model: claude-cli/claude-opus-5-5
    caps: [text, reasoning, json, "ctx:200000"]
    params: {effort: high}
    sealed_ok: true
"""


def setup(tmp_path, monkeypatch):
    import model_bench.cli as cli

    p = tmp_path / "models.yaml"
    p.write_text(BASE)
    monkeypatch.setattr(cli, "MODELS_PATH", p)
    return p


def test_add_model_like_copies_settings_and_keeps_comments(tmp_path, monkeypatch):
    p = setup(tmp_path, monkeypatch)
    r = CliRunner().invoke(app, ["add-model", "claude-cli-opus-5-6", "--like", "claude-cli-opus-5-5",
                                 "--model", "claude-cli/claude-opus-5-6", "--label", "Claude Opus 5.6"])
    assert r.exit_code == 0, r.output
    m = load_models(p)["claude-cli-opus-5-6"]
    assert m.access == "cli" and m.inspect_model == "claude-cli/claude-opus-5-6"
    assert m.params == {"effort": "high"} and m.sealed_ok is True
    assert "ctx:200000" in m.caps and m.label == "Claude Opus 5.6"
    assert p.read_text().startswith("# keep this comment")
    assert "bench run --models claude-cli-opus-5-6 --tier smoke" in r.output


def test_add_model_effort_overrides_like(tmp_path, monkeypatch):
    p = setup(tmp_path, monkeypatch)
    r = CliRunner().invoke(app, ["add-model", "opus-low", "--like", "claude-cli-opus-5-5",
                                 "--model", "claude-cli/claude-opus-5-5", "--effort", "low"])
    assert r.exit_code == 0, r.output
    assert load_models(p)["opus-low"].params == {"effort": "low"}


def test_add_model_api_needs_price_and_is_not_sealed(tmp_path, monkeypatch):
    p = setup(tmp_path, monkeypatch)
    before = p.read_text()
    r = CliRunner().invoke(app, ["add-model", "gpt-x", "--model", "openai/gpt-x", "--caps", "text,json"])
    assert r.exit_code != 0 and "requires a price" in r.output
    assert p.read_text() == before  # nothing written on a bad entry

    r = CliRunner().invoke(app, ["add-model", "gpt-x", "--model", "openai/gpt-x", "--caps", "text,json",
                                 "--price-in", "2", "--price-out", "8"])
    assert r.exit_code == 0, r.output
    m = load_models(p)["gpt-x"]
    assert m.access == "api" and m.sealed_ok is False
    assert m.price.input == 2 and m.price.as_of == date.today()


def test_add_model_access_inferred_from_provider(tmp_path, monkeypatch):
    p = setup(tmp_path, monkeypatch)
    for mid, model, access in [("codex-x", "codex-cli/gpt-7", "cli"), ("olla", "ollama/qwen3:14b", "local")]:
        r = CliRunner().invoke(app, ["add-model", mid, "--model", model, "--caps", "text"])
        assert r.exit_code == 0, r.output
        m = load_models(p)[mid]
        assert m.access == access and m.sealed_ok is True


def test_add_model_rejects_duplicate_and_unknown_like(tmp_path, monkeypatch):
    p = setup(tmp_path, monkeypatch)
    before = p.read_text()
    r = CliRunner().invoke(app, ["add-model", "claude-cli-opus-5-5", "--like", "claude-cli-opus-5-5",
                                 "--model", "claude-cli/x"])
    assert r.exit_code != 0 and "already exists" in r.output
    r = CliRunner().invoke(app, ["add-model", "new", "--like", "nope", "--model", "claude-cli/x"])
    assert r.exit_code != 0 and "unknown model: nope" in r.output
    r = CliRunner().invoke(app, ["add-model", "new", "--model", "claude-cli/x"])
    assert r.exit_code != 0 and "--caps" in r.output
    assert p.read_text() == before
