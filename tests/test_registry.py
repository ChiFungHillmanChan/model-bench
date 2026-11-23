from pathlib import Path

import pytest

from model_bench.registry import RegistryError, load_models, load_tasks

ROOT = Path(__file__).resolve().parents[1]

API_MODEL = """
models:
  - id: m1
    access: api
    inspect_model: deepseek/deepseek-chat
    caps: [text, json]
    price: {input: 0.3, output: 1.2, as_of: 2026-11-22}
"""


def write(tmp_path, text, name="r.yaml"):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_load_models_valid(tmp_path):
    models = load_models(write(tmp_path, API_MODEL))
    m = models["m1"]
    assert m.access == "api"
    assert m.inspect_model == "deepseek/deepseek-chat"
    assert isinstance(m.caps, frozenset)
    assert m.price.input == 0.3
    assert m.sealed_ok is False


def test_unknown_capability_rejected(tmp_path):
    text = API_MODEL.replace("[text, json]", "[text, telepathy]")
    with pytest.raises(RegistryError, match="telepathy"):
        load_models(write(tmp_path, text))


def test_ctx_capability_parsed(tmp_path):
    ok = API_MODEL.replace("[text, json]", '[text, "ctx:128000"]')
    assert "ctx:128000" in load_models(write(tmp_path, ok))["m1"].caps
    bad = API_MODEL.replace("[text, json]", '[text, "ctx:abc"]')
    with pytest.raises(RegistryError, match="ctx:abc"):
        load_models(write(tmp_path, bad, "b.yaml"))


def test_api_model_requires_price(tmp_path):
    text = API_MODEL.replace("    price: {input: 0.3, output: 1.2, as_of: 2026-11-22}\n", "")
    with pytest.raises(RegistryError, match="price"):
        load_models(write(tmp_path, text))


def test_cli_and_local_price_optional(tmp_path):
    text = """
models:
  - {id: c, access: cli, inspect_model: claude-cli/sonnet, caps: [text], price: null}
  - {id: l, access: local, inspect_model: ollama/qwen3:8b, caps: [text]}
"""
    models = load_models(write(tmp_path, text))
    assert models["c"].price is None and models["l"].price is None


def test_duplicate_ids_rejected(tmp_path):
    text = API_MODEL + API_MODEL.replace("models:\n", "")
    with pytest.raises(RegistryError, match="duplicate"):
        load_models(write(tmp_path, text))


TASK = """
tasks:
  - name: gpqa_diamond
    source: {source}
    category: reasoning
    needs: [text]
    tiers:
      smoke: {{limit: 10, epochs: 1}}
      full: {{limit: null, epochs: 3}}
    split: test
    version: "1"
    scorer_kind: deterministic
    est_tokens_per_sample: [1500, 1500]
"""


def test_task_tiers_and_source(tmp_path):
    tasks = load_tasks(write(tmp_path, TASK.format(source="inspect_evals/gpqa_diamond")))
    t = tasks["gpqa_diamond"]
    assert t.tiers["smoke"].limit == 10
    assert t.tiers["full"].limit is None
    assert t.est_tokens_per_sample == (1500, 1500)
    with pytest.raises(RegistryError, match="source"):
        load_tasks(write(tmp_path, TASK.format(source="somewhere/else"), "b.yaml"))


def test_repo_registries_load():
    assert load_models(ROOT / "models.yaml")
    assert load_tasks(ROOT / "tasks" / "registry.yaml")


def test_private_overlay_merges_when_present(tmp_path):
    # Private task entries live in an untracked private/registry.yaml next to the public registry.
    pub = write(tmp_path, TASK.format(source="inspect_evals/gpqa_diamond"), "registry.yaml")
    (tmp_path / "private").mkdir()
    (tmp_path / "private" / "registry.yaml").write_text(
        TASK.format(source="private:tasks/private/x.py@x").replace("gpqa_diamond", "my_private")
        .replace("category: reasoning", 'category: private\n    title: "My task"\n    about: "Mine."'))
    tasks = load_tasks(pub)
    assert set(tasks) == {"gpqa_diamond", "my_private"}
    assert tasks["my_private"].title == "My task" and tasks["my_private"].about == "Mine."


def test_private_overlay_absent_is_fine(tmp_path):
    assert set(load_tasks(write(tmp_path, TASK.format(source="inspect_evals/gpqa_diamond"), "registry.yaml"))) == {"gpqa_diamond"}


def test_private_overlay_cannot_shadow_a_public_task(tmp_path):
    pub = write(tmp_path, TASK.format(source="inspect_evals/gpqa_diamond"), "registry.yaml")
    (tmp_path / "private").mkdir()
    (tmp_path / "private" / "registry.yaml").write_text(TASK.format(source="private:tasks/private/x.py@x"))
    with pytest.raises(RegistryError, match="duplicate"):
        load_tasks(pub)
