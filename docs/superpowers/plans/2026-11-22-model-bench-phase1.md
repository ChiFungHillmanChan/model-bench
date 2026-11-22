# model-bench Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A working `bench` CLI that plans, runs, stores, scores with confidence intervals, and
reproduction-checks a curated set of public benchmarks across Ollama, DeepSeek, Claude CLI and
Codex CLI models.

**Architecture:** Inspect AI does task/solver/scorer execution and logging. model-bench adds a
model and task registry (YAML), a capability-matching planner with cost/quota estimates, two
custom Inspect model providers that drive `claude -p` and `codex exec`, a runner that wraps
`eval_set` with a run manifest and budget guard, an ingest step into DuckDB, and statistics,
report and repro commands on top.

**Tech Stack:** Python 3.13, uv, inspect-ai 0.3.291, inspect-evals 0.29.0, DuckDB, pydantic v2,
PyYAML, Typer, pytest.

**Spec:** `docs/superpowers/specs/2026-11-22-model-bench-design.md` (read it before Task 1).

## Global Constraints

- Project root `~/Desktop/model-bench/`, its own git repo, branch `main`; remote is a **private**
  GitHub repo (owner creates it; do not push to a public repo).
- Pin exactly: `inspect-ai==0.3.291`, `inspect-evals==0.29.0`. Upgrades are explicit commits.
- Python `>=3.13,<3.14`; environment and lockfile via `uv`; run everything as `uv run ...`.
- No emojis in code, output, commits.
- API keys only in a gitignored `.env` (`DEEPSEEK_API_KEY`); never printed, never logged, never
  committed. Loaded with python-dotenv at CLI start.
- Gitignored: `.env`, `logs/`, `results.duckdb`, `reports/`, `blind/keys/`, `.venv/`.
- Every report row for an `access: cli` model is labelled `via CLI`; CLI models are excluded from
  `bench repro`.
- No number is shown without its 95% CI (or `n/a` when it cannot be computed).
- Sealed private tasks may only be planned for models with `sealed_ok: true`.
- Phase 1 needs no Docker, no AI judges, no tool-calling tasks on CLI providers.
- Paid or quota-consuming live runs (DeepSeek, Claude CLI, Codex CLI) happen only in Task 10 and
  only after the owner says go. All other tests use `mockllm/model` or recorded fixtures.

## Review Focus

1. **A CLI failure must never be scored as a wrong answer.** Expired login, usage limit, non-JSON
   output or timeout must become a sample *error* (Inspect records `error`), not an empty
   completion graded 0. Tests in Task 4.
2. **The CLI must not see the owner's environment.** No `CLAUDE.md`, user settings, MCP servers,
   memory or repo context may leak into a benchmark call. Tests in Task 4 assert the exact flags
   and that `cwd` is a fresh empty temp dir.
3. **Resuming an interrupted run must not double-count.** Re-running ingest, or ingesting a run
   that was resumed after a usage limit, must leave exactly one row per (run, task, model, sample,
   epoch). Test in Task 6.
4. **Nothing is silently skipped.** A task a model cannot run (missing capability, sealed, missing
   price) appears in the plan output as skipped with its reason. Tests in Task 3.
5. **Small samples must not crash or mislead.** n < 2 items, epochs = 1, or all-identical scores
   give `n/a` for CI / pass^k instead of NaN, zero-width intervals or exceptions. Tests in Task 7.

---

## File Structure

```
model-bench/
  pyproject.toml                    # deps, `bench` script, inspect_ai entry point
  .gitignore  .env.example  README.md
  models.yaml                       # model registry (Task 2)
  tasks/registry.yaml               # task registry (Task 2)
  tasks/fixtures/tiny_task.py       # 3-sample local task for tests (Task 5)
  reference_scores.yaml             # published scores (Task 9)
  src/model_bench/
    __init__.py
    cli.py                          # Typer app, one subcommand per module
    registry.py                     # load + validate both registries
    planner.py                      # capability match, tiers, estimates
    providers/__init__.py
    providers/_registry.py          # @modelapi registrations (entry point target)
    providers/cli_common.py         # message serialisation, subprocess, error mapping
    providers/claude_cli.py
    providers/codex_cli.py
    runner.py                       # eval_set wrapper, manifest, budget guard, ledger
    ingest.py                       # logs -> DuckDB
    stats.py                        # pure statistics
    report.py                       # report + compare
    repro.py                        # reproduction check
  tests/
    fixtures/claude_ok.json  claude_limit.txt  codex_ok.jsonl  codex_limit.txt
    test_registry.py  test_planner.py  test_cli_providers.py  test_runner.py
    test_ingest.py  test_stats.py  test_report.py  test_repro.py
```

---

### Task 1: Project skeleton

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `README.md`,
  `src/model_bench/__init__.py`, `src/model_bench/cli.py`, `tests/test_cli_smoke.py`

**Interfaces:**
- Produces: console script `bench` -> `model_bench.cli:app` (Typer). Entry point group
  `inspect_ai`: `model_bench = "model_bench.providers._registry"` (module created empty here,
  filled in Task 4).

- [ ] **Step 1:** `git init -b main` in `~/Desktop/model-bench/` (the `docs/` folder already exists
  there; keep it).
- [ ] **Step 2:** Write `pyproject.toml` (hatchling build, `src/` layout) with the pinned deps from
  Global Constraints plus `duckdb`, `pydantic>=2`, `pyyaml`, `typer`, `python-dotenv`, dev group
  `pytest`. Register the script and entry point above. Configure pytest marker `live` and
  `addopts = "-m 'not live'"`.
- [ ] **Step 3:** `.gitignore` per Global Constraints; `.env.example` with `DEEPSEEK_API_KEY=`.
- [ ] **Step 4: Write the failing test** `tests/test_cli_smoke.py::test_help_lists_commands`:
  invoke `app` with `--help` via `typer.testing.CliRunner`; assert exit code 0 and that the output
  contains `models`, `plan`, `run`, `ingest`, `report`, `compare`, `repro`.
- [ ] **Step 5:** Run `uv run pytest tests/test_cli_smoke.py -v`. Expected: FAIL (commands missing).
- [ ] **Step 6:** In `cli.py` create the Typer `app` with those seven subcommands as stubs that
  print `not implemented` and exit 2; load `.env` with `load_dotenv()` at app callback.
- [ ] **Step 7:** Run `uv sync && uv run pytest -v`. Expected: PASS.
- [ ] **Step 8:** Commit: `chore: project skeleton with pinned inspect-ai and bench CLI`
  (include `docs/`).

### Task 2: Model and task registries

**Files:**
- Create: `src/model_bench/registry.py`, `models.yaml`, `tasks/registry.yaml`,
  `tests/test_registry.py`
- Modify: `src/model_bench/cli.py` (implement `bench models`)

**Interfaces:**
- Produces (pydantic models in `registry.py`):
  - `Price(input: float, output: float, input_cache_read: float = 0, input_cache_write: float = 0, as_of: date)` ($/million tokens)
  - `ModelEntry(id: str, access: Literal["cli","api","local"], inspect_model: str, pinned: str | None, caps: frozenset[str], params: dict[str, Any] = {}, price: Price | None, sealed_ok: bool = False, role: Literal["reference"] | None = None)`
  - `TierSpec(limit: int | None, epochs: int)` (`limit=None` means all items)
  - `TaskEntry(name: str, source: str, category: str, needs: frozenset[str], tiers: dict[Literal["smoke","standard","full","canary"], TierSpec], split: Literal["dev","test"], version: str, scorer_kind: Literal["deterministic","judge","human"], sealed: bool = False, est_tokens_per_sample: tuple[int, int], task_args: dict[str, Any] = {})` (`est_tokens_per_sample` = (input, output))
  - `CAPS: frozenset[str]` = `{"text","json","tools","vision","reasoning","sandbox_agent","lang:yue","lang:zh-hant"}` plus pattern `ctx:<int>`
  - `def load_models(path: Path) -> dict[str, ModelEntry]`
  - `def load_tasks(path: Path) -> dict[str, TaskEntry]`
  - `class RegistryError(ValueError)`

- [ ] **Step 1: Write failing tests** in `tests/test_registry.py`:
  - `test_load_models_valid`: loads a tmp YAML with one `api` model; asserts fields, `caps` is a
    frozenset.
  - `test_unknown_capability_rejected`: `caps: [text, telepathy]` -> `RegistryError` mentioning
    `telepathy`.
  - `test_ctx_capability_parsed`: `ctx:128000` accepted; `ctx:abc` -> `RegistryError`.
  - `test_api_model_requires_price`: `access: api` with no `price` -> `RegistryError`.
  - `test_cli_and_local_price_optional`: `access: cli` and `access: local` load with `price: null`.
  - `test_duplicate_ids_rejected`.
  - `test_task_tiers_and_source`: a task with `source: inspect_evals/gpqa_diamond` and two tiers
    loads; a `source` not starting with `inspect_evals/` or `private:` or `local:` -> `RegistryError`.
  - `test_repo_registries_load`: the real `models.yaml` and `tasks/registry.yaml` load without error.
- [ ] **Step 2:** Run `uv run pytest tests/test_registry.py -v`. Expected: FAIL (module missing).
- [ ] **Step 3:** Implement `registry.py` with the interfaces above.
- [ ] **Step 4:** Write `models.yaml` with these entries (builder confirms exact model names the
  CLIs accept with `claude --help` / `codex exec --help` and DeepSeek's model list):
  - `ollama-qwen3-8b` (local, `ollama/qwen3:8b`, caps `[text, reasoning, json, "ctx:32000"]`,
    `role: reference`, `sealed_ok: true`, `pinned` = weights digest from `ollama show`, filled
    in Task 10)
  - `deepseek-chat` (api, `deepseek/deepseek-chat`, caps `[text, json, tools, "ctx:64000"]`, price
    from DeepSeek's pricing page with `as_of`, `sealed_ok: false`)
  - `deepseek-reasoner` (api, `deepseek/deepseek-reasoner`, adds `reasoning`)
  - `claude-cli-opus` (cli, `claude-cli/opus`, caps `[text, reasoning, json, "ctx:200000"]`,
    `params: {effort: high}`, `sealed_ok: true`)
  - `claude-cli-sonnet` (cli, `claude-cli/sonnet`, same caps)
  - `codex-cli-default` (cli, `codex-cli/<model>`, caps `[text, reasoning, json, "ctx:200000"]`,
    `sealed_ok: true`)

  No `tools` capability on any CLI model (spec 5.14).
- [ ] **Step 5:** Write `tasks/registry.yaml` with `gpqa_diamond`, `mmlu_pro`, `ifeval`, `aime2025`
  (sources `inspect_evals/<name>`; needs `[text]`, AIME adds `reasoning`; all
  `scorer_kind: deterministic`, `split: test`). Tier specs: smoke `{limit: 10, epochs: 1}`,
  standard `{limit: 100, epochs: 3}` (AIME: `{limit: null, epochs: 3}`), full `{limit: null,
  epochs: 3}`. Set `est_tokens_per_sample` per task from one Ollama smoke run in Task 10; until
  then use (1500, 1500) for MCQ tasks and (500, 8000) for AIME.
- [ ] **Step 6:** Implement `bench models`: prints one row per model: id, access, caps,
  sealed_ok, price-or-`quota`.
- [ ] **Step 7:** Run `uv run pytest -v`. Expected: PASS.
- [ ] **Step 8:** Commit: `feat: model and task registries with capability validation`

### Task 3: Planner

**Files:**
- Create: `src/model_bench/planner.py`, `tests/test_planner.py`
- Modify: `src/model_bench/cli.py` (implement `bench plan`)

**Interfaces:**
- Consumes: `ModelEntry`, `TaskEntry`, `TierSpec` from Task 2.
- Produces:
  - `PlanItem(model_id: str, task_name: str, limit: int | None, epochs: int, n_items: int, est_cost_usd: float | None, est_samples: int)` (`est_cost_usd` is None for cli/local)
  - `Skip(model_id: str, task_name: str, reason: str)`
  - `Plan(tier: str, items: list[PlanItem], skips: list[Skip], est_cost_usd: float, quota_samples: dict[str, int])`
  - `def capable(model: ModelEntry, task: TaskEntry) -> str | None` -- None if runnable, else the
    reason string.
  - `def build_plan(models: list[ModelEntry], tasks: list[TaskEntry], tier: str, dataset_sizes: dict[str, int]) -> Plan`
  - `def dataset_size(task: TaskEntry) -> int` -- loads the Inspect task header and returns sample
    count (cached to `~/.cache/model-bench/sizes.json`).

- [ ] **Step 1: Write failing tests** in `tests/test_planner.py` (pass `dataset_sizes` directly;
  no network):
  - `test_missing_capability_skipped_with_reason`: model caps `[text]`, task needs
    `[text, reasoning]` -> one `Skip` whose reason is `missing capability: reasoning`.
  - `test_ctx_compared_numerically`: model `ctx:32000`, task needs `ctx:128000` -> skip
    `missing capability: ctx:128000`; model `ctx:200000` -> runnable.
  - `test_sealed_task_only_for_sealed_ok`: sealed task + `sealed_ok: false` -> skip
    `sealed task, model not sealed_ok`.
  - `test_task_not_in_tier_is_not_listed`: task with only `full` tier is absent from both items and
    skips for `smoke`.
  - `test_limit_capped_by_dataset_size`: limit 100, dataset 40 -> `n_items == 40`,
    `est_samples == 40 * epochs`.
  - `test_cost_estimate`: api model price input 1.0 / output 2.0, est tokens (1000, 500), 10
    items, 1 epoch -> `est_cost_usd == pytest.approx(0.02)`.
  - `test_cli_counts_toward_quota_not_cost`: cli model -> `est_cost_usd is None`,
    `quota_samples["claude-cli-opus"] == est_samples`.
- [ ] **Step 2:** Run `uv run pytest tests/test_planner.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `planner.py`.
- [ ] **Step 4:** Implement `bench plan --models a,b --tier smoke [--tasks x,y]`: prints the items
  table, the skips table with reasons, total estimated USD, and quota samples per CLI model.
  `--models all` selects every registry model.
- [ ] **Step 5:** Run `uv run pytest -v`. Expected: PASS.
- [ ] **Step 6:** Commit: `feat: capability-matching planner with cost and quota estimates`

### Task 4: Claude CLI and Codex CLI model providers

**Files:**
- Create: `src/model_bench/providers/cli_common.py`, `providers/claude_cli.py`,
  `providers/codex_cli.py`, `providers/_registry.py`, `tests/test_cli_providers.py`,
  `tests/fixtures/{claude_ok.json,claude_limit.txt,codex_ok.jsonl,codex_limit.txt}`

**Interfaces:**
- Consumes: Inspect `ModelAPI`, `modelapi`, `ModelOutput.from_content`, `ModelUsage`
  (verified in 0.3.291: `ModelAPI.generate(self, input, tools, tool_choice, config) -> ModelOutput`).
- Produces:
  - `def serialise_messages(messages: list[ChatMessage]) -> tuple[str | None, str]` -- returns
    (system_prompt, prompt). System messages are joined into the system prompt; remaining turns
    become one prompt with each turn prefixed `[user]` / `[assistant]` on its own line. A single
    user message is passed through unprefixed.
  - `class CLIError(RuntimeError)`, subclasses `CLIUsageLimit`, `CLIAuthError`, `CLIBadOutput`,
    `CLITimeout`.
  - `async def run_cli(argv: list[str], stdin: str | None, timeout_s: int) -> tuple[int, str, str]`
    -- runs in a fresh `tempfile.TemporaryDirectory()` as cwd, environment copied minus
    `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` (so the subscription login is used, not a stray key).
  - `def claude_argv(model: str, system_prompt: str | None, effort: str | None) -> list[str]` ->
    `["claude", "-p", "--output-format", "json", "--model", model, "--tools", "",
    "--strict-mcp-config", "--setting-sources", "", "--no-session-persistence",
    "--system-prompt", system_prompt or ""]` plus `["--effort", effort]` when set. Prompt goes on
    stdin.
  - `def parse_claude(stdout: str, stderr: str, code: int) -> tuple[str, ModelUsage, str]` ->
    (text, usage, reported_model_id). Raises the matching `CLIError`.
  - `def codex_argv(model: str, out_file: Path) -> list[str]` -> `["codex", "exec", "--json",
    "--ephemeral", "--skip-git-repo-check", "-s", "read-only", "-m", model, "-o", str(out_file),
    "-"]` (prompt on stdin; system prompt prepended to the prompt as a `[system]` block, documented
    as a known non-isolation).
  - `def parse_codex(stdout: str, stderr: str, code: int, last_message: str) -> tuple[str, ModelUsage]`
  - `@modelapi(name="claude-cli")` and `@modelapi(name="codex-cli")` in `_registry.py`, returning
    `ClaudeCLIAPI` / `CodexCLIAPI` (`ModelAPI` subclasses). Each module docstring lists what cannot
    be switched off for that CLI.

- [ ] **Step 1: Record real fixtures** (owner present; one tiny call each, negligible quota):
  run `claude_argv("haiku", "Answer with one word.", None)` with prompt `Say OK` and save stdout to
  `tests/fixtures/claude_ok.json`; same with `codex_argv` to `codex_ok.jsonl`. For the limit
  fixtures, copy the exact usage-limit / not-logged-in text each CLI prints (from the CLI's docs or
  by running it logged out in a throwaway shell) into `*_limit.txt`. Confirm `--setting-sources ""`
  is accepted by `claude --help` in 2.1.330; if not, use the narrowest accepted value and note it
  in the docstring.
- [ ] **Step 2: Write failing tests** in `tests/test_cli_providers.py`:
  - `test_serialise_single_user_message_passthrough`
  - `test_serialise_system_and_multiturn`: system + user + assistant + user -> system prompt equals
    system text; prompt has `[user]`, `[assistant]`, `[user]` lines in order.
  - `test_claude_argv_isolation`: argv contains `--tools`, `""`, `--strict-mcp-config`,
    `--no-session-persistence`, `--setting-sources`; never contains `--continue` or `--resume`.
  - `test_run_cli_uses_empty_temp_cwd`: run `["python", "-c", "import os;print(os.getcwd(), len(os.listdir()))"]`;
    assert cwd is under the system temp dir and listing count is `0`.
  - `test_run_cli_strips_api_keys`: with `ANTHROPIC_API_KEY=x` set, child env lacks it.
  - `test_parse_claude_ok`: fixture -> non-empty text, `usage.output_tokens > 0`, model id non-empty.
  - `test_parse_claude_usage_limit_raises`: `claude_limit.txt` as stdout/stderr with non-zero code
    -> `CLIUsageLimit`.
  - `test_parse_claude_garbage_raises_bad_output`: stdout `not json`, code 0 -> `CLIBadOutput`.
  - `test_parse_codex_ok` and `test_parse_codex_usage_limit_raises` (same pattern).
  - `test_generate_error_is_sample_error_not_empty_answer`: run a 1-sample Inspect task with
    `claude-cli/haiku` where `run_cli` is monkeypatched to return the limit fixture; assert the
    log's sample has `error` set and **no** score of 0 was recorded for it (Review Focus 1).
  - `test_timeout_raises_cli_timeout`: `run_cli` with a `sleep 5` command and `timeout_s=1` ->
    `CLITimeout`, child process killed.
- [ ] **Step 3:** Run `uv run pytest tests/test_cli_providers.py -v`. Expected: FAIL.
- [ ] **Step 4:** Implement `cli_common.py`, `claude_cli.py`, `codex_cli.py`, `_registry.py`.
  `generate` raises `CLIError` subclasses (Inspect turns a raised exception into a sample error).
  Default per-provider `max_connections` = 1, `timeout_s` = 600. Ignore `tools` (planner never
  sends tool tasks to CLI models); if `tools` is non-empty raise `CLIError("tool tasks unsupported
  on CLI providers")`.
- [ ] **Step 5:** Add a `@pytest.mark.live` test `test_live_claude_and_codex_one_sample` that
  runs one real `Say OK` sample through each provider via Inspect. Not run by default.
- [ ] **Step 6:** Run `uv run pytest -v`. Expected: PASS (live test deselected).
- [ ] **Step 7:** Commit: `feat: claude-cli and codex-cli Inspect providers with isolation and error mapping`

### Task 5: Runner, manifest, budget guard, ledger

**Files:**
- Create: `src/model_bench/runner.py`, `tasks/fixtures/tiny_task.py`, `tests/test_runner.py`
- Modify: `src/model_bench/cli.py` (implement `bench run`)

**Interfaces:**
- Consumes: `Plan`, `PlanItem` (Task 3); registries (Task 2); providers registered via entry point
  (Task 4).
- Produces:
  - `Manifest(run_id: str, created: datetime, tier: str, git_sha: str, git_dirty: bool, inspect_ai: str, inspect_evals: str, cli_versions: dict[str, str], models: list[ModelEntry], tasks: list[TaskEntry], budget_usd: float, spend_usd: float, status: Literal["running","complete","stopped_budget","stopped_quota","failed"])`
    written as `logs/<run_id>/manifest.json` (rewritten at each status change).
  - `class BudgetExceeded(RuntimeError)` -- raised before any call when the plan estimate exceeds
    the budget.
  - `def run(plan: Plan, budget_usd: float, run_id: str | None = None, log_root: Path = Path("logs")) -> Manifest`
    -- `run_id=None` creates `YYYYMMDD-HHMMSS-<tier>`; passing an existing `run_id` resumes it.
  - `def build_task(entry: TaskEntry, item: PlanItem) -> inspect_ai.Task` -- loads the Inspect task
    (registry name for `inspect_evals/*`, file path for `local:`), applies `task_args`, shuffles
    the dataset with seed `20261122` and slices to `item.limit` so every model and every run of a
    tier gets the same items, then `task_with(..., epochs=item.epochs)`. Builds a fresh task object
    per call (`task_with` mutates in place).
  - Ledger: one JSON line per run appended to `logs/ledger.jsonl`:
    `{run_id, date, tier, spend_usd, quota_samples}`.

- [ ] **Step 1:** Create `tasks/fixtures/tiny_task.py`: an `@task def tiny()` with 3 samples
  (ids `a`,`b`,`c`), `generate()` solver, `match()` scorer. Register it in a test-only registry YAML
  with `source: local:tasks/fixtures/tiny_task.py@tiny`.
- [ ] **Step 2: Write failing tests** in `tests/test_runner.py` using `mockllm/model` registered as
  an `api` model with a price:
  - `test_run_writes_logs_and_manifest`: run tiny task smoke -> `logs/<run_id>/manifest.json`
    exists, status `complete`, at least one `.eval` file in `logs/<run_id>/`.
  - `test_manifest_records_versions_and_git`: manifest `inspect_ai == "0.3.291"`, `git_sha`
    non-empty, `git_dirty` is a bool.
  - `test_refuses_when_estimate_exceeds_budget`: plan estimate 5.0, budget 1.0 -> raises
    `BudgetExceeded` before any log is written.
  - `test_stops_between_chunks_when_spend_crosses_budget`: two (task, model) chunks; monkeypatch
    measured spend after chunk 1 above budget -> chunk 2 never starts, status `stopped_budget`.
  - `test_same_items_every_run`: two runs of the same tier -> identical sample id sets.
  - `test_resume_reuses_run_id`: run with forced failure on chunk 2 then `run(plan, run_id=<same>)`
    -> manifest `complete`, chunk 1 not re-executed (its `.eval` file mtime unchanged).
  - `test_quota_error_stops_model`: a chunk whose samples errored with `CLIUsageLimit` -> manifest
    status `stopped_quota`, later chunks for that model not started, other models continue.
- [ ] **Step 3:** Run `uv run pytest tests/test_runner.py -v`. Expected: FAIL.
- [ ] **Step 4:** Implement `runner.py`. Execute one `eval_set(...)` per (model, task) chunk into
  `logs/<run_id>/` (`log_dir` shared so resume works), with `retry_attempts=0`, `fail_on_error`
  left at the Inspect default for API/local models and `fail_on_error=False` for CLI models so one
  bad sample does not lose the chunk. Pass `model_cost_config` built from `models.yaml` prices so
  Inspect records `total_cost`. Measured spend = sum of `model_usage[*].total_cost` from the
  chunk's logs. Budget is checked before the run (estimate) and after every chunk (measured);
  overshoot is bounded by one chunk (designer call: Inspect's `cost_limit` is per sample, not per
  run). CLI versions come from `claude --version` / `codex --version`.
- [ ] **Step 5:** Implement `bench run --models ... --tier ... --budget <usd> [--run-id <id>]`:
  prints the plan, asks for `y` confirmation when any paid or CLI model is included (skip with
  `--yes`), then runs.
- [ ] **Step 6:** Run `uv run pytest -v`. Expected: PASS.
- [ ] **Step 7:** Commit: `feat: runner with manifest, budget guard, resume and ledger`

### Task 6: Ingest into DuckDB

**Files:**
- Create: `src/model_bench/ingest.py`, `tests/test_ingest.py`
- Modify: `src/model_bench/cli.py` (implement `bench ingest [--rebuild]`)

**Interfaces:**
- Consumes: `logs/<run_id>/manifest.json`, `.eval` logs via `inspect_ai.log.list_eval_logs` /
  `read_eval_log`.
- Produces DuckDB file `results.duckdb` with tables (primary keys enforce Review Focus 3):
  - `runs(run_id PK, created, tier, git_sha, git_dirty, inspect_ai, inspect_evals, status, spend_usd, manifest_json)`
  - `samples(run_id, task, model_id, sample_id, epoch, reported_model, access, input_tokens, output_tokens, total_cost, total_time, error, PRIMARY KEY(run_id, task, model_id, sample_id, epoch))`
  - `scores(run_id, task, model_id, sample_id, epoch, scorer, value_num, value_raw, PRIMARY KEY(run_id, task, model_id, sample_id, epoch, scorer))`
  - `def ingest(log_root: Path, db_path: Path, rebuild: bool = False) -> int` (rows upserted)
  - `value_num`: Inspect `C`/`I` -> 1.0/0.0, numeric values as-is, otherwise NULL.

- [ ] **Step 1: Write failing tests** in `tests/test_ingest.py` (produce logs with the Task 5
  runner on `mockllm/model` + tiny task, epochs 2):
  - `test_ingest_row_counts`: 3 samples x 2 epochs -> 6 `samples` rows, 6 `scores` rows.
  - `test_ingest_idempotent`: ingest twice -> still 6 rows.
  - `test_ingest_after_resume_no_duplicates`: resumed run (Task 5 resume scenario) -> one row per
    key.
  - `test_errored_sample_has_error_and_no_score`: a sample that errored -> `samples.error` set,
    no row in `scores`.
  - `test_rebuild_recreates_from_logs`: delete DB, `ingest(rebuild=True)` -> same counts.
  - `test_model_id_maps_back_to_registry`: `samples.model_id` is the registry id, not the Inspect
    model string; `access` filled from the manifest.
- [ ] **Step 2:** Run `uv run pytest tests/test_ingest.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `ingest.py` (upsert with `INSERT OR REPLACE`).
- [ ] **Step 4:** Run `uv run pytest -v`. Expected: PASS.
- [ ] **Step 5:** Commit: `feat: ingest Inspect logs into DuckDB`

### Task 7: Statistics

**Files:**
- Create: `src/model_bench/stats.py`, `tests/test_stats.py`

**Interfaces:**
- Produces (pure functions over a long-format list of `(item_id, epoch, value)`):
  - `Estimate(mean: float | None, se: float | None, ci_low: float | None, ci_high: float | None, n_items: int, n_obs: int)`
  - `def clustered_mean(obs: list[tuple[str, int, float]]) -> Estimate` -- item-clustered SE:
    average epochs within each item, then SE of the item means (`sd / sqrt(n_items)`), 95% CI
    `mean +/- 1.96 * se`. `se`/CI are None when `n_items < 2`.
  - `def paired_diff(a: list[tuple[str,int,float]], b: list[tuple[str,int,float]], n_boot: int = 10000, seed: int = 20261122) -> Estimate`
    -- over shared item ids only; bootstrap resampling of items; CI from 2.5/97.5 percentiles.
  - `def mcnemar(a: dict[str, bool], b: dict[str, bool]) -> tuple[int, int, float | None]` ->
    (b_only, a_only, exact two-sided p) over shared items, using item-majority correctness.
  - `def pass_hat_k(obs: list[tuple[str,int,float]], k: int) -> float | None` -- share of items
    whose first k epochs all score 1.0; None if any item has fewer than k epochs or `k < 2`.

- [ ] **Step 1: Write failing tests** in `tests/test_stats.py`:
  - `test_clustered_mean_known_values`: items x,y,z with item means 1.0, 0.0, 0.5 (2 epochs each)
    -> mean 0.5, se `pytest.approx(0.5/ (3**0.5))`.
  - `test_clustered_differs_from_naive_when_epochs_correlate`: 2 items x 5 identical epochs ->
    `n_items == 2`, SE computed on 2 items not 10.
  - `test_single_item_gives_none_ci` (Review Focus 5).
  - `test_all_identical_scores`: all 1.0 -> mean 1.0, se 0.0, CI (1.0, 1.0) and a flag is not
    needed; report layer prints it as `1.000 [1.000, 1.000]`.
  - `test_paired_diff_uses_shared_items_only`: a has items {1,2,3}, b has {2,3,4} -> `n_items == 2`.
  - `test_paired_diff_deterministic_seed`: two calls -> identical CI.
  - `test_mcnemar_exact`: b_only 0, a_only 6 -> p `pytest.approx(0.03125)`.
  - `test_pass_hat_k_none_when_epochs_1` (Review Focus 5) and `test_pass_hat_k_value`.
- [ ] **Step 2:** Run `uv run pytest tests/test_stats.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `stats.py` (numpy allowed; `math.comb` for McNemar).
- [ ] **Step 4:** Run `uv run pytest -v`. Expected: PASS.
- [ ] **Step 5:** Commit: `feat: clustered CIs, paired bootstrap, McNemar and pass^k`

### Task 8: Report and compare

**Files:**
- Create: `src/model_bench/report.py`, `tests/test_report.py`
- Modify: `src/model_bench/cli.py` (implement `bench report`, `bench compare`)

**Interfaces:**
- Consumes: DuckDB tables (Task 6), `stats` functions (Task 7).
- Produces:
  - `def report(db: Path, run_id: str | None = None) -> str` -- markdown; latest run when None.
    Written to `reports/<run_id>.md`.
  - `def compare(db: Path, a: str, b: str, tier: str | None = None) -> str` -- markdown; uses the
    latest run that contains both models (or each model's latest run of that tier if no shared run,
    with a warning line that items may differ).

- [ ] **Step 1: Write failing tests** in `tests/test_report.py` on a DB built from mockllm runs:
  - `test_report_every_score_has_ci`: every task x model row matches
    `\d\.\d{3} \[\d\.\d{3}, \d\.\d{3}\]` or contains `n/a`.
  - `test_cli_rows_labelled`: a model with `access: cli` shows `via CLI` in its row.
  - `test_dirty_tree_flagged`: run with `git_dirty=True` -> report header contains
    `WARNING: run from a dirty working tree`.
  - `test_errors_counted_not_scored`: a run with 1 errored sample shows `errors: 1` and the mean
    uses the remaining samples.
  - `test_report_includes_cost_latency_tokens`: columns `cost_usd`, `p50_s`, `p95_s`, `out_tokens`.
  - `test_pass_k_column_na_when_epochs_1`.
  - `test_quality_vs_cost_section`: report has a `Quality vs cost` table with one row per model
    (mean score across tasks, total cost, cost per correct answer) and marks models no other
    model beats on both quality and cost as `frontier`; CLI rows show `quota` instead of cost.
  - `test_compare_paired_diff_and_mcnemar`: compare output has `diff`, 95% CI and McNemar p.
- [ ] **Step 2:** Run `uv run pytest tests/test_report.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `report.py` and the two CLI commands.
- [ ] **Step 4:** Run `uv run pytest -v`. Expected: PASS.
- [ ] **Step 5:** Commit: `feat: markdown report and paired model comparison`

### Task 9: Reproduction check

**Files:**
- Create: `src/model_bench/repro.py`, `reference_scores.yaml`, `tests/test_repro.py`
- Modify: `src/model_bench/cli.py` (implement `bench repro`)

**Interfaces:**
- Consumes: DuckDB (Task 6), `clustered_mean` (Task 7), registries (Task 2).
- Produces:
  - `reference_scores.yaml` entries: `{model_id, task, score, source_url, source_kind: provider|third_party, settings: str, retrieved: date, explanation: str | null}`
  - `ReproRow(model_id, task, ours: Estimate, published: float, verdict: Literal["match","gap","no_data"], explanation: str | None)`
  - `def repro(db: Path, refs: Path) -> list[ReproRow]` -- uses the latest `full` run for that
    model+task, else latest `standard` (and says which). `match` iff published lies within our
    95% CI. CLI-access models are skipped with a line saying why.

- [ ] **Step 1: Write failing tests** in `tests/test_repro.py`:
  - `test_match_when_published_inside_ci`, `test_gap_when_outside_ci`,
    `test_no_data_when_model_task_not_run`.
  - `test_cli_models_excluded`: a reference row for a cli model -> excluded, reason printed.
  - `test_provider_source_preferred`: two refs for same model+task (provider and third_party) ->
    provider used.
  - `test_gap_without_explanation_marked_open`: output line contains `OPEN: needs explanation`.
- [ ] **Step 2:** Run `uv run pytest tests/test_repro.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `repro.py`; fill `reference_scores.yaml` with published GPQA Diamond,
  MMLU-Pro and IFEval numbers for `deepseek-chat` / `deepseek-reasoner` and `ollama-qwen3-8b`
  from the providers' model cards or technical reports (source URL + stated settings for each;
  third-party only where the provider published none).
- [ ] **Step 4:** Run `uv run pytest -v`. Expected: PASS.
- [ ] **Step 5:** Commit: `feat: reproduction check against published scores`

### Task 10: Live acceptance (owner present)

**Files:**
- Modify: `models.yaml` (Ollama digest, confirmed model names), `tasks/registry.yaml`
  (`est_tokens_per_sample`), `README.md` (how to run, what each label means)
- Create: `reports/` outputs (gitignored), `docs/acceptance-2026-11-XX.md` (committed summary,
  no private data)

**Interfaces:**
- Consumes: everything above. No new code unless a defect is found (fix it in the owning task's
  module with a regression test, then continue).

- [ ] **Step 1:** Owner: put `DEEPSEEK_API_KEY` in `.env`; `ollama pull qwen3:8b`; record the digest
  from `ollama show qwen3:8b` into `models.yaml`; turn off "use chats for training" in the Claude
  and ChatGPT apps.
- [ ] **Step 2:** `uv run bench run --models ollama-qwen3-8b --tier smoke --budget 0` (free).
  Expected: status `complete`; then update `est_tokens_per_sample` from the measured averages.
- [ ] **Step 3:** **Ask the owner to approve** before continuing. Then:
  `uv run bench plan --models deepseek-chat,claude-cli-sonnet,codex-cli-default --tier smoke`;
  show the estimate; on approval
  `uv run bench run --models deepseek-chat,claude-cli-sonnet,codex-cli-default --tier smoke --budget 2`.
  Expected: status `complete` or `stopped_quota` with a resumable run id.
- [ ] **Step 4:** `uv run pytest -m live -v`. Expected: PASS.
- [ ] **Step 5:** Repro run, on approval:
  `uv run bench run --models deepseek-chat,ollama-qwen3-8b --tier standard --tasks gpqa_diamond,mmlu_pro,ifeval --budget 8`,
  then `uv run bench ingest && uv run bench repro`. Expected: one row per model x task with
  verdict. For every `gap`, spend at most half a day finding the cause (prompt template, answer
  extraction, reasoning settings, item subset) and record it in `reference_scores.yaml`
  `explanation`.
- [ ] **Step 6:** `uv run bench report` and
  `uv run bench compare claude-cli-sonnet codex-cli-default`; write
  `docs/acceptance-<date>.md` with: what ran, spend, repro verdicts and explanations, known CLI
  non-isolations.
- [ ] **Step 7:** Commit: `docs: phase 1 acceptance run` (owner pushes to the private remote).

---

## Deferred to Phase 2/3 plans (per spec section 6)

Judge panel + calibration, XSTest, MMMU, long context, BFCL, private task format, `bench backup`
(encrypted iCloud archive; Time Machine covers Phase 1 logs, which hold no private items yet),
blind side-by-side, drift + scheduling, Gemini CLI provider, Docker sandboxes, coding / SWE-bench,
private-harness provider.
