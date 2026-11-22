# model-bench -- Private Model Benchmark and Test Harness (Design)

Date: 2026-11-22
Status: draft, awaiting owner approval
Location: `~/Desktop/model-bench/` (new standalone project)

---

## 1. Intent and why

The owner wants a private, standing benchmark they control, so that:

1. They **understand how public online benchmarks actually run** -- the software decisions behind
   them (dataset / solver / scorer split, prompt templates, answer extraction, sandboxes, judges,
   error bars) -- by running real public benchmarks through a mainstream harness and reproducing
   published numbers.
2. They can **point the system at any model and have it automatically run the tests that model is
   capable of** (text, tools, vision, long context, reasoning, coding sandbox, Cantonese, ...),
   without hand-picking tests per model.
3. They can **re-run the same suite later and detect silent degradation** of a model (provider-side
   changes, quantisation, routing) with statistics, not vibes.
4. They can **test an AI harness, not just a bare model** -- specifically a private AI agent
   loop -- in the same grid, so they can tell whether a gain came from the model or from the
   harness.
5. They keep a **blind side-by-side human review** (the original ask) for qualities only a human
   can judge.

Prior art on this machine: `~/Desktop/dream-record/secret/model-eval/` already has a one-project
blind A/B builder (`lens-blind-build.mjs`: seeded shuffle, owner-only key, results JSON, budget
caps, re-tabulation without re-calling). model-bench generalises those habits into a standalone,
cross-model, longitudinal tool. It does not import that code.

## 2. Goals, non-goals, anti-goals

**Goals**
- G1. Run a curated set of public benchmarks via Inspect AI + `inspect_evals` on any supported model.
- G2. Capability-driven planning: each task declares what it needs, each model declares what it
  has; the planner selects the applicable tasks automatically.
- G3. Tiers (`smoke`, `standard`, `full`, `canary`) with a cost estimate before every run and a hard
  budget cap during it.
- G4. Every run is reproducible and auditable: immutable manifest + raw responses kept; scores can
  be recomputed from stored logs without new API calls.
- G5. Honest statistics: per-task mean with clustered standard errors and 95% CI; paired comparisons
  between models; pass^k reliability where epochs > 1.
- G6. Reproduction check: harness numbers for at least 3 public tasks fall within error bars of
  published numbers, or the gap is explained (prompt/extraction difference).
- G7. Drift detection over time using a fixed canary set plus an unchanged reference model.
- G8. Private task format so the owner can add their own sealed tasks (Cantonese / 書面語, HK
  context, document generation, etc.).
- G9. Blind side-by-side review page over stored responses, with Bradley-Terry ranking.
- G10. Harness-under-test: a custom Inspect model provider that sends tasks through a private
  agent loop.

**Non-goals**
- Not a public leaderboard. Nothing is published.
- Not a training pipeline. No fine-tuning, no RL.
- Not a re-implementation of benchmarks that `inspect_evals` already provides.
- Not a web app or multi-user service. Single owner, local CLI first.
- Not log-likelihood scoring of open-weight models (lm-evaluation-harness territory). Generation
  only, so open and closed models are measured the same way.

**Anti-goals (would be a failure even if it technically works)**
- Numbers without confidence intervals, or comparisons that ignore pairing -- a leaderboard of noise.
- Tuning prompts on the sealed test split (Goodhart: measuring our own overfitting).
- A full run silently spending hundreds of dollars.
- Results that cannot be traced to an exact model ID, prompt, dataset version and harness version.
- Private test items leaking into a public repo, an artifact, or a public page.
- An AI judge used for a task without first being calibrated against human labels.

## 3. Constraints (verified 2026-11-22 on the owner's Mac)

- Python 3.13 available; `uv` available at `~/.local/bin/uv`. Project uses `uv` for env + lockfile.
- `inspect-ai` latest = 0.3.291, `inspect-evals` latest = 0.29.0 on PyPI. Pin exact versions in
  `pyproject.toml`; upgrades are explicit commits, because harness upgrades can move scores.
- **Docker is NOT installed.** Every task needing a sandbox (code execution, SWE-bench, agentic
  tool tasks) is blocked until a container runtime exists (Docker Desktop, OrbStack or Colima --
  see Open Questions). Phase 1 uses only tasks that need no sandbox.
- Ollama is installed (`/usr/local/bin/ollama`) -> local models are a free target for smoke tests
  and for developing the harness without spending.
- The owner's HK-registered AWS account cannot call Anthropic models on Bedrock (blocked); Bedrock
  does serve Qwen3 / DeepSeek / Nova / gpt-oss in Tokyo. Claude models therefore go through the
  Anthropic API directly, not Bedrock.
- macOS privacy blocks launchd jobs from reading `~/Desktop` (learned in another repo). A
  scheduled canary run therefore cannot be a plain launchd job pointing at this folder.
- **The owner uses subscriptions, not API keys, for Claude, ChatGPT and Google** (Claude Code,
  ChatGPT desktop / Codex, Antigravity). DeepSeek is a pay-as-you-go API key. Scriptable access
  verified 2026-11-22:
  - Claude: `claude -p` (installed) supports `--system-prompt`, `--tools ""`, `--model`,
    `--effort`, `--output-format json`, `--no-session-persistence`, `--strict-mcp-config`,
    `--setting-sources`.
  - ChatGPT: `codex exec` (installed) supports `--json`, `-m`, `-c key=value`, `-s` sandbox,
    `--ephemeral`, `--skip-git-repo-check`, `-o`.
  - Google: Antigravity has no headless mode. Gemini CLI (not installed) with Google sign-in is
    the candidate; its quota on the owner's plan must be checked before relying on it.
- Subscription access means: no dated snapshot pinning, no temperature control, usage windows and
  weekly caps instead of per-token cost, and the CLI's own scaffolding is part of what is measured.
  Heavy unattended batch use may hit fair-use limits; the owner checks each plan's terms before
  scheduling full-tier runs.
- Consumer-plan training settings must be OFF in Claude and ChatGPT before private items are sent.
- API keys (DeepSeek) are never written to files in the repo; loaded from a gitignored `.env`.
- House rule: no emojis in code, output, commits.

## 4. Approaches considered

| | Approach | Trade-off | Decision |
|---|---|---|---|
| A | **Build on Inspect AI** (+ `inspect_evals`), add our own registry, planner, results DB, stats, drift, blind review and a private-harness provider on top | Inherits a mainstream task/solver/scorer model, sandboxes, logging and ~100 public benchmarks; reproducible against published numbers. Python, not the owner's usual TypeScript. | **Chosen by owner** |
| B | Wrapper around several native harnesses (lm-eval-harness, Inspect, official SWE-bench harness) with a normalising results layer | Most faithful to "how each benchmark officially runs", but three heterogeneous toolchains to maintain and messy cross-harness comparability | Rejected |
| C | Write from scratch in TypeScript | Maximum learning and stack match, but re-builds sandboxing/scoring and cannot tell harness bugs from model gaps on public benchmarks | Rejected |

**Model access** (decided 2026-11-22 after learning the owner uses subscriptions):

| | Access approach | Trade-off | Decision |
|---|---|---|---|
| A | **Subscriptions first**: Claude / ChatGPT / Google through their own CLIs wrapped as Inspect model providers; DeepSeek API and Ollama as true bare models | No new spend; measures what the owner actually uses day to day; CLI numbers are "model through its CLI" and not comparable to published scores | **Chosen by owner** |
| B | A plus a small Anthropic/OpenAI API top-up (~US$20-30/month) for bare-model numbers | Adds published-score comparability for Claude/GPT | Deferred; add later if A shows the need, no redesign required |
| C | API only for every model | Cleanest science; pays per token on top of existing subscriptions | Rejected |

Learning the "software decisions" is served under A by (a) reading Inspect / `inspect_evals`
task code for each benchmark we adopt, and (b) the reproduction check (G6), which forces us to
find where scores move.

## 5. Architecture

```
models.yaml ----+                     +---- tasks/registry.yaml
 (caps, price,  |                     |      (needs, tiers, category,
  pinned IDs)   v                     v       source, split, version)
              +-------------------------+
              |        planner          |  capability match x tier x budget
              |  -> plan (dry run shows |  -> cost estimate, abort if > cap
              |     tasks, items, $)    |
              +-----------+-------------+
                          v
              +-------------------------+
              | runner (Inspect eval_set)|  epochs, fixed params, per-provider
              |  -> logs/*.eval          |  concurrency, retries, budget guard
              +-----------+-------------+
                          v
              +-------------------------+
              | ingest -> results.duckdb |  runs, samples, scores, usage, manifest
              +-----------+-------------+
          +---------------+----------------+-----------------+
          v               v                v                 v
     report/compare    drift check     blind review       repro check
     (CI, paired,      (canary vs      (pairs from        (vs reference_
      pass^k, cost,     baseline,       stored outputs,    scores.yaml)
      Pareto)           anchor model)   Bradley-Terry)
```

### 5.1 Repo layout

```
model-bench/
  pyproject.toml, uv.lock          # pinned inspect-ai, inspect-evals, duckdb, etc.
  models.yaml                      # model registry
  tasks/registry.yaml              # task registry (public + private)
  tasks/private/<task>/            # private task code + data (see 5.8)
  reference_scores.yaml            # published numbers used by the repro check
  src/model_bench/
    cli.py                         # `bench` entry point
    registry.py  planner.py  runner.py  ingest.py  stats.py
    report.py  drift.py  blind/  judges/  providers/harness.py
  logs/                            # Inspect .eval logs (gitignored, backed up -- see Open Q)
  results.duckdb                   # derived; rebuildable from logs/ (gitignored)
  reports/                         # generated markdown/HTML reports
  docs/superpowers/specs/          # this document
```

### 5.2 Model registry (`models.yaml`)

One entry per model under test:

- `id` (our stable key), `access` (`cli` / `api` / `local`), `inspect_model` (Inspect provider
  string: `claude-cli/<model>`, `codex-cli/<model>`, `gemini-cli/<model>` for our CLI providers
  (5.14); built-in strings such as `openai-api/deepseek/...` or `ollama/<model>` otherwise).
- `pinned` -- exact dated snapshot ID where the provider offers one (API only). For CLI access
  record the alias; the ingest step stores the model ID the CLI/API actually reported back, plus
  the CLI version. For Ollama, record the weights digest.
- `privacy` -- `sealed_ok: true|false`: whether sealed private tasks may be sent to this model
  (5.8).
- `caps` -- capability list from a fixed vocabulary: `text`, `json`, `tools`, `vision`, `reasoning`,
  `ctx:<tokens>`, `lang:yue`, `lang:zh-hant`, `sandbox_agent`.
- `params` -- defaults applied to every run (temperature, max tokens, reasoning effort). Fixed per
  model for comparability; overriding creates a different `id` (e.g. `claude-opus-5-5@effort-high`).
- `price` -- per-million input/output (and cached) token prices, used for estimates and cost
  reports. Updated by hand; date-stamped.
- `role` -- optional `reference` flag for the drift anchor model.

Capabilities are declared by the owner, not auto-probed. (A probe command that checks a declared
capability actually works is left to the builder as optional.)

### 5.3 Task registry (`tasks/registry.yaml`)

Each task entry:

- `name`, `source` (`inspect_evals:<task>` or `private:<path>`), `category`, `needs` (capability
  list, same vocabulary), `tiers` (which of smoke/standard/full/canary include it, with per-tier
  `limit` and `epochs`), `split` (`dev` or `test`), `version`, `scorer_kind`
  (`deterministic` / `judge` / `human`), `est_tokens_per_sample` (for the cost estimate).

Planner rule: a task runs on a model iff `needs` is a subset of `caps` (with `ctx:` compared
numerically) and the task is in the requested tier.

### 5.4 Initial task suite

Public tasks via `inspect_evals` (builder confirms exact task names exist at the pinned version
and substitutes the nearest equivalent if one has moved or been renamed):

| Category | Task | Needs | Scorer | Phase |
|---|---|---|---|---|
| Graduate reasoning | GPQA Diamond | text | deterministic (choice) | 1 |
| Broad knowledge | MMLU-Pro (subset per tier) | text | deterministic | 1 |
| Instruction following | IFEval | text | deterministic | 1 |
| Math | AIME (latest year available) | text, reasoning | deterministic | 1 |
| Over-refusal | XSTest | text | judge | 2 |
| Vision | MMMU (subset) | vision | deterministic | 2 |
| Long context | a needle/long-doc task from `inspect_evals` | ctx:128000 | deterministic | 2 |
| Tool calling | a function-calling task from `inspect_evals` (e.g. BFCL if present) | tools | deterministic | 2 |
| Coding | HumanEval / a LiveCodeBench-style task | text, sandbox_agent | unit tests in sandbox | 3 |
| Agentic SWE | SWE-bench Verified (small fixed subset) | tools, sandbox_agent | hidden tests in sandbox | 3 |

Private tasks (format built in phase 2; content authored by the owner -- see 5.8):
Cantonese and 書面語 writing/comprehension, HK-specific knowledge, and owner workload samples.

Reproduction check targets (phase 1): GPQA Diamond, MMLU-Pro, IFEval, compared to numbers the
model provider or a reputable third party published, recorded in `reference_scores.yaml` with the
source URL and the settings they used. **The repro check runs only on bare models** (DeepSeek API
and open Ollama models such as Qwen3 with published scores). CLI-access models are excluded from
repro and labelled "via CLI" in every report.

### 5.5 Tiers and cost control

- `smoke`: every applicable task, small `limit` (~10-20), epochs 1. Purpose: does the pipeline work
  for this model. Target: minutes and about US$1 on a frontier model.
- `standard`: larger limits, epochs 3 on cheap tasks. The default comparison tier.
- `full`: all items, task-defined epochs. Run deliberately; most expensive.
- `canary`: a fixed ~50-item mix across categories drawn from the test split, epochs 3, used only
  for drift.

Before any run: `bench plan` prints tasks x models x items x epochs and an estimated cost from
`est_tokens_per_sample` x `price`. `bench run` refuses to start if the estimate exceeds `--budget`
(default set in config), and stops launching new samples when measured spend (from Inspect usage
records x price) crosses the cap. Every run appends to a spend ledger.

- **Money budget** applies to API models (DeepSeek): monthly cap about US$10 by default.
- **Quota budget** applies to CLI models: the plan shows samples per CLI model; when a CLI reports
  a usage-limit error the runner stops that model cleanly and the run is resumable later via
  `eval_set` (no finished sample is redone). Per-CLI concurrency defaults low (1-2) to avoid
  burning a usage window. Full-tier runs on CLI models are never scheduled unattended by default.

### 5.6 Runner and run manifest

- Uses Inspect's `eval_set` so a partly failed set can be resumed without redoing finished work.
- Per-provider concurrency limits and retries come from Inspect; values set in config.
- Each run writes a manifest: harness git SHA (and whether the tree was dirty), `inspect-ai` and
  `inspect-evals` versions, task versions + dataset content hash, model registry entry (including
  params), model ID reported by the API, provider/region, start/end timestamps, tier, budget,
  actual spend.
- A run on a dirty tree is allowed but flagged in every report.

### 5.7 Ingest, statistics and reports

- `ingest` reads Inspect `.eval` logs into DuckDB tables: `runs`, `samples` (one row per
  item x epoch), `scores`, `usage`. DuckDB is derived data -- `bench ingest --rebuild` recreates it
  from `logs/`.
- Statistics:
  - per task x model: mean, standard error clustered by item (epochs are not independent items),
    95% CI;
  - model vs model: paired difference over shared items with paired bootstrap CI; McNemar for
    binary tasks;
  - reliability: pass^k (all k epochs correct) next to the mean;
  - cost per task and per correct answer, latency (p50/p95), tokens;
  - a quality-vs-cost Pareto view across models.
- `bench report` writes a markdown + static HTML report; `bench compare <a> <b>` writes the paired
  view. No number is shown without its interval.

### 5.8 Private tasks and confidentiality

- Private tasks live in `tasks/private/` as normal Inspect tasks (dataset file + solver + scorer),
  split into `dev` (may tune prompts on) and `test` (sealed: run only, never inspected while
  tuning).
- Each private item carries a canary GUID string so leakage into training data can later be tested.
- The repo is a **private** GitHub repo. Private items are committed (losing them is worse than the
  risk of a private repo). They are never sent to artifacts or public pages.
- Sending items to a provider exposes them to that provider. Owner decision (2026-11-22): private
  tasks marked `sealed` may go **only** to models with `sealed_ok: true` -- by default the Claude
  CLI, the Codex CLI and Ollama (training settings off in the consumer apps). DeepSeek and
  Gemini default to `sealed_ok: false`. The planner skips sealed tasks for other models and says
  so. Every run's manifest records which models saw which private tasks.
- Backups (owner decision): `logs/`, `blind/keys/` and the private tasks are covered by Time
  Machine, plus a weekly encrypted archive (`bench backup`) written to a private iCloud Drive
  folder. The encryption key never lives in the repo.
- Builder delivers the format, one example private task with ~5 items, and an authoring guide; the
  owner writes the real items.

### 5.9 AI judges

- Judges are Inspect model-graded scorers, configured as a **panel** of at least two models from
  different providers; a model is never the only judge of its own family. Default panel: Claude
  CLI + Codex CLI (DeepSeek may join for public tasks only, since it is not `sealed_ok`). Judge
  calls count against the same quotas as 5.5.
- Pairwise judging runs both orders (A/B and B/A); disagreement between orders counts as a tie.
- Before a judge-scored task is trusted, it is calibrated: the owner labels ~50 items, and the
  panel's agreement with the owner (Cohen's kappa) must be >= 0.6. Tasks whose judge has not passed
  calibration are reported as "uncalibrated" and excluded from headline comparisons.

### 5.10 Drift detection

- The `canary` tier runs on a schedule (weekly target) on the models being tracked plus the
  `reference` model.
- A drop is flagged when the paired difference vs that model's baseline canary run is below zero
  with 95% confidence. If the reference model also dropped, the flag is marked "likely harness /
  environment", not "model degraded".
- Scheduling mechanism is left to the builder within the constraint in section 3 (launchd cannot
  read `~/Desktop`); a manual `bench drift run` must always work.

### 5.11 Blind side-by-side review

- `bench blind build --task <t> --models <a,b,...>` samples items with stored responses, pairs
  model outputs, randomises left/right with a recorded seed, strips model-identifying text
  (provider names, self-references) and renders both sides with the same renderer.
- Writes a self-contained local HTML page with no model names in it; the side-to-model key is
  written separately (`blind/keys/`, gitignored) and never into the page.
- Choices: A, B, tie, both bad. About 10% of pairs are repeats (and some model-vs-itself pairs)
  to measure the owner's own consistency.
- Served by a tiny local server (`bench blind serve`) that records votes into DuckDB; nothing
  leaves the machine. `bench blind score` fits Bradley-Terry with bootstrap CIs and reports
  self-consistency.

### 5.12 A private agent harness as a model under test

- A custom Inspect model provider (`harness/<variant>`) that sends a sample's messages through the
  private agent's loop and returns the final answer (plus, where available, which underlying model
  the harness router chose, and turn/tool counts) so it sits in the same grid as bare models.
- Grid convention: for the same base model, compare `bare API` vs `harness/*` to separate model
  effect from harness effect.
- Owner default: target a **local dev instance** with a dedicated test account, never
  production. Whether the harness's router can be forced onto one base model is verified at the start
  of Phase 3; if it cannot, the grid reports the harness against the model the router actually chose
  (recorded per sample) instead of a matched bare baseline.
- It must not add any auth bypass to the harness (owner rule: no bypass anywhere).

### 5.14 CLI subscription providers

- Three custom Inspect model providers (`claude-cli`, `codex-cli`, `gemini-cli`) that each run one
  sample as a fresh, non-persistent, headless CLI call and return the final message as the model
  output, with usage numbers when the CLI reports them.
- Isolation for a fair "model" measurement: the task's system prompt replaces the CLI's
  (`claude -p --system-prompt`), built-in tools are off (`--tools ""`), no MCP servers
  (`--strict-mcp-config`), no user/project settings or memory loaded, run in an empty temp
  directory, no session saved. Codex runs with `--ephemeral`, read-only sandbox, empty temp dir.
  Whatever cannot be switched off for a CLI is listed in that provider's docstring and in reports.
- Multi-turn chat inputs are serialised into one prompt in a fixed, documented format; tool-use
  tasks are not run through CLI providers in Phase 1 (their `caps` omit `tools`).
- The CLI version is recorded in the run manifest, because a CLI update can move scores just like a
  model update.
- Phase 1 builds `claude-cli` and `codex-cli`. `gemini-cli` is built once Gemini CLI is installed
  and its quota on the owner's plan is confirmed.

### 5.13 CLI surface

```
bench models                       # list registry + caps
bench plan  --models a,b --tier smoke [--tasks ...]   # dry run with cost
bench run   --models a,b --tier standard --budget 20
bench ingest [--rebuild]
bench report [--run <id>]
bench compare a b [--tier standard]
bench repro                        # reproduction check vs reference_scores.yaml
bench drift run | bench drift status
bench blind build | serve | score
```

## 6. Phasing

- **Phase 1 -- foundations (no Docker, no judges):** project setup, model + task registries,
  planner with cost estimate and budget cap, runner via `eval_set`, manifest, ingest to DuckDB,
  stats + report + compare, reproduction check on GPQA Diamond / MMLU-Pro / IFEval, AIME.
  `claude-cli` and `codex-cli` providers. Develop against Ollama first, then DeepSeek, then the
  CLIs.
  Done when: `bench run --tier smoke` works end to end on Ollama, DeepSeek, Claude CLI and Codex
  CLI, and `bench repro` reports results for the 3 reference tasks on DeepSeek and one Ollama
  model.
- **Phase 2 -- judgement and private data:** judge panel + calibration workflow, XSTest, vision,
  long context, tool calling, private task format + example, blind side-by-side, drift.
- **Phase 3 -- sandboxes and harnesses:** container runtime, coding + SWE-bench subset, private agent harness
  provider, scheduled canary.

## 7. Open questions for the owner

Answered by the owner on 2026-11-22:
- Access: subscriptions via CLIs for Claude / ChatGPT / Google, DeepSeek API key, Ollama local
  (Approach A in section 4).
- Private items: sealed tasks only to Claude CLI, Codex CLI, Ollama (5.8).
- Backups: Time Machine + weekly encrypted archive to private iCloud Drive (5.8).
- Private agent harness: Phase 3, local dev instance, test account (5.12).

Still open, to be checked by the owner before the step that needs it:
- Each plan's terms on scripted/batch use (before any scheduled CLI run).
- Gemini CLI quota on the owner's Google plan (before building `gemini-cli`).

### Calls made by the designer after the builder check (owner may override)

- **Repro check in Phase 1 reports, it does not gate.** Phase 1 is done when `bench repro` prints
  harness score vs published score with CI and the settings difference for each of the 3 tasks.
  Each gap outside the CI gets a timeboxed investigation (max half a day per task) and a written
  explanation in the report. Source priority: the provider's own published number with its stated
  settings; otherwise a reputable third party (e.g. Epoch AI, Artificial Analysis), source URL
  recorded.
- **Drift anchors are two models:** a local Ollama model with a fixed weights digest (cannot
  drift, so a drop there means harness/environment change) and DeepSeek via API (a cheap provider
  anchor). Canary size starts at ~50 items within the quota/money budgets; every drift report
  states the minimum detectable drop for that size instead of implying it catches everything.
- **Plan scope:** the implementation plan details Phase 1 only. Phases 2 and 3 stay at the level
  of this document and get their own plans when started.
- **No AI provider keys are in the shell environment today** (checked 2026-11-22, names only).
  Keys go in a gitignored `.env` loaded at run time.

## 8. Left to the builder

- Exact `inspect_evals` task names and the nearest substitutes at the pinned version.
- Config file format details, CLI framework, DuckDB schema details, report styling.
- Default per-tier limits/epochs (within the intent in 5.5) and default budget value.
- Optional capability probe command.
- Scheduling mechanism for the canary within the section 3 constraint.
