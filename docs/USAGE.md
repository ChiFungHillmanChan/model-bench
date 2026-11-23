# Model Bench 使用指南

私人 benchmark 同 test harness。用嚟：
- 測新出嘅 model；
- 隔一排重跑，睇吓 model 有冇退步；
- 比較唔同 model 嘅分數、速度、成本同穩定度。

全部結果只留喺呢部 Mac（`results.duckdb`、`logs/`、`reports/`）。

---

## 1. 一句講晒點用

```bash
cd ~/Desktop/model-bench
uv run bench plan --models claude-cli-opus-5-5 --tier smoke     # 睇計劃，唔會跑
uv run bench run  --models claude-cli-opus-5-5 --tier smoke --budget 0
uv run bench ingest --rebuild                                    # 將 logs 入 DB
uv run bench dashboard                                           # 生成 reports/dashboard.html
open reports/dashboard.html
```

---

## 2. 基本概念

| 名 | 意思 |
|---|---|
| **model** | `models.yaml` 入面一行。有 id、access、inspect_model、caps、params（例如 thinking 程度）、price |
| **access** | `cli` = 用訂閱嘅 CLI（Claude Code、Codex），唔使逐次俾錢，但會用 quota；`api` = API key，按 token 收錢；`local` = Ollama，喺本機跑，免費 |
| **task** | `tasks/registry.yaml` 入面一個測驗：GPQA Diamond（科學推理）、MMLU-Pro（知識）、IFEval（跟指令）、AIME 2025（數學） |
| **tier** | 跑幾多題：`smoke` = 每個 task 10 題 × 1 次（試水溫）；`standard` = 100 題 × 3 次（AIME 全部 30 題 × 3），正式比較用呢個；`full` = 全部題目 × 3 次 |
| **run** | 一次執行，有 run id（例如 `20261122-opus55-standard`），所有設定記錄喺 `logs/<run_id>/manifest.json` |

**比較原則：** 只可以喺同一個 tier 入面比較。smoke 嘅 10 題信心區間好闊，唔可以用嚟排名。

---

## 3. 每個指令

| 指令 | 做乜 |
|---|---|
| `bench models` | 列出所有 model 同佢哋嘅 access、價錢、capabilities |
| `bench plan --models A,B --tier smoke` | 顯示會跑邊啲 task、幾多題、預計 API 成本、用幾多 quota。唔會跑任何嘢 |
| `bench run --models A,B --tier standard --budget 2` | 真正去跑。`--budget` 係 API 真錢上限（USD），CLI 同 Ollama 唔計錢，所以可以 `--budget 0` |
| `bench run ... --run-id <id>` | 續跑一個中斷咗嘅 run（例如撞到訂閱 usage limit）。已完成嘅部分唔會重跑 |
| `bench run ... --yes` | 跳過「確認要用錢 / 用 quota」嘅問題 |
| `bench ingest --rebuild` | 由 `logs/` 重建 `results.duckdb`。每次跑完都要做 |
| `bench report [--run-id X]` | 寫 Markdown 報告去 `reports/<run_id>.md`：分數、信心區間、錯誤、時間、成本，同所有設定 |
| `bench dashboard` | 寫 `reports/dashboard.html`。有 Overview 頁同每個 model 自己嘅分析頁 |
| `bench compare A B [--tier standard]` | 兩個 model 對比：paired bootstrap 同 McNemar 檢定，話你個差距係真定運氣 |
| `bench repro` | 將 Ollama / DeepSeek 嘅分數同官方公佈嘅分數對比，檢查個 harness 有冇整錯 |
| `bench add-model ...` | 加新 model（見第 4 節） |

---

## 4. 加新 model（例如聽日出 Opus 5.6）

最快嘅方法：用 `--like` 複製一個現有 model 嘅設定，只改 model 名。

```bash
# Claude 新 model，沿用 Opus 5.5 嘅設定（CLI、thinking high、200k context）
uv run bench add-model claude-cli-opus-5-6 --like claude-cli-opus-5-5 \
    --model claude-cli/claude-opus-5-6 --label "Claude Opus 5.6"

# 同一個 model，唔同 thinking 程度（當成另一個 model 嚟比較）
uv run bench add-model claude-cli-opus-5-5-low --like claude-cli-opus-5-5 \
    --model claude-cli/claude-opus-5-5 --effort low --label "Claude Opus 5.5 (low)"

# GPT 新 model（經 Codex CLI，用 ChatGPT 訂閱）
uv run bench add-model codex-cli-gpt-6-3 --like codex-cli-default \
    --model codex-cli/gpt-6.3 --effort high --label "GPT-6.3 (Codex)"

# Ollama 本機 model（先要 `ollama pull qwen3:14b`）
uv run bench add-model ollama-qwen3-14b --model ollama/qwen3:14b \
    --caps text,json,reasoning,ctx:32000 --label "Qwen3 14B (Ollama)"

# API model：一定要俾價錢（USD / 每 1M token）
uv run bench add-model deepseek-v5 --model deepseek/deepseek-v5 \
    --caps text,json,tools,reasoning,ctx:1000000 --price-in 0.5 --price-out 2 \
    --label "DeepSeek V5"
```

規則：
- 冇 `--like` 嘅話，一定要俾 `--caps`。
- access 會由 provider 自動判斷：`claude-cli/` 同 `codex-cli/` 係 cli，`ollama/` 係 local，其他全部係 api。
- api model 一定要有價錢，而且自動設定為唔可以收私人題目（`sealed_ok: false`）。
- id 重複或者設定錯誤，就乜都唔會寫入，`models.yaml` 原封不動。
- 加完之後，先用 `bench plan --models <id> --tier smoke` 睇一次，再跑 smoke。

Model 名要用準確嘅 id（例如 `claude-opus-5-6`），唔好用 `opus` 呢類會變嘅別名。否則日後比較，會唔知自己測緊邊個 model。

---

## 5. 每個供應商點測

| 供應商 | 點樣連 | 收費 | 狀態 |
|---|---|---|---|
| **Claude** | `claude -p`（Claude Code CLI，用你嘅訂閱） | 用 quota | 已經可以用：`claude-cli-opus-5-5`、`claude-cli-sonnet-5-5` |
| **GPT** | `codex exec`（Codex CLI，用 ChatGPT 訂閱） | 用 quota | 已經可以用：`codex-cli-default`（gpt-6.2-astra） |
| **DeepSeek** | API key | 按 token 收錢 | 已設定：`deepseek-flash`、`deepseek-v4-pro` |
| **Ollama** | 本機 | 免費 | 已設定：`ollama-qwen3-8b` |
| **Gemini** | 用 plan 做唔到：Gemini CLI 已經被 Google 封咗個人帳戶（2026-11-22 試過："This client is no longer supported for Gemini Code Assist for individuals"），而 Antigravity 冇 headless 模式（`agy-desktop-helper` 只有 `probe` 同 `serve`，係用嚟 pairing）。唯一方法係 Gemini API（要 `google-genai` 同 key）。API 價錢（2026-11-22，每 1M token，輸入 / 輸出）：Gemini 4 Neon $3 / $15；Hydrogen $1.5 / $7.5；Carbon $2 / $10（12 月 1 號起 $4 / $20） | 收錢 | 未整；用 key 之前要你講明 OK |
| **Grok** | xAI API（要裝 `xai-sdk` 同 `XAI_API_KEY`） | 收錢 | 未整 |

### 重要：API key 嘅規則
`.env` 入面嘅 key 要付費。**除非你逐次講明「OK」，否則唔可以用嚟跑要錢嘅 test。** 上一次講過 OK，唔代表今次都 OK。

預設只用訂閱 CLI（Claude、Codex）同本機 Ollama。

### CLI 測試點樣保持公平
- 每次 call 都喺一個空嘅臨時 folder 入面跑，只會傳低必要嘅環境變數，冇 API key。
- Claude 會關掉 tools、MCP，同埋用戶設定（`--tools "" --strict-mcp-config --setting-sources ""`）。
- Codex 會忽略用戶設定同規則，而且係 read-only。
- 如果 `~/.claude/CLAUDE.md` 存在，或者 `~/.codex/AGENTS.md` 有內容，harness 會拒絕開始，因為佢哋會影響答案。
- 同一時間最多 2 個 CLI process（`CLI_MAX_CONNECTIONS = 2`）。
- 撞到 usage limit 會即刻停低、記錄喺 manifest。之後用 `--run-id` 續跑就得。

---

## 6. 點睇結果

| 指標 | 意思 |
|---|---|
| **score / Bench score** | 答啱嘅百分比。Bench score = 各 task 分數嘅平均 |
| **95% CI** | 信心區間。兩個 model 嘅區間重疊太多，就唔可以話邊個好啲 |
| **errors / error rate** | 冇答到（例如 CLI 出錯、timeout），**唔會當答錯**，直接排除喺分數之外 |
| **pass^k** | 同一題跑 k 次，次次都啱嘅比例（要 standard tier，即係跑 3 次） |
| **agreement** | 同一題幾次嘅答案有幾一致，用嚟量度穩定性 |
| **p50 s** | 每題時間嘅中位數，包括 thinking |
| **cost / q** | 每題真實 API 成本。CLI 顯示 "quota" |
| **tokens / q** | 每題輸出幾多 token（verbosity）。thinking model 會高好多 |

Dashboard：
- 頂部可以揀 tier，同埋揀 model 打開佢自己嘅分析頁。
- 分析頁有排名（#x / N）、文字總結、所有設定、同其他 model 對比嘅圖、歷史走勢，同埋佢答錯嘅題目。

---

## 7. 偵測退步（degradation）

1. 建立 baseline：每個 model 跑一次 `standard`。
2. 之後定期（例如每月，或者懷疑佢變差嘅時候）用同一個 tier 再跑一次。
3. `bench ingest --rebuild && bench dashboard`，睇「Degradation watch」個圖。
4. 新分數跌出上一次嘅信心區間，就值得再跑一次確認。

題目揀法係固定嘅：先排序，再用 seed 20261122 洗牌，然後取頭 N 題。所以每次 standard 都係同一批題目，可以直接比較。

注意：公開 benchmark（例如 AIME 2025）可能已經入咗新 model 嘅訓練數據，分數高唔一定代表能力高。真正量度能力要靠私人、冇公開過嘅題目（Phase 2）。

---

## 8. 常見問題

| 問題 | 解決 |
|---|---|
| 「usage limit」停咗 | 等 quota 重置，然後 `bench run ... --run-id <同一個 id>` 續跑 |
| `isolation check failed` | 暫時移走 `~/.claude/CLAUDE.md`，或者清空 `~/.codex/AGENTS.md` |
| Dashboard 冇新數據 | 未 ingest：`bench ingest --rebuild && bench dashboard` |
| Ollama 好慢 | 正常。8B model 喺 Mac 上面，一題可以要幾分鐘 |
| 想測私人題目 | 放喺 `tasks/private/`（只喺本機，唔入 git），登記喺 `tasks/private/registry.yaml`，`bench` 會自動讀埋 |

---

## 9. 檔案位置

| 位置 | 內容 |
|---|---|
| `models.yaml` | 所有 model 設定 |
| `tasks/registry.yaml` | 所有 task 同 tier 設定 |
| `.env` | API keys（chmod 600，唔會入 git） |
| `logs/<run_id>/` | 每次 run 嘅原始 Inspect logs 同 manifest |
| `results.duckdb` | 所有結果嘅資料庫 |
| `reports/` | Markdown 報告同 `dashboard.html` |
