"""Markdown report and paired comparison from results.duckdb."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np

import json
from datetime import datetime

from model_bench.stats import (
    Estimate,
    clustered_mean,
    epoch_agreement,
    majority_correct,
    mcnemar,
    paired_diff,
    pass_hat_k,
)

THINKING_KEYS = ("effort", "reasoning_effort", "reasoning_tokens")
CLI_BINARIES = {"claude-cli": "claude", "codex-cli": "codex", "gemini-cli": "gemini"}
SHUFFLE_SEED = 20261122


def fmt(est: Estimate) -> str:
    if est.mean is None:
        return "n/a"
    if est.ci_low is None:
        return f"{est.mean:.3f} [n/a]"
    return f"{est.mean:.3f} [{est.ci_low:.3f}, {est.ci_high:.3f}]"


def _num(v: float | None, digits: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{digits}f}"


def _headline(scorers: list[str]) -> str:
    plain = sorted(s for s in scorers if "." not in s)
    if plain:
        return plain[0]
    strict = sorted(s for s in scorers if s.endswith("prompt_level_strict"))
    if strict:
        return strict[0]
    score = sorted(s for s in scorers if s.endswith(".score"))  # e.g. HLE {score, confidence}
    return score[0] if score else sorted(scorers)[0]


def _headlines(con, run_id: str) -> dict[str, str]:
    rows = con.execute(
        "select task, list(distinct scorer) from scores where run_id=? and value_num is not null group by task",
        [run_id],
    ).fetchall()
    return {task: _headline(scorers) for task, scorers in rows}


def _obs(con, run_id: str, task: str, model: str, scorer: str) -> list[tuple[str, int, float]]:
    return con.execute(
        "select sample_id, epoch, value_num from scores where run_id=? and task=? and model_id=? and scorer=?"
        " and value_num is not null",
        [run_id, task, model, scorer],
    ).fetchall()


def _latest_run(con) -> str | None:
    row = con.execute("select run_id from runs order by created desc limit 1").fetchone()
    return row[0] if row else None


def report(db: Path, run_id: str | None = None) -> str:
    con = duckdb.connect(str(db), read_only=True)
    run_id = run_id or _latest_run(con)
    if run_id is None:
        return "no runs ingested"
    tier, sha, dirty, ia, ie, status, spend, created = con.execute(
        "select tier, git_sha, git_dirty, inspect_ai, inspect_evals, status, spend_usd, created"
        " from runs where run_id=?",
        [run_id],
    ).fetchone()
    heads = _headlines(con, run_id)
    out = [
        f"# Run {run_id}",
        "",
        f"tier: {tier} | status: {status} | created: {created} | git: {sha[:7]} | "
        f"inspect-ai {ia} / inspect-evals {ie} | API spend: ${spend:.4f}",
    ]
    if dirty:
        out.append("")
        out.append("WARNING: run from a dirty working tree")
    out += _settings(con, run_id)
    n_err = con.execute("select count(*) from samples where run_id=? and error is not null", [run_id]).fetchone()[0]
    out += ["", f"errors: {n_err}", "", "## Scores", ""]
    out.append("| task | model | access | score [95% CI] | n | errors | pass^k | agreement | cost_usd | p50_s | p95_s | out_tokens |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    pairs = con.execute(
        "select task, model_id, any_value(access), count(*) filter (where error is not null),"
        " sum(total_cost), list(total_time), sum(output_tokens), max(epoch)"
        " from samples where run_id=? group by task, model_id order by task, model_id",
        [run_id],
    ).fetchall()
    per_model: dict[str, dict] = defaultdict(lambda: {"means": [], "cost": 0.0, "correct": 0.0, "access": ""})
    for task, model, access, errors, cost, times, out_tokens, epochs in pairs:
        scorer = heads.get(task)
        obs = _obs(con, run_id, task, model, scorer) if scorer else []
        est = clustered_mean(obs)
        times = [t for t in times if t is not None]
        p50, p95 = (np.percentile(times, [50, 95]) if times else (None, None))
        label = f"{model} (via CLI)" if access == "cli" else model
        cost_cell = "quota" if access == "cli" else f"{cost or 0:.4f}"
        out.append(
            f"| {task} | {label} | {access} | {fmt(est)} | {est.n_items} | {errors} | "
            f"{_num(pass_hat_k(obs, epochs))} | {_num(epoch_agreement(obs))} | {cost_cell} | {_num(p50, 1)} | {_num(p95, 1)} | {out_tokens} |"
        )
        pm = per_model[model]
        pm["access"] = access
        if est.mean is not None:
            pm["means"].append(est.mean)
        pm["cost"] += cost or 0.0
        pm["correct"] += sum(v for _, _, v in obs if v == 1.0)
    out += ["", "## Quality vs cost", "", "| model | mean score | total cost | cost per correct | frontier |", "|---|---|---|---|---|"]
    priced = {m: d for m, d in per_model.items() if d["access"] != "cli" and d["means"]}
    for model, d in sorted(per_model.items()):
        quality = sum(d["means"]) / len(d["means"]) if d["means"] else None
        if d["access"] == "cli":
            out.append(f"| {model} | {_num(quality)} | quota | quota | |")
            continue
        per_correct = d["cost"] / d["correct"] if d["correct"] else None
        frontier = model in priced and not any(
            o != model
            and sum(od["means"]) / len(od["means"]) >= quality
            and od["cost"] <= d["cost"]
            and (sum(od["means"]) / len(od["means"]) > quality or od["cost"] < d["cost"])
            for o, od in priced.items()
        )
        out.append(f"| {model} | {_num(quality)} | {d['cost']:.4f} | {_num(per_correct, 4)} | {'frontier' if frontier else ''} |")
    con.close()
    return "\n".join(out) + "\n"


def _settings(con, run_id: str) -> list[str]:
    """Everything needed to reproduce or trust the run: times, models, thinking, params, items."""
    (raw,) = con.execute("select manifest_json from runs where run_id=?", [run_id]).fetchone()
    try:
        m = json.loads(raw or "{}")
    except json.JSONDecodeError:
        m = {}
    if not m.get("models"):
        return ["", "## Run settings", "", "settings not recorded"]
    started, finished = m.get("created"), m.get("finished")
    duration = "n/a"
    if started and finished:
        secs = (datetime.fromisoformat(finished.replace("Z", "+00:00"))
                - datetime.fromisoformat(started.replace("Z", "+00:00"))).total_seconds()
        duration = f"{secs / 60:.1f} min"
    out = ["", "## Run settings", "",
           f"started: {started} | finished: {finished or 'n/a'} | duration: {duration} | "
           f"budget: ${m.get('budget_usd', 0):.2f} | API spend: ${m.get('spend_usd', 0):.4f}", "",
           "| model | access | inspect model | reported model | thinking | other params | CLI version | price |",
           "|---|---|---|---|---|---|---|---|"]
    clis = m.get("cli_versions") or {}
    for e in m["models"]:
        params = e.get("params") or {}
        thinking = next((str(params[k]) for k in THINKING_KEYS if k in params), "default")
        other = ", ".join(f"{k}={v}" for k, v in params.items() if k not in THINKING_KEYS)
        reported = [r for (r,) in con.execute(
            "select distinct reported_model from samples where run_id=? and model_id=? and reported_model is not null"
            " order by 1", [run_id, e["id"]]).fetchall()]
        cli = clis.get(CLI_BINARIES.get(e["inspect_model"].split("/", 1)[0], ""), "")
        price = e.get("price")
        if e["access"] == "cli":
            price_cell = "subscription quota"
        elif price:
            price_cell = f"{price['input']:g}/{price['output']:g} per M ({price['as_of']})"
        else:
            price_cell = "free (local)"
        out.append(f"| {e['id']} | {e['access']} | {e['inspect_model']} | {', '.join(reported) or 'n/a'} | "
                   f"{thinking} | {other} | {cli} | {price_cell} |")
    out += ["", "| task | tier | items | epochs | seed | version |", "|---|---|---|---|---|---|"]
    for t in m.get("tasks") or []:
        spec = (t.get("tiers") or {}).get(m.get("tier"), {})
        limit = spec.get("limit")
        out.append(f"| {t['name']} | {m.get('tier')} | {'all' if limit is None else limit} | "
                   f"{spec.get('epochs', 'n/a')} | {SHUFFLE_SEED} | {t.get('version')} |")
    return out


def _runs_with(con, model: str, tier: str | None) -> list[str]:
    sql = ("select s.run_id from samples s join runs r using (run_id) where s.model_id=?"
           + (" and r.tier=?" if tier else "") + " group by s.run_id, r.created order by r.created desc")
    return [r[0] for r in con.execute(sql, [model, tier] if tier else [model]).fetchall()]


def compare(db: Path, a: str, b: str, tier: str | None = None) -> str:
    con = duckdb.connect(str(db), read_only=True)
    runs_a, runs_b = _runs_with(con, a, tier), _runs_with(con, b, tier)
    for name, runs in ((a, runs_a), (b, runs_b)):
        if not runs:
            con.close()
            return f"no runs found for {name}\n"
    shared = [r for r in runs_a if r in runs_b]
    out = [f"# Compare {a} vs {b}", ""]
    if shared:
        run_a = run_b = shared[0]
        out.append(f"run: {run_a} (both models, same items)")
    else:
        run_a, run_b = runs_a[0], runs_b[0]
        out.append(f"WARNING: no shared run; {a} from {run_a}, {b} from {run_b}; items may differ")
    out += ["", f"| task | {a} | {b} | diff ({a} - {b}) [95% CI] | n shared | McNemar p |", "|---|---|---|---|---|---|"]
    heads_a, heads_b = _headlines(con, run_a), _headlines(con, run_b)
    for task in sorted(set(heads_a) & set(heads_b)):
        oa = _obs(con, run_a, task, a, heads_a[task])
        ob = _obs(con, run_b, task, b, heads_b[task])
        if not oa or not ob:
            continue
        d = paired_diff(oa, ob)
        _, _, p = mcnemar(majority_correct(oa), majority_correct(ob))
        out.append(f"| {task} | {fmt(clustered_mean(oa))} | {fmt(clustered_mean(ob))} | {fmt(d)} | {d.n_items} | {_num(p)} |")
    con.close()
    return "\n".join(out) + "\n"
