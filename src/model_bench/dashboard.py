"""Self-contained HTML dashboard over results.duckdb (local file; private data never leaves the Mac)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np

from model_bench.registry import RegistryError, load_models
from model_bench.report import CLI_BINARIES, THINKING_KEYS, _headlines, _obs
from model_bench.stats import clustered_mean, epoch_agreement, pass_hat_k

TEMPLATE = Path(__file__).with_name("dashboard_template.html")

# Display titles and one-line descriptions for known tasks (unknown tasks fall back to their name).
TASK_INFO = {
    "gpqa_diamond": ("GPQA Diamond", "198 graduate-level biology, physics and chemistry multiple-choice questions "
                                     "written so that search does not help."),
    "mmlu_pro": ("MMLU-Pro", "12,032 ten-option multiple-choice questions across 14 academic and professional subjects."),
    "ifeval": ("IFEval", "541 prompts with checkable instructions (length, keywords, format, case); "
                         "prompt-level strict accuracy."),
    "aime2025": ("AIME 2025", "30 American Invitational Mathematics Examination 2025 problems; integer answers."),
}


def _pct(v: float | None) -> float | None:
    return None if v is None else round(v * 100, 1)


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def build_data(db: Path, models_path: Path | None = None) -> dict:
    labels = {}
    if models_path and Path(models_path).exists():
        try:
            labels = {k: m.label for k, m in load_models(models_path).items() if m.label}
        except RegistryError:
            labels = {}
    con = duckdb.connect(str(db), read_only=True)
    try:
        return _build(con, labels)
    finally:
        con.close()


def _build(con, labels: dict[str, str]) -> dict:
    runs_out, results, questions = [], [], []
    model_info: dict[str, dict] = {}
    tasks_info: dict[str, dict] = {}
    run_rows = con.execute(
        "select run_id, created, tier, status, spend_usd, manifest_json from runs order by created desc"
    ).fetchall()
    for run_id, created, tier, status, spend, raw in run_rows:
        try:
            manifest = json.loads(raw or "{}")
        except json.JSONDecodeError:
            manifest = {}
        fields = {t["name"]: t.get("category", "") for t in manifest.get("tasks") or []}
        for name, field in fields.items():
            title, about = TASK_INFO.get(name, (name, ""))
            tasks_info.setdefault(name, {"title": title, "field": field, "about": about})
        thinking = {}
        for e in manifest.get("models") or []:
            params = e.get("params") or {}
            thinking[e["id"]] = next((str(params[k]) for k in THINKING_KEYS if k in params), "default")
            if e["id"] not in model_info:  # runs are newest first: first sighting = latest settings
                reported = [r for (r,) in con.execute(
                    "select distinct reported_model from samples where run_id=? and model_id=?"
                    " and reported_model is not null and reported_model <> ? order by 1",
                    [run_id, e["id"], e.get("inspect_model", "")]).fetchall()]  # errored samples echo the inspect name
                binary = CLI_BINARIES.get(e.get("inspect_model", "").split("/", 1)[0], "")
                ctx = next((c.split(":", 1)[1] for c in e.get("caps") or [] if c.startswith("ctx:")), None)
                model_info[e["id"]] = {
                    "label": labels.get(e["id"], e["id"]), "access": e.get("access"),
                    "inspect_model": e.get("inspect_model"), "reported_model": ", ".join(reported) or None,
                    "thinking": thinking[e["id"]],
                    "params": {k: v for k, v in params.items() if k not in THINKING_KEYS},
                    "cli_version": (manifest.get("cli_versions") or {}).get(binary) if binary else None,
                    "price": e.get("price"), "ctx": int(ctx) if ctx else None,
                    "caps": sorted(c for c in e.get("caps") or [] if not c.startswith("ctx:")),
                    "sealed_ok": e.get("sealed_ok"), "last_run": run_id, "last_date": None, "runs": 0,
                }
        start, end = _parse_time(manifest.get("created")), _parse_time(manifest.get("finished"))
        duration = round((end - start).total_seconds() / 60, 1) if start and end else None
        date = created.isoformat() if created else manifest.get("created")

        n_samples, n_errors = con.execute(
            "select count(*), count(*) filter (where error is not null) from samples where run_id=?", [run_id]
        ).fetchone()
        run_models = [m for (m,) in con.execute(
            "select distinct model_id from samples where run_id=? order by 1", [run_id]).fetchall()]
        runs_out.append({
            "run_id": run_id, "date": date, "tier": tier, "status": status, "spend_usd": spend,
            "duration_min": duration, "models": run_models, "samples": n_samples,
            "error_rate_pct": round(100 * n_errors / n_samples, 1) if n_samples else None,
        })

        heads = _headlines(con, run_id)
        pairs = con.execute(
            "select task, model_id, any_value(access), count(*), count(*) filter (where error is not null),"
            " sum(total_cost), list(total_time), max(epoch), sum(output_tokens)"
            " from samples where run_id=? group by task, model_id order by task, model_id",
            [run_id],
        ).fetchall()
        for task, model, access, total, errors, cost, times, epochs, out_tokens in pairs:
            scorer = heads.get(task)
            obs = _obs(con, run_id, task, model, scorer) if scorer else []
            est = clustered_mean(obs)
            times = [t for t in times if t is not None]
            results.append({
                "run_id": run_id, "date": date, "tier": tier, "task": task, "field": fields.get(task, ""),
                "model": model, "access": access, "thinking": thinking.get(model, "default"),
                "scorer": scorer,
                "score_pct": _pct(est.mean), "ci_low_pct": _pct(est.ci_low), "ci_high_pct": _pct(est.ci_high),
                "items": est.n_items, "correct": int(sum(1 for _, _, v in obs if v == 1.0)),
                "answered": len(obs), "samples": total, "errors": errors,
                "error_rate_pct": round(100 * errors / total, 1) if total else None,
                "cost_usd": None if access == "cli" else round(cost or 0.0, 6),
                "p50_s": round(float(np.percentile(times, 50)), 1) if times else None,
                "pass_k_pct": _pct(pass_hat_k(obs, epochs)),
                "agreement_pct": _pct(epoch_agreement(obs)),
                "out_tokens": out_tokens, "epochs": epochs,
                "cost_per_q": None if access == "cli" else round((cost or 0.0) / total, 6) if total else None,
                "out_tokens_per_q": round(out_tokens / total, 1) if total and out_tokens is not None else None,
            })

        rows = _question_rows(con, run_id, heads)
        by_q: dict[tuple, dict] = {}
        for task, sid, model, text, target, answer, errored, value in rows:
            q = by_q.setdefault((task, sid), {
                "run_id": run_id, "date": date, "task": task, "field": fields.get(task, ""),
                "sample_id": sid, "input": (text or "")[:1500], "target": (target or "")[:300], "answers": {},
            })
            if errored and value is None:
                mark = "error"
            elif value is None:
                mark = "unscored"
            else:
                mark = "correct" if value >= 0.5 else "wrong"
            q["answers"][model] = {"answer": (answer or "")[:800], "mark": mark,
                                   "score": None if value is None else round(value, 3)}
        questions.extend(by_q.values())

    models = [m for (m,) in con.execute("select distinct model_id from samples order by 1").fetchall()]
    for run in runs_out:
        for m in run["models"]:
            info = model_info.setdefault(m, {"label": labels.get(m, m), "last_run": run["run_id"], "runs": 0})
            info["runs"] += 1
            if info.get("last_run") == run["run_id"]:
                info["last_date"] = run["date"]
    return {
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "models": models, "model_info": model_info, "tasks": tasks_info,
        "runs": runs_out, "results": results, "questions": questions,
    }


def _question_rows(con, run_id: str, heads: dict[str, str]) -> list[tuple]:
    rows = []
    for task, scorer in sorted(heads.items()):
        rows += con.execute(
            "select s.task, s.sample_id, s.model_id, any_value(s.input_text), any_value(s.target),"
            " any_value(s.answer) filter (where s.epoch = 1), bool_or(s.error is not null), avg(sc.value_num)"
            " from samples s left join scores sc on sc.run_id=s.run_id and sc.task=s.task"
            " and sc.model_id=s.model_id and sc.sample_id=s.sample_id and sc.epoch=s.epoch and sc.scorer=?"
            " where s.run_id=? and s.task=? group by s.task, s.sample_id, s.model_id order by s.sample_id",
            [scorer, run_id, task],
        ).fetchall()
    unscored_tasks = [t for (t,) in con.execute(
        "select distinct task from samples where run_id=?", [run_id]).fetchall() if t not in heads]
    for task in unscored_tasks:  # every sample errored: still list the questions
        rows += con.execute(
            "select task, sample_id, model_id, any_value(input_text), any_value(target),"
            " any_value(answer) filter (where epoch = 1), bool_or(error is not null), null"
            " from samples where run_id=? and task=? group by task, sample_id, model_id order by sample_id",
            [run_id, task],
        ).fetchall()
    return rows


def render(data: dict) -> str:
    # < keeps any "</script>" inside question text from closing the data block
    payload = json.dumps(data, default=str).replace("<", "\\u003c")
    return TEMPLATE.read_text().replace("__BENCH_DATA__", payload)


def write_dashboard(db: Path, out: Path, models_path: Path | None = None) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(build_data(db, models_path)))
    return out
