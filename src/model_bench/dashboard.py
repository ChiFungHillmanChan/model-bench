"""Self-contained HTML dashboard over results.duckdb (local file; private data never leaves the Mac)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np

from model_bench.report import THINKING_KEYS, _headlines, _obs
from model_bench.stats import clustered_mean, epoch_agreement, pass_hat_k

TEMPLATE = Path(__file__).with_name("dashboard_template.html")


def _pct(v: float | None) -> float | None:
    return None if v is None else round(v * 100, 1)


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def build_data(db: Path) -> dict:
    con = duckdb.connect(str(db), read_only=True)
    try:
        return _build(con)
    finally:
        con.close()


def _build(con) -> dict:
    runs_out, results, questions = [], [], []
    run_rows = con.execute(
        "select run_id, created, tier, status, spend_usd, manifest_json from runs order by created desc"
    ).fetchall()
    for run_id, created, tier, status, spend, raw in run_rows:
        try:
            manifest = json.loads(raw or "{}")
        except json.JSONDecodeError:
            manifest = {}
        fields = {t["name"]: t.get("category", "") for t in manifest.get("tasks") or []}
        thinking = {}
        for e in manifest.get("models") or []:
            params = e.get("params") or {}
            thinking[e["id"]] = next((str(params[k]) for k in THINKING_KEYS if k in params), "default")
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
    return {
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "models": models, "runs": runs_out, "results": results, "questions": questions,
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


def write_dashboard(db: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(build_data(db)))
    return out
