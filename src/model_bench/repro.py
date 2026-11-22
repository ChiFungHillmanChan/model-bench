"""Reproduction check: our score vs published score, bare models only."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import duckdb
import yaml
from pydantic import BaseModel

from model_bench.report import _headlines, _obs, fmt
from model_bench.stats import Estimate, clustered_mean

TIER_PREFERENCE = ("full", "standard", "smoke")


class Reference(BaseModel):
    model_id: str
    task: str
    score: float  # fraction 0-1
    source_url: str
    source_kind: Literal["provider", "third_party"]
    settings: str
    retrieved: date
    explanation: str | None = None


@dataclass(frozen=True)
class ReproRow:
    model_id: str
    task: str
    ours: Estimate
    published: float
    verdict: Literal["match", "gap", "no_data"]
    explanation: str | None
    run_id: str | None = None
    tier: str | None = None
    scorer: str | None = None
    source_url: str | None = None


def load_refs(path: Path) -> list[Reference]:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return [Reference.model_validate(r) for r in raw.get("references") or []]


def _pick(refs: list[Reference]) -> dict[tuple[str, str], Reference]:
    out: dict[tuple[str, str], Reference] = {}
    for r in refs:
        key = (r.model_id, r.task)
        if key not in out or (r.source_kind == "provider" and out[key].source_kind != "provider"):
            out[key] = r
    return out


def _access(con, model_id: str) -> str | None:
    row = con.execute("select any_value(access) from samples where model_id=?", [model_id]).fetchone()
    return row[0] if row else None


def _latest(con, model_id: str, task: str) -> tuple[str, str] | None:
    rows = con.execute(
        "select r.run_id, r.tier from runs r join samples s using (run_id)"
        " where s.model_id=? and s.task=? group by r.run_id, r.tier, r.created order by r.created desc",
        [model_id, task],
    ).fetchall()
    for tier in TIER_PREFERENCE:
        for run_id, t in rows:
            if t == tier:
                return run_id, t
    return None


def _check(con, refs: Path) -> tuple[list[ReproRow], list[str]]:
    rows, excluded = [], []
    for (model_id, task), ref in sorted(_pick(load_refs(refs)).items()):
        if _access(con, model_id) == "cli":
            excluded.append(model_id)
            continue
        found = _latest(con, model_id, task)
        if found is None:
            rows.append(ReproRow(model_id, task, Estimate(None, None, None, None, 0, 0), ref.score, "no_data",
                                 ref.explanation, source_url=ref.source_url))
            continue
        run_id, tier = found
        scorer = _headlines(con, run_id).get(task)
        est = clustered_mean(_obs(con, run_id, task, model_id, scorer)) if scorer else clustered_mean([])
        if est.ci_low is None:
            verdict = "no_data"
        else:
            verdict = "match" if est.ci_low <= ref.score <= est.ci_high else "gap"
        rows.append(ReproRow(model_id, task, est, ref.score, verdict, ref.explanation, run_id, tier, scorer,
                             ref.source_url))
    return rows, sorted(set(excluded))


def repro(db: Path, refs: Path) -> list[ReproRow]:
    con = duckdb.connect(str(db), read_only=True)
    try:
        return _check(con, refs)[0]
    finally:
        con.close()


def format_repro(db: Path, refs: Path) -> str:
    con = duckdb.connect(str(db), read_only=True)
    try:
        rows, excluded = _check(con, refs)
    finally:
        con.close()
    out = ["# Reproduction check", "", "| model | task | ours [95% CI] | published | verdict | run (tier, scorer) | note |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        note = r.explanation or ("OPEN: needs explanation" if r.verdict == "gap" else "")
        if r.tier == "smoke":
            note = (note + " " if note else "") + "smoke tier only: too few items to trust"
        where = f"{r.run_id} ({r.tier}, {r.scorer})" if r.run_id else "not run"
        out.append(f"| {r.model_id} | {r.task} | {fmt(r.ours)} | {r.published:.3f} | {r.verdict} | {where} | {note} |")
    for m in excluded:
        out.append(f"\n{m}: excluded (via CLI, not a bare model)")
    return "\n".join(out) + "\n"
