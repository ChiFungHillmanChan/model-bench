"""Load Inspect .eval logs plus run manifests into DuckDB (derived, rebuildable)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
from inspect_ai.log import read_eval_log

from model_bench.runner import Manifest, usage_cost

SCHEMA = """
create table if not exists runs(
  run_id varchar primary key, created timestamptz, tier varchar, git_sha varchar, git_dirty boolean,
  inspect_ai varchar, inspect_evals varchar, status varchar, spend_usd double, manifest_json varchar);
create table if not exists samples(
  run_id varchar, task varchar, model_id varchar, sample_id varchar, epoch integer,
  reported_model varchar, access varchar, input_tokens bigint, output_tokens bigint,
  total_cost double, total_time double, error varchar,
  input_text varchar, target varchar, answer varchar,
  primary key(run_id, task, model_id, sample_id, epoch));
create table if not exists scores(
  run_id varchar, task varchar, model_id varchar, sample_id varchar, epoch integer,
  scorer varchar, value_num double, value_raw varchar,
  primary key(run_id, task, model_id, sample_id, epoch, scorer));
"""

# Columns added after the first schema; applied to existing databases on ingest.
MIGRATIONS = [
    "alter table samples add column if not exists input_text varchar",
    "alter table samples add column if not exists target varchar",
    "alter table samples add column if not exists answer varchar",
]

_LETTERS = {"C": 1.0, "I": 0.0, "P": 0.5, "N": 0.0}


def value_num(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return _LETTERS.get(value)
    return None


def _text(value: Any) -> str | None:
    """Question / target as plain text: strings pass through, message lists are joined."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(getattr(v, "text", None) or str(v) for v in value)
    return str(value)


def _score_rows(key: tuple, scores: dict) -> list[tuple]:
    rows = []
    for name, score in (scores or {}).items():
        v = score.value
        rows.append((*key, name, value_num(v), json.dumps(v, default=str)))
        if isinstance(v, dict):  # e.g. IFEval: one row per sub-metric
            for sub, sv in v.items():
                rows.append((*key, f"{name}.{sub}", value_num(sv), json.dumps(sv, default=str)))
    return rows


def ingest(log_root: Path, db_path: Path, rebuild: bool = False) -> int:
    if rebuild and Path(db_path).exists():
        Path(db_path).unlink()
    con = duckdb.connect(str(db_path))
    con.execute(SCHEMA)
    for stmt in MIGRATIONS:
        con.execute(stmt)
    n = 0
    for manifest_file in sorted(Path(log_root).glob("*/manifest.json")):
        m = Manifest.model_validate_json(manifest_file.read_text())
        models = {e.id: e for e in m.models}
        con.execute(
            "insert or replace into runs values (?,?,?,?,?,?,?,?,?,?)",
            [m.run_id, m.created, m.tier, m.git_sha, m.git_dirty, m.inspect_ai, m.inspect_evals,
             m.status, m.spend_usd, m.model_dump_json()],
        )
        for chunk_dir in sorted(p for p in manifest_file.parent.iterdir() if p.is_dir()):
            model_id, _, task = chunk_dir.name.partition("__")
            model = models.get(model_id)
            if model is None:
                continue
            for log_file in sorted(chunk_dir.glob("*.eval")):
                log = read_eval_log(str(log_file))
                for s in log.samples or []:
                    key = (m.run_id, task, model_id, str(s.id), s.epoch)
                    usages = list((s.model_usage or {}).values())
                    con.execute(
                        "insert or replace into samples (run_id, task, model_id, sample_id, epoch,"
                        " reported_model, access, input_tokens, output_tokens, total_cost, total_time, error,"
                        " input_text, target, answer) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        [*key, s.output.model if s.output else None, model.access,
                         sum(u.input_tokens for u in usages), sum(u.output_tokens for u in usages),
                         sum(usage_cost(u, model) for u in usages), s.total_time,
                         s.error.message if s.error else None,
                         _text(s.input), _text(s.target), s.output.completion if s.output else None],
                    )
                    con.execute(
                        "delete from scores where run_id=? and task=? and model_id=? and sample_id=? and epoch=?",
                        list(key),
                    )
                    if s.error is None:
                        rows = _score_rows(key, s.scores)
                        if rows:
                            con.executemany("insert or replace into scores values (?,?,?,?,?,?,?,?)", rows)
                    n += 1
    con.close()
    return n
