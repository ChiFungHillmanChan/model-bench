"""Statistics over long-format observations (item_id, epoch, value). Pure functions."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

Obs = list[tuple[str, int, float]]
Z95 = 1.96
SEED = 20261122


@dataclass(frozen=True)
class Estimate:
    mean: float | None
    se: float | None
    ci_low: float | None
    ci_high: float | None
    n_items: int
    n_obs: int


def _by_item(obs: Obs) -> dict[str, list[tuple[int, float]]]:
    out: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for item, epoch, value in obs:
        out[str(item)].append((epoch, float(value)))
    return out


def item_means(obs: Obs) -> dict[str, float]:
    return {k: sum(v for _, v in vs) / len(vs) for k, vs in _by_item(obs).items()}


def clustered_mean(obs: Obs) -> Estimate:
    """Mean with standard error clustered by item (epochs of one item are not independent)."""
    means = np.array(list(item_means(obs).values()))
    n = len(means)
    if n == 0:
        return Estimate(None, None, None, None, 0, 0)
    mean = float(means.mean())
    if n < 2:
        return Estimate(mean, None, None, None, n, len(obs))
    se = float(means.std(ddof=1) / math.sqrt(n))
    if se == 0:  # a zero-width interval would claim certainty we do not have
        return Estimate(mean, se, None, None, n, len(obs))
    return Estimate(mean, se, max(0.0, mean - Z95 * se), min(1.0, mean + Z95 * se), n, len(obs))


def paired_diff(a: Obs, b: Obs, n_boot: int = 10000, seed: int = SEED) -> Estimate:
    """a - b over shared items, bootstrap CI by resampling items."""
    ma, mb = item_means(a), item_means(b)
    shared = sorted(set(ma) & set(mb))
    diffs = np.array([ma[i] - mb[i] for i in shared])
    n = len(diffs)
    n_obs = sum(1 for o in a if str(o[0]) in shared) + sum(1 for o in b if str(o[0]) in shared)
    if n == 0:
        return Estimate(None, None, None, None, 0, 0)
    mean = float(diffs.mean())
    if n < 2:
        return Estimate(mean, None, None, None, n, n_obs)
    rng = np.random.default_rng(seed)
    boots = diffs[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    se = float(boots.std(ddof=1))
    if se == 0:
        return Estimate(mean, se, None, None, n, n_obs)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return Estimate(mean, se, max(-1.0, float(lo)), min(1.0, float(hi)), n, n_obs)


def mcnemar(a: dict[str, bool], b: dict[str, bool]) -> tuple[int, int, float | None]:
    """(b_only, a_only, exact two-sided p) over shared items."""
    shared = set(a) & set(b)
    a_only = sum(1 for i in shared if a[i] and not b[i])
    b_only = sum(1 for i in shared if b[i] and not a[i])
    n = a_only + b_only
    if n == 0:
        return b_only, a_only, None
    k = min(a_only, b_only)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return b_only, a_only, min(1.0, p)


def majority_correct(obs: Obs) -> dict[str, bool]:
    return {k: v > 0.5 for k, v in item_means(obs).items()}


def pass_hat_k(obs: Obs, k: int) -> float | None:
    """Share of items whose first k epochs all scored 1.0 (tau-bench style reliability)."""
    if k < 2:
        return None
    items = _by_item(obs)
    if not items or any(len(vs) < k for vs in items.values()):
        return None
    ok = sum(1 for vs in items.values() if all(v == 1.0 for _, v in sorted(vs)[:k]))
    return ok / len(items)
