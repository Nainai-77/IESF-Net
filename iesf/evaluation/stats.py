"""Date-paired inference (Section 3.6.6, Appendix B.1).

* Diebold-Mariano on the squared-loss differential d_t = e_ref^2 - e_F^2
  (positive favours IESF-Net), Bartlett-HAC (Newey-West) long-run variance with
  lag floor(4 (N/100)^(2/9)) (= 5 for N = 393), two-sided normal probabilities.
* Holm step-down adjustment within each market and comparison family.
* Paired circular-block bootstrap (1000 resamples, block length 8): the same
  resampled dates are applied to both forecast series; percentile 95% interval.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as sps


def hac_lag(n: int) -> int:
    return int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def bartlett_lrv(d: np.ndarray, lag: int) -> float:
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    u = d - d.mean()
    v = float(u @ u) / n                                             # γ̂_0
    for j in range(1, lag + 1):
        gamma = float(u[j:] @ u[:-j]) / n
        v += 2.0 * (1.0 - j / (lag + 1.0)) * gamma
    return v


def dm_test(sq_ref, sq_f, lag: int | None = None) -> dict:
    """Eq. (25): DM = d̄ / sqrt(v̂ / N)."""
    d = np.asarray(sq_ref, dtype=np.float64) - np.asarray(sq_f, dtype=np.float64)
    n = len(d)
    q = hac_lag(n) if lag is None else int(lag)
    v = bartlett_lrv(d, q)
    stat = float(d.mean() / math.sqrt(v / n)) if v > 0 else float("nan")
    p = float(2.0 * sps.norm.sf(abs(stat))) if np.isfinite(stat) else float("nan")
    return {"DM": stat, "p": p, "lag": q, "mean_diff": float(d.mean()), "N": n}


def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, dtype=np.float64)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def circular_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    n_blocks = int(math.ceil(n / block))
    starts = rng.integers(0, n, size=n_blocks)
    idx = (starts[:, None] + np.arange(block)[None, :]) % n
    return idx.ravel()[:n]


def bootstrap_reduction(sq_ref, sq_f, n_boot=1000, block=8, seed=0, labels=None, select=None) -> dict:
    """Percentile interval of r = 100 (1 - RMSE_F / RMSE_ref) under paired circular-block resampling.

    ``labels`` (e.g. change-size groups) stay attached to the residual pairs during
    resampling; ``select`` restricts the statistic to resampled dates carrying that label.
    """
    sq_ref, sq_f = np.asarray(sq_ref, np.float64), np.asarray(sq_f, np.float64)
    rng = np.random.default_rng(seed)
    n = len(sq_ref)
    keep_all = np.ones(n, dtype=bool) if labels is None else (np.asarray(labels) == select)
    point = 100.0 * (1.0 - math.sqrt(sq_f[keep_all].mean() / sq_ref[keep_all].mean()))
    draws = np.empty(n_boot)
    for b in range(n_boot):
        ix = circular_block_indices(n, block, rng)
        if labels is not None:
            ix = ix[keep_all[ix]]
        draws[b] = 100.0 * (1.0 - math.sqrt(sq_f[ix].mean() / sq_ref[ix].mean())) if len(ix) else np.nan
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return {"reduction": point, "ci_low": float(lo), "ci_high": float(hi), "N": int(keep_all.sum())}
