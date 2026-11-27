"""Loading forecast series and aligning them on common target dates."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..experiments.runner import run_dir


def seeds_available(cfg: dict, market: str, model: str) -> list[str]:
    root = run_dir(cfg, "forecasts", market, model)
    if not root.exists():
        return []
    return sorted(p.name[4:] for p in root.iterdir() if p.is_dir() and p.name.startswith("seed"))


def load_series(cfg: dict, market: str, model: str, split: str = "test", seed=None) -> pd.DataFrame | None:
    """Forecast table of one run (``seed=None`` -> designated seed, or the deterministic run)."""
    seeds = seeds_available(cfg, market, model)
    if not seeds:
        return None
    if seed is None:
        want = str(cfg["seeds"]["designated"])
        seed = want if want in seeds else ("deterministic" if "deterministic" in seeds else seeds[0])
    path = run_dir(cfg, "forecasts", market, model, seed) / f"{split}_predictions.csv"
    return pd.read_csv(path) if path.exists() else None


def load_runs(cfg: dict, market: str, model: str, split: str = "test") -> list[pd.DataFrame]:
    out = []
    for s in seeds_available(cfg, market, model):
        df = load_series(cfg, market, model, split, s)
        if df is not None:
            out.append(df)
    return out


def squared_losses(cfg: dict, market: str, model: str, split: str = "test", average_runs: bool = False) -> pd.Series | None:
    """Date-indexed squared losses.  ``average_runs`` averages losses within date across runs
    (repeated numerical benchmarks); otherwise the designated series is used."""
    if average_runs:
        runs = load_runs(cfg, market, model, split)
        if not runs:
            return None
        mats = [r.set_index("target_index").eval("(prediction - actual) ** 2") for r in runs]
        return pd.concat(mats, axis=1).mean(axis=1)
    df = load_series(cfg, market, model, split)
    if df is None:
        return None
    return df.set_index("target_index").eval("(prediction - actual) ** 2")


def common_index(*series) -> np.ndarray:
    idx = None
    for s in series:
        idx = s.index if idx is None else idx.intersection(s.index)
    return np.asarray(sorted(idx))
