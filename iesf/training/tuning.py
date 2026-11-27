"""Validation-only hyperparameter selection with Optuna TPE (Section 3.6.5, Appendix A.3, Table A4).

TPESampler(seed 4401, n_startup_trials 12, n_ei_candidates 24, multivariate /
group / constant_liar disabled, prior weight 1, magic clipping on, endpoint
adjustment off), NopPruner, sequential search (n_jobs = 1).  Identical
configurations reuse a cached objective; non-finite or failed trials receive
an infinite objective.  The selected configuration has the lowest validation
RMSE; ties within 1e-6 are resolved by fewer parameters, shorter total
look-back and earliest trial.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..utils import ensure_dir, get_logger, save_json

LOG = get_logger(__name__)


def make_sampler(seed: int, n_startup: int, n_ei: int):
    import optuna

    return optuna.samplers.TPESampler(
        seed=int(seed), n_startup_trials=int(n_startup), n_ei_candidates=int(n_ei),
        multivariate=False, group=False, constant_liar=False, prior_weight=1.0,
        consider_magic_clip=True, consider_endpoints=False,
    )


def select_best(trials: pd.DataFrame, tol: float = 1e-6) -> pd.Series:
    ok = trials[np.isfinite(trials["value"])]
    if ok.empty:
        raise RuntimeError("no finite trial")
    best = ok["value"].min()
    near = ok[ok["value"] <= best + tol].copy()
    for col in ("n_params", "total_lookback"):
        if col not in near:
            near[col] = 0
    return near.sort_values(["n_params", "total_lookback", "trial"]).iloc[0]


def run_tpe(space: dict, objective: Callable[[dict], tuple[float, dict]], tcfg: dict, sampler_seed: int,
            out_dir: str | Path, name: str = "study") -> dict:
    """Run one study; ``objective(params) -> (mean validation RMSE, info)`` where info may carry
    ``n_params`` and ``total_lookback`` for tie-breaking."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    out_dir = ensure_dir(out_dir)
    study = optuna.create_study(direction="minimize", study_name=name, pruner=optuna.pruners.NopPruner(),
                                sampler=make_sampler(sampler_seed, tcfg["n_startup_trials"], tcfg["n_ei_candidates"]))
    cache: dict[str, tuple[float, dict]] = {}
    records: list[dict] = []

    def _objective(trial):
        params = {k: trial.suggest_categorical(k, list(v)) for k, v in space.items()}
        key = json.dumps(params, sort_keys=True, default=str)
        cached = key in cache
        if not cached:
            try:
                value, info = objective(params)
            except Exception as exc:  # unconverged / failed trials receive an infinite objective
                LOG.warning("trial %d failed: %s", trial.number, exc)
                value, info = math.inf, {"error": repr(exc)}
            value = float(value) if np.isfinite(value) else math.inf
            cache[key] = (value, info)
        value, info = cache[key]
        records.append({"trial": trial.number, **params, "value": value, "cached": cached,
                        **{k: v for k, v in info.items() if np.isscalar(v) or v is None}})
        LOG.info("[%s] trial %2d  value %.5f %s", name, trial.number, value, "(cached)" if cached else "")
        return value

    study.optimize(_objective, n_trials=int(tcfg["n_trials"]), n_jobs=1, catch=())
    trials = pd.DataFrame(records)
    trials.to_csv(out_dir / "trials.csv", index=False)
    best = select_best(trials, float(tcfg.get("tie_tolerance", 1e-6)))
    best_params = {k: _py(best[k]) for k in space}
    result = {"best_params": best_params, "best_value": float(best["value"]), "best_trial": int(best["trial"]),
              "n_trials": len(trials), "n_unique": len(cache)}
    save_json(result, out_dir / "best.json")
    return result


def _py(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v
