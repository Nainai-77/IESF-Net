"""Supplementary timing, reconstruction and partition checks (Appendix B.2)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.dataset import MarketData


def timing_check(df: pd.DataFrame) -> dict:
    """Regress the forecast on the preceding observed price and the contemporaneous target.

    A large weight on the contemporaneous target flags possible look-ahead; the
    diagnostic does not replace checking input availability at the origin."""
    X = np.column_stack([np.ones(len(df)), df["origin_price"], df["actual"]])
    beta, *_ = np.linalg.lstsq(X, df["prediction"].to_numpy(), rcond=None)
    return {"const": float(beta[0]), "beta_prev_price": float(beta[1]), "beta_target": float(beta[2]), "N": len(df)}


def reconstruction_check(md: MarketData) -> dict:
    """Observed price vs the sum of its four archived components."""
    err = md["R"].sum(axis=1) - md["close"]
    return {"max_abs_error": float(np.max(np.abs(err))), "mean_abs_error": float(np.mean(np.abs(err)))}


def partition_check(md: MarketData, market_cfg: dict, reserve: int) -> dict:
    p = md.partition
    out = {"n_obs": md.n_obs, "reserve": p.reserve, "train/val/test": [p.n_train, p.n_val, p.n_test],
           "matches_config": [p.n_train, p.n_val, p.n_test] == list(market_cfg["splits"]) and p.reserve == reserve,
           "first_val_target": str(md.dates[p.train_end]), "first_test_target": str(md.dates[p.val_end]),
           "last_target": str(md.dates[-1])}
    if market_cfg.get("test_start"):
        out["test_start_matches"] = str(md.dates[p.val_end]) == str(market_cfg["test_start"])
    return out


def forecast_alignment_check(df: pd.DataFrame, md: MarketData, split: str) -> dict:
    idx = md.partition.targets(split)
    ok_idx = np.array_equal(np.sort(df["target_index"].to_numpy()), idx)
    ok_actual = np.allclose(df.sort_values("target_index")["actual"].to_numpy(), md["close"][idx])
    ok_prev = np.allclose(df.sort_values("target_index")["origin_price"].to_numpy(), md["close"][idx - 1])
    return {"target_dates_match": bool(ok_idx), "actual_matches": bool(ok_actual), "origin_price_matches": bool(ok_prev)}
