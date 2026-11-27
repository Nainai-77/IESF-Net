"""Forecast metrics (Section 3.6.6, Eqs. 20-24, Table B1)."""
from __future__ import annotations

import numpy as np


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def rmse(pred, y) -> float:
    return float(np.sqrt(np.mean((_arr(pred) - _arr(y)) ** 2)))


def mae(pred, y) -> float:
    return float(np.mean(np.abs(_arr(pred) - _arr(y))))


def mse(pred, y) -> float:
    return float(np.mean((_arr(pred) - _arr(y)) ** 2))


def smape(pred, y) -> float:
    """Eq. (20): 200/N Σ |ŷ - y| / (|ŷ| + |y|), in percent."""
    p, a = _arr(pred), _arr(y)
    return float(np.mean(200.0 * np.abs(p - a) / (np.abs(p) + np.abs(a))))


def r2(pred, y) -> float:
    """Eq. (22): 1 - SSE / SST about the mean of the test targets."""
    p, a = _arr(pred), _arr(y)
    return float(1.0 - np.sum((a - p) ** 2) / np.sum((a - a.mean()) ** 2))


def directional_accuracy(pred, y, prev) -> float:
    """Eq. (22): three-way direction (up / down / unchanged) of predicted vs observed change from the preceding close."""
    p, a, b = _arr(pred), _arr(y), _arr(prev)
    return float(100.0 * np.mean(np.sign(p - b) == np.sign(a - b)))


def all_metrics(pred, y, prev) -> dict:
    return {"RMSE": rmse(pred, y), "MAE": mae(pred, y), "MSE": mse(pred, y), "sMAPE": smape(pred, y),
            "R2": r2(pred, y), "DA": directional_accuracy(pred, y, prev), "N": int(len(_arr(y)))}


def relative_reduction(rmse_f: float, rmse_m: float) -> float:
    """Eq. (23): r_m = 100 (1 - RMSE_F / RMSE_m)."""
    return 100.0 * (1.0 - rmse_f / rmse_m)


def relative_increase(rmse_f: float, rmse_m: float) -> float:
    """Eq. (23): i_m = 100 (RMSE_m / RMSE_F - 1)."""
    return 100.0 * (rmse_m / rmse_f - 1.0)


def cumulative_gain(sq_ref, sq_f) -> np.ndarray:
    """Eq. (24): G_m(k) = 100 Σ_{t<=k}(e_m^2 - e_F^2) / Σ_t e_m^2; G_m(N) = 100[1 - (1 - r_m/100)^2]."""
    sq_ref, sq_f = _arr(sq_ref), _arr(sq_f)
    return 100.0 * np.cumsum(sq_ref - sq_f) / np.sum(sq_ref)


def rolling_rmse(pred, y, window: int = 21) -> np.ndarray:
    """Centred rolling RMSE (Figures 6-7)."""
    import pandas as pd

    se = pd.Series((_arr(pred) - _arr(y)) ** 2)
    return np.sqrt(se.rolling(window, center=True, min_periods=max(1, window // 2)).mean().to_numpy())
