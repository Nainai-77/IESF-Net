"""Persistence, ARIMA, Ridge and XGBoost benchmarks (Section 3.6.4, Appendix A.2).

Ridge / XGBoost read the flattened four-variable window at positions
{i-L, ..., i-1} (training-based scaling) and predict the standardised
one-observation change; the close forecast adds the latest observed close.
ARIMA is fitted to the training price sequence (including the reserved
warm-up) and produces one-step predictive states with fixed parameters.
"""
from __future__ import annotations

import warnings

import numpy as np

from ..data.dataset import MarketData


class WindowData:
    """Sliding windows of the scaled numerical input and change targets."""

    def __init__(self, md: MarketData):
        st = md.stats()
        self.md = md
        self.X = ((md["X_raw"] - st["X_mean"]) / st["X_std"]).astype(np.float32)
        self.y = md["close"].astype(np.float64)
        self.dy_mean, self.dy_std = st["dy_mean"], st["dy_std"]

    def windows(self, idx: np.ndarray, L: int) -> np.ndarray:
        pos = idx[:, None] - L + np.arange(L)[None, :]          # i-L .. i-1
        return self.X[pos]                                      # [N, L, 4]

    def target(self, idx: np.ndarray) -> np.ndarray:
        return ((self.y[idx] - self.y[idx - 1] - self.dy_mean) / self.dy_std).astype(np.float32)

    def restore(self, idx: np.ndarray, d_hat: np.ndarray) -> np.ndarray:
        return self.y[idx - 1] + self.dy_mean + self.dy_std * np.asarray(d_hat, dtype=np.float64)


# ----------------------------------------------------------------------------------- Persistence
def persistence_forecast(md: MarketData, idx: np.ndarray) -> np.ndarray:
    return md["close"][idx - 1].astype(np.float64)


# ----------------------------------------------------------------------------------------- ARIMA
def arima_fit_forecast(md: MarketData, p: int, d: int, q: int) -> tuple[np.ndarray, int]:
    """One-step predictions on the whole calendar from a fit on the training price sequence."""
    from statsmodels.tsa.arima.model import ARIMA

    y = md["close"].astype(np.float64)
    train = y[: md.partition.train_end]
    trend = "c" if d == 0 else "t"                               # d = 0: constant; d = 1: linear trend
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ARIMA(train, order=(p, d, q), trend=trend).fit()
        if not res.mle_retvals.get("converged", True):
            raise RuntimeError(f"ARIMA({p},{d},{q}) did not converge")
        full = res.apply(y, refit=False)
        pred = np.asarray(full.get_prediction(start=0, dynamic=False).predicted_mean, dtype=np.float64)
    return pred, int(len(res.params))


# ----------------------------------------------------------------------------------------- Ridge
def ridge_fit_forecast(wd: WindowData, L: int, alpha: float, splits: dict) -> tuple[dict, int]:
    from sklearn.linear_model import Ridge

    tr = splits["train"]
    model = Ridge(alpha=float(alpha)).fit(wd.windows(tr, L).reshape(len(tr), -1), wd.target(tr))
    out = {s: wd.restore(ix, model.predict(wd.windows(ix, L).reshape(len(ix), -1))) for s, ix in splits.items()}
    return out, int(model.coef_.size + 1)


# --------------------------------------------------------------------------------------- XGBoost
def xgb_fit_forecast(wd: WindowData, L: int, max_depth: int, lr: float, fixed: dict, seed: int, splits: dict):
    from xgboost import XGBRegressor

    tr = splits["train"]
    model = XGBRegressor(n_estimators=int(fixed["n_estimators"]), max_depth=int(max_depth), learning_rate=float(lr),
                         tree_method=fixed["tree_method"], max_bin=int(fixed["max_bin"]),
                         subsample=float(fixed["subsample"]), colsample_bytree=float(fixed["colsample_bytree"]),
                         reg_lambda=float(fixed["reg_lambda"]), random_state=int(seed), n_jobs=1)
    model.fit(wd.windows(tr, L).reshape(len(tr), -1), wd.target(tr))
    out = {s: wd.restore(ix, model.predict(wd.windows(ix, L).reshape(len(ix), -1))) for s, ix in splits.items()}
    return out, int(fixed["n_estimators"])
