"""Descriptive statistics and unit-root tests (Table 3, Section 3.6.1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps


def describe_series(x: np.ndarray) -> dict:
    from statsmodels.tsa.stattools import adfuller

    x = np.asarray(x, dtype=np.float64)
    adf = adfuller(x, regression="c", autolag="AIC")            # constant; lag order by AIC; MacKinnon p-values
    return {
        "N": len(x), "Mean": x.mean(), "SD": x.std(ddof=1), "Min": x.min(), "Max": x.max(),
        "Skew.": sps.skew(x, bias=True), "Ex. kurt.": sps.kurtosis(x, fisher=True, bias=True),   # population moments
        "AR(1)": float(np.corrcoef(x[:-1], x[1:])[0, 1]), "ADF": float(adf[0]), "p": float(adf[1]),
    }


def ljung_box(x: np.ndarray, lags: int = 10) -> dict:
    from statsmodels.stats.diagnostic import acorr_ljungbox

    r = acorr_ljungbox(np.asarray(x, dtype=np.float64), lags=[lags], return_df=True)
    return {"LB stat": float(r["lb_stat"].iloc[0]), "LB p": float(r["lb_pvalue"].iloc[0])}


def table3(closes: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for market, close in closes.items():
        rows.append({"Panel": "A: closing price level", "Market": market, **describe_series(close)})
    for market, close in closes.items():
        d = np.diff(np.asarray(close, dtype=np.float64))
        rows.append({"Panel": "B: one-observation price change", "Market": market, **describe_series(d),
                     **ljung_box(d), "Mean abs. change": float(np.mean(np.abs(d)))})
    return pd.DataFrame(rows)
