"""Selection and evaluation of the numerical benchmarks (Table A4 protocol, Table A2, Table 7).

Each searchable benchmark is selected by Optuna TPE on validation RMSE (mean over
the tuning seeds for stochastic models).  Stochastic benchmarks are then
re-estimated with the five evaluation seeds; deterministic references produce one
forecast sequence.  Forecast files share the schema of the multiscale models.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data.dataset import MarketData
from ..experiments.runner import processed_path, run_dir
from ..training.tuning import run_tpe
from ..utils import ensure_dir, get_device, get_logger, load_json, save_json
from .classical import WindowData, arima_fit_forecast, persistence_forecast, ridge_fit_forecast, xgb_fit_forecast
from .neural import fit_neural

LOG = get_logger(__name__)


def _splits(md: MarketData) -> dict:
    return {s: md.partition.targets(s) for s in ("train", "val", "test")}


def _rmse(pred, y) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(y)) ** 2)))


def forecast_once(cfg: dict, name: str, md: MarketData, params: dict, seed: int) -> tuple[dict, int]:
    """Fit one benchmark configuration and return native-unit forecasts per split."""
    bcfg = cfg["benchmarks"][name]
    kind, splits = bcfg["kind"], _splits(md)
    if kind == "persistence":
        return {s: persistence_forecast(md, ix) for s, ix in splits.items()}, 0
    if kind == "arima":
        pred, size = arima_fit_forecast(md, int(params["p"]), int(params["d"]), int(params["q"]))
        return {s: pred[ix] for s, ix in splits.items()}, size
    wd = WindowData(md)
    if kind == "ridge":
        return ridge_fit_forecast(wd, int(params["lookback"]), float(params["alpha"]), splits)
    if kind == "xgboost":
        return xgb_fit_forecast(wd, int(params["lookback"]), int(params["max_depth"]), float(params["learning_rate"]),
                                bcfg["fixed"], seed, splits)
    if kind == "neural":
        device = get_device(cfg["optim"].get("device", "auto"))
        preds, size, _ = fit_neural(bcfg["arch"], wd, splits, params, bcfg.get("fixed", {}), cfg["optim"], seed, device)
        return preds, size
    raise ValueError(kind)


def tune_benchmark(cfg: dict, market: str, name: str) -> dict:
    md = MarketData.load(processed_path(cfg, market))
    bcfg = cfg["benchmarks"][name]
    if "search_space" not in bcfg:
        return {"best_params": {}, "best_value": float("nan")}
    seeds = cfg["seeds"]["tuning"] if bcfg.get("stochastic") else [0]
    y_val = md["close"][md.partition.targets("val")]

    def objective(params):
        vals, size = [], 0
        for seed in seeds:
            preds, size = forecast_once(cfg, name, md, params, int(seed))
            vals.append(_rmse(preds["val"], y_val))
        return float(np.mean(vals)), {"n_params": size, "total_lookback": int(params.get("lookback", 0))}

    return run_tpe(bcfg["search_space"], objective, cfg["tuning"], int(cfg["seeds"]["sampler"]),
                   run_dir(cfg, "tuning", market, name), name=f"{market}-{name}")


def selected_params(cfg: dict, market: str, name: str, source: str = "tuned") -> dict:
    bcfg = cfg["benchmarks"][name]
    path = run_dir(cfg, "tuning", market, name) / "best.json"
    if source == "tuned" and path.exists():
        return load_json(path)["best_params"]
    params = dict(bcfg.get("paper_selected", {}).get(market, {}))
    if bcfg["kind"] == "neural":                       # settings not reported in Table A2: IESF-Net values of Table A3
        params = {"lr": 0.001, "weight_decay": 0.0001, "dropout": 0.1, "batch_size": 64, **params}
    if not params and "search_space" in bcfg:
        params = {k: v[0] for k, v in bcfg["search_space"].items()}
    return params


def evaluate_benchmark(cfg: dict, market: str, name: str, params: dict | None = None) -> pd.DataFrame:
    """Forecast sequences for every evaluation seed (one sequence for deterministic models)."""
    md = MarketData.load(processed_path(cfg, market))
    bcfg = cfg["benchmarks"][name]
    params = params if params is not None else selected_params(cfg, market, name)
    seeds = cfg["seeds"]["evaluation"] if bcfg.get("stochastic") else ["deterministic"]
    rows = []
    for seed in seeds:
        preds, size = forecast_once(cfg, name, md, params, 0 if seed == "deterministic" else int(seed))
        out = ensure_dir(run_dir(cfg, "forecasts", market, name, seed))
        for split in ("val", "test"):
            ix = md.partition.targets(split)
            pd.DataFrame({
                "market": market, "model": name, "seed": seed, "split": split, "target_index": ix,
                "origin_index": ix - 1,
                "origin_date": pd.to_datetime(md.dates[ix - 1]).strftime("%Y-%m-%d"),
                "target_date": pd.to_datetime(md.dates[ix]).strftime("%Y-%m-%d"),
                "origin_price": md["close"][ix - 1], "actual": md["close"][ix], "prediction": preds[split],
            }).to_csv(out / f"{split}_predictions.csv", index=False)
        val_rmse = _rmse(preds["val"], md["close"][md.partition.targets("val")])
        save_json({"market": market, "model": name, "seed": seed, "params": params, "size": size,
                   "val_rmse": val_rmse}, out / "run.json")
        rows.append({"market": market, "model": name, "seed": seed, "val_rmse": val_rmse, "size": size,
                     "params": str(params)})
        LOG.info("%s | %s | seed %s | val RMSE %.4f", market, name, seed, val_rmse)
    return pd.DataFrame(rows)
