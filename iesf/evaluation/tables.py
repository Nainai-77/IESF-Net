"""Builders of the result tables (Tables 7-16, C1-C2, A2 and the core repeat evaluations)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..experiments.registry import LABELS, SHORT_LABELS
from ..experiments.runner import run_dir
from ..utils import load_json
from . import metrics as M
from .forecasts import common_index, load_runs, load_series, squared_losses
from .stats import bootstrap_reduction, dm_test, holm

F = "IESF-Net"


def _metrics_of(df: pd.DataFrame) -> dict:
    return M.all_metrics(df["prediction"], df["actual"], df["origin_price"])


def _fmt(mean, sd=None, nd=3):
    return f"{mean:.{nd}f}" if sd is None or not np.isfinite(sd) else f"{mean:.{nd}f} ± {sd:.{nd}f}"


# ------------------------------------------------------------------------------------------- Table 7
def table7(cfg: dict, markets: list[str]) -> pd.DataFrame:
    rows = []
    for mk in markets:
        for model in ["Numeric", F]:
            df = load_series(cfg, mk, model)
            if df is not None:
                rows.append({"Market": mk, "Group": "multiscale", "Model": model, **_metrics_of(df), "runs": 1})
        for model in cfg["experiments"]["families"]["benchmarks"]:
            runs = load_runs(cfg, mk, model)
            if not runs:
                continue
            per = pd.DataFrame([_metrics_of(r) for r in runs])
            row = {"Market": mk, "Group": "numerical benchmark", "Model": model, "runs": len(runs)}
            for col in ("RMSE", "MAE", "MSE", "sMAPE", "R2", "DA"):
                row[col] = per[col].mean()
                row[f"{col}_sd"] = per[col].std(ddof=1) if len(per) > 1 else np.nan   # sample SD across runs
            row["N"] = int(per["N"].iloc[0])
            rows.append(row)
    return pd.DataFrame(rows)


def format_table7(t7: pd.DataFrame) -> pd.DataFrame:
    out = []
    for r in t7.to_dict("records"):
        f = lambda c, nd=3: _fmt(r[c], r.get(f"{c}_sd"), nd)
        out.append({"Market": r["Market"], "Model": r["Model"], "RMSE": f("RMSE"), "MAE": f("MAE"),
                    "sMAPE (%)": f("sMAPE"), "R2": f("R2"), "DA (%)": f("DA", 2)})
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------ paired helper
def paired_family(cfg: dict, market: str, references: list[str], target: str = F, average_benchmarks: bool = True,
                  boot_seed: int = 0, family: str = "") -> pd.DataFrame:
    """RMSE reduction of ``target`` vs each reference with bootstrap CI, DM-HAC and Holm (within family)."""
    bench = set(cfg["experiments"]["families"]["benchmarks"])
    sq_f = squared_losses(cfg, market, target)
    if sq_f is None:
        return pd.DataFrame()
    ev = cfg["evaluation"]
    rows = []
    for j, ref in enumerate(references):
        sq_r = squared_losses(cfg, market, ref, average_runs=average_benchmarks and ref in bench)
        if sq_r is None:
            continue
        ix = common_index(sq_f, sq_r)
        a, b = sq_r.loc[ix].to_numpy(), sq_f.loc[ix].to_numpy()
        dm = dm_test(a, b)
        bs = bootstrap_reduction(a, b, ev["bootstrap_resamples"], ev["bootstrap_block"], seed=boot_seed)
        rows.append({"Reference": SHORT_LABELS.get(ref, ref), "model": ref, "Family": family,
                     "Reduction (%)": M.relative_reduction(np.sqrt(b.mean()), np.sqrt(a.mean())),
                     "CI low": bs["ci_low"], "CI high": bs["ci_high"], "DM": dm["DM"], "p": dm["p"],
                     "lag": dm["lag"], "N": len(ix)})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["Adj. p"] = holm(df["p"].to_numpy())
    return df


def tables9_10(cfg: dict, market: str) -> pd.DataFrame:
    """Numeric baseline and architectural controls (Holm within the eight controlled comparisons)."""
    return paired_family(cfg, market, cfg["experiments"]["families"]["controls"],
                         boot_seed=int(cfg["seeds"]["bootstrap_main"].get(market, 0)), family="Controls")


def tables_c(cfg: dict, market: str) -> pd.DataFrame:
    """Appendix C: IESF-Net vs the nine numerical benchmarks (losses averaged over runs within date)."""
    return paired_family(cfg, market, cfg["experiments"]["families"]["benchmarks"],
                         boot_seed=int(cfg["seeds"]["bootstrap_main"].get(market, 0)) + 500, family="Benchmarks")


# ------------------------------------------------------------------------------------------- Table 8
def table8(cfg: dict, market: str) -> pd.DataFrame:
    """Numerical benchmarks vs Persistence: MSE reduction and Holm-adjusted DM probabilities."""
    sq_p = squared_losses(cfg, market, "Persistence")
    rows = []
    for model in cfg["experiments"]["families"]["benchmarks"]:
        if model == "Persistence":
            continue
        sq_m = squared_losses(cfg, market, model, average_runs=True)
        if sq_m is None or sq_p is None:
            continue
        ix = common_index(sq_p, sq_m)
        a, b = sq_p.loc[ix].to_numpy(), sq_m.loc[ix].to_numpy()
        dm = dm_test(a, b)
        rows.append({"Market": market, "Model": model, "MSE reduction (%)": 100.0 * (1.0 - b.mean() / a.mean()),
                     "DM": dm["DM"], "p": dm["p"]})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["Adj. p"] = holm(df["p"].to_numpy())
    return df


# ------------------------------------------------------------------------------------------ Table 11
def table11(cfg: dict, market: str) -> pd.DataFrame:
    """Subgroup RMSE reductions vs Numeric: chronological halves and observed change size."""
    ev = cfg["evaluation"]
    f, n = load_series(cfg, market, F), load_series(cfg, market, "Numeric")
    if f is None or n is None:
        return pd.DataFrame()
    m = f.merge(n[["target_index", "prediction"]], on="target_index", suffixes=("", "_ref"))
    sq_f = ((m["prediction"] - m["actual"]) ** 2).to_numpy()
    sq_r = ((m["prediction_ref"] - m["actual"]) ** 2).to_numpy()
    seeds = list(cfg["seeds"]["bootstrap_subgroup"].get(market, [0, 1, 2, 3, 4]))
    B, blk = ev["bootstrap_resamples"], ev["bootstrap_block"]
    N = len(m)
    h = N // 2                                                  # 196 / 197 for N = 393
    change = np.abs(m["actual"] - m["origin_price"]).to_numpy()
    thr = float(np.quantile(change, float(ev["large_change_quantile"])))
    large = np.where(change >= thr, "large", "small")           # ties assigned to the large-change group
    rows = []
    full = bootstrap_reduction(sq_r, sq_f, B, blk, seeds[0])
    rows.append({"Market": market, "Target subset": "Full test", **full})
    for name, sl, seed in (("First half", slice(0, h), seeds[1]), ("Second half", slice(h, N), seeds[2])):
        rows.append({"Market": market, "Target subset": name, **bootstrap_reduction(sq_r[sl], sq_f[sl], B, blk, seed)})
    for name, lab, seed in (("Large observed changes", "large", seeds[3]), ("Small observed changes", "small", seeds[4])):
        rows.append({"Market": market, "Target subset": name,
                     **bootstrap_reduction(sq_r, sq_f, B, blk, seed, labels=large, select=lab)})
    df = pd.DataFrame(rows)
    df["threshold"] = thr
    return df


# ------------------------------------------------------------------------------ Tables 12-16 (point)
def variant_metrics(cfg: dict, markets: list[str], variants: list[str]) -> pd.DataFrame:
    rows = []
    for mk in markets:
        for v in variants:
            df = load_series(cfg, mk, v)
            if df is not None:
                rows.append({"Market": mk, "Variant": v, "Label": LABELS.get(v, v), **_metrics_of(df)})
    return pd.DataFrame(rows)


def with_increase(df: pd.DataFrame, reference: str = F) -> pd.DataFrame:
    """Table 14 / 15: RMSE increase relative to the complete model (Eq. 23, i_m)."""
    out = df.copy()
    ref = out[out["Variant"] == reference].set_index("Market")["RMSE"]
    out["Increase (%)"] = [M.relative_increase(ref.get(mk, np.nan), r) for mk, r in zip(out["Market"], out["RMSE"])]
    return out


def table16(cfg: dict, markets: list[str]) -> pd.DataFrame:
    rows = []
    for mk in markets:
        f, n = load_series(cfg, mk, F), load_series(cfg, mk, "Numeric")
        if f is None or n is None:
            continue
        mf, mn = _metrics_of(f), _metrics_of(n)
        rows.append({"Market": mk, "N": mf["N"], "Numeric RMSE": mn["RMSE"], "IESF-Net RMSE": mf["RMSE"],
                     "RMSE reduction (%)": M.relative_reduction(mf["RMSE"], mn["RMSE"]),
                     "MAE reduction (%)": M.relative_reduction(mf["MAE"], mn["MAE"]), "IESF-Net R2": mf["R2"]})
    return pd.DataFrame(rows)


def core_repeats(cfg: dict, markets: list[str], models=("IESF-Net", "Numeric", "Bidirectional")) -> pd.DataFrame:
    """Independent repeat evaluations of the three core models (distinct from the designated series)."""
    rows = []
    for mk in markets:
        for model in models:
            runs = load_runs(cfg, mk, model)
            if len(runs) < 1:
                continue
            per = pd.DataFrame([_metrics_of(r) for r in runs])
            rows.append({"Market": mk, "Model": model, "runs": len(runs),
                         **{f"{c}_mean": per[c].mean() for c in ("RMSE", "MAE", "sMAPE", "R2", "DA")},
                         **{f"{c}_sd": per[c].std(ddof=1) if len(per) > 1 else np.nan for c in ("RMSE", "MAE")}})
    return pd.DataFrame(rows)


def tableA2(cfg: dict, markets: list[str]) -> pd.DataFrame:
    rows = []
    for mk in markets:
        for model in cfg["experiments"]["families"]["benchmarks"]:
            best = run_dir(cfg, "tuning", mk, model) / "best.json"
            runs = sorted(run_dir(cfg, "forecasts", mk, model).glob("seed*/run.json"))
            if not runs:
                continue
            info = load_json(runs[0])
            val = load_json(best)["best_value"] if best.exists() else info.get("val_rmse")
            rows.append({"Market": mk, "Model": model, "Selected configuration": info["params"],
                         "Validation RMSE": val, "Size": info["size"], "Runs": len(runs)})
    return pd.DataFrame(rows)
