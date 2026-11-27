"""Enumerate the full experiment matrix of the manuscript and the status of every run.

One row per (market, model or variant, seed) estimation unit and per selection study, with the
manuscript tables it feeds and whether its output exists.  Writes reference/experiment_matrix.csv.
"""
from pathlib import Path

import pandas as pd
from _common import config_from, parser

from iesf.config import market_names
from iesf.experiments.registry import LABELS, VARIANTS
from iesf.experiments.runner import run_dir

SUITE_TABLES = {"main": "7, 11, 16", "representation": "12", "fusion": "13",
                "ablation": "14", "window": "15", "cross_market": "16"}

if __name__ == "__main__":
    a = parser(__doc__).parse_args()
    cfg = config_from(a)
    ecfg, seeds = cfg["experiments"], cfg["seeds"]
    core = {"Numeric", "IESF-Net", "Bidirectional"}
    controls = set(ecfg["families"]["controls"])
    units: dict = {}

    def add(market, model, seed, family, tables, purpose):
        key = (market, model, str(seed))
        u = units.setdefault(key, {"market": market, "model": model, "label": LABELS.get(model, model), "seed": str(seed),
                                   "family": family, "tables": set(), "purpose": set()})
        u["tables"].update(t.strip() for t in tables.split(","))
        u["purpose"].add(purpose)

    for m in cfg["primary_markets"]:                                             # numerical benchmarks
        for b in ecfg["families"]["benchmarks"]:
            stoch = cfg["benchmarks"][b].get("stochastic")
            for s in (seeds["evaluation"] if stoch else ["deterministic"]):
                add(m, b, s, "benchmark", "7, 8, A2, C1, C2", "five-run summary" if stoch else "deterministic reference")
    for suite, scfg in ecfg["suites"].items():                                    # multiscale family
        sm = scfg["markets"]
        for m in (market_names(cfg, sm) if isinstance(sm, str) else sm):
            for v in scfg["variants"]:
                rep = scfg.get("repeat_seeds") and v in core
                for s in (seeds["evaluation"] if rep else [seeds["designated"]]):
                    tabs = SUITE_TABLES[suite] if not (suite == "main" and v == "Bidirectional") else "Fig. 6, Fig. 7"
                    tabs += ", 9, 10" if v in controls or v == "IESF-Net" else ""
                    tabs += ", C1, C2" if v == "IESF-Net" else ""
                    designated = str(s) == str(seeds["designated"])
                    add(m, v, s, "multiscale", tabs, "designated series" if designated else "independent repeat")
    rows = []
    for (m, model, s), u in sorted(units.items()):
        done = (run_dir(cfg, "forecasts", m, model, s) / "test_predictions.csv").exists()
        rows.append({**u, "tables": ", ".join(sorted(u["tables"], key=lambda t: (len(t), t))),
                     "purpose": "; ".join(sorted(u["purpose"])), "kind": "estimation",
                     "status": "done" if done else "pending",
                     "output": str(run_dir(cfg, "forecasts", m, model, s).relative_to(Path(cfg["paths"]["output_dir"]).parent))})
    for m in cfg["primary_markets"]:                                             # validation-only selection studies
        for model in ["IESF-Net"] + [b for b in ecfg["families"]["benchmarks"] if "search_space" in cfg["benchmarks"][b]]:
            done = (run_dir(cfg, "tuning", m, model) / "best.json").exists()
            rows.append({"market": m, "model": model, "label": LABELS.get(model, model), "seed": "tuning seeds",
                         "family": "selection", "tables": "A2" if model != "IESF-Net" else "A3",
                         "purpose": f"Optuna TPE, {cfg['tuning']['n_trials']} trials", "kind": "selection",
                         "status": "done" if done else "pending (optional when using previously selected settings)",
                         "output": f"outputs/tuning/{m}/{model}"})
    df = pd.DataFrame(rows)
    out = Path(cfg["paths"]["output_dir"]).parent / "reference" / "experiment_matrix.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")
    est = df[df["kind"] == "estimation"]
    print(f"variants registered: {len(VARIANTS)};  estimation units: {len(est)}  "
          f"(multiscale {int((est.family == 'multiscale').sum())}, benchmark {int((est.family == 'benchmark').sum())});  "
          f"selection studies: {int((df.kind == 'selection').sum())}")
    print(est.groupby(["family", "market", "status"]).size().to_string())
    print(f"-> {out}")
