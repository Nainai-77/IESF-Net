"""Run records of every completed run and the core models against the manuscript.

For every forecast folder: configuration source, whether an Optuna record exists, the
early-stopping rule replayed on the saved training log, whether the saved forecasts come from
the selected (best-validation) checkpoint, test metrics, the time the forecast file was written
and the line of the run log that started the run.  The core models (Persistence, Numeric,
IESF-Net, Bidirectional) are then summarised per seed and set against the values reported in
the manuscript (an optional privately supplied reference CSV).

Writes to outputs/audit/: run_records.csv, core_models_per_seed.csv, core_models_vs_manuscript.csv.
"""
import re
from pathlib import Path

import pandas as pd
from _common import ROOT, config_from, parser

from iesf.evaluation import leakage as L
from iesf.utils import ensure_dir, get_logger

LOG = get_logger("audit")
START = re.compile(r"^(\d\d:\d\d:\d\d) \| INFO \| reproduce \| >>> (\S+) \| (?:benchmark (\S+)|(\S+) \| seed (\S+))")


def log_starts(paths) -> dict:
    """(market, model, seed) -> 'file HH:MM:SS' of the line that started the run."""
    starts = {}
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
            m = START.match(line)
            if m:
                t, market, bench, model, seed = m.groups()
                starts[(market, bench or model, seed or "benchmark")] = f"{Path(path).name} {t}"
    return starts


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--paper-values", default=str(ROOT / "reference" / "paper_reported_values.csv"))
    p.add_argument("--logs", nargs="*", default=[str(ROOT / "logs" / "reproduce.log")])
    a = p.parse_args()
    cfg = config_from(a)
    out = ensure_dir(Path(cfg["paths"]["output_dir"]) / "audit")
    rec = L.run_records(cfg)
    starts = log_starts([x for x in a.logs if Path(x).exists()])
    rec["run_log_start"] = [starts.get((r.market, r.model, r.seed)) or starts.get((r.market, r.model, "benchmark"))
                            for r in rec.itertuples()]
    rec.to_csv(out / "run_records.csv", index=False)
    core = rec[rec["model"].isin(L.CORE_MODELS)].copy()
    keep = ["market", "model", "seed", "n_test", "test_RMSE", "test_MAE", "test_sMAPE", "test_R2", "test_DA",
            "logged_best_val_rmse", "best_epoch", "epochs_run", "early_stopping_consistent",
            "saved_forecasts_are_selected_checkpoint", "configuration", "optuna_record_present", "run_log_start"]
    core[[c for c in keep if c in core.columns]].to_csv(out / "core_models_per_seed.csv", index=False)
    paper = pd.read_csv(a.paper_values) if Path(a.paper_values).exists() else None
    if paper is None:
        LOG.warning("no manuscript values at %s; the comparison has no manuscript column", a.paper_values)
    L.core_comparison(rec, paper, int(cfg["seeds"]["designated"])).to_csv(out / "core_models_vs_manuscript.csv", index=False)
    LOG.info("%d runs -> %s", len(rec), out)
