"""Leakage checks on the decomposition, the partitions, the training statistics and the trained models.

* CEEMDAN: sampled origins of every split are re-decomposed from their own price prefix and
  compared with the archive the models read; two test origins are re-decomposed again with
  random-walk future prices; the grouping is refitted on training origins only, with and
  without perturbed later frequencies; call sites of the full-sample (Figure 3) decomposition.
* Partitions: full date index of every market and the ordering assertions.
* Statistics: training statistics with validation and test rows replaced by noise.
* Checkpoints: every saved core-model checkpoint is reloaded and its saved forecasts are
  reproduced; for the designated seed, all inputs at and after three cut points are replaced
  by noise and the forecasts of earlier targets are compared.

Writes to outputs/audit/: {Market}_date_index.csv, {Market}_decomposition_recheck.csv and
leakage_checks.json.
"""
from pathlib import Path

import pandas as pd
from _common import ROOT, config_from, parser

from iesf.config import market_names
from iesf.evaluation import leakage as L
from iesf.evaluation.forecasts import seeds_available
from iesf.experiments.runner import run_dir
from iesf.utils import ensure_dir, get_logger, save_json

LOG = get_logger("audit")

if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--origins-per-split", type=int, default=10)
    p.add_argument("--models", nargs="*", default=["Numeric", "IESF-Net", "Bidirectional"])
    p.add_argument("--skip-decomposition", action="store_true")
    a = p.parse_args()
    cfg = config_from(a)
    out = ensure_dir(Path(cfg["paths"]["output_dir"]) / "audit")
    designated = int(cfg["seeds"]["designated"])
    report = {"full_sample_decomposition_call_sites": L.full_sample_call_sites(ROOT / "iesf"), "markets": {}}
    for m in a.markets or market_names(cfg):
        rep = {}
        idx, rep["partition"] = L.partition_audit(cfg, m)
        idx.to_csv(out / f"{m}_date_index.csv", index=False)
        rep["statistics"] = L.statistics_check(cfg, m)
        if not a.skip_decomposition:
            LOG.info("%s: re-decomposing sampled origins", m)
            dec, rep["decomposition"] = L.decomposition_check(cfg, m, a.origins_per_split)
            dec.to_csv(out / f"{m}_decomposition_recheck.csv", index=False)
        rep["checkpoints"] = []
        for model in a.models:
            for seed in seeds_available(cfg, m, model):
                if not (run_dir(cfg, "checkpoints", m, model) / f"seed{seed}.pt").exists():
                    continue
                LOG.info("%s | %s | seed %s: checkpoint check", m, model, seed)
                rep["checkpoints"].append(L.model_causality_check(cfg, m, model, int(seed), perturb=int(seed) == designated))
        report["markets"][m] = rep
    save_json(report, out / "leakage_checks.json")
    rows = [{**{k: v for k, v in c.items() if k != "cuts"}, **cut} for r in report["markets"].values()
            for c in r["checkpoints"] for cut in (c.get("cuts") or [{}])]
    pd.DataFrame(rows).to_csv(out / "checkpoint_checks.csv", index=False)
    LOG.info("-> %s", out)
