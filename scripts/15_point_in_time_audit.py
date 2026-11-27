"""Point-in-time information audit.

Target i is forecast at origin i-1.  Every input must satisfy
    max(input timestamp) <= forecast cut-off of the origin,
where documents and institutional actions count when they are available at or before the
market cut-off (15:00, UTC+08:00) of the origin date, and prices up to and including the origin
date's close.  For every target the audit lists the latest price date, document and action in
the information set; it also checks record-level timing, deduplication, and rebuilds the daily
text and PII arrays from the raw timestamps to compare them with the arrays the models read.

Writes to outputs/audit/: {Market}_point_in_time.csv, {Market}_actions_timing.csv and
point_in_time_summary.json.
"""
from pathlib import Path

from _common import config_from, parser

from iesf.config import market_names
from iesf.evaluation import leakage as L
from iesf.utils import ensure_dir, get_logger, save_json

LOG = get_logger("audit")

if __name__ == "__main__":
    a = parser(__doc__).parse_args()
    cfg = config_from(a)
    out = ensure_dir(Path(cfg["paths"]["output_dir"]) / "audit")
    tz = cfg["alignment"]["timezone"]
    docs = L.document_metadata(cfg)
    summary = {"rule": "max(input timestamp) <= forecast cut-off of the origin "
                       f"({cfg['alignment']['market_cutoff']}, UTC{tz}); prices up to the origin date",
               "documents": L.document_checks(docs), "deduplication": L.deduplication_check(cfg, docs), "markets": {}}
    for m in a.markets or market_names(cfg):
        actions, s_act = L.action_checks(cfg, m, docs)
        pit, s_pit = L.point_in_time_table(cfg, m, docs, actions)
        pit.to_csv(out / f"{m}_point_in_time.csv", index=False)
        L.actions_for_export(actions, tz).to_csv(out / f"{m}_actions_timing.csv", index=False)
        summary["markets"][m] = {"point_in_time": s_pit, "actions": s_act,
                                 "text_arrays": L.text_array_check(cfg, m, docs),
                                 "pii_arrays": L.pii_array_check(cfg, m, docs)}
        LOG.info("%s: %d targets, all compliant: %s", m, len(pit), s_pit["all_compliant"])
    save_json(summary, out / "point_in_time_summary.json")
    LOG.info("-> %s", out)
