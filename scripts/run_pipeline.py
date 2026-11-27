"""Run the whole reproduction pipeline in order (or a subset of stages).

Full paper run (real data under data/raw):
    python scripts/run_pipeline.py
Synthetic smoke test (minutes, no model download):
    python scripts/run_pipeline.py --config configs/smoke.yaml --synthetic --mock-encoders
"""
from _common import config_from, parser

from iesf import pipeline as P

STAGES = ["corpus", "encode", "pii", "decompose", "build", "descriptive", "benchmarks", "tune",
          "experiments", "statistics", "interpret", "figures", "audit", "checks"]

if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--stages", nargs="*", default=STAGES, choices=STAGES)
    p.add_argument("--synthetic", action="store_true", help="generate synthetic raw data first")
    p.add_argument("--mock-encoders", action="store_true")
    p.add_argument("--import-encoded", default=None, help="directory with precomputed encodings (see 02_encode_text.py)")
    p.add_argument("--actions", default=None, help="external reviewed action register (JSONL)")
    p.add_argument("--no-tune", action="store_true", help="use paper-selected settings instead of Optuna")
    a = p.parse_args()
    cfg = config_from(a)
    if a.synthetic:
        P.stage_synthetic(cfg)
    run = {
        "corpus": lambda: P.stage_corpus(cfg, a.markets),
        "encode": lambda: (P.stage_import_encoded(cfg, a.import_encoded) if a.import_encoded
                           else P.stage_encode(cfg, mock=a.mock_encoders)),
        "pii": lambda: P.stage_pii(cfg, a.actions),
        "decompose": lambda: P.stage_decompose(cfg, a.markets),
        "build": lambda: P.stage_build(cfg, a.markets),
        "descriptive": lambda: P.stage_descriptive(cfg),
        "benchmarks": lambda: P.stage_benchmarks(cfg, a.markets, tune=not a.no_tune),
        "tune": lambda: None if a.no_tune else P.stage_tune(cfg, a.markets),
        "experiments": lambda: P.stage_experiments(cfg, None, a.markets),
        "statistics": lambda: P.stage_statistics(cfg),
        "interpret": lambda: P.stage_interpret(cfg, a.markets),
        "figures": lambda: P.stage_figures(cfg),
        "audit": lambda: P.stage_audit(cfg, "sample") if not a.actions else None,
        "checks": lambda: P.stage_checks(cfg),
    }
    for stage in STAGES:
        if stage in a.stages:
            P.LOG.info("=== stage: %s", stage)
            run[stage]()
