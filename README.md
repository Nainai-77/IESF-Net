# IESF-Net

Source code for **Persistent Institutional Memory and Multiscale Text–Price Fusion for Carbon Allowance Price Forecasting in China's Regional Markets**.

This repository contains the forecasting architecture, rule-based institutional action extraction and stratified expert-audit utilities, causal origin-wise CEEMDAN wrapper, experimental configurations, and verification scripts. It does not contain the original datasets or empirical forecast outputs.

## Repository layout

- `configs/`: research protocol, model and benchmark settings, synthetic smoke-test configuration.
- `iesf/`: data preparation, decomposition, text processing, models, training, evaluation.
- `scripts/`: numbered pipeline stages, experiment and audit entry points.
- `tests/`: unit tests, including model-level future-input perturbation.

## Installation

Use a clean Python environment (the supplied original requirements identify Python 3.12.7 as the tested version). Install dependencies:

```bash
python -m pip install -r requirements.txt
```

The requirements retain the version constraints in the supplied original requirements file, with `tqdm` added as an explicit unpinned dependency. This is not a complete transitive environment lock, and installation of the pinned versions has not been independently verified for this package. The third-party PyEMD source is not redistributed here because the supplied files did not include the complete package metadata and license.

## Verification without research data

```bash
python -m pytest tests -q
python scripts/run_pipeline.py --config configs/smoke.yaml --synthetic --mock-encoders --stages corpus encode pii decompose build
```

The second command is a synthetic preprocessing smoke test only. It writes under `data_smoke/`; its outputs are not evidence for the manuscript's empirical conclusions. Synthetic data are not used in the reported experiments. No pretrained language model download is needed for the mock encoder run.

## Research workflow (requires authorized private data)

The default configuration expects market price CSVs under `data/raw/prices/` and the source corpus at `data/raw/documents.jsonl`. See the loader code for required fields. The default PII path extracts candidates automatically, consolidates actions and writes `data/interim/pii/actions.jsonl`; the expert sampling script is `scripts/13_extraction_audit.py`. The optional `--actions` interface imports an external register and is not required by the automatic extraction workflow.

```bash
python scripts/run_pipeline.py
```

The causal decomposition implementation is in `iesf/decomposition/ceemdan.py`, training-period frequency grouping in `iesf/decomposition/grouping.py`, and forecasting in `iesf/models/iesf_net.py` and `iesf/training/trainer.py`. Audit utilities are `scripts/14_consistency_checks.py`, `scripts/15_point_in_time_audit.py`, and `scripts/16_leakage_checks.py`. Some audit scripts require private model artifacts or data and cannot be run from this source-only repository alone.

## Scope and limitations

The original price datasets, text corpus, action registers, decomposition archives, trained checkpoints, logs, expert responses, and empirical outputs are excluded. Configuration files retain sample partitions and validation-selected settings for transparency. Synthetic test fixtures are not empirical observations.

The model-level perturbation tests do not establish end-to-end point-in-time integrity of the raw-text pipeline. In particular, global deduplication and action consolidation must be checked for retrospective attribute merging; cache provenance and supporting-document timestamps also require verification. This source-only release does not assert independent reproduction or a complete absence of future-information leakage.

## License

A project license has not been supplied with the source materials. The authors must determine the licensing terms before public release. Third-party dependencies retain their respective licenses.
