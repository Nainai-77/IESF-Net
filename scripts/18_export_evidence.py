"""Assemble the reviewer evidence folder from the audit outputs and the completed runs.

No document text is exported.  The folder contains
  metadata/     documents_time_metadata.csv (identifiers, URLs, publication and availability
                times, text hashes), actions_time_metadata.csv (register fields without the
                evidence text), and per market the document alignment, the PII event table and
                the date index with partitions
  audit/        outputs of 15_point_in_time_audit.py, 16_leakage_checks.py, 17_run_records.py
  forecasts/    every completed run: val / test predictions, run.json, training_log.csv
  checkpoints/  trained weights of the core models (--checkpoints core | all | none) + SHA256SUMS
  logs/         the run logs passed with --logs
  data_manifest.csv   size and SHA-256 of every data file the experiments read
  processed_arrays_manifest.csv  SHA-256 of every model input array (checked by 19_verify_rebuild.py)
  environment.json    interpreter, platform and package versions; text-encoder environment
  clean_environment/  logs and reports of scripts/clean_run.sh (--clean-run DIR), if supplied
  EVIDENCE.md         summary generated from the files above

Run 15, 16 and 17 first.
"""
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

import pandas as pd
from _common import ROOT, config_from, parser

from iesf.config import market_names
from iesf.evaluation import leakage as L
from iesf.utils import ensure_dir, get_logger, load_json, read_jsonl, save_json

LOG = get_logger("evidence")
CORE = ("Numeric", "IESF-Net", "Bidirectional")
ACTION_TEXT_FIELDS = {"evidence", "additional_evidence", "evidence_spans", "review_note", "reason",
                      "effective_period_text", "title", "source_title"}
PACKAGES = ["EMD-signal", "optuna", "numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "torch",
            "transformers", "xgboost", "joblib", "matplotlib", "PyYAML", "pytest"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def copy(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def environment(cfg) -> dict:
    import importlib.metadata as md

    import torch

    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = md.version(p)
        except md.PackageNotFoundError:
            versions[p] = None
    pinned = {}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        spec = line.split("#")[0].strip()
        if "==" in spec:
            name, ver = spec.split("==")
            pinned[name.strip()] = ver.strip()
    enc = Path(cfg["paths"]["interim_dir"]) / "encoded" / "encoding_manifest.json"
    src = load_json(enc).get("source_manifest", {}) if enc.exists() else {}
    return {
        "python": sys.version.split()[0], "platform": platform.platform(), "processor": platform.processor(),
        "torch": {"version": torch.__version__, "cuda_build": torch.version.cuda,
                  "cuda_available": torch.cuda.is_available(), "threads": torch.get_num_threads()},
        "packages": versions,
        "differs_from_requirements_pin": {k: {"installed": versions.get(k), "requirements.txt": v}
                                          for k, v in pinned.items() if versions.get(k) not in (None, v)},
        "text_encoder_environment": {k: src.get(k) for k in ("torch_version", "device", "dtype", "threads", "tf32_enabled")}
        | {"semantic_revision": (src.get("semantic") or {}).get("model_revision"),
           "sentiment_revision": (src.get("sentiment") or {}).get("model_revision"),
           "corpus_sha256": src.get("corpus_sha256")},
    }


def export_metadata(cfg, out: Path, actions_path: Path | None) -> dict:
    tz = cfg["alignment"]["timezone"]
    _, interim, proc, audit = (Path(cfg["paths"][k]) for k in ("raw_dir", "interim_dir", "processed_dir", "output_dir"))
    audit = audit / "audit"
    meta = ensure_dir(out / "metadata")
    docs = L.document_metadata(cfg)
    docs = docs.assign(markets=docs["markets"].map(lambda ms: "|".join(ms or [])),
                       publication_time=L._iso(docs["publication_ns"], tz), availability_time=L._iso(docs["availability_ns"], tz))
    docs.drop(columns=["publication_ns", "availability_ns", "text_length"]).to_csv(meta / "documents_time_metadata.csv", index=False)
    info = {"documents": len(docs)}
    if actions_path is not None and actions_path.exists():
        rows = []
        for a in read_jsonl(actions_path):
            r = {k: v for k, v in a.items() if k not in ACTION_TEXT_FIELDS}
            rows.append({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v) for k, v in r.items()})
        pd.DataFrame(rows).to_csv(meta / "actions_time_metadata.csv", index=False)
        info["actions"] = len(rows)
        info["register_used_by_build_equals_file"] = read_jsonl(actions_path) == read_jsonl(interim / "pii" / "actions.jsonl")
    for m in market_names(cfg):
        for src, name in ((interim / "alignment" / f"{m}_document_alignment.csv", f"{m}_document_alignment.csv"),
                          (interim / "decomposition" / f"{m}_grouping.json", f"{m}_grouping.json"),
                          (proc / f"{m}_pii_events.csv", f"{m}_pii_events.csv"),
                          (audit / f"{m}_date_index.csv", f"{m}_date_index.csv")):
            if src.exists():
                copy(src, meta / name)
    return info


def data_manifest(cfg, out: Path) -> int:
    roots = [Path(cfg["paths"][k]) for k in ("raw_dir", "interim_dir", "processed_dir")]
    rows = []
    for root in roots:
        for f in sorted(p for p in root.rglob("*") if p.is_file()):
            rows.append({"path": f.resolve().relative_to(ROOT).as_posix(), "bytes": f.stat().st_size, "sha256": sha256(f)})
    pd.DataFrame(rows).to_csv(out / "data_manifest.csv", index=False)
    return len(rows)


def export_runs(cfg, out: Path, which: str) -> dict:
    src = Path(cfg["paths"]["output_dir"])
    n_files, sums = 0, []
    for f in sorted((src / "forecasts").rglob("*")):
        if f.is_file() and f.suffix in (".csv", ".json"):
            copy(f, out / "forecasts" / f.relative_to(src / "forecasts"))
            n_files += 1
    if which != "none":
        for f in sorted((src / "checkpoints").rglob("seed*.pt")):
            if which == "all" or f.parent.name in CORE:
                rel = f.relative_to(src / "checkpoints")
                copy(f, out / "checkpoints" / rel)
                sums.append(f"{sha256(f)}  {rel.as_posix()}")
        if sums:
            (out / "checkpoints" / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    return {"forecast_files": n_files, "checkpoints": len(sums)}


# ----------------------------------------------------------------------------- summary
def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{nd}g}" if (abs(v) < 1e-3 and v != 0) else f"{v:.{nd}f}"
    return str(v)


def table(rows, header) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(fmt(c) for c in r) + " |" for r in rows]
    return lines + [""]


def summary_md(cfg, out: Path, info: dict) -> str:
    audit = out / "audit"
    pit = load_json(audit / "point_in_time_summary.json")
    leak = load_json(audit / "leakage_checks.json")
    rec = pd.read_csv(audit / "run_records.csv")
    core = pd.read_csv(audit / "core_models_vs_manuscript.csv")
    env = load_json(out / "environment.json")
    markets = list(pit["markets"])
    L_ = ["# Reviewer evidence", "",
          "Generated by `scripts/18_export_evidence.py` from the audit files in `audit/`; every number below is read "
          "from those files.", "",
          f"Information-set rule: {pit['rule']}.", "",
          f"Scope: the {len(rec)} completed runs in `forecasts/`; see the run records for configuration and log provenance.", "",
          "## 1. Point-in-time information audit", "",
          "Per-target tables: `audit/{Market}_point_in_time.csv` (origin, forecast cut-off, target, latest price date, "
          "latest document and latest institutional action in the information set, compliance flags).", ""]
    rows = []
    for m in markets:
        p = pit["markets"][m]["point_in_time"]
        rows.append([m, p["targets"]["train"], p["targets"]["val"], p["targets"]["test"], p["non_compliant_targets"],
                     p["min_hours_between_latest_document_and_cutoff"], p["min_hours_between_latest_action_and_cutoff"]])
    L_ += table(rows, ["Market", "Train targets", "Val targets", "Test targets", "Non-compliant targets",
                       "Min. hours latest document → cut-off", "Min. hours latest action → cut-off"])
    d, dd = pit["documents"], pit["deduplication"]
    L_ += ["Documents:", "",
           f"* {d['documents']} raw documents; availability before publication: {d['availability_before_publication']}; "
           f"availability basis: " + ", ".join(f"`{k}` ({v})" for k, v in d["availability_basis"].items()) + ".",
           f"* Deduplication: {dd['duplicates_removed']} duplicates removed, {dd['retained_documents']} retained; the retained "
           f"record is the earliest available one of its identity: {fmt(dd['retained_is_earliest_available_record_of_each_identity'])}; "
           f"retained records whose market scope comes only from a later duplicate: "
           f"{sum(dd['retained_records_with_market_scope_only_from_later_duplicates'].values())}.", ""]
    rows = []
    for m in markets:
        t, a = pit["markets"][m]["text_arrays"], pit["markets"][m]["pii_arrays"]
        rows.append([m, t["documents_aligned"], t["entry_recomputed_from_availability"], t["arrays_identical"],
                     a["arrays_identical"], a["attributes_identical_with_point_in_time_support_counts"]])
    L_ += ["Model input arrays rebuilt from the raw timestamps and compared with the arrays the models read:", ""]
    L_ += table(rows, ["Market", "Aligned documents", "Entry index recomputed from availability", "Text arrays identical",
                       "PII arrays identical", "PII attributes identical with point-in-time support counts"])
    rows = []
    for m in markets:
        a = pit["markets"][m]["actions"]
        rows.append([m, a["register_entries"], a["entered"], a["entry_is_first_cutoff_on_or_after_availability"],
                     a["each_identifier_enters_once"], a["effective_date_before_availability_date"],
                     a["supporting_documents_available_after_entry"], a["actions_available_before_earliest_supporting_document"],
                     "; ".join(f"{k}: {v}" for k, v in a["reviewed_text_hash"].items()),
                     "; ".join(f"{k}: {v}" for k, v in a["reviewer_type"].items())])
    L_ += ["Institutional actions (`audit/{Market}_actions_timing.csv`):", ""]
    L_ += table(rows, ["Market", "Register entries", "Entered", "Entry = first cut-off ≥ availability", "Enters once",
                       "Effective date before availability", "Supporting documents available after entry",
                       "Entry before earliest supporting document", "Reviewed-text hash vs. source document", "Reviewer type"])

    L_ += ["## 2. Causal decomposition", "",
           "Sampled origins are re-decomposed from their own price prefix (`audit/{Market}_decomposition_recheck.csv`).", ""]
    rows = []
    for m in markets:
        c = leak["markets"][m].get("decomposition")
        if c:
            rows.append([m, f"{c['sampled_origins_reproduced']}/{c['sampled_origins']}", c["max_abs_change_from_future_perturbation"],
                         c["grouping_fit_last_origin_date"], c["stored_centres_minus_training_refit"],
                         c["training_refit_change_when_later_frequencies_perturbed"], c["full_calendar_refit_minus_stored_centres"],
                         c["model_components_minus_causal_archive"]])
    L_ += table(rows, ["Market", "Origins reproduced exactly", "Change when future prices perturbed", "Grouping fitted through",
                       "Stored − training refit centres", "Refit change, later frequencies perturbed",
                       "Full-calendar refit − stored centres", "Model components − causal archive"])
    L_ += ["Call sites of the full-sample (descriptive, Figure 3) decomposition: "
           + "; ".join(f"`{s}`" for s in leak["full_sample_decomposition_call_sites"]) + ".", ""]

    L_ += ["## 3. Partitions and training statistics", "", "Full date indices: `metadata/{Market}_date_index.csv`.", ""]
    rows = []
    for m in markets:
        p, s = leak["markets"][m]["partition"], leak["markets"][m]["statistics"]
        rows.append([m, f"{p['train'][0]} – {p['train'][1]} ({p['train'][2]})", f"{p['val'][0]} – {p['val'][1]} ({p['val'][2]})",
                     f"{p['test'][0]} – {p['test'][1]} ({p['test'][2]})", p["train_before_val"] and p["val_before_test"],
                     s["max_change_when_validation_and_test_rows_perturbed"], s["max_change_when_training_rows_perturbed_control"]])
    L_ += table(rows, ["Market", "Train targets", "Validation targets", "Test targets", "Chronological",
                       "Statistics change, val/test rows perturbed", "Control: training rows perturbed"])

    L_ += ["## 4. Trained checkpoints", "",
           "Each saved core-model checkpoint is reloaded and its saved forecasts are recomputed; for the designated seed, every "
           "input at and after a cut point is replaced by noise (`audit/checkpoint_checks.csv`). Forecasts are computed in "
           "float32 and in batches of different composition, so differences of order 1e-5 (price units) or smaller are rounding.", ""]
    rows = []
    for m in markets:
        for c in leak["markets"][m]["checkpoints"]:
            for cut in c.get("cuts") or [None]:
                rows.append([m, c["model"], c["seed"], c["max_abs_diff_vs_saved_test_forecasts"],
                             cut["cut_date"] if cut else "—", cut["max_abs_change_targets_up_to_cut"] if cut else None,
                             cut["max_abs_change_next_targets"] if cut else None])
    L_ += table(rows, ["Market", "Model", "Seed", "Max. abs. difference, recomputed vs. saved test forecasts", "Cut date",
                       "Max. change, targets up to cut", "Max. change, next 5 targets"])

    multi = rec[rec["training_log_columns"].notna()]
    L_ += ["## 5. Runs, selection and early stopping", "",
           f"* Completed runs: {len(rec)} (`audit/run_records.csv`; forecasts in `forecasts/`). Runs listed in the run log: "
           f"{int(rec['run_log_start'].notna().sum())}.",
           f"* Configuration source: " + "; ".join(f"{k}: {v}" for k, v in rec["configuration"].value_counts().items())
           + f". Runs with an Optuna selection record: {int(rec['optuna_record_present'].sum())}.",
           f"* Runs with a per-epoch training log: {len(multi)}; early-stopping rule replayed on the log matches the saved best "
           f"and stopping epochs: {int(multi['early_stopping_consistent'].sum())}/{len(multi)}; training logs containing a "
           f"test-set quantity: {int(multi['test_quantity_in_training_log'].sum())}.",
           f"* Runs without a per-epoch training log (numerical benchmarks; `run.json` holds the validation RMSE of the "
           f"evaluated configuration and the run log one line per seed): {len(rec) - len(multi)}.",
           f"* Saved validation forecasts reproduce the logged best validation RMSE (forecasts come from the selected checkpoint): "
           f"{int(rec['saved_forecasts_are_selected_checkpoint'].sum())}/{int(rec['saved_forecasts_are_selected_checkpoint'].notna().sum())}.",
           f"* Validation targets precede test targets in every run: {fmt(bool(rec['val_targets_before_test_targets'].all()))}.", ""]

    L_ += ["## 6. Core models: per-seed results and the manuscript", "",
           "Per-seed values: `audit/core_models_per_seed.csv`; all metrics: `audit/core_models_vs_manuscript.csv`.", ""]
    rows = []
    for r in core[core["metric"].isin(["RMSE", "MAE"])].itertuples():
        rows.append([r.market, r.model, r.metric, r.manuscript_value, r.manuscript_source, r.designated_run, r.designated_run_value,
                     f"{r.mean:.3f} ± {r.sd:.3f}" if pd.notna(r.sd) else f"{r.mean:.3f}", f"{r.min:.3f} – {r.max:.3f}", r.runs])
    L_ += table(rows, ["Market", "Model", "Metric", "Manuscript", "Source", "Designated run", "Designated run value",
                       "Mean ± SD over runs", "Range", "Runs"])

    L_ += ["## 7. Environment", "",
           f"* Experiments: Python {env['python']}, torch {env['torch']['version']} (CUDA build: {env['torch']['cuda_build']}), "
           f"{env['platform']}.",
           "* Packages: " + ", ".join(f"{k} {v}" for k, v in env["packages"].items() if v) + ".",
           "* Installed versions that differ from `requirements.txt`: "
           + (", ".join(f"{k} {v['installed']} (requirements.txt {v['requirements.txt']})" for k, v in env["differs_from_requirements_pin"].items()) or "none") + ".",
           "* Text encodings were computed in a separate environment: "
           + ", ".join(f"{k} {v}" for k, v in env["text_encoder_environment"].items() if v is not None) + ".",
           f"* Data files and SHA-256: `data_manifest.csv` ({info['manifest_files']} files); model input arrays: "
           "`processed_arrays_manifest.csv` (verified after a rebuild by `scripts/19_verify_rebuild.py`). "
           "Checkpoints: `checkpoints/SHA256SUMS`.", ""]
    L_ += clean_run_section(out)
    return "\n".join(L_)


def clean_run_section(out: Path) -> list[str]:
    """Level-1 summary from the files of scripts/clean_run.sh copied into clean_environment/."""
    d = out / "clean_environment"
    if not (d / "clean_run.log").exists():
        return []
    log = (d / "clean_run.log").read_text(encoding="utf-8", errors="replace").splitlines()
    pick = lambda key: next((x.strip() for x in log if key in x), "—")
    L_ = ["## 8. Level 1: clean environment", "",
          "`scripts/clean_run.sh` in a new virtual environment with only `requirements.txt` installed "
          "(`clean_environment/install.log` lists the installed packages; `clean_environment/clean_run.log` is the full log).", "",
          f"* Interpreter: {pick('executable')}.",
          f"* Unit tests: {pick(' passed')}.",
          f"* Synthetic end-to-end run: {'all stages completed' if any('=== stage: checks' in x for x in log) else 'not completed'}.",
          f"* Exit status: {next((x.strip() for x in log if x.startswith('exit ')), '—')}."]
    if (d / "NOTE.txt").exists():
        L_ += ["* Note: " + " ".join((d / "NOTE.txt").read_text(encoding="utf-8").split())]
    ver = d / "rebuild_verification.json"
    if not ver.exists():
        line = next((x.strip() for x in log if "arrays identical" in x), None)
        if line:
            L_ += [f"* Rebuild of the model inputs from the raw files (stages 01–05), as logged: `{line.split(' -> ')[0]}`."]
    else:
        v = load_json(ver)
        L_ += [f"* Model inputs rebuilt from the raw files by stages 01–05: {v['arrays_identical']}/{v['arrays_compared']} arrays and "
               f"{v['files_identical']}/{v['files_compared']} alignment, component, grouping and PII-event files byte-identical to "
               "the published hashes."]
        for x in v["arrays_differing"]:
            s = x.get("summary_differences") or {}
            L_ += [f"  * `{x['file']}:{x['array']}` differs in its bytes; same shape and dtype: {fmt(x['same_shape_and_dtype'])}; "
                   + ", ".join(f"{k} {val:.2g}" for k, val in s.items()) + "."]
        for x in v["files_differing"]:
            L_ += [f"  * `{x['file']}` differs; largest numerical difference from the published copy: "
                   f"{fmt(x.get('max_abs_numerical_difference'))}."]
    chk = d / "leakage_checks.json"
    if chk.exists():
        c = [x for r in load_json(chk)["markets"].values() for x in r["checkpoints"]]
        cuts = [k for x in c for k in x.get("cuts") or []]
        line = (f"* Saved checkpoints on the rebuilt inputs: {len(c)} checkpoints; max. abs. difference between recomputed and "
                f"saved forecasts {max(x['max_abs_diff_vs_saved_test_forecasts'] for x in c):.2g} (test), "
                f"{max(x['max_abs_diff_vs_saved_val_forecasts'] for x in c):.2g} (validation)")
        if cuts:
            line += (f"; inputs after cut points replaced by noise: max. change up to the cut "
                     f"{max(k['max_abs_change_targets_up_to_cut'] for k in cuts):.2g}, smallest max. change over the next 5 targets "
                     f"{min(k['max_abs_change_next_targets'] for k in cuts):.3f}")
        L_ += [line + "."]
    return L_ + [""]


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--out", default=str(ROOT / "evidence"))
    p.add_argument("--actions", default=str(ROOT / "data" / "raw" / "actions_reviewed.jsonl"))
    p.add_argument("--logs", nargs="*", default=[])
    p.add_argument("--checkpoints", choices=["core", "all", "none"], default="core")
    p.add_argument("--clean-run", default=None, help="folder with clean_run.log, install.log and the rebuild reports")
    a = p.parse_args()
    cfg = config_from(a)
    out = ensure_dir(a.out)
    info = export_metadata(cfg, out, Path(a.actions) if a.actions else None)
    src_audit = Path(cfg["paths"]["output_dir"]) / "audit"
    for f in sorted(src_audit.glob("*")):
        if f.is_file() and not f.name.endswith("_date_index.csv"):
            copy(f, out / "audit" / f.name)
    info.update(export_runs(cfg, out, a.checkpoints))
    for log in a.logs:
        if Path(log).exists():
            copy(Path(log), out / "logs" / Path(log).name)
    if a.clean_run:
        for f in sorted(Path(a.clean_run).glob("*")):
            if f.is_file():
                copy(f, out / "clean_environment" / f.name)
    info["manifest_files"] = data_manifest(cfg, out)
    L.array_hashes(cfg["paths"]["processed_dir"]).to_csv(out / "processed_arrays_manifest.csv", index=False)
    save_json(environment(cfg), out / "environment.json")
    (out / "EVIDENCE.md").write_text(summary_md(cfg, out, info), encoding="utf-8")
    save_json(info, out / "export_info.json")
    LOG.info("evidence -> %s  %s", out, info)
