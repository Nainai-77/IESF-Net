"""Pipeline stages.  Every numbered script in ``scripts/`` calls exactly one stage."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import market_names
from .data.corpus import align_documents, deduplicate, load_corpus
from .data.dataset import MarketData
from .data.prices import check_test_start, load_prices, make_partition, numeric_features, price_file
from .utils import ensure_dir, get_logger, load_json, read_jsonl, save_json, write_jsonl

LOG = get_logger("iesf.pipeline")


def _p(cfg, key) -> Path:
    return Path(cfg["paths"][key])


def _markets(cfg, markets=None) -> list[str]:
    return list(markets) if markets else market_names(cfg)


def _write_table(df: pd.DataFrame, out_dir: Path, name: str, floatfmt: str = "{:.4f}") -> None:
    ensure_dir(out_dir)
    df.to_csv(out_dir / f"{name}.csv", index=False)
    if df.empty:
        return
    cols = list(df.columns)
    fmt = lambda v: floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    (out_dir / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================== 0. synthetic data
def stage_synthetic(cfg) -> None:
    from .data.synthetic import write_synthetic

    write_synthetic(cfg)
    LOG.info("synthetic raw data written to %s", cfg["paths"]["raw_dir"])


# =============================================================================== 1. corpus
def stage_corpus(cfg, markets=None) -> None:
    al = cfg["alignment"]
    raw, interim = _p(cfg, "raw_dir"), ensure_dir(_p(cfg, "interim_dir"))
    docs = deduplicate(load_corpus(raw / "documents.jsonl", al["timezone"], al["date_only_time"]))
    docs.to_pickle(interim / "documents_clean.pkl")
    summary = {}
    for mk in _markets(cfg, markets):
        prices = load_prices(price_file(raw, mk))
        aligned = align_documents(docs, mk, prices["date"], al["market_cutoff"], al["timezone"],
                                  al.get("pre_calendar_days", 1))
        ensure_dir(interim / "alignment")
        aligned[["document_id", "origin_index", "origin_date", "availability_ts", "text_identity"]].to_csv(
            interim / "alignment" / f"{mk}_document_alignment.csv", index=False)
        T = len(prices)
        has = np.zeros(T, dtype=bool)
        has[aligned["origin_index"].to_numpy()] = True
        last_doc = int(aligned["origin_index"].max()) if len(aligned) else -1
        summary[mk] = {                                           # Table 5 inventory
            "documents": int(len(aligned)), "dates_with_text": int(has[: last_doc + 1].sum()),
            "dates_without_new_text": int((~has[: last_doc + 1]).sum()), "inventory_calendar": last_doc + 1,
            "price_observations_after_inventory": int(T - last_doc - 1), "price_total": T,
            "last_document_date": str(prices["date"].iloc[last_doc].date()) if last_doc >= 0 else None,
        }
    save_json(summary, interim / "corpus_summary.json")
    LOG.info("corpus: %d deduplicated documents; inventory %s", len(docs), summary)


# =============================================================================== 2. frozen encoders
def stage_encode(cfg, mock: bool = False, chunk: int = 2000) -> None:
    import torch

    from .text.encoders import FrozenSemanticEncoder, FrozenSentimentEncoder, HashingSemanticEncoder, LexiconSentimentEncoder
    from .utils import get_device

    interim = _p(cfg, "interim_dir")
    docs = pd.read_pickle(interim / "documents_clean.pkl")
    out = ensure_dir(interim / "encoded")
    tcfg = cfg["text"]
    if mock:
        sem_enc, sent_enc = HashingSemanticEncoder(), LexiconSentimentEncoder()
    else:
        device = get_device(tcfg.get("device", "auto"))
        sem_enc = FrozenSemanticEncoder.from_pretrained(tcfg["semantic_model"], tcfg["semantic_revision"], device,
                                                        tcfg["max_content_tokens"], tcfg["batch_size"])
        sent_enc = FrozenSentimentEncoder.from_pretrained(tcfg["sentiment_model"], tcfg["sentiment_revision"], device,
                                                          tcfg["sentiment_labels"], tcfg["max_content_tokens"], tcfg["batch_size"])
    titles, bodies = docs["title"].tolist(), docs["body"].tolist()
    parts_s, parts_p = [], []
    for s in range(0, len(docs), chunk):                          # resumable chunked encoding
        fs, fp = out / f"part_{s:08d}_sem.npy", out / f"part_{s:08d}_sent.npy"
        if not (fs.exists() and fp.exists()):
            with torch.no_grad():
                np.save(fs, sem_enc.encode(titles[s:s + chunk], bodies[s:s + chunk]))
                np.save(fp, sent_enc.encode(titles[s:s + chunk], bodies[s:s + chunk]))
            LOG.info("encoded documents %d-%d / %d", s, min(s + chunk, len(docs)), len(docs))
        parts_s.append(np.load(fs)); parts_p.append(np.load(fp))
    np.save(out / "semantic.npy", np.concatenate(parts_s))
    np.save(out / "sentiment.npy", np.concatenate(parts_p))
    save_json(docs["document_id"].tolist(), out / "document_ids.json")
    save_json({"mock_encoders": mock, "semantic_model": tcfg["semantic_model"], "semantic_revision": tcfg["semantic_revision"],
               "sentiment_model": tcfg["sentiment_model"], "sentiment_revision": tcfg["sentiment_revision"],
               "sentiment_order": ["neutral", "positive", "negative"], "n_documents": len(docs)},
              out / "encoding_manifest.json")


def stage_import_encoded(cfg, src_dir: str) -> None:
    """Import document encodings computed elsewhere with the same frozen encoders.

    ``src_dir`` must hold document_ids.json, semantic.npy [n, 768], sentiment.npy [n, 3]
    (neutral, positive, negative) and encoding_manifest.json whose model revisions match
    configs/default.yaml.  Every cleaned document must have an encoding.
    """
    src, interim = Path(src_dir), _p(cfg, "interim_dir")
    man = load_json(src / "encoding_manifest.json")
    tcfg = cfg["text"]
    revs = {"semantic": tcfg["semantic_revision"], "sentiment": tcfg["sentiment_revision"]}
    for key, rev in revs.items():
        got = (man.get(key) or {}).get("model_revision")
        if got != rev:
            raise ValueError(f"{key} revision {got} differs from configured {rev}")
        if not man[key].get("completed_documents") == man[key].get("total_documents"):
            raise ValueError(f"{key} encoding incomplete in {src}")
    ids = load_json(src / "document_ids.json")
    sem, sent = np.load(src / "semantic.npy"), np.load(src / "sentiment.npy")
    if sem.shape != (len(ids), 768) or sent.shape != (len(ids), 3):
        raise ValueError("encoded arrays do not match document_ids.json")
    if np.any(sent < 0) or np.abs(sent.sum(1) - 1).max() > 1e-4:
        raise ValueError("sentiment rows are not probability vectors")
    docs = pd.read_pickle(interim / "documents_clean.pkl")
    row = {d: i for i, d in enumerate(ids)}
    missing = [d for d in docs["document_id"] if d not in row]
    if missing:
        raise ValueError(f"{len(missing)} cleaned documents have no imported encoding, e.g. {missing[:3]}")
    take = np.array([row[d] for d in docs["document_id"]])
    out = ensure_dir(interim / "encoded")
    np.save(out / "semantic.npy", sem[take].astype(np.float32))
    np.save(out / "sentiment.npy", sent[take].astype(np.float32))
    save_json(docs["document_id"].tolist(), out / "document_ids.json")
    save_json({"imported_from": str(src), "source_manifest": man, "mock_encoders": False,
               "sentiment_order": ["neutral", "positive", "negative"], "n_documents": len(docs)},
              out / "encoding_manifest.json")
    LOG.info("imported encodings for %d documents from %s", len(docs), src)


# =============================================================================== 3. institutional actions
def stage_pii(cfg, actions_path: str | None = None) -> None:
    from .text.pii_extractor import candidate_type_counts, consolidate, extract_corpus

    interim = _p(cfg, "interim_dir")
    out = ensure_dir(interim / "pii")
    if actions_path:
        actions = read_jsonl(actions_path)                        # externally reviewed register
        LOG.info("using %d external action records from %s", len(actions), actions_path)
    else:
        docs = pd.read_pickle(interim / "documents_clean.pkl")
        cands = extract_corpus(docs, market_names(cfg))
        write_jsonl(cands.to_dict("records"), out / "candidates.jsonl")
        actions = consolidate(cands)
        save_json({"candidates": len(cands), "actions": len(actions), "candidate_types": candidate_type_counts(cands)},
                  out / "extraction_summary.json")
        LOG.info("extractor: %d candidates -> %d canonical actions", len(cands), len(actions))
    write_jsonl(actions, out / "actions.jsonl")


# =============================================================================== 4. causal decomposition
def stage_decompose(cfg, markets=None) -> None:
    from .decomposition.ceemdan import build_origin_archive
    from .decomposition.grouping import component_archive, fit_grouping

    raw, out = _p(cfg, "raw_dir"), ensure_dir(_p(cfg, "interim_dir") / "decomposition")
    for mk in _markets(cfg, markets):
        prices = load_prices(price_file(raw, mk))
        part = make_partition(len(prices), cfg["markets"][mk], int(cfg["reserve"]))
        arch_path = out / f"{mk}_origin_archive.npz"
        if arch_path.exists():
            from .decomposition.ceemdan import OriginArchive
            arch = OriginArchive.load(arch_path)
        else:
            LOG.info("%s: origin-wise CEEMDAN over %d origins", mk, len(prices))
            arch = build_origin_archive(prices["close"].to_numpy(), cfg["decomposition"],
                                        cache_dir=out / f"{mk}_origin_parts")   # resumable chunks
            arch.save(arch_path)
        grouping = fit_grouping(arch, part.train_end, cfg["decomposition"]["grouping"])   # training boundary only
        R = component_archive(arch, grouping)
        df = pd.DataFrame({"date": prices["date"].dt.strftime("%Y-%m-%d"), "close": prices["close"]})
        for k in range(R.shape[1]):
            df[f"scale_{k + 1}"] = R[:, k]
        df["reconstruction_error"] = R.sum(1) - prices["close"].to_numpy()
        df.to_csv(out / f"{mk}_components.csv", index=False)
        save_json({"market": mk, **grouping.to_dict(), "train_end": part.train_end,
                   "max_abs_reconstruction_error": float(np.abs(df["reconstruction_error"]).max()),
                   "mean_modes_per_origin": float(arch.mask.sum(1).mean())}, out / f"{mk}_grouping.json")


# =============================================================================== 5. model-ready dataset
def stage_build(cfg, markets=None) -> None:
    from .text.daily import daily_count, daily_csi, daily_mean, doc_availability
    from .text.lexical import LexicalEncoder
    from .text.pii_register import build_pii_arrays

    raw, interim, proc = _p(cfg, "raw_dir"), _p(cfg, "interim_dir"), ensure_dir(_p(cfg, "processed_dir"))
    docs = pd.read_pickle(interim / "documents_clean.pkl").set_index("document_id")
    ids = load_json(interim / "encoded" / "document_ids.json")
    row_of = {d: i for i, d in enumerate(ids)}
    sem_all = np.load(interim / "encoded" / "semantic.npy", mmap_mode="r")
    sent_all = np.load(interim / "encoded" / "sentiment.npy")
    actions = read_jsonl(interim / "pii" / "actions.jsonl")
    for mk in _markets(cfg, markets):
        prices = load_prices(price_file(raw, mk))
        T = len(prices)
        part = make_partition(T, cfg["markets"][mk], int(cfg["reserve"]))
        check_test_start(prices["date"], part, cfg["markets"][mk].get("test_start"), mk)
        comp = pd.read_csv(interim / "decomposition" / f"{mk}_components.csv")
        assert (pd.to_datetime(comp["date"]).to_numpy() == prices["date"].to_numpy()).all(), "component dates mismatch"
        al = pd.read_csv(interim / "alignment" / f"{mk}_document_alignment.csv")
        rows = np.array([row_of[d] for d in al["document_id"]], dtype=np.int64)
        t_idx = al["origin_index"].to_numpy(np.int64)
        # lexical model fitted on documents available within the training period of this market
        texts = [f"{docs.at[d, 'title']}\n{docs.at[d, 'body']}" for d in al["document_id"]]
        lcfg = cfg["text"]["lexical"]
        n_train_docs = int((t_idx < part.train_end).sum())
        lex_enc = LexicalEncoder(lcfg["analyzer"], lcfg["ngram_range"], lcfg["max_features"],
                                 lcfg["min_df"] if n_train_docs >= 1000 else 1, lcfg["svd_dim"], lcfg["seed"])
        lex_enc.fit([t for t, i in zip(texts, t_idx) if i < part.train_end])
        lex_docs = lex_enc.transform(texts)
        pii = build_pii_arrays(actions, mk, prices["date"], cfg)
        arrays = {
            "close": prices["close"].to_numpy(np.float64),
            "X_raw": numeric_features(prices),
            "R": comp[[f"scale_{k + 1}" for k in range(4)]].to_numpy(np.float64),
            "sem": daily_mean(np.asarray(sem_all[rows]), t_idx, T),
            "csi": daily_csi(sent_all[rows], t_idx, T),
            "lex": daily_mean(lex_docs, t_idx, T),
            "count": daily_count(t_idx, T),
            "doc_avail": doc_availability(t_idx, T),
            **{k: v for k, v in pii.items() if k != "events"},
        }
        md = MarketData(mk, prices["date"].to_numpy().astype("datetime64[D]"), arrays, part)
        md.save(proc / f"{mk}.npz")
        pii["events"].to_csv(proc / f"{mk}_pii_events.csv", index=False)
        LOG.info("%s: T=%d, text dates=%d, PII event dates=%d, train/val/test=%d/%d/%d", mk, T,
                 int(arrays["doc_avail"].sum()), int(arrays["pii_new"].max(1).sum()), part.n_train, part.n_val, part.n_test)


# =============================================================================== 6. descriptive analysis
def stage_descriptive(cfg) -> None:
    from .decomposition.ceemdan import decompose_prefix
    from .decomposition.grouping import FrequencyGrouping, group_full_series
    from .evaluation.descriptive import table3
    from .viz import figures as V

    V.setup_style()
    raw, out = _p(cfg, "raw_dir"), Path(cfg["paths"]["output_dir"])
    closes, fig5, fig3, t2 = {}, {}, {}, []
    for mk in market_names(cfg):
        path = price_file(raw, mk)
        if not path.exists():
            continue
        prices = load_prices(path)
        closes[mk] = prices["close"].to_numpy()
        part = make_partition(len(prices), cfg["markets"][mk], int(cfg["reserve"]))
        test = part.targets("test")
        y = prices["close"].to_numpy()
        t2.append({"Market": mk, "Product": cfg["markets"][mk].get("product", ""), "Observations": len(prices),
                   "Date range": f"{prices['date'].iloc[0].date()} – {prices['date'].iloc[-1].date()}",
                   "Train / val. / test targets": f"{part.n_train} / {part.n_val} / {part.n_test}",
                   "Test start": str(prices["date"].iloc[part.val_end].date()),
                   "Unchanged test closes (%)": 100.0 * float(np.mean(y[test] == y[test - 1]))})   # §4.2
        if mk in cfg["primary_markets"]:
            fig5[mk] = (prices["date"], prices["close"], prices["date"].iloc[part.train_end], prices["date"].iloc[part.val_end])
            gfile = _p(cfg, "interim_dir") / "decomposition" / f"{mk}_grouping.json"
            if gfile.exists():                                     # full-sample descriptive decomposition only
                dcfg = {**cfg["decomposition"], "history_cap": len(prices)}
                imfs, res = decompose_prefix(prices["close"].to_numpy(), int(cfg["decomposition"]["seed_base"]), dcfg)
                comps = group_full_series(imfs, res, FrequencyGrouping(np.array(load_json(gfile)["log_centres"])))
                fig3[mk] = (prices["date"], prices["close"], comps)
    _write_table(pd.DataFrame(t2), out / "tables", "Table2_samples_partitions")
    _write_table(table3(closes), out / "tables", "Table3_descriptive_unit_root")
    inv = _p(cfg, "interim_dir") / "corpus_summary.json"
    if inv.exists():                                                 # Table 5 text inventory, computed from the corpus
        _write_table(pd.DataFrame([{"Market": m, **v} for m, v in load_json(inv).items()]), out / "tables",
                     "Table5_text_inventory")
    if fig5:
        V.fig5_prices(fig5, out / "figures")
    if fig3:
        V.fig3_decomposition(fig3, out / "figures")


# =============================================================================== 7. numerical benchmarks
def stage_benchmarks(cfg, markets=None, models=None, tune: bool = True) -> None:
    from .benchmarks.runner import evaluate_benchmark, selected_params, tune_benchmark

    models = models or cfg["experiments"]["families"]["benchmarks"]
    for mk in markets or cfg["primary_markets"]:
        for name in models:
            if tune:
                tune_benchmark(cfg, mk, name)
            evaluate_benchmark(cfg, mk, name, selected_params(cfg, mk, name, "tuned" if tune else "paper"))


# =============================================================================== 8. IESF-Net selection
def stage_tune(cfg, markets=None, variants=("IESF-Net",)) -> None:
    from .experiments.runner import tune_variant

    for mk in markets or cfg["primary_markets"]:
        for v in variants:
            res = tune_variant(cfg, mk, v)
            LOG.info("%s | %s selected %s (val RMSE %.4f)", mk, v, res["best_params"], res["best_value"])


# =============================================================================== 9. experiment suites
def stage_experiments(cfg, suites=None, markets=None, force: bool = False) -> None:
    from .experiments.runner import run_dir, train_and_forecast

    ecfg = cfg["experiments"]
    core = {"Numeric", "IESF-Net", "Bidirectional"}
    for suite in suites or list(ecfg["suites"]):
        scfg = ecfg["suites"][suite]
        sm = scfg["markets"]
        for mk in markets or (market_names(cfg, sm) if isinstance(sm, str) else sm):
            if mk not in cfg["markets"]:
                continue
            for v in scfg["variants"]:
                seeds = cfg["seeds"]["evaluation"] if (scfg.get("repeat_seeds") and v in core) else [cfg["seeds"]["designated"]]
                for seed in seeds:
                    done = run_dir(cfg, "forecasts", mk, v, seed) / "test_predictions.csv"
                    if done.exists() and not force:
                        continue
                    train_and_forecast(cfg, mk, v, int(seed))


# =============================================================================== 10. statistics / tables
def stage_statistics(cfg) -> None:
    from .evaluation import tables as TB
    from .evaluation.forecasts import common_index, squared_losses
    from .evaluation.metrics import cumulative_gain

    out = Path(cfg["paths"]["output_dir"]) / "tables"
    prim = [m for m in cfg["primary_markets"] if m in cfg["markets"]]
    suites = cfg["experiments"]["suites"]
    t7 = TB.table7(cfg, prim)
    _write_table(t7, out, "Table7_raw")
    _write_table(TB.format_table7(t7) if not t7.empty else t7, out, "Table7_performance")
    _write_table(pd.concat([TB.table8(cfg, m) for m in prim] or [pd.DataFrame()]), out, "Table8_benchmarks_vs_persistence")
    for m in prim:
        _write_table(TB.tables9_10(cfg, m), out, f"Table9_10_controls_{m}")
        _write_table(TB.tables_c(cfg, m), out, f"TableC_benchmarks_{m}")
    _write_table(pd.concat([TB.table11(cfg, m) for m in prim] or [pd.DataFrame()]), out, "Table11_subgroups")
    _write_table(TB.variant_metrics(cfg, prim, suites["representation"]["variants"]), out, "Table12_representations")
    _write_table(TB.variant_metrics(cfg, prim, suites["fusion"]["variants"]), out, "Table13_fusion")
    abl = TB.variant_metrics(cfg, prim, ["IESF-Net"] + suites["ablation"]["variants"])
    _write_table(TB.with_increase(abl) if not abl.empty else abl, out, "Table14_ablations")
    win = TB.variant_metrics(cfg, prim, suites["window"]["variants"])
    _write_table(TB.with_increase(win) if not win.empty else win, out, "Table15_windows")
    _write_table(TB.table16(cfg, list(cfg["markets"])), out, "Table16_cross_market")
    _write_table(TB.tableA2(cfg, prim), out, "TableA2_benchmark_configurations")
    _write_table(TB.core_repeats(cfg, list(cfg["markets"])), out, "Core_repeat_evaluations")
    # cumulative normalised squared-loss gains (Figures 8b and 9b)
    rows = []
    for m in prim:
        sq_f = squared_losses(cfg, m, "IESF-Net")
        for ref in ("Numeric", "Bidirectional", "Fusion-FeatureCond"):
            sq_r = squared_losses(cfg, m, ref)
            if sq_f is None or sq_r is None:
                continue
            ix = common_index(sq_f, sq_r)
            for k, g in zip(ix, cumulative_gain(sq_r.loc[ix], sq_f.loc[ix])):
                rows.append({"Market": m, "Reference": ref, "target_index": k, "G": g})
    pd.DataFrame(rows).to_csv(out / "cumulative_gains.csv", index=False)


# =============================================================================== 11. interpretability
def stage_interpret(cfg, markets=None) -> None:
    from .evaluation.interpret import attention_maps, collect_aux, conditioning_contrast, routing_frame, save_npz
    from .experiments.runner import load_trained

    out = ensure_dir(Path(cfg["paths"]["output_dir"]) / "interpretability")
    for mk in markets or cfg["primary_markets"]:
        md, data, model = load_trained(cfg, mk, "IESF-Net", int(cfg["seeds"]["designated"]))
        idx = md.partition.targets("test")
        aux = collect_aux(model, data, idx)
        save_npz(attention_maps(aux), out / f"{mk}_attention.npz")
        routing_frame(aux, idx, md.dates, model.constructs, lookbacks=model.cfg.lookbacks).to_csv(
            out / f"{mk}_routing.csv", index=False)
        save_npz(conditioning_contrast(model, data, idx), out / f"{mk}_contrast.npz")
        if hasattr(model, "pii_tam"):
            save_json(dict(zip(cfg["pii"]["categories"], model.pii_tam.half_lives().tolist())), out / f"{mk}_half_lives.json")


# =============================================================================== 12. figures
def stage_figures(cfg) -> None:
    from .evaluation.forecasts import common_index, load_series, squared_losses
    from .viz import figures as V

    V.setup_style()
    root = Path(cfg["paths"]["output_dir"])
    tdir, fdir, idir = root / "tables", ensure_dir(root / "figures"), root / "interpretability"
    prim = [m for m in cfg["primary_markets"] if m in cfg["markets"]]
    read = lambda n: pd.read_csv(tdir / f"{n}.csv") if (tdir / f"{n}.csv").exists() and (tdir / f"{n}.csv").stat().st_size > 1 else pd.DataFrame()
    for j, m in enumerate(prim):                                         # Figures 6 and 7
        frames = {v: load_series(cfg, m, v) for v in ("Numeric", "Bidirectional", "IESF-Net")}
        frames = {k: v for k, v in frames.items() if v is not None}
        if frames:
            V.fig_forecasts(m, frames, fdir, f"Figure{6 + j}_forecasts_{m}", window=int(cfg["evaluation"]["rolling_window"]))
    t7 = read("Table7_raw")
    if not t7.empty:                                                     # Figure 8
        cum = {}
        bench = [b for b in cfg["experiments"]["families"]["benchmarks"] if b != "Persistence"]
        for m in prim:
            val = {}
            for b in bench:
                best = root / "tuning" / m / b / "best.json"
                runs = sorted((root / "forecasts" / m / b).glob("seed*/run.json"))
                if best.exists():
                    val[b] = load_json(best)["best_value"]
                elif runs:
                    val[b] = float(np.mean([load_json(r)["val_rmse"] for r in runs]))
            sq_p = squared_losses(cfg, m, "Persistence")
            if val and sq_p is not None:
                sel = min(val, key=val.get)
                sq_m = squared_losses(cfg, m, sel, average_runs=True)
                ix = common_index(sq_p, sq_m)
                cum[m] = (sel, 100 * np.cumsum(sq_p.loc[ix] - sq_m.loc[ix]) / sq_p.loc[ix].sum())
        V.fig8_benchmarks(t7, cum, fdir)
    paired = {m: read(f"Table9_10_controls_{m}") for m in prim}
    paired = {m: d for m, d in paired.items() if not d.empty}
    cg = read("cumulative_gains")
    if paired and not cg.empty:                                          # Figure 9
        names = {"Numeric": "Numeric", "Bidirectional": "Bidirectional", "Fusion-FeatureCond": "Feature conditioning"}
        cum = {m: {names[r]: g["G"].to_numpy() for r, g in cg[cg["Market"] == m].groupby("Reference", sort=False)} for m in paired}
        V.fig9_paired(paired, cum, fdir)
    t11 = read("Table11_subgroups")
    if not t11.empty:
        V.fig10_subgroups(t11, fdir)
    rep, fus = read("Table12_representations"), read("Table13_fusion")
    if not rep.empty or not fus.empty:
        V.fig11_rep_fusion(rep, fus, fdir)
    abl, win = read("Table14_ablations"), read("Table15_windows")
    if not abl.empty or not win.empty:
        V.fig12_ablation_window(abl[abl["Variant"] != "IESF-Net"] if not abl.empty else abl, win, fdir)
    t16 = read("Table16_cross_market")
    if not t16.empty:
        V.fig13_cross_market(t16, fdir)
    maps, routes = {}, {}
    for m in prim:                                                       # Figures 14-16
        f = idir / f"{m}_attention.npz"
        if f.exists():
            z = np.load(f)
            maps[m] = {(int(k[1]) - 1, "N→T" if k.endswith("n2t") else "T→N"): z[k] for k in z.files}
        f = idir / f"{m}_contrast.npz"
        if f.exists():
            z = np.load(f)
            con = {}
            for k in z.files:
                tag, sub = k.rsplit("_", 1)
                key = (int(tag[1]) - 1, "N→T" if tag.endswith("n2t") else "T→N")
                con.setdefault(key, {})[sub] = z[k]
            V.fig15_contrast(con, m, fdir)
        f = idir / f"{m}_routing.csv"
        if f.exists():
            routes[m] = pd.read_csv(f)
    if maps:
        V.fig14_attention(maps, fdir)
    if routes:
        V.fig16_routing(routes, fdir)
    LOG.info("figures written to %s", fdir)


# =============================================================================== 13. audit / 14. checks
def stage_audit(cfg, mode: str = "sample", sheet: str | None = None) -> None:
    from .text.audit import sample_audit, summarise_audit

    interim, out = _p(cfg, "interim_dir"), ensure_dir(Path(cfg["paths"]["output_dir"]) / "audit")
    if mode == "sample":
        cands = pd.DataFrame(read_jsonl(interim / "pii" / "candidates.jsonl"))
        docs = pd.read_pickle(interim / "documents_clean.pkl")
        s = sample_audit(cands, docs, int(cfg["pii"]["audit_per_type"]), int(cfg["pii"]["audit_seed"]))
        s.to_csv(out / "audit_sheet.csv", index=False, encoding="utf-8-sig")
        LOG.info("audit sheet with %d documents written to %s", len(s), out / "audit_sheet.csv")
    else:
        _write_table(summarise_audit(pd.read_csv(sheet or out / "audit_sheet.csv", encoding="utf-8-sig").fillna("")),
                     Path(cfg["paths"]["output_dir"]) / "tables", "Table6_extraction_audit")


def stage_checks(cfg) -> None:
    from .evaluation.checks import forecast_alignment_check, partition_check, reconstruction_check, timing_check
    from .evaluation.forecasts import load_series

    report = {}
    for mk in market_names(cfg):
        f = _p(cfg, "processed_dir") / f"{mk}.npz"
        if not f.exists():
            continue
        md = MarketData.load(f)
        rep = {"partition": partition_check(md, cfg["markets"][mk], int(cfg["reserve"])),
               "reconstruction": reconstruction_check(md), "timing": {}, "alignment": {}}
        for model in ("IESF-Net", "Numeric", "Bidirectional"):
            df = load_series(cfg, mk, model)
            if df is not None:
                rep["timing"][model] = timing_check(df)
                rep["alignment"][model] = forecast_alignment_check(df, md, "test")
        report[mk] = rep
    save_json(report, Path(cfg["paths"]["output_dir"]) / "consistency_checks.json")
    LOG.info("consistency checks: %s", json.dumps(report, ensure_ascii=False, default=str)[:2000])
