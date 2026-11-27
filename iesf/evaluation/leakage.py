"""Point-in-time information audit and leakage checks.

Each check re-derives what it audits from the files the experiments read, instead of
trusting the stage that wrote them:

* ``point_in_time_table``: for every target, the forecast cut-off of its origin and the
  latest price, document and institutional action in the information set;
* ``document_checks`` and ``action_checks``: record-level timing rules (availability never
  before publication, entry at the first cut-off on or after availability, entry by
  availability rather than effective date, supporting documents available at entry);
* ``text_array_check`` and ``pii_array_check``: the daily arrays consumed by the models are
  rebuilt from the raw timestamps and compared with the processed arrays;
* ``decomposition_check``: sampled origins are re-decomposed from their own price prefix,
  with and without perturbed future prices, and the grouping is refitted on training origins;
* ``statistics_check``: preprocessing statistics do not move when validation and test rows
  are perturbed;
* ``partition_audit``: the full date index of every market and the ordering assertions;
* ``model_causality_check``: a trained checkpoint is re-run with every input at and after a
  cut point replaced by noise; forecasts of earlier targets must not change;
* ``run_records``: early-stopping replay, validation selection and test metrics of every
  completed run, from its saved training log and forecasts.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.alignment import _tz, assign_to_calendar, market_cutoffs, to_timestamp
from ..data.dataset import ARRAY_KEYS, MarketData
from ..data.prices import load_prices, make_partition, price_file
from ..utils import load_json, read_jsonl

NS_MIN = np.iinfo(np.int64).min                  # missing timestamp
DOC_FIELDS = ["document_id", "source_type", "source_url", "publish_datetime", "date_only_publish_flag",
              "availability", "availability_basis", "markets", "text_sha256"]
ACTION_HASH_FIELDS = ("source_body_sha256", "body_sha256", "source_text_sha256")


# ----------------------------------------------------------------------------- helpers
def _dirs(cfg) -> tuple[Path, Path, Path, Path]:
    p = cfg["paths"]
    return Path(p["raw_dir"]), Path(p["interim_dir"]), Path(p["processed_dir"]), Path(p["output_dir"])


def _calendar(cfg, market: str):
    """Prices, partition and the cut-off (int64 ns, UTC) of every observation of one market."""
    prices = load_prices(price_file(cfg["paths"]["raw_dir"], market))
    part = make_partition(len(prices), cfg["markets"][market], int(cfg["reserve"]))
    al = cfg["alignment"]
    return prices, part, market_cutoffs(prices["date"], al["market_cutoff"], al["timezone"])


def _ns(values) -> np.ndarray:
    """int64 ns (UTC) of timestamp-like values; NS_MIN where missing."""
    s = pd.to_datetime(pd.Series(list(values), dtype=object), utc=True)
    out = np.full(len(s), NS_MIN, dtype=np.int64)
    ok = s.notna().to_numpy()
    out[ok] = s[ok].astype("int64").to_numpy()
    return out


def _iso(ns, tz: str) -> list:
    """Local ISO-8601 strings of int64 ns values (None where missing)."""
    ns = np.asarray(ns, dtype=np.int64)
    out = [None] * len(ns)
    ok = np.flatnonzero(ns != NS_MIN)
    if len(ok):
        local = pd.to_datetime(ns[ok], utc=True).tz_convert(_tz(tz))
        for i, v in zip(ok, local):
            out[i] = v.isoformat()
    return out


def _running_latest(origin: np.ndarray, ns: np.ndarray, ids: np.ndarray, T: int):
    """Latest timestamp, and its record, among the records entering at or before each observation."""
    best_ns, best_id = np.full(T, NS_MIN, dtype=np.int64), np.full(T, None, dtype=object)
    for o, n, i in zip(origin, ns, ids):
        if 0 <= o < T and n >= best_ns[o]:
            best_ns[o], best_id[o] = n, i
    cur_ns, cur_id = NS_MIN, None
    for j in range(T):
        if best_ns[j] > cur_ns:
            cur_ns, cur_id = best_ns[j], best_id[j]
        best_ns[j], best_id[j] = cur_ns, cur_id
    return best_ns, best_id


def _max_abs(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b))) if a.size else 0.0


# ----------------------------------------------------------------------------- raw records
def document_metadata(cfg) -> pd.DataFrame:
    """Timing metadata of every raw document (no title or body), with the publication and
    availability timestamps derived exactly as in ``iesf.data.corpus.load_corpus``.

    ``content_hashes`` holds the SHA-256 of the stored text versions (raw body, cleaned body,
    title + body, and the record's own ``text_sha256``) so that register entries can be matched
    to the document they were reviewed from without redistributing the text."""
    from ..data.corpus import clean_text, normalised_identity
    from ..utils import sha256_text

    tz, eod = cfg["alignment"]["timezone"], cfg["alignment"]["date_only_time"]
    rows = []
    with open(Path(cfg["paths"]["raw_dir"]) / "documents.jsonl", "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                row = {k: rec.get(k) for k in DOC_FIELDS}
                body, title = rec.get("body") or "", rec.get("title") or ""
                hashes = {sha256_text(body), sha256_text(clean_text(body)), sha256_text(title + "\n" + body)}
                row["content_hashes"] = "|".join(sorted(hashes | ({rec["text_sha256"]} if rec.get("text_sha256") else set())))
                ct, cb = clean_text(title), clean_text(body)
                row["text_identity"], row["text_length"] = normalised_identity(ct, cb), len(ct) + len(cb)
                rows.append(row)
    df = pd.DataFrame(rows)
    pub, avail = [], []
    for r in df.to_dict("records"):
        date_only = bool(r.get("date_only_publish_flag") or 0)
        p = to_timestamp(r.get("publish_datetime"), tz, eod, date_only=date_only or None)
        stated = r.get("availability")
        pub.append(p)
        avail.append(to_timestamp(stated, tz, eod) if isinstance(stated, str) and stated else p)
    df["publication_ns"], df["availability_ns"] = _ns(pub), _ns(avail)
    return df


def document_checks(docs: pd.DataFrame) -> dict:
    has_pub = docs["publication_ns"] != NS_MIN
    early = has_pub & (docs["availability_ns"] < docs["publication_ns"])
    return {
        "documents": int(len(docs)),
        "availability_before_publication": int(early.sum()),
        "missing_availability": int((docs["availability_ns"] == NS_MIN).sum()),
        "date_only_publications": int(pd.to_numeric(docs["date_only_publish_flag"], errors="coerce").fillna(0).sum()),
        "availability_basis": docs["availability_basis"].fillna("not stated").value_counts().to_dict(),
    }


def action_checks(cfg, market: str, docs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Entry timing of every register entry of one market, as the PII builder used it."""
    from ..text.pii_register import _markets_of

    prices, part, cut = _calendar(cfg, market)
    _, interim, proc, _ = _dirs(cfg)
    tz, eod = cfg["alignment"]["timezone"], cfg["alignment"]["date_only_time"]
    cats = cfg["pii"]["categories"]
    dates = prices["date"].dt.strftime("%Y-%m-%d").to_numpy()
    seen: dict[str, dict] = {}                     # earliest availability per canonical identifier
    for a in read_jsonl(interim / "pii" / "actions.jsonl"):
        ts = to_timestamp(a.get("availability"), tz, eod)
        if market in _markets_of(a) and not pd.isna(ts):
            cid = a["canonical_action_id"]
            if cid not in seen or ts < seen[cid]["_ts"]:
                seen[cid] = {**a, "_ts": ts}
    entered: dict[str, set] = {}                   # observations at which the model saw each identifier
    ev = pd.read_csv(proc / f"{market}_pii_events.csv")
    for o, ids in zip(ev["origin_index"], ev["canonical_action_ids"].astype(str)):
        for cid in ids.split("|"):
            entered.setdefault(cid, set()).add(int(o))
    doc = docs.set_index("document_id")
    rows = []
    for cid, a in seen.items():
        got = sorted(entered.get(cid, ()))
        o = got[0] if got else -1
        support = list(a.get("supporting_document_ids") or [])
        known = [int(doc.at[d, "availability_ns"]) for d in support if d in doc.index]
        src = a.get("source_document_id")
        stated = {a[k] for k in ACTION_HASH_FIELDS if a.get(k)}
        if src not in doc.index:
            text_status = "source document not in corpus"
        elif not stated:
            text_status = "no text hash in register"
        else:
            stored = set(str(doc.at[src, "content_hashes"]).split("|"))
            text_status = ("matches source document" if stated & stored
                           else "matches no stored version of the source document")
        rows.append({
            "canonical_action_id": cid, "market": market, "categories": "|".join(a.get("categories") or []),
            "has_model_category": any(c in cats for c in a.get("categories") or []),
            "availability": int(a["_ts"].value), "effective_date": a.get("effective_date"),
            "expected_entry_index": int(assign_to_calendar(pd.Series([a["_ts"]]), cut)[0]),
            "entry_index": o, "entry_date": dates[o] if o >= 0 else None,
            "entry_cutoff": int(cut[o]) if o >= 0 else NS_MIN, "entry_observations": len(got),
            "supporting_documents": len(support), "supporting_found_in_corpus": len(known),
            "earliest_supporting_availability": min(known) if known else NS_MIN,
            "supporting_available_after_entry": sum(1 for s in known if o >= 0 and s > cut[o]),
            "source_document_id": src, "reviewed_text_hash": text_status,
            "reviewer_type": a.get("reviewer_type"), "review_status": a.get("review_status"),
        })
    df = pd.DataFrame(rows).sort_values(["entry_index", "canonical_action_id"]).reset_index(drop=True)
    e = df[df["entry_index"] >= 0]
    avail_day = pd.to_datetime(df["availability"], utc=True).dt.tz_convert(_tz(tz)).dt.tz_localize(None).dt.normalize()
    eff = pd.to_datetime(df["effective_date"], errors="coerce")
    has_sup = e["earliest_supporting_availability"] != NS_MIN
    summary = {
        "register_entries": int(len(df)), "entered": int(len(e)),
        "not_entered_after_calendar": int(((df["entry_index"] < 0) & (df["expected_entry_index"] < 0)).sum()),
        "not_entered_no_model_category": int(((df["entry_index"] < 0) & ~df["has_model_category"]).sum()),
        "entry_is_first_cutoff_on_or_after_availability": bool((e["entry_index"] == e["expected_entry_index"]).all()),
        "availability_not_after_entry_cutoff": bool((e["availability"] <= e["entry_cutoff"]).all()),
        "each_identifier_enters_once": bool((e["entry_observations"] == 1).all()),
        "effective_date_before_availability_date": int((eff < avail_day).sum()),
        "effective_date_after_availability_date": int((eff > avail_day).sum()),
        "supporting_documents_total": int(e["supporting_documents"].sum()),
        "supporting_documents_not_in_corpus": int((e["supporting_documents"] - e["supporting_found_in_corpus"]).sum()),
        "supporting_documents_available_after_entry": int(e["supporting_available_after_entry"].sum()),
        "actions_available_before_earliest_supporting_document":
            int((has_sup & (e["availability"] < e["earliest_supporting_availability"])).sum()),
        "reviewed_text_hash": e["reviewed_text_hash"].value_counts().to_dict(),
        "reviewer_type": e["reviewer_type"].fillna("not stated").value_counts().to_dict(),
    }
    return df, summary


def deduplication_check(cfg, docs: pd.DataFrame) -> dict:
    """Deduplication keeps the earliest available record of every normalised text identity; also
    counts market scopes that a retained record receives only from a strictly later duplicate."""
    _, interim, _, _ = _dirs(cfg)
    kept = set(pd.read_pickle(interim / "documents_clean.pkl")["document_id"])
    valid = docs[(docs["text_length"] > 0) & (docs["availability_ns"] != NS_MIN)]
    ordered = valid.sort_values(["availability_ns", "document_id"])
    first = ordered.drop_duplicates("text_identity", keep="first").set_index("text_identity")
    later_scope: dict[str, int] = {}
    for ident, grp in ordered.groupby("text_identity", sort=False):
        if len(grp) < 2:
            continue
        own_ns, own = first.at[ident, "availability_ns"], set(first.at[ident, "markets"] or [])
        same_time = {m for ms, n in zip(grp["markets"], grp["availability_ns"]) if n == own_ns for m in (ms or [])}
        for m in {m for ms in grp["markets"] for m in (ms or [])} - own - same_time:
            later_scope[m] = later_scope.get(m, 0) + 1
    return {
        "raw_documents": int(len(docs)), "with_text_and_availability": int(len(valid)),
        "identity_groups": int(len(first)), "duplicates_removed": int(len(valid) - len(first)),
        "retained_documents": int(len(kept)),
        "retained_is_earliest_available_record_of_each_identity": set(first["document_id"]) == kept,
        "retained_records_with_market_scope_only_from_later_duplicates": dict(sorted(later_scope.items())),
    }


def actions_for_export(actions: pd.DataFrame, tz: str) -> pd.DataFrame:
    out = actions.copy()
    for col in ("availability", "entry_cutoff", "earliest_supporting_availability"):
        out[col] = _iso(out[col].to_numpy(np.int64), tz)
    return out


# ----------------------------------------------------------------------------- point-in-time table
def point_in_time_table(cfg, market: str, docs: pd.DataFrame, actions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """One row per target: origin, forecast cut-off and the latest input of every kind."""
    prices, part, cut = _calendar(cfg, market)
    T, tz = len(prices), cfg["alignment"]["timezone"]
    _, interim, _, _ = _dirs(cfg)
    dates = prices["date"].dt.strftime("%Y-%m-%d").to_numpy()
    al = pd.read_csv(interim / "alignment" / f"{market}_document_alignment.csv")
    d_origin = al["origin_index"].to_numpy(np.int64)
    d_ns, d_id = _running_latest(d_origin, _ns(al["availability_ts"]), al["document_id"].to_numpy(object), T)
    e = actions[actions["entry_index"] >= 0]
    a_origin = e["entry_index"].to_numpy(np.int64)
    a_ns, a_id = _running_latest(a_origin, e["availability"].to_numpy(np.int64), e["canonical_action_id"].to_numpy(object), T)
    pub = dict(zip(docs["document_id"], docs["publication_ns"]))
    eff = dict(zip(actions["canonical_action_id"], actions["effective_date"]))
    i = part.targets("all")
    o = i - 1
    df = pd.DataFrame({
        "market": market, "split": [part.split_of(int(k)) for k in i],
        "origin_index": o, "origin_date": dates[o], "forecast_cutoff": _iso(cut[o], tz),
        "target_index": i, "target_date": dates[i],
        "latest_price_date": dates[o],
        "documents_entering_at_origin": np.bincount(d_origin, minlength=T)[o],
        "latest_document_id": d_id[o],
        "latest_document_published": _iso([pub.get(x, NS_MIN) if x is not None else NS_MIN for x in d_id[o]], tz),
        "latest_document_available": _iso(d_ns[o], tz),
        "actions_entering_at_origin": np.bincount(a_origin, minlength=T)[o],
        "latest_action_id": a_id[o],
        "latest_action_available": _iso(a_ns[o], tz),
        "latest_action_effective_date": [eff.get(x) if x is not None else None for x in a_id[o]],
    })
    df["price_ok"] = dates[o] < dates[i]
    df["documents_ok"] = (d_ns[o] == NS_MIN) | (d_ns[o] <= cut[o])
    df["actions_ok"] = (a_ns[o] == NS_MIN) | (a_ns[o] <= cut[o])
    df["compliant"] = df["price_ok"] & df["documents_ok"] & df["actions_ok"]
    lead = lambda ns: (cut[o] - ns)[ns != NS_MIN] / 3.6e12          # hours from the latest input to the cut-off
    summary = {
        "targets": {s: int((df["split"] == s).sum()) for s in ("train", "val", "test")},
        "first_target": dates[i[0]], "last_target": dates[i[-1]],
        "all_compliant": bool(df["compliant"].all()), "non_compliant_targets": int((~df["compliant"]).sum()),
        "min_hours_between_latest_document_and_cutoff": float(lead(d_ns[o]).min()) if (d_ns[o] != NS_MIN).any() else None,
        "min_hours_between_latest_action_and_cutoff": float(lead(a_ns[o]).min()) if (a_ns[o] != NS_MIN).any() else None,
    }
    return df, summary


# ----------------------------------------------------------------------------- model input arrays
def text_array_check(cfg, market: str, docs: pd.DataFrame) -> dict:
    """Rebuild the daily semantic / CSI / count arrays from the raw timestamps and compare."""
    from ..text.daily import daily_count, daily_csi, daily_mean, doc_availability

    prices, part, cut = _calendar(cfg, market)
    T, tz = len(prices), cfg["alignment"]["timezone"]
    _, interim, proc, _ = _dirs(cfg)
    md = MarketData.load(proc / f"{market}.npz")
    al = pd.read_csv(interim / "alignment" / f"{market}_document_alignment.csv")
    t_idx = al["origin_index"].to_numpy(np.int64)
    ns = _ns(al["availability_ts"])
    raw_ns = docs.set_index("document_id").loc[al["document_id"], "availability_ns"].to_numpy(np.int64)
    ids = load_json(interim / "encoded" / "document_ids.json")
    row_of = {d: k for k, d in enumerate(ids)}
    rows = np.array([row_of[d] for d in al["document_id"]], dtype=np.int64)
    sem = np.load(interim / "encoded" / "semantic.npy", mmap_mode="r")
    sent = np.load(interim / "encoded" / "sentiment.npy")
    rebuilt = {"sem": daily_mean(np.asarray(sem[rows]), t_idx, T), "csi": daily_csi(sent[rows], t_idx, T),
               "count": daily_count(t_idx, T), "doc_avail": doc_availability(t_idx, T)}
    first = pd.Timestamp(prices["date"].min()).tz_localize(_tz(tz))
    floor = int((first - pd.Timedelta(days=int(cfg["alignment"].get("pre_calendar_days") or 0))).value)
    diffs = {k: _max_abs(v, md[k]) for k, v in rebuilt.items()}
    return {
        "documents_aligned": int(len(al)),
        "availability_equals_raw_record": bool(np.array_equal(ns, raw_ns)),
        "entry_recomputed_from_availability": bool(np.array_equal(assign_to_calendar(pd.Series(pd.to_datetime(ns, utc=True)), cut), t_idx)),
        "availability_not_after_entry_cutoff": bool((ns <= cut[t_idx]).all()),
        "entered_at_first_eligible_cutoff": bool(((t_idx == 0) | (ns > cut[np.maximum(t_idx - 1, 0)])).all()),
        "no_document_older_than_pre_calendar_floor": bool((ns >= floor).all()),
        "max_abs_difference_rebuilt_vs_model_arrays": diffs,
        "arrays_identical": all(v == 0.0 for v in diffs.values()),
    }


def pii_array_check(cfg, market: str, docs: pd.DataFrame) -> dict:
    """Rebuild the PII arrays from the register, once as the pipeline does and once with every
    supporting document carrying its corpus availability (counted only if available at entry)."""
    from ..text.pii_register import build_pii_arrays

    prices, _, _ = _calendar(cfg, market)
    tz = cfg["alignment"]["timezone"]
    _, interim, proc, _ = _dirs(cfg)
    md = MarketData.load(proc / f"{market}.npz")
    actions = read_jsonl(interim / "pii" / "actions.jsonl")
    keys = ["pii_attr", "pii_new", "pii_age", "pii_dt", "pii_avail"]
    rebuilt = build_pii_arrays(actions, market, prices["date"], cfg)
    avail = dict(zip(docs["document_id"], _iso(docs["availability_ns"], tz)))
    strict_actions = [{**a, "supporting_documents": [{"document_id": d, "availability": avail.get(d)}
                                                     for d in a.get("supporting_document_ids") or []]} for a in actions]
    strict = build_pii_arrays(strict_actions, market, prices["date"], cfg)
    d = np.abs(strict["pii_attr"].astype(np.float64) - md["pii_attr"])
    diffs = {k: _max_abs(rebuilt[k], md[k]) for k in keys}
    return {
        "max_abs_difference_rebuilt_vs_model_arrays": diffs,
        "arrays_identical": all(v == 0.0 for v in diffs.values()),
        "attributes_identical_with_point_in_time_support_counts": bool(d.max() == 0.0),
        "attribute_cells_differing": int((d > 0).sum()),
        "event_dates": int(md["pii_new"].max(axis=1).sum()),
    }


# ----------------------------------------------------------------------------- decomposition
def decomposition_check(cfg, market: str, n_per_split: int = 5, seed: int = 2026) -> tuple[pd.DataFrame, dict]:
    """Re-decompose sampled origins from their own prefix and audit the frequency grouping."""
    from joblib import Parallel, delayed

    from ..decomposition.ceemdan import OriginArchive, _one_origin
    from ..decomposition.grouping import FrequencyGrouping, component_archive, fit_grouping

    prices, part, _ = _calendar(cfg, market)
    close = prices["close"].to_numpy(np.float64)
    T = len(close)
    dates = prices["date"].dt.strftime("%Y-%m-%d").to_numpy()
    _, interim, proc, _ = _dirs(cfg)
    dec, dcfg = interim / "decomposition", cfg["decomposition"]
    arch = OriginArchive.load(dec / f"{market}_origin_archive.npz")
    stored = load_json(dec / f"{market}_grouping.json")
    md = MarketData.load(proc / f"{market}.npz")
    rng = np.random.default_rng(seed)

    g_train = fit_grouping(arch, part.train_end, dcfg["grouping"])
    pert = dataclasses.replace(arch, freqs=arch.freqs.copy())
    tail = pert.freqs[part.train_end:]
    pert.freqs[part.train_end:] = np.exp(rng.uniform(np.log(1e-3), np.log(0.5), size=tail.shape))
    g_pert = fit_grouping(pert, part.train_end, dcfg["grouping"])
    g_full = fit_grouping(arch, T, dcfg["grouping"])
    R = component_archive(arch, FrequencyGrouping(np.asarray(stored["log_centres"])))
    comp = pd.read_csv(dec / f"{market}_components.csv")

    label = np.array([part.split_of(t + 1) if t + 1 < T else "test" for t in range(T)])   # split of the target
    sample = []
    for s in ("reserve", "train", "val", "test"):
        cand = np.flatnonzero(label == s)
        k = min(len(cand), 2 if s == "reserve" else n_per_split)
        sample += sorted(int(t) for t in rng.choice(cand, size=k, replace=False))
    future = [t for t in sample if label[t] == "test"][:2]
    perturbed = {}
    for t in future:                               # random-walk future prices after the origin
        c = close.copy()
        c[t + 1:] = close[t] + np.cumsum(rng.normal(0.0, close.std(), size=T - t - 1))
        perturbed[t] = c
    jobs = [(t, close) for t in sample] + [(t, perturbed[t]) for t in future]
    out = Parallel(n_jobs=int(dcfg.get("n_jobs", -1)))(delayed(_one_origin)(t, c, dcfg) for t, c in jobs)
    plain = {t: r for (t, _), r in zip(jobs[:len(sample)], out[:len(sample)])}
    rows = []
    cap = int(dcfg["history_cap"])
    for t in sample:
        _, ends, fr, res = plain[t]
        n, m = len(ends), int(arch.mask[t].sum())
        rows.append({"market": market, "origin_index": t, "origin_date": dates[t], "split_of_target": label[t],
                     "prefix_first_index": max(0, t + 1 - cap), "prefix_last_index": t,
                     "modes_archived": m, "modes_recomputed": n,
                     "max_abs_endpoint_difference": _max_abs(ends, arch.endpoints[t, :n]) if n == m else float("inf"),
                     "max_abs_frequency_difference": _max_abs(fr, arch.freqs[t, :n]) if n == m else float("inf"),
                     "abs_residual_difference": abs(res - float(arch.residual[t])),
                     "future_prices_perturbed": False, "abs_change_from_future_perturbation": None})
    for (t, _), r in zip(jobs[len(sample):], out[len(sample):]):
        _, ends, fr, res = r
        _, e0, f0, r0 = plain[t]
        change = max(_max_abs(ends, e0) if len(ends) == len(e0) else float("inf"), _max_abs(fr, f0) if len(fr) == len(f0) else float("inf"), abs(res - r0))
        rows.append({"market": market, "origin_index": t, "origin_date": dates[t], "split_of_target": label[t],
                     "prefix_first_index": max(0, t + 1 - cap), "prefix_last_index": t,
                     "modes_archived": int(arch.mask[t].sum()), "modes_recomputed": len(ends),
                     "max_abs_endpoint_difference": None, "max_abs_frequency_difference": None, "abs_residual_difference": None,
                     "future_prices_perturbed": True, "abs_change_from_future_perturbation": change})
    df = pd.DataFrame(rows)
    rec = df[~df["future_prices_perturbed"]]
    summary = {
        "archive_close_equals_observed_prices": bool(np.array_equal(arch.close, close)),
        "origins_in_archive": int(T),
        "grouping_fit_rows": [0, int(part.train_end) - 1],
        "grouping_fit_last_origin_date": dates[part.train_end - 1],
        "stored_train_end": int(stored["train_end"]),
        "stored_centres_minus_training_refit": _max_abs(stored["log_centres"], g_train.log_centres),
        "training_refit_change_when_later_frequencies_perturbed": _max_abs(g_pert.log_centres, g_train.log_centres),
        "full_calendar_refit_minus_stored_centres": _max_abs(g_full.log_centres, stored["log_centres"]),
        "model_components_minus_causal_archive": _max_abs(R, md["R"]),
        "components_file_minus_causal_archive": _max_abs(comp[[f"scale_{k + 1}" for k in range(R.shape[1])]].to_numpy(), R),
        "max_abs_reconstruction_error": _max_abs(md["R"].sum(axis=1), close),
        "sampled_origins": len(rec),
        "sampled_origins_reproduced": int(((rec["modes_archived"] == rec["modes_recomputed"])
                                           & (rec["max_abs_endpoint_difference"] <= 1e-9)
                                           & (rec["abs_residual_difference"] <= 1e-9)).sum()),
        "max_abs_change_from_future_perturbation": float(df.loc[df["future_prices_perturbed"], "abs_change_from_future_perturbation"].max())
        if future else None,
    }
    return df, summary


def full_sample_call_sites(package_dir: Path) -> list[str]:
    """Source lines that call the descriptive full-sample grouping (Figure 3 only)."""
    hits = []
    for path in sorted(Path(package_dir).rglob("*.py")):
        if path.resolve() == Path(__file__).resolve():
            continue
        for k, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "group_full_series(" in line and not line.lstrip().startswith("def "):
                hits.append(f"{path.relative_to(Path(package_dir).parent).as_posix()}:{k}: {line.strip()}")
    return hits


# ----------------------------------------------------------------------------- partitions and statistics
def partition_audit(cfg, market: str) -> tuple[pd.DataFrame, dict]:
    prices, part, cut = _calendar(cfg, market)
    tz = cfg["alignment"]["timezone"]
    dates = prices["date"].dt.strftime("%Y-%m-%d").to_numpy()
    idx = np.arange(len(prices))
    split = [part.split_of(int(k)) for k in idx]
    df = pd.DataFrame({"market": market, "index": idx, "date": dates, "split": split,
                       "role": ["input history" if s == "reserve" else "target" for s in split],
                       "forecast_cutoff_as_origin": _iso(cut, tz)})
    tr, va, te = (part.targets(s) for s in ("train", "val", "test"))
    mcfg = cfg["markets"][market]
    summary = {
        "observations": int(len(prices)), "reserve": int(part.reserve),
        "train": [dates[tr[0]], dates[tr[-1]], int(len(tr))], "val": [dates[va[0]], dates[va[-1]], int(len(va))],
        "test": [dates[te[0]], dates[te[-1]], int(len(te))],
        "train_before_val": bool(dates[tr].max() < dates[va].min()),
        "val_before_test": bool(dates[va].max() < dates[te].min()),
        "counts_match_configuration": [len(tr), len(va), len(te)] == [int(v) for v in mcfg["splits"]],
        "first_test_target_matches_configuration": (dates[te[0]] == mcfg["test_start"]) if mcfg.get("test_start") else None,
    }
    return df, summary


def statistics_check(cfg, market: str, seed: int = 2026) -> dict:
    """Training statistics with validation/test rows perturbed (must not move) and, as a control,
    with training rows perturbed (must move); also the lexical-encoder fitting window."""
    prices, part, cut = _calendar(cfg, market)
    _, interim, proc, _ = _dirs(cfg)
    md = MarketData.load(proc / f"{market}.npz")
    e, rng = part.train_end, np.random.default_rng(seed)

    def stats_with_noise(lo, hi):
        arrays = {k: v.copy() for k, v in md.arrays.items()}
        for k in ARRAY_KEYS:
            if k in arrays and arrays[k].shape[0] == md.n_obs and arrays[k].dtype.kind == "f":
                arrays[k][lo:hi] = rng.normal(0.0, 10.0, size=arrays[k][lo:hi].shape)
        return MarketData(md.market, md.dates, arrays, part).stats()

    def change(a, b):
        return max(_max_abs(a[k], b[k]) for k in a)

    base = stats_with_noise(e, e)                    # same code path, nothing perturbed
    al = pd.read_csv(interim / "alignment" / f"{market}_document_alignment.csv")
    fit_docs = al[al["origin_index"] < e]
    dates = prices["date"].dt.strftime("%Y-%m-%d").to_numpy()
    return {
        "statistics_rows": [0, int(e) - 1], "statistics_last_date": dates[e - 1],
        "first_validation_target": dates[e],
        "max_change_when_validation_and_test_rows_perturbed": change(base, stats_with_noise(e, md.n_obs)),
        "max_change_when_training_rows_perturbed_control": change(base, stats_with_noise(part.reserve, e)),
        "lexical_fit_documents": int(len(fit_docs)),
        "lexical_fit_latest_availability_not_after_last_training_cutoff":
            bool((_ns(fit_docs["availability_ts"]) <= cut[e - 1]).all()) if len(fit_docs) else None,
    }


def array_hashes(processed_dir) -> pd.DataFrame:
    """SHA-256 of the bytes of every array in every processed market file (with shape and dtype),
    independent of the compression metadata of the .npz container, plus summary statistics that
    let a rebuild be compared within a tolerance when the bytes differ (e.g. another BLAS)."""
    import hashlib

    rows = []
    for f in sorted(Path(processed_dir).glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        for k in sorted(z.files):
            arr = np.ascontiguousarray(z[k])
            row = {"file": f.name, "array": k, "shape": "x".join(map(str, arr.shape)) or "scalar",
                   "dtype": str(arr.dtype), "sha256": hashlib.sha256(arr.tobytes()).hexdigest()}
            if arr.dtype.kind in "fiu" and arr.size:
                x = arr.astype(np.float64)
                row.update({"sum": float(x.sum()), "sum_abs": float(np.abs(x).sum()), "min": float(x.min()), "max": float(x.max())})
            rows.append(row)
    return pd.DataFrame(rows)


def compare_summaries(published, rebuilt) -> dict | None:
    """Differences between the summary statistics of two versions of an array: absolute for
    the minimum and maximum, relative to the published sum of absolute values for the sums."""
    keys = ("sum", "sum_abs", "min", "max")
    if any(pd.isna(published.get(k)) or pd.isna(rebuilt.get(k)) for k in keys):
        return None
    scale = max(abs(float(published["sum_abs"])), 1e-300)
    return {"min_abs_diff": abs(float(published["min"]) - float(rebuilt["min"])),
            "max_abs_diff": abs(float(published["max"]) - float(rebuilt["max"])),
            "sum_rel_diff": abs(float(published["sum"]) - float(rebuilt["sum"])) / scale,
            "sum_abs_rel_diff": abs(float(published["sum_abs"]) - float(rebuilt["sum_abs"])) / scale}


# ----------------------------------------------------------------------------- trained models
def _perturb_from(data, cut: int, seed: int):
    """Copy of a MarketTensors view whose time-indexed inputs at positions >= cut are noise."""
    import torch

    T = data.partition.n_obs
    g = torch.Generator().manual_seed(int(seed))
    t2 = {}
    for k, x in data.t.items():
        if x.dim() >= 1 and x.shape[0] == T and x.is_floating_point():
            y = x.clone()
            scale = float(x[:cut].float().std()) + 1.0
            y[cut:] = (torch.randn(y[cut:].shape, generator=g) * scale).to(y.dtype)
            t2[k] = y
        else:
            t2[k] = x
    rng = np.random.default_rng(seed)
    close, R = data.close.copy(), data.R.copy()
    close[cut:] = rng.normal(close[:cut].mean(), close[:cut].std() + 1.0, size=close[cut:].shape)
    R[cut:] = rng.normal(0.0, R[:cut].std() + 1.0, size=R[cut:].shape)
    return dataclasses.replace(data, t=t2, close=close, R=R)


def model_causality_check(cfg, market: str, variant: str, seed: int, n_after: int = 5,
                          noise_seed: int = 2026, perturb: bool = True) -> dict:
    """Reload a checkpoint, reproduce its saved forecasts and, optionally, replace every input at
    and after three cut points by noise: forecasts of targets up to each cut must not change."""
    from ..experiments.runner import load_trained, run_dir
    from ..training.trainer import predict

    md, data, model = load_trained(cfg, market, variant, int(seed))
    p = data.partition
    bs = int(cfg["optim"].get("eval_batch_size", 256))
    idx = p.targets("all")
    base = predict(model, data, idx, bs)["y_hat"]
    out = {"market": market, "model": variant, "seed": int(seed)}
    for split in ("val", "test"):
        saved = pd.read_csv(run_dir(cfg, "forecasts", market, variant, seed) / f"{split}_predictions.csv")
        saved = saved.sort_values("target_index")
        pos = np.searchsorted(idx, saved["target_index"].to_numpy())
        out[f"max_abs_diff_vs_saved_{split}_forecasts"] = _max_abs(base[pos], saved["prediction"])
    if not perturb:
        return out
    rows = []
    for c in (p.train_end, p.val_end, p.val_end + p.n_test // 2):
        d2 = _perturb_from(data, c, noise_seed + c)
        before, after = idx[idx <= c], idx[(idx > c) & (idx <= c + n_after)]
        pb = predict(model, d2, before, bs)["y_hat"]
        pa = predict(model, d2, after, bs)["y_hat"]
        rows.append({"cut_index": int(c), "cut_date": str(md.dates[c]), "targets_up_to_cut": int(len(before)),
                     "max_abs_change_targets_up_to_cut": _max_abs(pb, base[np.searchsorted(idx, before)]),
                     "max_abs_change_next_targets": _max_abs(pa, base[np.searchsorted(idx, after)])})
    out["cuts"] = rows
    return out


# ----------------------------------------------------------------------------- run records
def _replay_early_stopping(log: pd.DataFrame, ocfg: dict) -> tuple[int, int]:
    """Best epoch and stopping epoch implied by the logged validation RMSE (trainer rules)."""
    best, best_epoch, wait, stop = float("inf"), 0, 0, int(log["epoch"].max())
    for ep, v in zip(log["epoch"], log["val_rmse"]):
        if v < best - float(ocfg["min_delta"]):
            best, best_epoch, wait = v, int(ep), 0
        else:
            wait += 1
        if wait >= int(ocfg["patience"]):
            stop = int(ep)
            break
    return best_epoch, stop


def run_records(cfg) -> pd.DataFrame:
    """One row per completed run: configuration source, early stopping, validation selection,
    test metrics and the time the forecast file was written."""
    from . import metrics as M

    out_dir = Path(cfg["paths"]["output_dir"])
    defaults = cfg["iesf"]["defaults"]
    rows = []
    for sdir in sorted(out_dir.glob("forecasts/*/*/seed*")):
        test_f, val_f = sdir / "test_predictions.csv", sdir / "val_predictions.csv"
        if not test_f.exists():
            continue
        market, model, seed = sdir.parent.parent.name, sdir.parent.name, sdir.name[4:]
        te = pd.read_csv(test_f)
        run = load_json(sdir / "run.json") if (sdir / "run.json").exists() else {}
        r = {"market": market, "model": model, "seed": seed, "n_test": len(te),
             "first_test_target": te["target_date"].min(), "last_test_target": te["target_date"].max()}
        r.update({f"test_{k}": v for k, v in M.all_metrics(te["prediction"], te["actual"], te["origin_price"]).items() if k != "N"})
        logged_val = run.get("best_val_rmse", run.get("val_rmse"))
        if val_f.exists():
            va = pd.read_csv(val_f)
            r["last_val_target"] = va["target_date"].max()
            r["val_targets_before_test_targets"] = bool(va["target_date"].max() < te["target_date"].min())
            r["val_rmse_from_saved_forecasts"] = M.rmse(va["prediction"], va["actual"])
            if logged_val is not None:
                r["saved_forecasts_are_selected_checkpoint"] = abs(r["val_rmse_from_saved_forecasts"] - float(logged_val)) < 1e-6
        r["logged_best_val_rmse"] = logged_val
        r["n_params"] = run.get("n_params", run.get("size"))
        hp = run.get("hparams", run.get("params"))
        if "hparams" in run:
            same = all(hp.get(k) == v for k, v in defaults.items() if k in hp)
            r["configuration"] = "Table A3 (configs/iesf_net.yaml)" if same else "other"
        elif "params" in run:
            r["configuration"] = "Table A2 (configs/benchmarks.yaml paper_selected)"
        r["optuna_record_present"] = (out_dir / "tuning" / market / model / "best.json").exists()
        log_f = sdir / "training_log.csv"
        if log_f.exists():
            log = pd.read_csv(log_f)
            be, stop = _replay_early_stopping(log, cfg["optim"])
            r.update({"best_epoch": run.get("best_epoch"), "epochs_run": run.get("epochs_run"),
                      "best_epoch_replayed_from_log": be, "stop_epoch_replayed_from_log": stop,
                      "early_stopping_consistent": be == run.get("best_epoch") and stop == run.get("epochs_run"),
                      "training_log_columns": "|".join(log.columns),
                      "test_quantity_in_training_log": any("test" in c.lower() for c in log.columns)})
        r["forecast_file_written"] = pd.Timestamp(test_f.stat().st_mtime, unit="s", tz="UTC").tz_convert(
            _tz(cfg["alignment"]["timezone"])).isoformat(timespec="seconds")
        rows.append(r)
    return pd.DataFrame(rows)


CORE_MODELS = {                                    # model -> (table, group template, row label)
    "Persistence": ("Table 7", "{m} — numerical benchmarks", "Persistence"),
    "Numeric": ("Table 7", "{m} — multiscale models", "Numeric"),
    "IESF-Net": ("Table 7", "{m} — multiscale models", "IESF-Net"),
    "Bidirectional": ("Table 13", None, "Ordinary bidirectional"),
}
CORE_METRICS = {"RMSE": "RMSE", "MAE": "MAE", "sMAPE (%)": "sMAPE", "R2": "R2", "DA (%)": "DA"}


def _reported(paper: pd.DataFrame, model: str, market: str, column: str):
    table, group, label = CORE_MODELS[model]
    sel = paper[(paper["table"] == table) & (paper["row"] == label)]
    sel = sel[(sel["group"] == group.format(m=market)) & (sel["column"] == column)] if group else \
        sel[sel["column"] == f"{market} {column}"]
    if sel.empty:                                  # secondary markets are reported in Table 16 only
        table, label = "Table 16", market
        sel = paper[(paper["table"] == table) & (paper["row"] == market) & (paper["column"] == f"{model} {column}")]
    if sel.empty:
        return None, None
    return f"{table}: {label}", float(str(sel["reported"].iloc[0]).replace("−", "-").split("±")[0])


def core_comparison(records: pd.DataFrame, paper: pd.DataFrame | None, designated: int) -> pd.DataFrame:
    """Per market, core model and metric: the designated-seed run, the five-run summary and the
    value reported in the manuscript (when the extracted manuscript values are supplied)."""
    rows = []
    core = records[records["model"].isin(CORE_MODELS)]
    for (market, model), grp in core.groupby(["market", "model"], sort=False):
        des = grp[grp["seed"].isin([str(designated), "deterministic"])]
        for column, key in CORE_METRICS.items():
            vals = grp[f"test_{key}"].astype(float)
            source, rep = _reported(paper, model, market, column) if paper is not None else (None, None)
            d = float(des[f"test_{key}"].iloc[0]) if len(des) else None
            rows.append({"market": market, "model": model, "metric": column, "manuscript_value": rep,
                         "manuscript_source": source, "designated_run": des["seed"].iloc[0] if len(des) else None,
                         "designated_run_value": d, "designated_minus_manuscript": (d - rep) if d is not None and rep is not None else None,
                         "runs": int(len(vals)), "seeds": "|".join(grp["seed"]),
                         "mean": float(vals.mean()), "sd": float(vals.std(ddof=1)) if len(vals) > 1 else None,
                         "min": float(vals.min()), "max": float(vals.max())})
    return pd.DataFrame(rows)
