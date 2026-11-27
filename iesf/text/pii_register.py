"""Daily PII input tensors on a market calendar (Table 1, Section 3.2.2).

For every observation t and category c the builder returns
    attr[t, c, :16]  the ordered attributes of Table 1 (zeros on no-action dates),
    new[t, c]        m_{c,t}: a new canonical action became available,
    age[t, c]        log(1 + pre-reset event age) fed to the category GRU,
    dt[t]            elapsed observation interval Δt used by the decay,
    avail[t]         institutional availability: 1 after the first accepted event.
Updates are indexed by canonical action identity; an identifier enters the
register once, at the first market cut-off on or after its first availability.
Accepts both extractor output and externally reviewed action records
(fields: canonical_action_id, categories, mechanisms, scope, issuer_tier,
confidence, effective_date, availability, market/markets, supporting_document_ids).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.alignment import assign_to_calendar, market_cutoffs, to_timestamp

ATTRIBUTE_NAMES = [
    "log1p_unique_actions", "log1p_sum_known_confidence", "known_confidence_fraction",
    "mech_increase", "mech_decrease", "mech_reschedule", "mech_unspecified",
    "scope_local", "scope_national", "scope_cross_market",
    "tier_regulator", "tier_exchange", "tier_other",
    "mean_signed_effective_offset", "known_effective_fraction",
    "log1p_supporting_documents",
]
N_ATTR = len(ATTRIBUTE_NAMES)


def _markets_of(action: dict) -> list[str]:
    ms = action.get("markets")
    if ms:
        return list(ms)
    return [action["market"]] if action.get("market") else []


def _known(x) -> bool:
    return x is not None and not (isinstance(x, float) and np.isnan(x))


def build_pii_arrays(actions: list[dict], market: str, dates, cfg: dict) -> dict:
    """Assemble the PII tensors of one market from the global action register."""
    cats, mechs = cfg["pii"]["categories"], cfg["pii"]["mechanisms"]
    scopes, tiers = cfg["pii"]["scopes"], cfg["pii"]["issuer_tiers"]
    tz, cutoff = cfg["alignment"]["timezone"], cfg["alignment"]["market_cutoff"]
    eod = cfg["alignment"]["date_only_time"]
    dates = pd.to_datetime(pd.Series(dates)).dt.normalize().reset_index(drop=True)
    T, C = len(dates), len(cats)
    cut = market_cutoffs(dates, cutoff, tz)

    # --- global register: one entry per canonical identifier and market --------------------
    seen: dict[str, dict] = {}
    for a in actions:
        if market not in _markets_of(a):
            continue
        cid = a["canonical_action_id"]
        ts = to_timestamp(a.get("availability"), tz, eod)
        if pd.isna(ts):
            continue
        if cid not in seen or ts < seen[cid]["_ts"]:
            seen[cid] = {**a, "_ts": ts}
    reg = list(seen.values())
    if reg:
        origin = assign_to_calendar(pd.Series([r["_ts"] for r in reg]), cut)
    else:
        origin = np.zeros(0, dtype=np.int64)

    buckets: dict[tuple[int, int], list[dict]] = {}
    for r, t in zip(reg, origin):
        if t < 0:
            continue
        for c in r.get("categories", []):
            if c in cats:
                buckets.setdefault((int(t), cats.index(c)), []).append(r)

    attr = np.zeros((T, C, N_ATTR), dtype=np.float32)
    new = np.zeros((T, C), dtype=np.float32)
    for (t, ci), acts in buckets.items():
        c = cats[ci]
        n = len(acts)
        conf = [float(a["confidence"]) for a in acts if _known(a.get("confidence"))]
        mech = [((a.get("mechanisms") or {}).get(c) or "unspecified") for a in acts]
        mech = [m if m in mechs else "unspecified" for m in mech]
        scope = [a.get("scope") if a.get("scope") in scopes else "local" for a in acts]
        tier = [a.get("issuer_tier") if a.get("issuer_tier") in tiers else "other" for a in acts]
        offsets = []
        for a in acts:
            eff = a.get("effective_date")
            if isinstance(eff, str) and eff:
                d = (pd.Timestamp(eff).normalize() - dates[t]).days / 365.0
                offsets.append(float(np.clip(d, -1.0, 1.0)))
        docs = set()
        for a in acts:
            for s in a.get("supporting_documents") or [{"document_id": d, "availability": None}
                                                       for d in (a.get("supporting_document_ids") or [])]:
                av = s.get("availability")
                if av is None or assign_to_calendar(pd.Series([to_timestamp(av, tz, eod)]), cut)[0] <= t:
                    docs.add(s["document_id"])           # supporting documents available by this origin
        vec = [np.log1p(n), np.log1p(sum(conf)), len(conf) / n]
        vec += [mech.count(m) / n for m in mechs]
        vec += [scope.count(s) / n for s in scopes]
        vec += [tier.count(x) / n for x in tiers]
        vec += [float(np.mean(offsets)) if offsets else 0.0, len(offsets) / n]
        vec += [np.log1p(len(docs))]
        attr[t, ci] = np.asarray(vec, dtype=np.float32)
        new[t, ci] = 1.0

    # --- event age (pre-reset) and elapsed observation interval ------------------------------
    age = np.zeros((T, C), dtype=np.float32)
    day_index = ((dates - dates.iloc[0]).dt.days).to_numpy(np.float64)
    for ci in range(C):
        last = 0
        for t in np.flatnonzero(new[:, ci] > 0):
            raw = (day_index[t] - day_index[last]) if cfg["pii"]["age_unit"] == "day" else float(t - last)
            age[t, ci] = np.log1p(max(raw, 0.0))
            last = t
    if cfg["pii"]["delta_unit"] == "observation":
        dt = np.ones(T, dtype=np.float32)
    else:                                                  # calendar days between observations
        dt = np.concatenate([[0.0], np.diff(day_index)]).astype(np.float32)
    dt[0] = 0.0
    avail = (np.cumsum(new.max(axis=1)) > 0).astype(np.float32)

    events = [{"origin_index": t, "origin_date": dates[t].date().isoformat(), "category": cats[ci],
               "n_actions": len(acts), "canonical_action_ids": "|".join(a["canonical_action_id"] for a in acts)}
              for (t, ci), acts in sorted(buckets.items())]
    return {"pii_attr": attr, "pii_new": new, "pii_age": age, "pii_dt": dt, "pii_avail": avail,
            "events": pd.DataFrame(events)}
