"""Rule-based candidate extractor of institutional actions (Sections 3.2.2 and 3.6.2).

For every document the extractor splits the text into clauses, keeps clauses in
which a category anchor and an operative expression co-occur, and requires an
identifiable release (formal instrument / issuing body) or an operative
provision.  Every qualifying clause becomes a candidate action carrying the
attributes needed by Table 1: categories (multi-label), category-specific
mechanism labels, scope, issuing tier, effective date and an extraction
confidence.  A canonical identifier built from issuer, instrument, operative
clause and scope links repeat reports of one action, so they never re-trigger
the memory.  The rule layer also emits one of nine candidate types used to
stratify the expert audit (Section 3.6.3): the seven single categories,
``multi_category`` and ``amendment_cancellation``.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from datetime import date

import pandas as pd

from ..utils import sha256_text
from . import pii_lexicon as lx

CANDIDATE_TYPES = lx.CATEGORIES + ["multi_category", "amendment_cancellation"]


def _norm(text: str) -> str:
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text or "").lower())


def split_clauses(text: str) -> list[str]:
    return [c.strip() for c in lx.CLAUSE_SPLIT.split(text or "") if len(c.strip()) >= 6]


def detect_categories(clause: str) -> list[str]:
    return [c for c in lx.CATEGORIES if any(a in clause for a in lx.CATEGORY_ANCHORS[c])]


def has_operative(clause: str) -> bool:
    return any(w in clause for w in lx.OPERATIVE)


def mechanism_label(clause: str, category: str) -> str:
    """Category-specific direction: increase / decrease / reschedule / unspecified."""
    if any(w in clause for w in lx.GENERIC_MECHANISM["reschedule"]):
        return "reschedule"
    spec = lx.CATEGORY_MECHANISM.get(category, {})
    score = Counter()
    for label in ("increase", "decrease"):
        score[label] += 2 * sum(w in clause for w in spec.get(label, []))
        score[label] += sum(w in clause for w in lx.GENERIC_MECHANISM[label])
    if score["increase"] == score["decrease"]:
        return "unspecified"
    return "increase" if score["increase"] > score["decrease"] else "decrease"


def detect_scope(clause: str, title: str, doc_markets: list[str], all_markets: list[str]) -> tuple[str, list[str]]:
    text = f"{title} {clause}"
    named = [m for m, names in lx.REGION_NAMES.items() if any(n in text for n in names)]
    named_known = [m for m in named if m in all_markets]
    if any(k in text for k in lx.CROSS_MARKET) or len(named) >= 2:
        return "cross_market", (named_known or list(doc_markets) or list(all_markets))
    if any(k in text for k in lx.NATIONAL) and not named:
        return "national", list(all_markets)
    markets = named_known or list(doc_markets)
    return "local", markets


def detect_issuer(title: str, body: str, publisher: str = "", issuer_field: str = "") -> tuple[str, str]:
    head = f"{issuer_field} {title} {body[:300]} {publisher}"
    m = lx.ISSUER_PATTERN.search(head)
    issuer = (issuer_field or (m.group(1) if m else "") or publisher or "").strip()
    probe = f"{issuer} {head}"
    if any(k in issuer for k in lx.EXCHANGE) or (not issuer and any(k in probe for k in lx.EXCHANGE)):
        return issuer, "exchange"
    if any(k in probe for k in lx.REGULATOR):
        return issuer, "regulator"
    if any(k in probe for k in lx.EXCHANGE):
        return issuer, "exchange"
    return issuer, "other"


def detect_effective_date(clause: str, body: str, release: pd.Timestamp | None) -> str | None:
    for text in (clause, body):
        for pat in lx.EFFECTIVE_PATTERNS:
            m = pat.search(text)
            if m:
                try:
                    return date(int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat()
                except ValueError:
                    continue
        if lx.EFFECTIVE_ON_RELEASE.search(text) and release is not None and not pd.isna(release):
            return pd.Timestamp(release).tz_convert("Asia/Shanghai").date().isoformat()
    return None


def instrument_of(clause: str, title: str) -> str:
    for text in (clause, title):
        m = lx.FORMAL_INSTRUMENT.search(text)
        if m:
            return m.group(1)
    return _norm(title)[:60]


def confidence_score(formal: bool, tier: str, mechanism_known: bool, effective_known: bool) -> float:
    score = 0.4 + 0.2 * formal + 0.2 * (tier in ("regulator", "exchange")) + 0.1 * mechanism_known + 0.1 * effective_known
    return round(float(score), 2)


def extract_document(doc: dict, all_markets: list[str]) -> list[dict]:
    """Candidate actions of one cleaned document record."""
    title, body = doc.get("title", "") or "", doc.get("body", "") or ""
    release = doc.get("availability_ts")
    issuer, tier = detect_issuer(title, body, doc.get("publisher", "") or "", doc.get("issuer", "") or "")
    title_formal = bool(lx.FORMAL_INSTRUMENT.search(title) or lx.TITLE_INSTRUMENT.search(title))
    doc_markets = list(doc.get("markets") or all_markets)
    out = []
    for k, clause in enumerate(split_clauses(f"{title}\n{body}")):
        cats = detect_categories(clause)
        if not cats or not has_operative(clause):
            continue
        formal = title_formal or bool(lx.FORMAL_INSTRUMENT.search(clause))
        provision = bool(lx.NORMATIVE.search(clause))
        if not (formal or provision or tier in ("regulator", "exchange")):
            continue                                            # no identifiable release or operative provision
        scope, markets = detect_scope(clause, title, doc_markets, all_markets)
        mech = {c: mechanism_label(clause, c) for c in cats}
        eff = detect_effective_date(clause, body, release)
        instrument = doc.get("instrument_id") or instrument_of(clause, title)
        canon = sha256_text("|".join([_norm(issuer), _norm(instrument), _norm(clause), scope]))[:16]
        amend = any(w in clause for w in lx.AMEND_CANCEL)
        ctype = "amendment_cancellation" if amend else ("multi_category" if len(cats) > 1 else cats[0])
        out.append({
            "candidate_id": f"{doc['document_id']}-{k:03d}",
            "canonical_action_id": f"PII-{canon}",
            "document_id": doc["document_id"],
            "clause_index": k,
            "evidence": clause[:500],
            "title": title,
            "categories": cats,
            "mechanisms": mech,
            "scope": scope,
            "markets": markets,
            "issuer": issuer,
            "issuer_tier": tier,
            "instrument": instrument,
            "effective_date": eff,
            "confidence": confidence_score(formal, tier, any(v != "unspecified" for v in mech.values()), eff is not None),
            "candidate_type": ctype,
            "availability": pd.Timestamp(release).isoformat() if release is not None else None,
        })
    return out


def extract_corpus(docs: pd.DataFrame, all_markets: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for doc in docs.to_dict("records"):
        rows.extend(extract_document(doc, all_markets))
    return pd.DataFrame(rows)


def consolidate(candidates: pd.DataFrame) -> list[dict]:
    """Merge candidates sharing a canonical identifier into one action of the global register."""
    actions = []
    if candidates.empty:
        return actions
    candidates = candidates.sort_values("availability")
    for canon, grp in candidates.groupby("canonical_action_id", sort=False):
        first = grp.iloc[0]
        cats: list[str] = []
        for cs in grp["categories"]:
            cats.extend(c for c in cs if c not in cats)
        mechs = {}
        for c in cats:
            labels = [m.get(c) for m in grp["mechanisms"] if c in m]
            directed = [l for l in labels if l and l != "unspecified"]
            mechs[c] = Counter(directed).most_common(1)[0][0] if directed else "unspecified"
        markets: list[str] = []
        for ms in grp["markets"]:
            markets.extend(m for m in ms if m not in markets)
        effs = sorted(e for e in grp["effective_date"] if isinstance(e, str))
        support = (grp.groupby("document_id")["availability"].min().sort_values())
        actions.append({
            "canonical_action_id": canon,
            "categories": cats,
            "mechanisms": mechs,
            "scope": first["scope"],
            "markets": markets,
            "issuer": first["issuer"],
            "issuer_tier": first["issuer_tier"],
            "instrument": first["instrument"],
            "effective_date": effs[0] if effs else None,
            "confidence": float(grp["confidence"].max()),
            "availability": first["availability"],
            "supporting_documents": [{"document_id": d, "availability": a} for d, a in support.items()],
            "supporting_document_ids": list(support.index),
            "candidate_type": first["candidate_type"],
            "evidence": first["evidence"],
            "source_document_id": first["document_id"],
        })
    return actions


def candidate_type_counts(candidates: pd.DataFrame) -> dict:
    return dict(Counter(candidates["candidate_type"])) if not candidates.empty else defaultdict(int)
