"""Document corpus: cleaning, normalised text identity, deduplication and market assignment.

Document records (one JSON object per line) follow the project schema:
    document_id, title, body, publish_datetime, date_only_publish_flag,
    availability (optional, ISO time), markets (list), source_url, source_type,
    publisher, issuer (optional), instrument_id (optional), version (optional)
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import get_logger, read_jsonl, sha256_text
from .alignment import _tz, assign_to_calendar, market_cutoffs, to_timestamp

LOG = get_logger(__name__)

_HTML_TAG = re.compile(r"<[^>]+>")
_SPACES = re.compile(r"[ \t　\xa0]+")
_BOILERPLATE = re.compile(r"(责任编辑|编辑|来源|原标题|声明|免责声明)[:：][^\n。]{0,60}")
_NON_CONTENT = re.compile(r"[\W_]+", flags=re.UNICODE)


def clean_text(text: str | None) -> str:
    """Remove markup, boilerplate bylines and redundant whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = _HTML_TAG.sub(" ", text)
    text = _BOILERPLATE.sub(" ", text)
    text = _SPACES.sub(" ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def normalised_identity(title: str, body: str) -> str:
    """Normalised text identity used for deduplication (case, width, whitespace and punctuation removed)."""
    key = unicodedata.normalize("NFKC", f"{title}\n{body}").lower()
    return sha256_text(_NON_CONTENT.sub("", key))


def load_corpus(path: str | Path, tz: str = "+08:00", date_only_time: str = "23:59:59.999999") -> pd.DataFrame:
    rows = read_jsonl(path)
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"empty corpus: {path}")
    for col, default in (("title", ""), ("body", ""), ("markets", None), ("source_type", "unknown"),
                         ("publisher", ""), ("issuer", ""), ("instrument_id", ""), ("version", "")):
        if col not in df.columns:
            df[col] = default
    df["title"] = df["title"].map(clean_text)
    df["body"] = df["body"].map(clean_text)
    df["text_identity"] = [normalised_identity(t, b) for t, b in zip(df["title"], df["body"])]
    if "document_id" not in df.columns:
        df["document_id"] = df["text_identity"].str[:32]

    def _availability(row) -> pd.Timestamp:
        if isinstance(row.get("availability"), str) and row.get("availability"):
            return to_timestamp(row["availability"], tz, date_only_time)
        date_only = bool(row.get("date_only_publish_flag", 0))
        return to_timestamp(row.get("publish_datetime"), tz, date_only_time, date_only=date_only or None)

    df["availability_ts"] = pd.to_datetime([_availability(r) for r in df.to_dict("records")], utc=True)
    df = df[(df["title"].str.len() + df["body"].str.len()) > 0]
    df = df[df["availability_ts"].notna()].copy()
    return df.reset_index(drop=True)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one record per normalised text identity (the earliest available), merging market scopes."""
    df = df.sort_values(["availability_ts", "document_id"]).copy()
    markets = df.groupby("text_identity")["markets"].apply(_union_lists)
    out = df.drop_duplicates("text_identity", keep="first").copy()
    out["markets"] = out["text_identity"].map(markets)
    LOG.info("deduplication: %d -> %d documents", len(df), len(out))
    return out.reset_index(drop=True)


def _union_lists(values) -> list:
    seen: list = []
    for v in values:
        for m in (v if isinstance(v, (list, tuple, np.ndarray)) else [v]):
            if m is not None and m not in seen:
                seen.append(m)
    return seen


def align_documents(df: pd.DataFrame, market: str, dates, cutoff: str, tz: str, pre_calendar_days: int | None = 1) -> pd.DataFrame:
    """Documents of one market with the index of the observation whose information set they enter.

    The daily semantic / CSI channels summarise *newly available* documents, so documents released
    more than ``pre_calendar_days`` days before the first observation of the calendar are not piled
    onto observation 0 (None keeps them).  Institutional actions are aligned separately and do enter
    at observation 0, where they initialise the category memories.
    """
    in_market = df["markets"].map(lambda ms: ms is None or len(ms) == 0 or market in ms)
    sub = df[in_market].copy()
    if pre_calendar_days is not None:
        first = pd.Timestamp(pd.to_datetime(pd.Series(dates)).min()).tz_localize(_tz(tz))
        sub = sub[sub["availability_ts"] >= first - pd.Timedelta(days=int(pre_calendar_days))].copy()
    cut = market_cutoffs(dates, cutoff, tz)
    sub["origin_index"] = assign_to_calendar(sub["availability_ts"], cut)
    sub = sub[sub["origin_index"] >= 0].copy()
    sub["origin_date"] = pd.to_datetime(np.asarray(dates)[sub["origin_index"].to_numpy()])
    return sub.sort_values(["origin_index", "document_id"]).reset_index(drop=True)
