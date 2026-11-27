"""Point-in-time alignment of documents and institutional actions (Section 3.6.2).

A record enters the information set at the first applicable market cut-off on
or after its release.  Date-only releases are placed at the end of the recorded
day; publication time, first availability and effective date are kept apart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def to_timestamp(value, tz: str = "+08:00", date_only_time: str = "23:59:59.999999", date_only: bool | None = None):
    """Parse an availability / publication value to a tz-aware timestamp.

    ``date_only`` forces the end-of-day rule; when None it is inferred (no time component given).
    """
    if value is None or (isinstance(value, float) and np.isnan(value)) or value == "":
        return pd.NaT
    text = str(value).strip()
    inferred_date_only = len(text) <= 10
    ts = pd.Timestamp(text)
    if date_only or (date_only is None and inferred_date_only):
        ts = pd.Timestamp(f"{ts.date()} {date_only_time}")
    if ts.tzinfo is None:
        ts = ts.tz_localize(_tz(tz))
    return ts


def _tz(offset: str):
    import datetime as dt

    sign = 1 if offset[0] == "+" else -1
    hh, mm = offset[1:].split(":")
    return dt.timezone(sign * dt.timedelta(hours=int(hh), minutes=int(mm)))


def market_cutoffs(dates, cutoff: str = "15:00", tz: str = "+08:00") -> np.ndarray:
    """Cut-off timestamp (int64 ns, UTC) of every observation on a market calendar."""
    dates = pd.to_datetime(pd.Series(dates)).dt.normalize()
    hh, mm = (int(x) for x in cutoff.split(":"))
    local = dates + pd.Timedelta(hours=hh, minutes=mm)
    return local.dt.tz_localize(_tz(tz)).dt.tz_convert("UTC").astype("int64").to_numpy()


def assign_to_calendar(availability: pd.Series, cutoffs_ns: np.ndarray) -> np.ndarray:
    """Index of the first observation whose cut-off is on or after availability (-1 if after the calendar).

    Records released before the first observation enter at observation 0.
    """
    avail = pd.to_datetime(availability, utc=True)
    out = np.full(len(avail), -1, dtype=np.int64)
    ok = avail.notna().to_numpy()
    ns = avail[ok].astype("int64").to_numpy()
    idx = np.searchsorted(cutoffs_ns, ns, side="left")
    idx[idx >= len(cutoffs_ns)] = -1
    out[ok] = idx
    return out
