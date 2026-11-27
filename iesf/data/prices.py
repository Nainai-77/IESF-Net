"""Market calendars, numerical inputs and chronological target partitions.

Section 3.6.1: positive-price observations define each market calendar; the
horizon is the next *recorded* observation.  Each market reserves its first 90
observations for input construction (Table 2).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import get_logger

LOG = get_logger(__name__)

PRICE_COLUMNS = ["date", "market", "product", "close", "volume", "turnover"]
NUMERIC_NAMES = ["close", "delta_close", "log1p_volume", "log1p_turnover"]


def price_file(raw_dir: str | Path, market: str) -> Path:
    return Path(raw_dir) / "prices" / f"{market}_observed_prices.csv"


def load_prices(path: str | Path) -> pd.DataFrame:
    """Load one market file (columns: date, market, product, close, volume, turnover)."""
    df = pd.read_csv(path)
    missing = {"date", "close"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    for col in ("volume", "turnover"):
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df[df["close"] > 0].copy()                       # positive-price observations define the calendar
    df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return df


def numeric_features(df: pd.DataFrame) -> np.ndarray:
    """Four-variable numerical market vector X: close, one-observation change, log(1+volume), log(1+turnover)."""
    close = df["close"].to_numpy(np.float64)
    delta = np.concatenate([[0.0], np.diff(close)])
    return np.stack(
        [close, delta, np.log1p(df["volume"].to_numpy(np.float64)), np.log1p(df["turnover"].to_numpy(np.float64))],
        axis=1,
    )


@dataclass(frozen=True)
class Partition:
    """Chronological target partition.  Target index i is forecast from origin i-1."""

    n_obs: int
    reserve: int
    n_train: int
    n_val: int
    n_test: int

    @property
    def train_end(self) -> int:
        """Exclusive end of the training rows (reserved history + training targets)."""
        return self.reserve + self.n_train

    @property
    def val_end(self) -> int:
        return self.train_end + self.n_val

    def targets(self, split: str) -> np.ndarray:
        bounds = {
            "train": (self.reserve, self.train_end),
            "val": (self.train_end, self.val_end),
            "test": (self.val_end, self.val_end + self.n_test),
            "all": (self.reserve, self.val_end + self.n_test),
        }
        lo, hi = bounds[split]
        return np.arange(lo, hi, dtype=np.int64)

    def split_of(self, index: int) -> str:
        if index < self.reserve:
            return "reserve"
        if index < self.train_end:
            return "train"
        if index < self.val_end:
            return "val"
        return "test"


def make_partition(n_obs: int, market_cfg: dict, reserve: int) -> Partition:
    n_train, n_val, n_test = (int(v) for v in market_cfg["splits"])
    if reserve + n_train + n_val + n_test != n_obs:
        raise ValueError(
            f"Partition mismatch: reserve {reserve} + targets {n_train}+{n_val}+{n_test} != {n_obs} observations "
            "(Table 2: target counts sum to the observation count minus 90)."
        )
    return Partition(n_obs, reserve, n_train, n_val, n_test)


def check_test_start(dates: pd.Series | np.ndarray, part: Partition, expected: str | None, market: str) -> None:
    if not expected:
        return
    first_test = pd.Timestamp(np.asarray(dates)[part.val_end]).normalize()
    if first_test != pd.Timestamp(expected):
        LOG.warning("%s: first test target %s differs from configured test start %s", market, first_test.date(), expected)
