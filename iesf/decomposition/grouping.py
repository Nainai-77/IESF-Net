"""Frequency grouping of IMFs into four additive scales (Eq. 6, Table A1).

K-means (K = 4, n_init 20, max_iter 300, seed 2026) is fitted on the log of the
positive dominant frequencies of the modes observed at training origins only;
the centres are then held fixed.  Bands are ordered from high to low
frequency, so scale 1 is the fastest oscillation (look-back 10) and scale 4
the slowest; every mode enters exactly one group, an empty group is zero and
the residual (and any mode without a positive peak) enters group 4.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ceemdan import OriginArchive


@dataclass
class FrequencyGrouping:
    log_centres: np.ndarray          # [K] sorted descending (high frequency first)

    @property
    def k(self) -> int:
        return len(self.log_centres)

    def assign(self, freqs: np.ndarray) -> np.ndarray:
        """Group index (0..K-1) of each frequency; non-positive frequencies -> last group."""
        freqs = np.asarray(freqs, dtype=np.float64)
        out = np.full(freqs.shape, self.k - 1, dtype=np.int64)
        pos = freqs > 0
        if pos.any():
            d = np.abs(np.log(freqs[pos])[..., None] - self.log_centres[None, :])
            out[pos] = np.argmin(d, axis=-1)
        return out

    def to_dict(self) -> dict:
        return {"log_centres": self.log_centres.tolist(), "centres_cycles_per_obs": np.exp(self.log_centres).tolist(),
                "periods_obs": (1.0 / np.exp(self.log_centres)).tolist()}


def fit_grouping(archive: OriginArchive, train_end: int, gcfg: dict) -> FrequencyGrouping:
    from sklearn.cluster import KMeans

    sel = archive.mask[:train_end] & (archive.freqs[:train_end] > 0)
    x = np.log(archive.freqs[:train_end][sel]).reshape(-1, 1)
    k = int(gcfg["k"])
    if len(np.unique(x)) < k:
        raise ValueError("not enough distinct training frequencies to fit the grouping")
    km = KMeans(n_clusters=k, n_init=int(gcfg["n_init"]), max_iter=int(gcfg["max_iter"]),
                random_state=int(gcfg["seed"])).fit(x)
    return FrequencyGrouping(np.sort(km.cluster_centers_.ravel())[::-1].copy())


def component_archive(archive: OriginArchive, grouping: FrequencyGrouping) -> np.ndarray:
    """R[t, k]: archived component endpoints; sum over k reconstructs the observed close."""
    groups = grouping.assign(archive.freqs)
    T, K = archive.endpoints.shape[0], grouping.k
    R = np.zeros((T, K), dtype=np.float64)
    for k in range(K):
        R[:, k] = np.where(archive.mask & (groups == k), archive.endpoints, 0.0).sum(1)
    R[:, K - 1] += archive.residual
    return R


def group_full_series(imfs: np.ndarray, residual: np.ndarray, grouping: FrequencyGrouping) -> np.ndarray:
    """Descriptive (full-sample) grouping used only by Figure 3."""
    from .ceemdan import dominant_frequency

    groups = grouping.assign(np.array([dominant_frequency(m) for m in imfs]))
    comps = np.zeros((grouping.k, imfs.shape[1] if imfs.size else len(residual)))
    for j, g in enumerate(groups):
        comps[g] += imfs[j]
    comps[-1] += residual
    return comps
