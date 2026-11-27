"""Daily aggregation of document-level representations onto a market calendar (Section 3.2.1).

Deduplicated documents receive equal weight within a market date.  On dates
without documents the observation is zero and the availability indicator is
zero, whereas an observed but neutral document has availability one.
"""
from __future__ import annotations

import numpy as np

CSI_FEATURES = [
    "mean_tone",            # mean positive-minus-negative tone
    "mean_positive",
    "mean_neutral",
    "mean_negative",
    "mean_abs_tone",        # tone intensity
    "tone_dispersion",      # population SD of tone (0 on single-document dates)
    "log1p_doc_count",
    "doc_available",
]


def doc_availability(origin_index: np.ndarray, n_obs: int) -> np.ndarray:
    avail = np.zeros(n_obs, dtype=np.float32)
    avail[np.asarray(origin_index, dtype=np.int64)] = 1.0
    return avail


def daily_mean(vectors: np.ndarray, origin_index: np.ndarray, n_obs: int) -> np.ndarray:
    """Equal-weight mean of document vectors per observation (zeros where no document)."""
    vectors = np.asarray(vectors, dtype=np.float64)
    idx = np.asarray(origin_index, dtype=np.int64)
    out = np.zeros((n_obs, vectors.shape[1]), dtype=np.float64)
    cnt = np.bincount(idx, minlength=n_obs).astype(np.float64)
    np.add.at(out, idx, vectors)
    ok = cnt > 0
    out[ok] /= cnt[ok, None]
    return out.astype(np.float32)


def daily_csi(probs: np.ndarray, origin_index: np.ndarray, n_obs: int) -> np.ndarray:
    """Eight daily CSI features from document probabilities ordered (neutral, positive, negative)."""
    probs = np.asarray(probs, dtype=np.float64)
    idx = np.asarray(origin_index, dtype=np.int64)
    neu, pos, neg = probs[:, 0], probs[:, 1], probs[:, 2]
    tone = pos - neg
    cnt = np.bincount(idx, minlength=n_obs).astype(np.float64)
    ok = cnt > 0

    def _mean(values):
        s = np.bincount(idx, weights=values, minlength=n_obs)
        m = np.zeros(n_obs)
        m[ok] = s[ok] / cnt[ok]
        return m

    mean_tone = _mean(tone)
    var = np.zeros(n_obs)
    var[ok] = _mean(tone**2)[ok] - mean_tone[ok] ** 2
    out = np.stack(
        [mean_tone, _mean(pos), _mean(neu), _mean(neg), _mean(np.abs(tone)),
         np.sqrt(np.clip(var, 0.0, None)), np.log1p(cnt), ok.astype(np.float64)],
        axis=1,
    )
    out[~ok] = 0.0
    return out.astype(np.float32)


def daily_count(origin_index: np.ndarray, n_obs: int) -> np.ndarray:
    """"News count" representation of Table 12: log(1 + count) and availability."""
    cnt = np.bincount(np.asarray(origin_index, dtype=np.int64), minlength=n_obs).astype(np.float64)
    return np.stack([np.log1p(cnt), (cnt > 0).astype(np.float64)], axis=1).astype(np.float32)
