"""Causal, origin-wise improved CEEMDAN (Section 3.3, Table A1).

At every forecast origin t the observable price prefix (capped at the last 256
observations) is decomposed with improved CEEMDAN (EMD-signal / PyEMD):
100 normal noise realisations, epsilon 0.01, noise_scale 1, beta_progress,
noise seed 2026 + t, at most six oscillatory modes plus a separate residual.
Only the *endpoint* of every mode (its value at t) and the dominant rFFT
frequency of the mode are archived: the endpoints of the grouped modes form the
component histories and targets (Eq. 5, Eq. 6).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils import get_logger

LOG = get_logger(__name__)


def make_ceemdan(dcfg: dict):
    from PyEMD import CEEMDAN, EMD

    e = dcfg["emd"]
    emd = EMD(spline_kind=e["spline_kind"], nbsym=int(e["nbsym"]), extrema_detection=e["extrema_detection"],
              DTYPE=np.float64, MAX_ITERATION=int(e["MAX_ITERATION"]), std_thr=e["std_thr"], svar_thr=e["svar_thr"],
              energy_ratio_thr=e["energy_ratio_thr"], range_thr=e["range_thr"], total_power_thr=e["total_power_thr"],
              FIXE=int(e["FIXE"]), FIXE_H=int(e["FIXE_H"]))
    # Attributes are also set explicitly because some PyEMD versions read a subset of kwargs only.
    for key in ("MAX_ITERATION", "FIXE", "FIXE_H"):
        setattr(emd, key, int(e[key]))
    for key in ("std_thr", "svar_thr", "energy_ratio_thr", "range_thr", "total_power_thr"):
        setattr(emd, key, float(e[key]))
    emd.extrema_detection = e["extrema_detection"]
    emd.DTYPE = np.float64
    ce = CEEMDAN(trials=int(dcfg["trials"]), epsilon=float(dcfg["epsilon"]), ext_EMD=emd, parallel=False,
                 noise_scale=float(dcfg["noise_scale"]), noise_kind=dcfg["noise_kind"],
                 range_thr=float(dcfg["range_thr"]), total_power_thr=float(dcfg["total_power_thr"]),
                 beta_progress=bool(dcfg["beta_progress"]))
    return ce


def dominant_frequency(imf: np.ndarray) -> float:
    """Peak frequency (cycles per observation) of the amplitude spectrum; 0 if the peak is at DC."""
    n = len(imf)
    if n < 4:
        return 0.0
    amp = np.abs(np.fft.rfft(imf - 0.0))
    k = int(np.argmax(amp))
    return float(k / n)


def decompose_prefix(prefix: np.ndarray, seed: int, dcfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (imfs [J, n], residual [n]) of one price prefix."""
    prefix = np.asarray(prefix, dtype=np.float64)
    if len(prefix) < int(dcfg["min_prefix"]) or np.std(prefix) < 1e-12:
        return np.zeros((0, len(prefix))), prefix.copy()
    ce = make_ceemdan(dcfg)
    ce.noise_seed(int(seed))
    comps = ce.ceemdan(prefix, max_imf=int(dcfg["max_imf"]))
    imfs = comps[:-1][: int(dcfg["max_imf"])]           # PyEMD appends the residual as the last row
    residual = prefix - imfs.sum(0)                     # residual kept separate; exact additivity in float64
    return imfs, residual


@dataclass
class OriginArchive:
    """Per-origin endpoints of every mode (padded to max_imf) and their dominant frequencies."""

    endpoints: np.ndarray      # [T, J] endpoint value of mode j at origin t (0 where absent)
    freqs: np.ndarray          # [T, J] dominant frequency of mode j at origin t
    mask: np.ndarray           # [T, J] mode present
    residual: np.ndarray       # [T]    residual endpoint
    close: np.ndarray          # [T]

    def save(self, path) -> None:
        np.savez_compressed(path, endpoints=self.endpoints, freqs=self.freqs, mask=self.mask,
                            residual=self.residual, close=self.close)

    @classmethod
    def load(cls, path) -> "OriginArchive":
        z = np.load(path)
        return cls(z["endpoints"], z["freqs"], z["mask"].astype(bool), z["residual"], z["close"])


def _one_origin(t: int, close: np.ndarray, dcfg: dict):
    cap = int(dcfg["history_cap"])
    prefix = close[max(0, t + 1 - cap): t + 1]
    seed = int(dcfg["seed_base"]) + t                  # zero-based origin index on the full calendar
    try:
        imfs, res = decompose_prefix(prefix, seed, dcfg)
    except Exception as exc:  # pragma: no cover - numerical failure is logged and kept as pure trend
        LOG.warning("CEEMDAN failed at origin %d (%s); prefix kept as residual", t, exc)
        imfs, res = np.zeros((0, len(prefix))), prefix.copy()
    return t, [float(m[-1]) for m in imfs], [dominant_frequency(m) for m in imfs], float(res[-1])


def build_origin_archive(close: np.ndarray, dcfg: dict, origins: np.ndarray | None = None,
                         cache_dir=None, chunk: int = 64) -> OriginArchive:
    """Run the causal decomposition at every origin of the calendar (parallel over origins).

    With ``cache_dir`` the origins are processed in chunks that are saved as they finish,
    so an interrupted run resumes from the last completed chunk.  Each origin depends only
    on its own prefix and seed (2026 + t), so chunking does not change any result.
    """
    from pathlib import Path

    from joblib import Parallel, delayed

    close = np.asarray(close, dtype=np.float64)
    T, J = len(close), int(dcfg["max_imf"])
    origins = np.arange(T) if origins is None else np.asarray(origins)
    endpoints, freqs = np.zeros((T, J)), np.zeros((T, J))
    mask, residual = np.zeros((T, J), dtype=bool), close.copy()
    cache = Path(cache_dir) if cache_dir is not None else None
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
    pool = Parallel(n_jobs=int(dcfg.get("n_jobs", -1)), verbose=int(dcfg.get("verbose", 0)))
    for s in range(0, len(origins), chunk):
        block = origins[s:s + chunk]
        part = cache / f"origins_{int(block[0]):05d}_{int(block[-1]):05d}.npz" if cache is not None else None
        if part is not None and part.exists():
            z = np.load(part)
            rows = [(int(t), list(e[m]), list(f[m]), float(r)) for t, e, f, m, r in
                    zip(z["t"], z["endpoints"], z["freqs"], z["mask"], z["residual"])]
        else:
            rows = pool(delayed(_one_origin)(int(t), close, dcfg) for t in block)
            if part is not None:
                e, f, m = np.zeros((len(rows), J)), np.zeros((len(rows), J)), np.zeros((len(rows), J), dtype=bool)
                for i, (_, ends, fr, _) in enumerate(rows):
                    e[i, :len(ends)], f[i, :len(fr)], m[i, :len(ends)] = ends, fr, True
                np.savez(part, t=np.array([r[0] for r in rows]), endpoints=e, freqs=f, mask=m,
                         residual=np.array([r[3] for r in rows]))
            LOG.info("origins %d-%d / %d done", int(block[0]), int(block[-1]), T)
        for t, ends, fr, res in rows:
            n = len(ends)
            endpoints[t, :n], freqs[t, :n], mask[t, :n], residual[t] = ends, fr, True, res
    return OriginArchive(endpoints, freqs, mask, residual, close)
