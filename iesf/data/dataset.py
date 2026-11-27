"""Model-ready market arrays and their torch view.

All arrays live on the full market observation calendar (length T).  Target
index i (>= reserve) is forecast from origin i-1; scale k reads the history
positions {i-L_k, ..., i-1} (Eq. 18).  Training statistics are estimated on
rows [0, train_end): the reserved input history plus the training targets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .prices import Partition

ARRAY_KEYS = ["close", "X_raw", "R", "sem", "csi", "lex", "count", "doc_avail",
              "pii_attr", "pii_new", "pii_age", "pii_dt", "pii_avail"]


def _safe_std(x: np.ndarray, axis=0) -> np.ndarray:
    s = np.std(x, axis=axis)                       # population standard deviation
    return np.where(s < 1e-6, 1.0, s)              # SDs below 1e-6 are set to one


@dataclass
class MarketData:
    market: str
    dates: np.ndarray
    arrays: dict
    partition: Partition

    @property
    def n_obs(self) -> int:
        return len(self.dates)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.arrays[key]

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        p = self.partition
        np.savez_compressed(path, market=self.market, dates=self.dates.astype("datetime64[D]").astype(str),
                            partition=np.array([p.n_obs, p.reserve, p.n_train, p.n_val, p.n_test]),
                            **{k: self.arrays[k] for k in ARRAY_KEYS if k in self.arrays})

    @classmethod
    def load(cls, path: str | Path) -> "MarketData":
        z = np.load(path, allow_pickle=False)
        n_obs, reserve, n_train, n_val, n_test = (int(v) for v in z["partition"])
        arrays = {k: z[k] for k in ARRAY_KEYS if k in z.files}
        return cls(str(z["market"]), z["dates"].astype("datetime64[D]"), arrays,
                   Partition(n_obs, reserve, n_train, n_val, n_test))

    # ------------------------------------------------------------------ statistics
    def stats(self) -> dict:
        e = self.partition.train_end
        X, R, y = self["X_raw"][:e], self["R"][:e], self["close"][:e]
        tr = self.partition.targets("train")
        dy = self["close"][tr] - self["close"][tr - 1]
        dR = self["R"][tr] - self["R"][tr - 1]
        return {
            "dR_mean": dR.mean(0), "dR_std": _safe_std(dR),
            "X_mean": X.mean(0), "X_std": _safe_std(X),
            "R_mean": R.mean(0), "R_std": _safe_std(R),
            "s_y": float(_safe_std(y)),
            "dy_mean": float(dy.mean()), "dy_std": float(_safe_std(dy)),
        }

    def tensors(self, device="cpu") -> "MarketTensors":
        return MarketTensors.from_market(self, device)


@dataclass
class MarketTensors:
    """Torch view used by the IESF family (float32 inputs, float64 restoration constants)."""

    market: str
    partition: Partition
    stats: dict
    close: np.ndarray                     # float64, native units
    R: np.ndarray                         # float64 archived components
    t: dict = field(default_factory=dict) # name -> torch tensor on device

    @classmethod
    def from_market(cls, md: MarketData, device="cpu") -> "MarketTensors":
        import torch

        st = md.stats()
        f32 = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32), device=device)
        t = {
            "X_std": f32((md["X_raw"] - st["X_mean"]) / st["X_std"]),
            "R_std": f32((md["R"] - st["R_mean"]) / st["R_std"]),
            "y": f32(md["close"]),
            "R_mean": f32(st["R_mean"]), "R_scale": f32(st["R_std"]),
            "R": f32(md["R"]), "dR_mean": f32(st["dR_mean"]), "dR_scale": f32(st["dR_std"]),
        }
        for key in ("sem", "csi", "lex", "count", "doc_avail", "pii_attr", "pii_new", "pii_age", "pii_dt", "pii_avail"):
            if key in md.arrays:
                t[key] = f32(md[key])
        if "pii_new" in t:
            t["pii_any_new"] = (t["pii_new"].amax(dim=1) > 0).float()
        return cls(md.market, md.partition, st, md["close"].astype(np.float64), md["R"].astype(np.float64), t)

    def __getitem__(self, key: str):
        return self.t[key]

    def has(self, key: str) -> bool:
        return key in self.t
