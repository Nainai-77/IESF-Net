"""Learned attention, routing and conditional-input contrasts of a trained IESF-Net (Figures 14-16)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch


@torch.no_grad()
def collect_aux(model, data, idx: np.ndarray, qk_ctx_override: float | None = None, batch_size: int = 64) -> dict:
    """Concatenate auxiliary outputs over target indices: per scale route [N, M], attn_* [N, H, L, L]."""
    model.eval()
    device = data["y"].device
    out: dict = {}
    for s in range(0, len(idx), batch_size):
        b = torch.as_tensor(idx[s:s + batch_size], dtype=torch.long, device=device)
        _, aux = model(data, b, return_aux=True, qk_ctx_override=qk_ctx_override)
        for k, d in aux.items():
            for key, val in d.items():
                if torch.is_tensor(val):
                    out.setdefault(k, {}).setdefault(key, []).append(val.cpu().numpy())
    return {k: {key: np.concatenate(v, 0) for key, v in d.items()} for k, d in out.items()}


def attention_maps(aux: dict, scales=(0, 1, 2)) -> dict:
    """Head- and origin-averaged attention matrices (query lag x source lag) per scale and direction."""
    maps = {}
    for k in scales:
        for key, name in (("attn_n2t", "N→T"), ("attn_t2n", "T→N")):
            if k in aux and key in aux[k]:
                maps[(k, name)] = aux[k][key].mean(axis=(0, 1))
    return maps


def routing_frame(aux: dict, idx: np.ndarray, dates: np.ndarray, constructs: tuple, scales=(0, 1, 2),
                  lookbacks: tuple = (10, 30, 60, 90)) -> pd.DataFrame:
    rows = []
    for k in scales:
        if k not in aux or "route" not in aux[k]:
            continue
        pi = aux[k]["route"]
        for j, i in enumerate(idx):
            rows.append({"scale": k + 1, "lookback": int(lookbacks[k]), "target_index": int(i), "origin_date": str(dates[i - 1]),
                         **{m: float(pi[j, c]) for c, m in enumerate(constructs)}})
    return pd.DataFrame(rows)


def conditioning_contrast(model, data, idx: np.ndarray, low: float = -0.9, high: float = 0.9, scales=(0, 1, 2)) -> dict:
    """Attention under constant low / high conditioning of the query-key pathway, and their difference."""
    lo = attention_maps(collect_aux(model, data, idx, qk_ctx_override=low), scales)
    hi = attention_maps(collect_aux(model, data, idx, qk_ctx_override=high), scales)
    return {key: {"low": lo[key], "high": hi[key], "diff": hi[key] - lo[key]} for key in lo if key in hi}


def save_npz(maps: dict, path) -> None:
    flat = {}
    for key, val in maps.items():
        tag = f"s{key[0] + 1}_{'n2t' if key[1] == 'N→T' else 't2n'}"
        if isinstance(val, dict):
            for sub, arr in val.items():
                flat[f"{tag}_{sub}"] = arr
        else:
            flat[tag] = val
    np.savez_compressed(path, **flat)
