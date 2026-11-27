"""Train / tune / forecast one multiscale variant in one market."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..data.dataset import MarketData
from ..models.iesf_net import IESFConfig, IESFNet
from ..training.trainer import fit, forecast_frame
from ..training.tuning import run_tpe
from ..utils import ensure_dir, get_device, get_logger, load_json, save_json
from .registry import variant_spec

LOG = get_logger(__name__)


# ------------------------------------------------------------------------------------------- paths
def processed_path(cfg: dict, market: str) -> Path:
    return Path(cfg["paths"]["processed_dir"]) / f"{market}.npz"


def run_dir(cfg: dict, kind: str, market: str, model: str, seed: int | None = None) -> Path:
    p = Path(cfg["paths"]["output_dir"]) / kind / market / model
    return p / f"seed{seed}" if seed is not None else p


def load_market(cfg: dict, market: str, device=None):
    md = MarketData.load(processed_path(cfg, market))
    return md, md.tensors(device or get_device(cfg["optim"].get("device", "auto")))


# ------------------------------------------------------------------------------------- hyperparams
def iesf_hparams(cfg: dict, market: str, variant: str = "IESF-Net") -> dict:
    """Validation-selected IESF-Net configuration (tuned JSON when present, else the paper setting)."""
    hp = dict(cfg["iesf"]["defaults"])
    own = run_dir(cfg, "tuning", market, variant) / "best.json"
    shared = run_dir(cfg, "tuning", market, "IESF-Net") / "best.json"
    for path in ((own,) if cfg["experiments"].get("tune_each_variant") else ()) + (shared,):
        if path.exists():
            hp.update(load_json(path)["best_params"])
            break
    return hp


def lookbacks_of(cfg: dict, value) -> tuple:
    if isinstance(value, str):
        return tuple(cfg["iesf"]["lookback_presets"][value])
    return tuple(value)


def build_model(cfg: dict, variant: str, hp: dict, md: MarketData) -> IESFNet:
    spec = variant_spec(variant)
    lookbacks = lookbacks_of(cfg, spec["lookbacks"] or hp["lookbacks"])
    dims = {"sem": int(md["sem"].shape[1]) if "sem" in md.arrays else 768,
            "csi": int(md["csi"].shape[1]) if "csi" in md.arrays else 8,
            "lex": int(md["lex"].shape[1]) if "lex" in md.arrays else 128,
            "count": int(md["count"].shape[1]) if "count" in md.arrays else 2}
    mcfg = IESFConfig(
        constructs=tuple(spec["constructs"]), fusion=spec["fusion"], router=spec["router"],
        lookbacks=lookbacks, text_scales=tuple(spec["text_scales"]), shared_numeric=bool(spec["shared_numeric"]),
        residual_anchor=bool(hp.get("residual_anchor", False)),
        d_model=int(hp["d_model"]), n_heads=int(hp["n_heads"]), temporal_layers=int(hp["temporal_layers"]),
        ff_dim=int(hp["ff_dim"]), dropout=float(hp["dropout"]), pii_hidden=int(hp["pii_hidden"]),
        half_life=float(hp["half_life"]), scale_emb_dim=int(hp["scale_emb_dim"]),
        n_categories=int(md["pii_attr"].shape[1]) if "pii_attr" in md.arrays else 7,
        pii_attr_dim=int(md["pii_attr"].shape[2]) if "pii_attr" in md.arrays else 16,
        input_dims=dims,
    )
    if max(lookbacks) > md.partition.reserve:
        raise ValueError("look-back exceeds the reserved input history")
    return IESFNet(mcfg)


# ------------------------------------------------------------------------------------------ runs
def train_and_forecast(cfg: dict, market: str, variant: str, seed: int, hp: dict | None = None,
                       splits=("val", "test"), save: bool = True):
    md, data = load_market(cfg, market)
    hp = hp or iesf_hparams(cfg, market, variant)
    model = build_model(cfg, variant, hp, md)
    LOG.info("%s | %s | seed %d | %d parameters", market, variant, seed, sum(p.numel() for p in model.parameters()))
    res = fit(model, data, hp, cfg["optim"], seed)
    frames = {s: forecast_frame(res.model, data, s, {"model": variant, "seed": seed}, md.dates,
                                int(cfg["optim"].get("eval_batch_size", 256))) for s in splits}
    if save:
        out = ensure_dir(run_dir(cfg, "forecasts", market, variant, seed))
        for s, df in frames.items():
            df.to_csv(out / f"{s}_predictions.csv", index=False)
        pd.DataFrame(res.history).to_csv(out / "training_log.csv", index=False)
        save_json({"market": market, "variant": variant, "seed": seed, "hparams": hp,
                   "model_config": res.model.cfg.to_dict(), "best_val_rmse": res.best_val_rmse,
                   "best_epoch": res.best_epoch, "epochs_run": res.epochs_run, "n_params": res.n_params,
                   "seconds": res.seconds}, out / "run.json")
        ck = ensure_dir(run_dir(cfg, "checkpoints", market, variant))
        torch.save({"state_dict": res.model.state_dict(), "model_config": res.model.cfg.to_dict(), "hparams": hp},
                   ck / f"seed{seed}.pt")
    return res, frames


def load_trained(cfg: dict, market: str, variant: str, seed: int, device=None):
    md, data = load_market(cfg, market, device)
    ck = torch.load(run_dir(cfg, "checkpoints", market, variant) / f"seed{seed}.pt", map_location=data["y"].device,
                    weights_only=False)
    mc = dict(ck["model_config"])
    for key in ("constructs", "lookbacks", "text_scales"):
        mc[key] = tuple(mc[key])
    model = IESFNet(IESFConfig(**mc)).to(data["y"].device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return md, data, model


def tune_variant(cfg: dict, market: str, variant: str = "IESF-Net") -> dict:
    """Table A4 steps 1-3 for one multiscale variant."""
    md, data = load_market(cfg, market)
    base = dict(cfg["iesf"]["defaults"])
    space = cfg["iesf"]["search_space"]
    if variant_spec(variant)["lookbacks"] is not None:          # window variants keep their own windows
        space = {k: v for k, v in space.items() if k != "lookbacks"}

    def objective(params: dict):
        hp = {**base, **params}
        if int(hp["d_model"]) % int(hp["n_heads"]):
            return float("inf"), {}
        vals, n_params = [], 0
        for seed in cfg["seeds"]["tuning"]:
            model = build_model(cfg, variant, hp, md)
            res = fit(model, data, hp, cfg["optim"], int(seed), log_every=0)
            vals.append(res.best_val_rmse)
            n_params = res.n_params
        lb = lookbacks_of(cfg, variant_spec(variant)["lookbacks"] or hp["lookbacks"])
        return float(np.mean(vals)), {"n_params": n_params, "total_lookback": int(sum(lb))}

    return run_tpe(space, objective, cfg["tuning"], int(cfg["seeds"]["sampler"]),
                   run_dir(cfg, "tuning", market, variant), name=f"{market}-{variant}")
