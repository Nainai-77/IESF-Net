"""Joint estimation of the multiscale family (Section 3.5, Eq. 17, Table A4 steps 4-7).

    L = 1/N Σ_t ((y_{t+1} - ŷ_{t+1}) / s_y)^2 + η/(4N) Σ_{t,k} ((R_{k,t+1} - R̂_{k,t+1}) / s_k)^2

AdamW (β = 0.9/0.999, ε = 1e-8) over every trainable parameter; the frozen
language encoders never enter the optimiser (their outputs are precomputed).
Before each parameter update the institutional states are replayed from the
available prefix under the current weights (inside ``IESFNet.forward``).
Validation RMSE in native price units is evaluated after every epoch; the best
checkpoint is retained under epoch-cap / patience / tolerance rules.  Testing
holds parameters fixed, replays the observed history and carries the
institutional state across partition boundaries.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from ..models.iesf_net import IESFNet, restore
from ..utils import count_parameters, get_logger, set_seed

LOG = get_logger(__name__)


def iesf_loss(r_hat: torch.Tensor, data, idx: torch.Tensor, eta: float, anchor: bool = False) -> torch.Tensor:
    R_hat = restore(r_hat, data, idx, anchor)
    y_hat = R_hat.sum(dim=1)
    y = data["y"][idx]
    s_y = float(data.stats["s_y"])
    price_term = ((y - y_hat) / s_y).pow(2).mean()
    if anchor:                                                  # standardised component-change target (diagnostic option)
        target = (data["R"][idx] - data["R"][idx - 1] - data["dR_mean"].unsqueeze(0)) / data["dR_scale"].unsqueeze(0)
    else:
        target = data["R_std"][idx]
    comp_term = (target - r_hat).pow(2).mean()                 # ((R - R̂)/s_k)^2 averaged over targets and 4 branches
    return price_term + eta * comp_term


@torch.no_grad()
def predict(model: IESFNet, data, idx: np.ndarray, batch_size: int = 256) -> dict:
    """Native-unit forecasts; components are restored and summed in float64."""
    model.eval()
    device = data["y"].device
    st = data.stats
    out_r = []
    for s in range(0, len(idx), batch_size):
        b = torch.as_tensor(idx[s:s + batch_size], dtype=torch.long, device=device)
        out_r.append(model(data, b).double().cpu().numpy())
    r = np.concatenate(out_r, axis=0) if out_r else np.zeros((0, 4))
    if getattr(model.cfg, "residual_anchor", False):
        R_hat = data.R[np.asarray(idx) - 1] + st["dR_mean"][None, :] + st["dR_std"][None, :] * r
    else:
        R_hat = st["R_mean"][None, :].astype(np.float64) + st["R_std"][None, :].astype(np.float64) * r
    return {"index": np.asarray(idx), "R_hat": R_hat, "y_hat": R_hat.sum(axis=1)}


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2)))


@dataclass
class FitResult:
    model: IESFNet
    best_val_rmse: float
    best_epoch: int
    epochs_run: int
    history: list = field(default_factory=list)
    seconds: float = 0.0
    n_params: int = 0


def fit(model: IESFNet, data, hp: dict, ocfg: dict, seed: int, log_every: int = 10) -> FitResult:
    """Train one model under the Appendix A protocol and keep the best validation checkpoint."""
    set_seed(seed)
    device = data["y"].device
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(hp["lr"]), weight_decay=float(hp["weight_decay"]),
                            betas=tuple(ocfg["betas"]), eps=float(ocfg["eps"]))
    part = data.partition
    train_idx = torch.as_tensor(part.targets("train"), dtype=torch.long, device=device)
    val_idx = part.targets("val")
    y_val = data.close[val_idx]
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    bs, eta = int(hp["batch_size"]), float(hp["eta"])
    best, best_state, best_epoch, wait = float("inf"), None, 0, 0
    history, t0 = [], time.time()
    epoch = 0
    for epoch in range(1, int(ocfg["max_epochs"]) + 1):
        model.train()
        perm = train_idx[torch.randperm(len(train_idx), generator=gen).to(device)]
        losses = []
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            loss = iesf_loss(model(data, b), data, b, eta, bool(getattr(model.cfg, "residual_anchor", False)))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, float(ocfg["clip_norm"]))
            opt.step()
            losses.append(float(loss.detach()))
        val = rmse(predict(model, data, val_idx, int(ocfg.get("eval_batch_size", 256)))["y_hat"], y_val)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_rmse": val})
        if not np.isfinite(val):
            LOG.warning("non-finite validation RMSE at epoch %d; stopping", epoch)
            break
        if val < best - float(ocfg["min_delta"]):
            best, best_epoch, wait = val, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
        if log_every and epoch % log_every == 0:
            LOG.info("epoch %3d  loss %.5f  val RMSE %.4f  (best %.4f @ %d)", epoch, history[-1]["train_loss"], val, best, best_epoch)
        if wait >= int(ocfg["patience"]):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return FitResult(model, best, best_epoch, epoch, history, time.time() - t0, count_parameters(model))


def forecast_frame(model: IESFNet, data, split: str, meta: dict, dates: np.ndarray, batch_size: int = 256) -> pd.DataFrame:
    """Observation-aligned forecast table of one split (columns used by every downstream analysis)."""
    idx = data.partition.targets(split)
    pred = predict(model, data, idx, batch_size)
    y = data.close
    df = pd.DataFrame({
        "market": data.market, **meta, "split": split,
        "target_index": idx, "origin_index": idx - 1,
        "origin_date": pd.to_datetime(dates[idx - 1]).strftime("%Y-%m-%d"),
        "target_date": pd.to_datetime(dates[idx]).strftime("%Y-%m-%d"),
        "origin_price": y[idx - 1], "actual": y[idx], "prediction": pred["y_hat"],
    })
    for k in range(pred["R_hat"].shape[1]):
        df[f"component_{k + 1}_pred"] = pred["R_hat"][:, k]
        df[f"component_{k + 1}_true"] = data.R[idx, k]
    return df
