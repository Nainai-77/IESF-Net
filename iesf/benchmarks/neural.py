"""Neural numerical benchmarks (Section 3.6.4, Appendix A.2, Table A2).

All read the scaled four-variable window and output the standardised
one-observation change.  The architectures reproduce the parameter counts of
Table A2 (e.g. width 64: LSTM L=30 -> 17,985; TCN L=10 -> 38,273, L=30 -> 87,681;
Transformer -> 33,857; PatchTST L=10 -> 34,689; iTransformer L=10 -> 34,241,
L=90 -> 39,361); see tests/test_param_counts.py.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from ..models.layers import SinusoidalPositionalEncoding, init_weights
from ..utils import set_seed

N_VARS = 4


class LSTMBench(nn.Module):
    def __init__(self, L, width, dropout=0.0, **_):
        super().__init__()
        self.rnn = nn.LSTM(N_VARS, width, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(width, 1)

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.out(self.drop(h[:, -1])).squeeze(-1)


class _CausalConv(nn.Module):
    def __init__(self, c_in, c_out, k, dilation):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, k, dilation=dilation)

    def forward(self, x):                                       # x [B, C, L]
        return self.conv(nn.functional.pad(x, (self.pad, 0)))


class _TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, dilation, dropout):
        super().__init__()
        self.c1, self.c2 = _CausalConv(c_in, c_out, 3, dilation), _CausalConv(c_out, c_out, 3, dilation)
        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None

    def forward(self, x):
        h = self.drop(torch.relu(self.c1(x)))
        h = self.drop(torch.relu(self.c2(h)))
        return torch.relu(h + (x if self.down is None else self.down(x)))


def tcn_dilations(L: int, kernel: int = 3) -> list[int]:
    """Dilations 1, 2, 4, ... until the receptive field covers the selected history."""
    dil, rf = [], 1
    while rf < L:
        d = 2 ** len(dil)
        dil.append(d)
        rf += 2 * (kernel - 1) * d                              # two convolutions per residual block
    return dil or [1]


class TCNBench(nn.Module):
    def __init__(self, L, width, dropout=0.0, **_):
        super().__init__()
        chans = [N_VARS] + [width] * len(tcn_dilations(L))
        self.blocks = nn.Sequential(*[_TCNBlock(chans[i], chans[i + 1], d, dropout) for i, d in enumerate(tcn_dilations(L))])
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(width, 1)

    def forward(self, x):
        h = self.blocks(x.transpose(1, 2))[:, :, -1]
        return self.out(self.drop(h)).squeeze(-1)


def _encoder(width, n_heads, ff_dim, dropout):
    return nn.TransformerEncoder(nn.TransformerEncoderLayer(width, n_heads, ff_dim, dropout, batch_first=True), 1,
                                 enable_nested_tensor=False)


class TransformerBench(nn.Module):
    """Temporal Transformer: sinusoidal positions and one two-head encoder layer."""

    def __init__(self, L, width, dropout=0.0, n_heads=2, ff_dim=128, **_):
        super().__init__()
        self.inp = nn.Linear(N_VARS, width)
        self.pe = SinusoidalPositionalEncoding(width, max(L, 8) + 1)
        self.enc = _encoder(width, n_heads, ff_dim, dropout)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(width, 1)

    def forward(self, x):
        h = self.enc(self.pe(self.inp(x)))
        return self.out(self.drop(h[:, -1])).squeeze(-1)


class PatchTSTBench(nn.Module):
    """Shared channel-independent patch embedding (length 8, stride 4, right replication padding),
    one encoder layer and a multivariate-to-single-price flatten readout."""

    def __init__(self, L, width, dropout=0.0, n_heads=2, ff_dim=128, patch_len=8, stride=4, **_):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        self.n_patches = (L + stride - patch_len) // stride + 1
        self.embed = nn.Linear(patch_len, width)
        self.pos = nn.Parameter(torch.zeros(self.n_patches, width))
        self.enc = _encoder(width, n_heads, ff_dim, dropout)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(N_VARS * self.n_patches * width, 1)

    def forward(self, x):                                       # x [B, L, V]
        B = x.shape[0]
        z = x.transpose(1, 2)                                   # [B, V, L]
        z = torch.cat([z, z[..., -1:].expand(-1, -1, self.stride)], dim=-1)    # right replication padding
        patches = z.unfold(-1, self.patch_len, self.stride)    # [B, V, P, patch_len]
        h = self.embed(patches) + self.pos                      # shared across channels
        h = self.enc(h.reshape(B * N_VARS, self.n_patches, -1))
        return self.out(self.drop(h.reshape(B, -1))).squeeze(-1)


class ITransformerBench(nn.Module):
    """Each variate history is a token; a single-target head reads the close token."""

    def __init__(self, L, width, dropout=0.0, n_heads=2, ff_dim=128, **_):
        super().__init__()
        self.embed = nn.Linear(L, width)
        self.enc = _encoder(width, n_heads, ff_dim, dropout)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(width, 1)

    def forward(self, x):
        h = self.enc(self.embed(x.transpose(1, 2)))             # [B, V, width]
        return self.out(self.drop(h[:, 0])).squeeze(-1)         # close is variate 0


ARCHS = {"lstm": LSTMBench, "tcn": TCNBench, "transformer": TransformerBench, "patchtst": PatchTSTBench,
         "itransformer": ITransformerBench}


def build_benchmark(arch: str, L: int, width: int, dropout: float, fixed: dict | None = None) -> nn.Module:
    kwargs = {k: v for k, v in (fixed or {}).items()}
    model = ARCHS[arch](L=L, width=width, dropout=dropout, **kwargs)
    init_weights(model)
    return model


def fit_neural(arch: str, wd, splits: dict, hp: dict, fixed: dict, ocfg: dict, seed: int, device) -> tuple[dict, int, dict]:
    """Train with AdamW (epoch cap 200, patience 20, clip 1, tolerance 1e-6); best validation checkpoint."""
    set_seed(seed)
    L = int(hp["lookback"])
    model = build_benchmark(arch, L, int(hp["width"]), float(hp["dropout"]), fixed).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(hp["lr"]), weight_decay=float(hp["weight_decay"]),
                            betas=tuple(ocfg["betas"]), eps=float(ocfg["eps"]))
    to_t = lambda a: torch.as_tensor(a, device=device)
    Xtr, ytr = to_t(wd.windows(splits["train"], L)), to_t(wd.target(splits["train"]))
    Xs = {s: to_t(wd.windows(ix, L)) for s, ix in splits.items()}
    gen = torch.Generator().manual_seed(int(seed))
    bs = int(hp["batch_size"])
    best, best_state, wait, best_epoch = float("inf"), None, 0, 0

    @torch.no_grad()
    def _pred(split):
        model.eval()
        out = [model(Xs[split][s:s + 1024]).double().cpu().numpy() for s in range(0, len(Xs[split]), 1024)]
        return wd.restore(splits[split], np.concatenate(out))

    for epoch in range(1, int(ocfg["max_epochs"]) + 1):
        model.train()
        perm = torch.randperm(len(Xtr), generator=gen).to(device)
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            loss = nn.functional.mse_loss(model(Xtr[b]), ytr[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(ocfg["clip_norm"]))
            opt.step()
        val = float(np.sqrt(np.mean((_pred("val") - wd.y[splits["val"]]) ** 2)))
        if not np.isfinite(val):
            break
        if val < best - float(ocfg["min_delta"]):
            best, best_state, wait, best_epoch = val, copy.deepcopy(model.state_dict()), 0, epoch
        else:
            wait += 1
            if wait >= int(ocfg["patience"]):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    preds = {s: _pred(s) for s in splits}
    n_params = int(sum(p.numel() for p in model.parameters()))
    return preds, n_params, {"best_val_rmse": best, "best_epoch": best_epoch}
