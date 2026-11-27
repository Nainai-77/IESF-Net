"""Building blocks shared by IESF-Net and its variants."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(module: nn.Module) -> None:
    """Appendix A.3 initialisation.

    Affine/input matrices: unit-gain Xavier-uniform; packed recurrent matrices:
    orthogonal; biases zero except the LSTM input forget-gate bias (= 1).
    Embeddings and the PII-TAM parameters are initialised by their owners.
    """
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.MultiheadAttention):
            if m.in_proj_weight is not None:
                nn.init.xavier_uniform_(m.in_proj_weight, gain=1.0)
            if m.in_proj_bias is not None:
                nn.init.zeros_(m.in_proj_bias)
        elif isinstance(m, (nn.GRU, nn.LSTM)):
            for name, p in m.named_parameters():
                if name.startswith("weight_ih"):
                    nn.init.xavier_uniform_(p, gain=1.0)
                elif name.startswith("weight_hh"):
                    nn.init.orthogonal_(p)
                elif name.startswith("bias"):
                    nn.init.zeros_(p)
                    if isinstance(m, nn.LSTM) and name.startswith("bias_ih"):
                        h = m.hidden_size
                        with torch.no_grad():
                            p[h:2 * h].fill_(1.0)          # gate order i, f, g, o
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # x [B, L, d]
        return x + self.pe[: x.shape[1]].unsqueeze(0)


class TemporalTransformer(nn.Module):
    """Temporal Transformer encoder for a construct history (one layer, two heads, FF 128)."""

    def __init__(self, d_model=64, n_heads=2, ff_dim=128, layers=1, dropout=0.1, max_len=512):
        super().__init__()
        self.pe = SinusoidalPositionalEncoding(d_model, max_len)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, ff_dim, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def forward(self, x: torch.Tensor, avail: torch.Tensor) -> torch.Tensor:
        """x [B, L, d] (already masked), avail [B, L] in {0, 1}."""
        pad = avail <= 0
        empty = pad.all(dim=1, keepdim=True)
        pad = pad & ~empty                          # windows without any text attend uniformly; outputs re-masked
        h = self.encoder(self.pe(x), src_key_padding_mask=pad)
        return h * avail.unsqueeze(-1)


class BoundedModulation(nn.Module):
    """r(c) = 1 + tanh(W_r c + b_r) in (0, 2), broadcast over the historical axis."""

    def __init__(self, ctx_dim: int, d: int):
        super().__init__()
        self.lin = nn.Linear(ctx_dim, d)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return 1.0 + torch.tanh(self.lin(c))


class ModulatedCrossAttention(nn.Module):
    """Multi-head cross-attention with optional context modulation of Q and K (Eqs. 11-13).

    Source positions with zero availability are excluded before normalisation;
    a zero-valued null candidate is enabled only when the whole source set is
    empty, so a text-free window produces no spurious attention mass.
    """

    def __init__(self, d_model: int, n_heads: int, ctx_dim: int | None = None, modulate: bool = False, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "common width must be divisible by the head count"
        self.h, self.dk = n_heads, d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.modulate = modulate
        if modulate:
            self.rq = BoundedModulation(ctx_dim, d_model)
            self.rk = BoundedModulation(ctx_dim, d_model)
        self.drop = nn.Dropout(dropout)

    def _split(self, x):                                        # [B, L, d] -> [B, h, L, dk]
        B, L, _ = x.shape
        return x.view(B, L, self.h, self.dk).transpose(1, 2)

    def forward(self, query: torch.Tensor, source: torch.Tensor, source_avail: torch.Tensor, ctx: torch.Tensor | None = None):
        q, k, v = self.Wq(query), self.Wk(source), self.Wv(source)
        if self.modulate:
            q = q * self.rq(ctx).unsqueeze(1)
            k = k * self.rk(ctx).unsqueeze(1)
        q, k, v = self._split(q), self._split(k), self._split(v)
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.dk)  # [B, h, Lq, Ls]
        valid = (source_avail > 0)[:, None, None, :]
        logits = logits.masked_fill(~valid, float("-inf"))
        empty = ~(source_avail > 0).any(dim=1)                  # [B]
        null = torch.where(empty, 0.0, float("-inf")).to(logits.dtype)[:, None, None, None]
        null = null.expand(-1, self.h, logits.shape[2], 1)
        attn = torch.softmax(torch.cat([logits, null], dim=-1), dim=-1)[..., :-1]   # null value is zero
        out = self.drop(attn) @ v                                # [B, h, Lq, dk]
        B, _, Lq, _ = out.shape
        return out.transpose(1, 2).reshape(B, Lq, self.h * self.dk), attn


class AttentionBiLSTMHead(nn.Module):
    """Attention-enhanced BiLSTM component head (Eqs. 15-16)."""

    def __init__(self, d_in: int, hidden: int = 64, attn_dim: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(d_in, hidden, batch_first=True, bidirectional=True)
        self.Wa = nn.Linear(2 * hidden, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)
        self.out = nn.Linear(2 * hidden, 1)

    def forward(self, seq: torch.Tensor):
        u, _ = self.lstm(seq)                                   # [B, L, 2h]
        beta = torch.softmax(self.v(torch.tanh(self.Wa(u))).squeeze(-1), dim=-1)
        q = (beta.unsqueeze(-1) * u).sum(1)
        return self.out(q).squeeze(-1), beta


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    w = mask.unsqueeze(-1).to(x.dtype)
    return (x * w).sum(dim) / w.sum(dim).clamp_min(1.0)


__all__ = ["init_weights", "SinusoidalPositionalEncoding", "TemporalTransformer", "BoundedModulation",
           "ModulatedCrossAttention", "AttentionBiLSTMHead", "masked_mean", "F"]
