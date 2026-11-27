"""IESF-Net and the multiscale family it is compared with (Sections 3.1-3.5, Figure 1).

Per scale k = 1..4 with look-back L_k (default 10, 30, 60, 90):
  * numerical branch: GRU_k over [X^std; R_k^std] at positions {i-L_k, ..., i-1}         (Eq. 18)
    -> sequence N_k and summary n_k;
  * scales in ``text_scales`` (default 1-3): conditioning vector
        c_k = tanh(W [p_t; n_k; e_k] + b)        (64 + 64 + 8 = 136 -> 64)              (Eq. 7)
    semantic / CSI histories (shared projection + temporal Transformer, branch-specific
    window), PII history p_i from PII-TAM, PII-CAF fusion                                (Eqs. 8-14);
  * scale 4 (and text-disabled scales) passes N_k directly to its head;
  * attention-enhanced BiLSTM head -> standardised component forecast r̂_k              (Eqs. 15-16);
restored as R̂_k = μ_k + σ_k r̂_k and summed:  ŷ = Σ_k R̂_k                              (Eq. 19).
With ``constructs=()`` the model is the numerical-only multiscale baseline "Numeric".
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn

from .layers import AttentionBiLSTMHead, TemporalTransformer, init_weights
from .pii_caf import PIICAF
from .pii_tam import PIIAttributeProjector, PIITAM

SEQUENCE_CONSTRUCTS = {"sem": "sem", "csi": "csi", "lex": "lex", "count": "count"}   # construct -> array key


@dataclass
class IESFConfig:
    constructs: tuple = ("sem", "csi", "pii")
    fusion: str = "full"
    router: str | None = None
    lookbacks: tuple = (10, 30, 60, 90)
    text_scales: tuple = (0, 1, 2)                 # zero-based: scales 1-3 receive text
    shared_numeric: bool = False
    residual_anchor: bool = False                 # NOT in the manuscript (Eq. 19 restores levels); diagnostic option:
                                                  # R̂_k,i = R_k,i-1 + μ^Δ_k + σ^Δ_k r̂_k, as in the project's earlier MS-ESMF v5 code
    d_model: int = 64
    n_heads: int = 2
    temporal_layers: int = 1
    ff_dim: int = 128
    dropout: float = 0.1
    pii_hidden: int = 16
    half_life: float = 12.0
    scale_emb_dim: int = 8
    n_scales: int = 4
    numeric_in: int = 5                           # 4 market variables + archived component
    n_categories: int = 7
    pii_attr_dim: int = 16
    input_dims: dict = field(default_factory=lambda: {"sem": 768, "csi": 8, "lex": 128, "count": 2})

    def to_dict(self) -> dict:
        return asdict(self)


class IESFNet(nn.Module):
    def __init__(self, cfg: IESFConfig):
        super().__init__()
        self.cfg = cfg
        d, S = cfg.d_model, cfg.n_scales
        self.constructs = tuple(cfg.constructs)
        self.uses_text = len(self.constructs) > 0

        # ---- numerical encoders (scale-specific unless the shared-encoder ablation) --------------
        def gru():
            return nn.GRU(cfg.numeric_in, d, num_layers=cfg.temporal_layers, batch_first=True,
                          dropout=cfg.dropout if cfg.temporal_layers > 1 else 0.0)
        self.num_enc = nn.ModuleList([gru()] if cfg.shared_numeric else [gru() for _ in range(S)])
        self.drop = nn.Dropout(cfg.dropout)
        self.heads = nn.ModuleList([AttentionBiLSTMHead(d, hidden=d, attn_dim=d) for _ in range(S)])

        if self.uses_text:
            max_len = max(cfg.lookbacks) + 1
            self.seq_constructs = [m for m in self.constructs if m in SEQUENCE_CONSTRUCTS]
            self.text_proj = nn.ModuleDict({m: nn.Linear(cfg.input_dims[m], d) for m in self.seq_constructs})
            self.text_enc = nn.ModuleDict({m: TemporalTransformer(d, cfg.n_heads, cfg.ff_dim, cfg.temporal_layers,
                                                                  cfg.dropout, max_len) for m in self.seq_constructs})
            if "pii" in self.constructs:
                self.pii_tam = PIITAM(cfg.n_categories, cfg.pii_attr_dim, cfg.pii_hidden, d, cfg.half_life)
            if "pii_attr" in self.constructs:
                self.pii_attr = PIIAttributeProjector(cfg.n_categories, cfg.pii_attr_dim, d)
            self.scale_emb = nn.Embedding(S, cfg.scale_emb_dim)
            self.ctx = nn.Linear(d + d + cfg.scale_emb_dim, d)                    # 136 -> 64, then tanh
            self.fusion = PIICAF(self.constructs, d, cfg.n_heads, d, cfg.fusion, cfg.router, cfg.dropout)
        init_weights(self)
        if self.uses_text:
            nn.init.normal_(self.scale_emb.weight, mean=0.0, std=0.3)
            if "pii" in self.constructs:                                          # keep PII-TAM's own init
                self.pii_tam.cell.reset_parameters()

    # ------------------------------------------------------------------------------------------
    def numeric_encoder(self, k: int) -> nn.GRU:
        return self.num_enc[0 if self.cfg.shared_numeric else k]

    def _full_streams(self, data, t_max: int) -> dict:
        """Calendar-level projections computed once per forward pass (prefix 0..t_max)."""
        full = {}
        T = t_max + 1
        for m in getattr(self, "seq_constructs", []):
            av = data["doc_avail"][:T]
            full[m] = (self.text_proj[m](data[SEQUENCE_CONSTRUCTS[m]][:T]) * av.unsqueeze(-1), av)
        if "pii" in self.constructs:                                              # replay the institutional prefix
            p = self.pii_tam(data["pii_attr"], data["pii_new"], data["pii_age"], data["pii_dt"],
                             data["pii_avail"], t_max=t_max)
            full["pii"] = (p, data["pii_avail"][:T])
        if "pii_attr" in self.constructs:
            p = self.pii_attr(data["pii_attr"], data["pii_new"], data["pii_any_new"], t_max=t_max)
            full["pii_attr"] = (p, data["pii_any_new"][:T])
        return full

    def forward(self, data, idx: torch.Tensor, return_aux: bool = False, qk_ctx_override: float | None = None):
        """idx: LongTensor of target indices.  Returns standardised component forecasts r̂ [B, 4]."""
        origin = idx - 1
        t_max = int(origin.max().item())
        full = self._full_streams(data, t_max) if self.uses_text else {}
        B = idx.shape[0]
        outs, aux_all = [], {}
        for k, L in enumerate(self.cfg.lookbacks):
            pos = origin.unsqueeze(1) - L + 1 + torch.arange(L, device=idx.device).unsqueeze(0)   # i-L_k .. i-1
            x = torch.cat([data["X_std"][pos], data["R_std"][pos, k:k + 1]], dim=-1)
            N, h_n = self.numeric_encoder(k)(x)
            n_k = h_n[-1]
            N = self.drop(N)
            if self.uses_text and k in self.cfg.text_scales:
                H, A = {}, {}
                for m in self.constructs:
                    z, av = full[m]
                    H[m], A[m] = z[pos], av[pos]
                    if m in self.text_enc:
                        H[m] = self.text_enc[m](H[m], A[m])
                if "pii" in full:
                    p_t = full["pii"][0][origin]
                elif "pii_attr" in full:
                    p_t = full["pii_attr"][0][origin]
                else:
                    p_t = torch.zeros(B, self.cfg.d_model, device=idx.device)
                e_k = self.scale_emb.weight[k].unsqueeze(0).expand(B, -1)
                c = torch.tanh(self.ctx(torch.cat([p_t, n_k, e_k], dim=-1)))                      # Eq. (7)
                F_, aux = self.fusion(N, H, A, c, qk_ctx_override)
                if return_aux:
                    aux_all[k] = {key: (v.detach() if torch.is_tensor(v) else v) for key, v in aux.items()}
            else:
                F_ = N
            r_k, beta = self.heads[k](self.drop(F_))
            if return_aux:
                aux_all.setdefault(k, {})["pool"] = beta.detach()
            outs.append(r_k)
        r_hat = torch.stack(outs, dim=1)
        return (r_hat, aux_all) if return_aux else r_hat


def restore(r_hat: torch.Tensor, data, idx: torch.Tensor | None = None, anchor: bool = False) -> torch.Tensor:
    """Eq. (19): native-unit components R̂_k = μ_k + σ_k r̂_k (differentiable, float32).

    With ``anchor`` (diagnostic, not in the manuscript) r̂_k is the standardised change from the
    last archived component value: R̂_k,i = R_k,i-1 + μ^Δ_k + σ^Δ_k r̂_k."""
    if anchor:
        return data["R"][idx - 1] + data["dR_mean"].unsqueeze(0) + data["dR_scale"].unsqueeze(0) * r_hat
    return data["R_mean"].unsqueeze(0) + data["R_scale"].unsqueeze(0) * r_hat
