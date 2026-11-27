"""PII-CAF: institution-conditioned attention fusion (Section 3.4, Eqs. 7-14, Figure 4) and its controls.

Fusion modes (Section 3.4 / Table 13):
    static       concatenation of the encoded streams
    gated        gated weighting of the routed text against the numerical path
    n2t / t2n    one retrieval direction (numerical-to-text / text-to-numerical)
    bidirectional both directions, global construct weights, no conditioning
    feature      contextual feature gates + context routing          (Eqs. 8-10)
    matching     context modulation of queries and keys, global weights (Eq. 11)
    full         both conditioning paths  (IESF-Net)
Router modes: context (Eq. 9) | global (learned constant logits) | equal (ablation).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .layers import ModulatedCrossAttention

MODE_FLAGS = {
    "static":        dict(kind="static"),
    "gated":         dict(kind="gated", router="global"),
    "n2t":           dict(kind="attn", dirs=("n2t",), gates=False, qk=False, router="global"),
    "t2n":           dict(kind="attn", dirs=("t2n",), gates=False, qk=False, router="global"),
    "bidirectional": dict(kind="attn", dirs=("n2t", "t2n"), gates=False, qk=False, router="global"),
    "feature":       dict(kind="attn", dirs=("n2t", "t2n"), gates=True, qk=False, router="context"),
    "matching":      dict(kind="attn", dirs=("n2t", "t2n"), gates=False, qk=True, router="global"),
    "full":          dict(kind="attn", dirs=("n2t", "t2n"), gates=True, qk=True, router="context"),
}
GATED_CONSTRUCTS = {"sem", "csi", "lex", "count"}   # Eq. (8): m in {sem, CSI}; PII stays content


class PIICAF(nn.Module):
    def __init__(self, constructs: tuple[str, ...], d_model: int = 64, n_heads: int = 2, ctx_dim: int = 64,
                 mode: str = "full", router: str | None = None, dropout: float = 0.1):
        super().__init__()
        if mode not in MODE_FLAGS:
            raise ValueError(f"unknown fusion mode {mode}")
        self.constructs = tuple(constructs)
        self.flags = dict(MODE_FLAGS[mode])
        self.mode = mode
        self.router = router or self.flags.get("router", "global")
        self.kind = self.flags["kind"]
        d, M = d_model, len(self.constructs)

        if self.kind == "static":
            self.static = nn.Linear(d * (1 + M), d)
        else:
            self.V = nn.ModuleDict({m: nn.Linear(d, d) for m in self.constructs})           # V_m of Eq. (10)
            if self.router == "context":
                self.route = nn.Linear(ctx_dim, M)                                           # w_m, b_m of Eq. (9)
            elif self.router == "global":
                self.route_logits = nn.Parameter(torch.zeros(M))
            if self.flags.get("gates"):
                self.gates = nn.ModuleDict({m: nn.Linear(ctx_dim, d) for m in self.constructs if m in GATED_CONSTRUCTS})
        if self.kind == "gated":
            self.gate = nn.Linear(2 * d, d)
        if self.kind == "attn":
            dirs = self.flags["dirs"]
            self.dirs = dirs
            qk = self.flags["qk"]
            if "n2t" in dirs:
                self.att_n2t = ModulatedCrossAttention(d, n_heads, ctx_dim, modulate=qk, dropout=dropout)
                self.b_x = nn.Parameter(torch.zeros(d))                                      # bounded output gate b_x
            if "t2n" in dirs:
                self.att_t2n = ModulatedCrossAttention(d, n_heads, ctx_dim, modulate=qk, dropout=dropout)
                self.b_z = nn.Parameter(torch.zeros(d))                                      # bounded output gate b_z
            self.W_f = nn.Linear(d * (1 + len(dirs)), d)
        self.norm = nn.LayerNorm(d)

    # ------------------------------------------------------------------------------------------
    def routing_weights(self, c: torch.Tensor) -> torch.Tensor:
        B, M = c.shape[0], len(self.constructs)
        if self.router == "context":
            return torch.softmax(self.route(c), dim=-1)                                       # Eq. (9)
        if self.router == "global":
            return torch.softmax(self.route_logits, dim=-1).unsqueeze(0).expand(B, M)
        return torch.full((B, M), 1.0 / M, device=c.device)                                  # equal-weight ablation

    def route_text(self, H: dict, c: torch.Tensor):
        """Eqs. (8)-(10): feature gates, construct routing and one textual vector per date."""
        pi = self.routing_weights(c)
        z, gates = 0.0, {}
        for j, m in enumerate(self.constructs):
            h = H[m]
            if self.flags.get("gates") and m in GATED_CONSTRUCTS:
                g = torch.sigmoid(self.gates[m](c))                                           # Eq. (8)
                gates[m] = g
                h = g.unsqueeze(1) * h
            z = z + pi[:, j, None, None] * self.V[m](h)                                       # Eq. (10)
        return z, pi, gates

    def forward(self, N: torch.Tensor, H: dict, A: dict, c: torch.Tensor, qk_ctx_override: float | None = None):
        """N [B, L, d] numerical sequence; H[m] [B, L, d] construct histories; A[m] [B, L] availability;
        c [B, ctx] conditioning vector.  Returns F [B, L, d] and auxiliary tensors for interpretation."""
        aux: dict = {}
        avail = torch.stack([A[m] for m in self.constructs], 0).amax(0)                       # text available at i
        if self.kind == "static":
            F_ = self.norm(self.static(torch.cat([N] + [H[m] for m in self.constructs], dim=-1)))
            return F_, aux
        Z, pi, gates = self.route_text(H, c)
        aux["route"], aux["gates"] = pi, gates
        if self.kind == "gated":
            g = torch.sigmoid(self.gate(torch.cat([N, Z], dim=-1))) * avail.unsqueeze(-1)
            return self.norm(N + g * Z), aux
        ctx = c if qk_ctx_override is None else torch.full_like(c, float(qk_ctx_override))
        parts = [N]
        if "n2t" in self.dirs:                                                                # numerical queries, text keys/values
            C_xz, a = self.att_n2t(N, Z, avail, ctx)
            parts.append(torch.sigmoid(self.b_x) * C_xz)
            aux["attn_n2t"] = a
        if "t2n" in self.dirs:                                                                # text queries, numerical keys/values
            ones = torch.ones(N.shape[:2], device=N.device)
            C_zx, a = self.att_t2n(Z, N, ones, ctx)
            parts.append(torch.sigmoid(self.b_z) * C_zx * avail.unsqueeze(-1))
            aux["attn_t2n"] = a
        return self.norm(self.W_f(torch.cat(parts, dim=-1))), aux                              # Eq. (14)
