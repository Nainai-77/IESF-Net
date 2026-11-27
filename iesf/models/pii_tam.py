"""PII-TAM: policy-instrument temporal aggregation module (Section 3.2.2, Eqs. 1-4, Figure 2).

Seven independent category memories h_c (16 units each):
    h̄_{c,t} = exp(-λ_c Δ_t) h_{c,t-1},          λ_c = softplus(γ_c)            (1)
    u_{c,t} = GRU_c([a_{c,t}; age_{c,t}], h̄_{c,t})                              (2)
    h_{c,t} = m_{c,t} u_{c,t} + (1 - m_{c,t}) h̄_{c,t}                           (3)
    p_t     = W_p [h_{1,t}; ...; h_{C,t}] + b_p        (112 -> 64)               (4)
Memory starts at zero; the availability indicator (1 after the first accepted
event) is applied separately to the projected state.  γ_c is initialised so
that every category starts from the same half-life (12 observations) and is
learned independently afterwards.

Replay is exact but event-sparse: between events Eq. (3) reduces to pure decay,
so the GRU only has to be evaluated on the dates at which some category
receives a new action, and every other state follows in closed form from the
cumulative elapsed interval.  The full prefix is replayed under the current
weights at every forward pass (gradients flow through the whole prefix).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedGRUCell(nn.Module):
    """C independent GRU cells evaluated in parallel (PyTorch gate convention r, z, n)."""

    def __init__(self, n_groups: int, input_size: int, hidden_size: int):
        super().__init__()
        self.C, self.D, self.H = n_groups, input_size, hidden_size
        self.W_ih = nn.Parameter(torch.empty(n_groups, 3 * hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.empty(n_groups, 3 * hidden_size, hidden_size))
        self.b_ih = nn.Parameter(torch.zeros(n_groups, 3 * hidden_size))
        self.b_hh = nn.Parameter(torch.zeros(n_groups, 3 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for c in range(self.C):
            nn.init.xavier_uniform_(self.W_ih.data[c], gain=1.0)
            nn.init.orthogonal_(self.W_hh.data[c])               # packed recurrent matrix
        nn.init.zeros_(self.b_ih)
        nn.init.zeros_(self.b_hh)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:   # x [C, D], h [C, H]
        gi = torch.einsum("cgd,cd->cg", self.W_ih, x) + self.b_ih
        gh = torch.einsum("cgh,ch->cg", self.W_hh, h) + self.b_hh
        H = self.H
        r = torch.sigmoid(gi[:, :H] + gh[:, :H])
        z = torch.sigmoid(gi[:, H:2 * H] + gh[:, H:2 * H])
        n = torch.tanh(gi[:, 2 * H:] + r * gh[:, 2 * H:])
        return (1.0 - z) * n + z * h


class PIITAM(nn.Module):
    def __init__(self, n_categories: int = 7, attr_dim: int = 16, hidden: int = 16, out_dim: int = 64,
                 half_life: float = 12.0):
        super().__init__()
        self.C, self.H = n_categories, hidden
        self.cell = GroupedGRUCell(n_categories, attr_dim + 1, hidden)   # 16 attributes + log(1 + event age)
        lam0 = math.log(2.0) / float(half_life)
        self.gamma = nn.Parameter(torch.full((n_categories,), math.log(math.expm1(lam0))))
        self.proj = nn.Linear(n_categories * hidden, out_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    @property
    def decay_rate(self) -> torch.Tensor:
        return F.softplus(self.gamma)

    def half_lives(self) -> torch.Tensor:
        return math.log(2.0) / self.decay_rate

    def forward(self, attr: torch.Tensor, new: torch.Tensor, age: torch.Tensor, dt: torch.Tensor,
                avail: torch.Tensor, t_max: int | None = None, return_states: bool = False):
        """attr [T, C, 16], new [T, C], age [T, C], dt [T], avail [T] -> p [T', 64] for T' = t_max + 1."""
        T = attr.shape[0] if t_max is None else int(t_max) + 1
        attr, new, age, dt, avail = attr[:T], new[:T], age[:T], dt[:T], avail[:T]
        lam = self.decay_rate                                       # [C]
        tau = torch.cumsum(dt, dim=0)                               # cumulative elapsed interval
        events = torch.nonzero(new.amax(dim=1) > 0, as_tuple=False).flatten()
        device = attr.device
        if events.numel() == 0:
            states = torch.zeros(T, self.C, self.H, device=device)
        else:
            h = torch.zeros(self.C, self.H, device=device)
            prev_tau = None
            hist = []
            for e in events.tolist():                               # sequential replay over event dates only
                h_bar = h if prev_tau is None else torch.exp(-lam * (tau[e] - prev_tau)).unsqueeze(-1) * h   # (1)
                x = torch.cat([attr[e], age[e].unsqueeze(-1)], dim=-1)
                u = self.cell(x, h_bar)                                                                     # (2)
                m = new[e].unsqueeze(-1)
                h = m * u + (1.0 - m) * h_bar                                                               # (3)
                hist.append(h)
                prev_tau = tau[e]
            H_e = torch.stack(hist)                                 # [E, C, H]
            pos = torch.arange(T, device=device)
            last = torch.searchsorted(events, pos, right=True) - 1  # latest event date <= t
            valid = (last >= 0).float()
            last_c = last.clamp_min(0)
            gap = (tau - tau[events[last_c]]).clamp_min(0.0)       # elapsed interval since that event
            decay = torch.exp(-lam.unsqueeze(0) * gap.unsqueeze(1)) * valid.unsqueeze(1)   # [T, C]
            states = decay.unsqueeze(-1) * H_e[last_c]
        p = self.proj(states.reshape(T, -1)) * avail.unsqueeze(-1)                             # (4) + availability
        return (p, states) if return_states else p


class PIIAttributeProjector(nn.Module):
    """Current-date PII attributes without memory (Table 12 '+ PII attributes', Table 14 'No institutional memory')."""

    def __init__(self, n_categories: int = 7, attr_dim: int = 16, out_dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(n_categories * (attr_dim + 1), out_dim)

    def forward(self, attr, new, any_new, t_max: int | None = None):
        T = attr.shape[0] if t_max is None else int(t_max) + 1
        x = torch.cat([attr[:T], new[:T].unsqueeze(-1)], dim=-1).reshape(T, -1)
        return self.proj(x) * any_new[:T].unsqueeze(-1)
