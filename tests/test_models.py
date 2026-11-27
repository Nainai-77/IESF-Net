"""Unit tests: parameter counts (Table A2), PII-TAM exactness, causality, masking, statistics."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iesf.benchmarks.neural import build_benchmark  # noqa: E402
from iesf.data.dataset import MarketTensors  # noqa: E402
from iesf.data.prices import Partition  # noqa: E402
from iesf.evaluation import metrics as M  # noqa: E402
from iesf.evaluation.stats import bootstrap_reduction, dm_test, hac_lag, holm  # noqa: E402
from iesf.experiments.registry import VARIANTS, variant_spec  # noqa: E402
from iesf.models.iesf_net import IESFConfig, IESFNet  # noqa: E402
from iesf.models.layers import ModulatedCrossAttention  # noqa: E402
from iesf.models.pii_tam import PIITAM  # noqa: E402


# ----------------------------------------------------------------------------- Table A2 sizes
@pytest.mark.parametrize("arch,L,expected", [
    ("lstm", 30, 17985), ("tcn", 10, 38273), ("tcn", 30, 87681), ("transformer", 90, 33857),
    ("transformer", 30, 33857), ("patchtst", 10, 34689), ("itransformer", 10, 34241), ("itransformer", 90, 39361),
])
def test_benchmark_parameter_counts(arch, L, expected):
    fixed = {"n_heads": 2, "ff_dim": 128} if arch in ("transformer", "patchtst", "itransformer") else {}
    if arch == "patchtst":
        fixed.update(patch_len=8, stride=4)
    model = build_benchmark(arch, L, 64, 0.1, fixed)
    assert sum(p.numel() for p in model.parameters()) == expected
    out = model(torch.randn(3, L, 4))
    assert out.shape == (3,)


# ----------------------------------------------------------------------------- PII-TAM
def _naive_tam(tam: PIITAM, attr, new, age, dt, avail):
    """Literal step-by-step Eqs. (1)-(4)."""
    T, C, _ = attr.shape
    h = torch.zeros(C, tam.H)
    lam = tam.decay_rate
    states = []
    for t in range(T):
        h_bar = torch.exp(-lam * dt[t]).unsqueeze(-1) * h
        u = tam.cell(torch.cat([attr[t], age[t].unsqueeze(-1)], -1), h_bar)
        m = new[t].unsqueeze(-1)
        h = m * u + (1 - m) * h_bar
        states.append(h)
    S = torch.stack(states)
    return tam.proj(S.reshape(T, -1)) * avail.unsqueeze(-1)


def test_pii_tam_event_sparse_replay_is_exact():
    torch.manual_seed(0)
    T, C = 60, 7
    tam = PIITAM(C, 16, 16, 64, half_life=12)
    new = (torch.rand(T, C) < 0.05).float()
    new[0] = 0
    attr = torch.randn(T, C, 16) * new.unsqueeze(-1)
    age = torch.rand(T, C) * new
    dt = torch.ones(T); dt[0] = 0
    avail = (torch.cumsum(new.amax(1), 0) > 0).float()
    fast = tam(attr, new, age, dt, avail)
    slow = _naive_tam(tam, attr, new, age, dt, avail)
    assert torch.allclose(fast, slow, atol=1e-5)
    lam = tam.decay_rate
    assert torch.allclose(math.log(2) / lam, torch.full((C,), 12.0), atol=1e-4)   # 12-observation half-life


def test_pii_tam_no_events_is_zero():
    tam = PIITAM()
    T = 10
    z = torch.zeros(T, 7)
    p = tam(torch.zeros(T, 7, 16), z, z, torch.ones(T), torch.zeros(T))
    assert torch.count_nonzero(p) == 0


# ----------------------------------------------------------------------------- attention masking
def test_null_candidate_gives_zero_context_on_empty_source():
    att = ModulatedCrossAttention(64, 2, 64, modulate=True)
    q, s = torch.randn(2, 5, 64), torch.randn(2, 5, 64)
    avail = torch.tensor([[0, 0, 0, 0, 0], [1, 0, 1, 0, 1]], dtype=torch.float32)
    out, a = att(q, s, avail, torch.randn(2, 64))
    assert torch.allclose(out[0], torch.zeros_like(out[0]))
    assert torch.allclose(a[1].sum(-1), torch.ones_like(a[1].sum(-1)), atol=1e-6)
    assert torch.all(a[1][..., [1, 3]] == 0)


# ----------------------------------------------------------------------------- full model
def _toy_data(T=140, reserve=90):
    rng = np.random.default_rng(0)
    close = 30 + np.cumsum(rng.normal(0, 0.5, T))
    R = np.stack([rng.normal(0, 0.3, T), rng.normal(0, 0.5, T), rng.normal(0, 0.8, T), np.zeros(T)], 1)
    R[:, 3] = close - R[:, :3].sum(1)
    new = (rng.random((T, 7)) < 0.04).astype(np.float32)
    arrays = {
        "close": close, "X_raw": np.stack([close, np.r_[0, np.diff(close)], rng.random(T), rng.random(T)], 1), "R": R,
        "sem": rng.normal(0, 1, (T, 768)).astype(np.float32), "csi": rng.random((T, 8)).astype(np.float32),
        "lex": rng.normal(0, 1, (T, 128)).astype(np.float32), "count": rng.random((T, 2)).astype(np.float32),
        "doc_avail": (rng.random(T) < 0.8).astype(np.float32),
        "pii_attr": (rng.random((T, 7, 16)) * new[..., None]).astype(np.float32), "pii_new": new,
        "pii_age": (rng.random((T, 7)) * new).astype(np.float32), "pii_dt": np.r_[0, np.ones(T - 1)].astype(np.float32),
        "pii_avail": (np.cumsum(new.max(1)) > 0).astype(np.float32),
    }
    from iesf.data.dataset import MarketData

    md = MarketData("Toy", np.arange(T).astype("datetime64[D]"), arrays, Partition(T, reserve, 30, 10, 10))
    return md, MarketTensors.from_market(md)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_every_variant_forward_backward(variant):
    md, data = _toy_data()
    spec = variant_spec(variant)
    presets = {"uniform10": (10,) * 4, "uniform30": (30,) * 4, "uniform60": (60,) * 4, "uniform90": (90,) * 4}
    lb = presets.get(spec["lookbacks"], (10, 30, 60, 90))
    model = IESFNet(IESFConfig(constructs=spec["constructs"], fusion=spec["fusion"], router=spec["router"],
                               lookbacks=lb, text_scales=spec["text_scales"], shared_numeric=spec["shared_numeric"]))
    idx = torch.tensor([95, 110, 120, 139])
    r = model(data, idx)
    assert r.shape == (4, 4) and torch.isfinite(r).all()
    from iesf.training.trainer import iesf_loss

    iesf_loss(r, data, idx, 0.01).backward()


def test_causality_future_inputs_do_not_change_forecast():
    md, data = _toy_data()
    model = IESFNet(IESFConfig()).eval()
    idx = torch.tensor([120])
    with torch.no_grad():
        before = model(data, idx).clone()
        for key in ("X_std", "R_std", "sem", "csi", "pii_attr", "pii_new", "doc_avail"):
            data.t[key][120:] = data.t[key][120:] * 0 + 7.0     # perturb the target date and beyond
        after = model(data, idx)
    assert torch.allclose(before, after)


# ----------------------------------------------------------------------------- statistics
def test_hac_lag_and_dm():
    assert hac_lag(393) == 5
    rng = np.random.default_rng(1)
    a, b = rng.random(393) + 0.5, rng.random(393)
    res = dm_test(a, b)
    assert res["lag"] == 5 and res["DM"] > 0 and 0 <= res["p"] <= 1


def test_holm_monotone():
    adj = holm([0.01, 0.04, 0.03, 0.5])
    assert np.allclose(adj, [0.04, 0.09, 0.09, 0.5])


def test_cumulative_gain_endpoint_identity():
    rng = np.random.default_rng(2)
    e_m, e_f = rng.normal(0, 1.0, 393), rng.normal(0, 0.7, 393)
    r = M.relative_reduction(M.rmse(e_f, 0), M.rmse(e_m, 0))
    g = M.cumulative_gain(e_m ** 2, e_f ** 2)
    assert math.isclose(g[-1], 100 * (1 - (1 - r / 100) ** 2), rel_tol=1e-9)    # Eq. (24)


def test_directional_accuracy_three_way():
    assert M.directional_accuracy([10, 10.5, 9], [10, 11, 8], [10, 10, 10]) == pytest.approx(100.0)
    assert M.directional_accuracy([10.5], [10], [10]) == 0.0                     # predicted move, no actual move


def test_bootstrap_interval_contains_point():
    rng = np.random.default_rng(3)
    a, b = rng.random(200) + 0.3, rng.random(200)
    out = bootstrap_reduction(a, b, 200, 8, 0)
    assert out["ci_low"] <= out["reduction"] <= out["ci_high"]


# ----------------------------------------------------------------------------- resumable decomposition
def test_chunked_origin_archive_matches_single_pass(tmp_path):
    pytest.importorskip("PyEMD")
    from iesf.config import load_config
    from iesf.decomposition.ceemdan import build_origin_archive

    d = {**load_config()["decomposition"], "trials": 5, "history_cap": 64, "n_jobs": 1}
    rng = np.random.default_rng(4)
    close = 30 + np.cumsum(rng.normal(0, 0.5, 40)) + np.sin(np.arange(40) / 3)
    a = build_origin_archive(close, d)
    build_origin_archive(close, d, cache_dir=tmp_path, chunk=16)                   # writes the parts
    b = build_origin_archive(close, d, cache_dir=tmp_path, chunk=16)               # resumes from the parts
    assert len(list(tmp_path.glob("*.npz"))) == 3
    for f in ("endpoints", "freqs", "mask", "residual"):
        assert np.allclose(getattr(a, f), getattr(b, f))


def test_residual_anchor_restoration_and_loss():
    from iesf.models.iesf_net import restore
    from iesf.training.trainer import iesf_loss, predict

    md, data = _toy_data()
    model = IESFNet(IESFConfig(constructs=(), residual_anchor=True)).eval()
    idx = torch.tensor([100, 120])
    zero = torch.zeros(2, 4)
    R_hat = restore(zero, data, idx, anchor=True)                    # r̂ = 0 -> last value + mean change
    expect = data["R"][idx - 1] + data["dR_mean"]
    assert torch.allclose(R_hat, expect)
    iesf_loss(model(data, idx), data, idx, 0.01, anchor=True).backward()
    out = predict(model, data, np.array([100, 120]))
    assert out["R_hat"].shape == (2, 4) and np.isfinite(out["y_hat"]).all()
