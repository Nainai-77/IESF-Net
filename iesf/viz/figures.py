"""Figures 3 and 5-16 (static PNG + PDF for the manuscript).

Colour roles: observed series in ink; categorical slots in fixed order (blue, orange,
aqua, yellow, ...) that follow the entity, never its rank; magnitude heatmaps use one
blue ramp; signed differences use a blue <-> red diverging map with a grey midpoint.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

INK, INK_2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MODEL_COLOR = {"Numeric": CAT[1], "Bidirectional": CAT[2], "IESF-Net": CAT[0]}
CONSTRUCT_COLOR = {"sem": CAT[0], "csi": CAT[1], "pii": CAT[2]}
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_blue_red", ["#104281", "#3987e5", "#f0efec", "#e66767", "#a8201f"])


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.2, "savefig.dpi": 300,
        "figure.dpi": 120,
    })


def _save(fig, out_dir, name) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _panel(ax, letter):
    ax.text(-0.08, 1.04, f"({letter})", transform=ax.transAxes, fontweight="bold", color=INK, va="bottom")


# --------------------------------------------------------------------------- Figure 3 / Figure 5
def fig3_decomposition(series: dict, out_dir) -> None:
    """series[market] = (dates, close, comps[4, T]); one column per market."""
    names = ["Observed price", "Component 1", "Component 2", "Component 3", "Component 4 (trend)"]
    fig, axes = plt.subplots(5, len(series), figsize=(3.4 * len(series), 6.4), sharex="col", squeeze=False)
    for c, (mk, (dates, close, comps)) in enumerate(series.items()):
        rows = [close] + list(comps)
        for r, y in enumerate(rows):
            ax = axes[r, c]
            ax.plot(pd.to_datetime(dates), y, color=INK if r == 0 else CAT[0], lw=0.8)
            ax.set_ylabel(names[r] if c == 0 else "")
            if r == 0:
                ax.set_title(f"({chr(97 + c)}) {mk}")
    _save(fig, out_dir, "Figure3_decomposition")


def fig5_prices(markets: dict, out_dir) -> None:
    """markets[name] = (dates, close, first_val_date, first_test_date)."""
    fig, axes = plt.subplots(1, len(markets), figsize=(3.4 * len(markets), 2.4), squeeze=False)
    for c, (mk, (dates, close, v0, t0)) in enumerate(markets.items()):
        ax = axes[0, c]
        ax.plot(pd.to_datetime(dates), close, color=INK, lw=0.8)
        ax.axvline(pd.Timestamp(v0), color=INK_2, ls="--", lw=0.8, label="first validation target")
        ax.axvline(pd.Timestamp(t0), color=INK_2, ls="-.", lw=0.8, label="first test target")
        ax.set_title(f"({chr(97 + c)}) {mk}")
        ax.set_ylabel("CNY/tCO$_2$")
        if c == 0:
            ax.legend(loc="upper left")
    _save(fig, out_dir, "Figure5_prices_partitions")


# --------------------------------------------------------------------------- Figures 6-7
def fig_forecasts(market: str, frames: dict, out_dir, name: str, zoom=(140, 200), window: int = 21) -> None:
    """frames[model] = test forecast table; panels (a) full, (b) zoom, (c) centred rolling RMSE."""
    from ..evaluation.metrics import rolling_rmse

    base = next(iter(frames.values())).sort_values("target_index")
    t = np.arange(1, len(base) + 1)
    if zoom[1] > len(base):                                     # short samples: zoom on the middle third
        zoom = (max(1, len(base) // 3), max(2, 2 * len(base) // 3))
    fig = plt.figure(figsize=(7.0, 5.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.3, 1.0, 0.9], hspace=0.45)
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[i]) for i in range(3))
    for ax in (ax_a, ax_b):
        ax.plot(t, base["actual"], color=INK, lw=1.0, label="Observed")
    for model, df in frames.items():
        df = df.sort_values("target_index")
        col = MODEL_COLOR.get(model, CAT[3])
        ax_a.plot(t, df["prediction"], color=col, lw=0.9, label=model)
        ax_b.plot(t, df["prediction"], color=col, lw=1.1, label=model)
        ax_c.plot(t, rolling_rmse(df["prediction"], df["actual"], window), color=col, lw=1.1, label=model)
    ax_a.axvspan(*zoom, color=GRID, alpha=0.5, lw=0, zorder=0)
    ax_b.set_xlim(*zoom)
    ax_a.set_ylabel("CNY/tCO$_2$"); ax_b.set_ylabel("CNY/tCO$_2$"); ax_c.set_ylabel(f"{window}-obs RMSE")
    ax_c.set_xlabel("Test target index")
    ax_a.legend(ncol=4, loc="lower left", bbox_to_anchor=(0.0, 1.02), fontsize=7)
    for ax, l in zip((ax_a, ax_b, ax_c), "abc"):
        _panel(ax, l)
    fig.suptitle(f"{market}: test-period forecasts", fontsize=9)
    _save(fig, out_dir, name)


# --------------------------------------------------------------------------- Figure 8
def fig8_benchmarks(t7: pd.DataFrame, cum: dict, out_dir) -> None:
    """(a) RMSE relative to Persistence; (b) cumulative Persistence-minus-model loss of the selected benchmark."""
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.8))
    bench = t7[t7["Group"] == "numerical benchmark"]
    markets = list(bench["Market"].unique())
    models = [m for m in bench["Model"].unique() if m != "Persistence"]
    w = 0.8 / max(1, len(markets))
    for j, mk in enumerate(markets):
        sub = bench[bench["Market"] == mk].set_index("Model")
        rel = [sub.loc[m, "RMSE"] / sub.loc["Persistence", "RMSE"] for m in models]
        ax_a.bar(np.arange(len(models)) + j * w, rel, width=w * 0.92, color=CAT[j], label=mk)
    ax_a.axhline(1.0, color=INK_2, lw=0.8)
    ax_a.set_xticks(np.arange(len(models)) + w * (len(markets) - 1) / 2, models, rotation=40, ha="right")
    ax_a.set_ylabel("RMSE / Persistence RMSE")
    ax_a.legend()
    for j, (mk, (model, g)) in enumerate(cum.items()):
        ax_b.plot(np.arange(1, len(g) + 1), g, color=CAT[j], label=f"{mk}: {model}")
    ax_b.axhline(0, color=INK_2, lw=0.8)
    ax_b.set_xlabel("Test target index"); ax_b.set_ylabel("Cumulative gain (%)")
    ax_b.legend()
    _panel(ax_a, "a"); _panel(ax_b, "b")
    _save(fig, out_dir, "Figure8_numerical_benchmarks")


# --------------------------------------------------------------------------- Figures 9-10
def _ci_bars(ax, labels, est, lo, hi, color):
    y = np.arange(len(labels))
    ax.errorbar(est, y, xerr=[np.subtract(est, lo), np.subtract(hi, est)], fmt="o", color=color, ms=4,
                capsize=2, lw=1.0)
    ax.set_yticks(y, labels)
    ax.axvline(0, color=INK_2, lw=0.8)
    ax.invert_yaxis()


def fig9_paired(paired: dict, cum: dict, out_dir, refs=("Numeric", "Bidirectional", "Feature conditioning")) -> None:
    """paired[market] = Tables 9/10 frame; cum[market][reference] = cumulative gain array."""
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.8))
    off = np.linspace(-0.15, 0.15, max(1, len(paired)))
    for j, (mk, df) in enumerate(paired.items()):
        sub = df.set_index("Reference").reindex(list(refs)).dropna(subset=["Reduction (%)"])
        y = np.arange(len(sub)) + off[j]
        ax_a.errorbar(sub["Reduction (%)"], y, xerr=[sub["Reduction (%)"] - sub["CI low"], sub["CI high"] - sub["Reduction (%)"]],
                      fmt="o", ms=4, capsize=2, color=CAT[j], label=mk, lw=1.0)
        ax_a.set_yticks(np.arange(len(sub)), sub.index)
    ax_a.invert_yaxis(); ax_a.axvline(0, color=INK_2, lw=0.8)
    ax_a.set_xlabel("RMSE reduction (%)"); ax_a.legend()
    styles = ["-", "--", ":"]
    for j, (mk, d) in enumerate(cum.items()):
        for s, (ref, g) in enumerate(d.items()):
            ax_b.plot(np.arange(1, len(g) + 1), g, color=CAT[j], ls=styles[s % 3], label=f"{mk} vs {ref}")
    ax_b.axhline(0, color=INK_2, lw=0.8)
    ax_b.set_xlabel("Test target index"); ax_b.set_ylabel("Cumulative squared-loss gain (%)")
    ax_b.legend(fontsize=6, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    _panel(ax_a, "a"); _panel(ax_b, "b")
    _save(fig, out_dir, "Figure9_paired_improvement")


def fig10_subgroups(t11: pd.DataFrame, out_dir) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    sub = t11[t11["Target subset"] != "Full test"]
    markets = list(sub["Market"].unique())
    off = np.linspace(-0.15, 0.15, max(1, len(markets)))
    for j, mk in enumerate(markets):
        d = sub[sub["Market"] == mk]
        y = np.arange(len(d)) + off[j]
        ax.errorbar(d["reduction"], y, xerr=[d["reduction"] - d["ci_low"], d["ci_high"] - d["reduction"]],
                    fmt="o", ms=4, capsize=2, color=CAT[j], label=mk, lw=1.0)
        ax.set_yticks(np.arange(len(d)), d["Target subset"])
    ax.invert_yaxis(); ax.axvline(0, color=INK_2, lw=0.8)
    ax.set_xlabel("RMSE reduction vs Numeric (%)"); ax.legend()
    _save(fig, out_dir, "Figure10_subgroup_reductions")


# --------------------------------------------------------------------------- Figures 11-13
def _grouped_bars(ax, df, label_col, value_col, markets):
    labels = list(dict.fromkeys(df[label_col]))
    w = 0.8 / max(1, len(markets))
    for j, mk in enumerate(markets):
        s = df[df["Market"] == mk].set_index(label_col).reindex(labels)[value_col]
        ax.bar(np.arange(len(labels)) + j * w, s.to_numpy(), width=w * 0.92, color=CAT[j], label=mk)
    ax.set_xticks(np.arange(len(labels)) + w * (len(markets) - 1) / 2, labels, rotation=35, ha="right")


def fig11_rep_fusion(rep: pd.DataFrame, fus: pd.DataFrame, out_dir) -> None:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for ax, df, l in ((ax_a, rep, "a"), (ax_b, fus, "b")):
        if not df.empty:
            _grouped_bars(ax, df, "Label", "RMSE", list(df["Market"].unique()))
        ax.set_ylabel("RMSE (CNY/tCO$_2$)"); _panel(ax, l)
    ax_a.legend()
    _save(fig, out_dir, "Figure11_representation_fusion")


def fig12_ablation_window(abl: pd.DataFrame, win: pd.DataFrame, out_dir) -> None:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.9))
    if not abl.empty:
        _grouped_bars(ax_a, abl, "Label", "Increase (%)", list(abl["Market"].unique()))
    ax_a.set_ylabel("RMSE increase vs IESF-Net (%)")
    if not win.empty:
        w2 = win.copy()
        w2["Label"] = w2["Label"].replace({"IESF-Net": "Hybrid (10,30,60,90)"})
        mks = list(w2["Market"].unique())
        _grouped_bars(ax_b, w2, "Label", "RMSE", mks)
        last, w = len(w2["Label"].unique()) - 1, 0.8 / max(1, len(mks))
        ax_b.axvspan(last - w / 2 - 0.05, last + (len(mks) - 0.5) * w + 0.05, color=GRID, alpha=0.6, lw=0, zorder=0)
        ax_b.set_axisbelow(True)
    ax_b.set_ylabel("RMSE (CNY/tCO$_2$)")
    ax_a.legend(); _panel(ax_a, "a"); _panel(ax_b, "b")
    _save(fig, out_dir, "Figure12_ablations_windows")


def fig13_cross_market(t16: pd.DataFrame, out_dir) -> None:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.7))
    x = np.arange(len(t16))
    ax_a.bar(x - 0.2, t16["RMSE reduction (%)"], 0.37, color=CAT[0], label="RMSE")
    ax_a.bar(x + 0.2, t16["MAE reduction (%)"], 0.37, color=CAT[1], label="MAE")
    ax_a.set_xticks(x, [f"{m}\n(N={n})" for m, n in zip(t16["Market"], t16["N"])])
    ax_a.set_ylabel("Reduction vs Numeric (%)"); ax_a.legend()
    ax_b.bar(x - 0.2, t16["Numeric RMSE"], 0.37, color=CAT[1], label="Numeric")
    ax_b.bar(x + 0.2, t16["IESF-Net RMSE"], 0.37, color=CAT[0], label="IESF-Net")
    ax_b.set_xticks(x, t16["Market"]); ax_b.set_ylabel("RMSE (CNY/tCO$_2$)"); ax_b.legend()
    _panel(ax_a, "a"); _panel(ax_b, "b")
    _save(fig, out_dir, "Figure13_cross_market")


# --------------------------------------------------------------------------- Figures 14-16
def fig14_attention(maps: dict, out_dir) -> None:
    """maps[market][(scale, direction)] = [L, L]; rows scales 1-3, columns market x direction."""
    cols = [(mk, d) for mk in maps for d in ("N→T", "T→N")]
    fig, axes = plt.subplots(3, len(cols), figsize=(1.9 * len(cols) + 0.8, 5.6), squeeze=False, constrained_layout=True)
    for r in range(3):
        row = [maps[mk].get((r, d)) for mk, d in cols]
        vals = [m for m in row if m is not None]
        if not vals:
            continue
        vmin, vmax = min(float(m.min()) for m in vals), max(float(m.max()) for m in vals)   # one scale per row
        im = None
        for c, ((mk, d), m) in enumerate(zip(cols, row)):
            ax = axes[r, c]
            ax.grid(False)
            if m is not None:
                im = ax.imshow(m, cmap=SEQ, aspect="auto", origin="upper", vmin=vmin, vmax=vmax)
            if r == 0:
                ax.set_title(f"{mk} {d}")
            if c == 0:
                ax.set_ylabel(f"Scale {r + 1} (L = {vals[0].shape[0]})\nquery lag")
            else:
                ax.set_yticklabels([])
            if r == 2:
                ax.set_xlabel("source lag")
        fig.colorbar(im, ax=list(axes[r]), fraction=0.025, pad=0.01, label="attention weight")
    _save(fig, out_dir, "Figure14_attention")


def fig15_contrast(contrast: dict, market: str, out_dir) -> None:
    keys = sorted(contrast)
    fig, axes = plt.subplots(len(keys), 3, figsize=(6.4, 1.7 * len(keys)), squeeze=False, constrained_layout=True)
    for r, key in enumerate(keys):
        d = contrast[key]
        lo = min(float(d["low"].min()), float(d["high"].min()))
        hi = max(float(d["low"].max()), float(d["high"].max()))
        dmax = max(float(np.abs(d["diff"]).max()), 1e-12)
        ims = []
        for c, (sub, cmap, vmin, vmax) in enumerate((("low", SEQ, lo, hi), ("high", SEQ, lo, hi), ("diff", DIV, -dmax, dmax))):
            ax = axes[r, c]
            ax.grid(False)
            ims.append(ax.imshow(d[sub], cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax))
            if r == 0:
                ax.set_title({"low": "Low (c = −0.90)", "high": "High (c = +0.90)", "diff": "High − low"}[sub])
            if c == 0:
                ax.set_ylabel(f"Scale {key[0] + 1} {key[1]}")
            else:
                ax.set_yticklabels([])
        fig.colorbar(ims[1], ax=list(axes[r, :2]), fraction=0.04, pad=0.01)
        fig.colorbar(ims[2], ax=axes[r, 2], fraction=0.08, pad=0.01)
    fig.suptitle(f"{market}: conditional-input contrast", fontsize=9)
    _save(fig, out_dir, f"Figure15_conditioning_contrast_{market}")


def fig16_routing(routes: dict, out_dir, n_display: int = 120) -> None:
    """routes[market] = routing frame (scale, target_index, sem, csi, pii); stacked areas per scale."""
    markets = list(routes)
    fig, axes = plt.subplots(3, len(markets), figsize=(3.5 * len(markets), 5.2), sharey=True, squeeze=False)
    for c, mk in enumerate(markets):
        df = routes[mk]
        for r in range(3):
            ax = axes[r, c]
            d = df[df["scale"] == r + 1].sort_values("target_index").tail(n_display)
            cons = [m for m in ("sem", "csi", "pii") if m in d.columns]
            x = np.arange(1, len(d) + 1)
            ax.stackplot(x, *[d[m].to_numpy() for m in cons], colors=[CONSTRUCT_COLOR[m] for m in cons],
                         labels=[{"sem": "Semantic", "csi": "CSI", "pii": "PII"}[m] for m in cons],
                         edgecolor="#fcfcfb", linewidth=0.3)
            ax.set_ylim(0, 1); ax.set_xlim(1, max(2, len(d)))
            if r == 0:
                ax.set_title(mk)
            if c == 0:
                lb = f"L = {int(d['lookback'].iloc[0])}" if "lookback" in d.columns and len(d) else f"Scale {r + 1}"
                ax.set_ylabel(f"{lb}\nrouting weight")
            if r == 2:
                ax.set_xlabel("Displayed observation")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7)
    _save(fig, out_dir, "Figure16_routing")
