# -*- coding: utf-8 -*-
"""
plot_qrl_fig7.py — Fig. 7: noise, shot and topology robustness of base vs light models (+ hardware validation).

a  Device-decision agreement with the noiseless policy vs 2Q depolarizing probability (1Q = p/10); thick lines =
   mean over the 7 algorithms and all seeds (95% CI band), thin lines = individual algorithms.
b  Agreement vs number of shots (noiseless circuit, finite-shot <Z> estimation).
c  Routing cost per topology: SWAPs inserted and compiled 2Q gates (mean over models).
d  Agreement under the combined superconducting-like noise after routing onto each topology (SWAP = 3 noisy CNOTs).
e  Closed-loop normalized score with the noisy circuit (noiseless, combined x1, combined x3), mean over algorithms.
f  Hardware validation (optional, --hw_dir): Origin Quantum WK_C180, Q-PPO, base vs light, mean and 95% CI over states:
   Kendall tau, device agreement, exact match, <Z> MAE.
Inputs   results_qrl_robust/mec/per_model.csv (exp_qrl_robust.py); results_hw/<run>/per_state.csv (optional)
Outputs  figures/fig7_qrl.png/.pdf, figures/fig7_summary.csv
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

from plot_common import PALETTE, INK, MUTED, DOUBLE_COL_IN, mean_ci, save

C = {"base": PALETTE[0], "light": PALETTE[1]}
TOPOS = ["All-to-all", "Ring", "Linear", "Heavy-hex (6Q)"]


def line_panel(ax, d, xcol, logx=True, rows=None, tag=""):
    for v in ("base", "light"):
        dv = d[d.variant == v]
        for al, da in dv.groupby("algo"):
            g = da.groupby(xcol).agreement.mean()
            ax.plot(g.index, g.values, color=C[v], lw=0.5, alpha=0.35)
        xs = sorted(dv[xcol].unique())
        mu, hw = zip(*[mean_ci(dv[dv[xcol] == x].agreement.values) for x in xs]) if xs else ([], [])
        mu, hw = np.array(mu), np.array(hw)
        ax.plot(xs, mu, color=C[v], lw=1.8, marker="o", ms=3, label=v)
        ax.fill_between(xs, mu - hw, mu + hw, color=C[v], alpha=0.15, lw=0)
        if rows is not None:
            rows += [dict(panel=tag, variant=v, x=x, agreement=m, ci95=h) for x, m, h in zip(xs, mu, hw)]
    if logx:
        ax.set_xscale("log")
    ax.set_ylim(top=1.005)
    ax.set_ylabel("Decision agreement")
    ax.yaxis.grid(True, color="#e6e6e2", lw=0.5)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robust_csv", default="results_qrl_robust/mec/per_model.csv")
    ap.add_argument("--hw_dir", default="results_hw/Q-PPO_seed0_WK_C180_s1000")
    ap.add_argument("--hw_label", default="WK_C180, Q-PPO")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(a.robust_csv)
    rows = []
    fig, axs = plt.subplots(2, 3, figsize=(DOUBLE_COL_IN, 4.9), gridspec_kw=dict(hspace=0.65, wspace=0.45))

    # a: depolarizing sweep
    d = df[df.condition == "depolarizing"].copy()
    d["x"] = d.level.astype(float)
    line_panel(axs[0, 0], d, "x", rows=rows, tag="a")
    axs[0, 0].set_xlabel("2Q depolarizing probability")
    axs[0, 0].legend(loc="lower left", fontsize=6.5)
    # b: shots
    d = df[df.condition == "shots"].copy()
    d["x"] = d.level.astype(float)
    line_panel(axs[0, 1], d, "x", rows=rows, tag="b")
    axs[0, 1].set_xlabel("Shots per circuit")
    # c: routing cost
    ax = axs[0, 2]
    d = df[df.condition == "topology"]
    w = 0.2
    for j, (v, metric, hatch) in enumerate((("base", "swaps", None), ("light", "swaps", None),
                                           ("base", "compiled_2q", "////"), ("light", "compiled_2q", "////"))):
        mu = [d[(d.variant == v) & (d.level == t)][metric].mean() for t in TOPOS]
        ax.bar(np.arange(len(TOPOS)) + (j - 1.5) * w, mu, width=w * 0.92, color=C[v] if hatch is None else "white",
               edgecolor=C[v], hatch=hatch, linewidth=0.8, zorder=2)
        rows += [dict(panel="c", variant=v, metric=metric, topology=t, value=m) for t, m in zip(TOPOS, mu)]
    ax.set_xticks(range(len(TOPOS)))
    ax.set_xticklabels(["All-to-all", "Ring", "Linear", "Heavy-hex"], rotation=30, ha="right")
    ax.set_ylabel("Gates per circuit")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=MUTED, label="SWAPs"),
                       plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=MUTED, hatch="////", label="compiled 2Q")],
              fontsize=6, loc="upper left")
    ax.yaxis.grid(True, color="#e6e6e2", lw=0.5)
    ax.set_axisbelow(True)
    # d: agreement after routing under combined noise
    ax = axs[1, 0]
    rng = np.random.default_rng(0)
    for j, v in enumerate(("base", "light")):
        for i, t in enumerate(TOPOS):
            s = d[(d.variant == v) & (d.level == t)].agreement.values
            if len(s) == 0:
                continue
            mu, hw = mean_ci(s)
            x = i + (j - 0.5) * 0.36
            ax.bar(x, mu, width=0.34, color=C[v], alpha=0.85, edgecolor="white", zorder=2)
            ax.errorbar(x, mu, yerr=hw, fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.5, zorder=3)
            rows.append(dict(panel="d", variant=v, topology=t, agreement=mu, ci95=hw))
    lo = d.agreement.min() if len(d) else 0.5
    ax.set_ylim(max(0.0, lo - 0.05), 1.0)
    ax.set_xticks(range(len(TOPOS)))
    ax.set_xticklabels(["All-to-all", "Ring", "Linear", "Heavy-hex"], rotation=30, ha="right")
    ax.set_ylabel("Decision agreement")
    ax.set_title("Combined noise after routing", fontsize=7, pad=3)
    ax.yaxis.grid(True, color="#e6e6e2", lw=0.5)
    ax.set_axisbelow(True)
    # e: closed loop
    ax = axs[1, 1]
    d = df[df.condition == "closed_loop"]
    labs = ["noiseless", "combined_x1", "combined_x3"]
    for j, v in enumerate(("base", "light")):
        for i, l in enumerate(labs):
            s = d[(d.variant == v) & (d.level == l)].score.values
            if len(s) == 0:
                continue
            mu, hw = mean_ci(s)
            x = i + (j - 0.5) * 0.36
            ax.bar(x, mu, width=0.34, color=C[v], alpha=0.85, edgecolor="white", zorder=2)
            ax.errorbar(x, mu, yerr=hw, fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.5, zorder=3)
            rows.append(dict(panel="e", variant=v, noise=l, score=mu, ci95=hw))
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Noiseless", "Combined ×1", "Combined ×3"], rotation=30, ha="right")
    ax.set_ylabel("Closed-loop score")
    ax.yaxis.grid(True, color="#e6e6e2", lw=0.5)
    ax.set_axisbelow(True)
    # f: hardware
    ax = axs[1, 2]
    hw_f = os.path.join(a.hw_dir, "per_state.csv")
    if os.path.exists(hw_f):
        h = pd.read_csv(hw_f)
        mets = [("kendall_tau", "Kendall\nτ"), ("device_agreement", "Agree-\nment"),
                ("exact_match", "Exact\nmatch"), ("expz_mae", "$\\langle Z\\rangle$\nMAE")]
        for j, v in enumerate(("base", "light")):
            for i, (m, _) in enumerate(mets):
                s = h[h.variant == v][m].values
                mu, hwid = mean_ci(s)
                x = i + (j - 0.5) * 0.36
                ax.bar(x, mu, width=0.34, color=C[v], alpha=0.85, edgecolor="white", zorder=2)
                ax.errorbar(x, mu, yerr=hwid, fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.5, zorder=3)
                rows.append(dict(panel="f", variant=v, metric=m, value=mu, ci95=hwid, n=len(s)))
        for i, (m, _) in enumerate(mets):
            b = h[h.variant == "base"].set_index("state")[m]
            l = h[h.variant == "light"].set_index("state")[m]
            dd = (l - b).dropna()
            p = stats.wilcoxon(dd).pvalue if len(dd) > 1 and np.any(dd != 0) else np.nan
            ptxt = "p<0.001" if p < 0.001 else f"p={p:.2f}"
            ax.text(i, 1.02, ptxt, ha="center", fontsize=5.2, color=INK, transform=ax.get_xaxis_transform())
        ax.set_xticks(range(len(mets)))
        ax.set_xticklabels([m[1] for m in mets], fontsize=6.3)
        ax.set_title(f"Hardware: {a.hw_label}\n(n = {h.state.nunique()} states)", fontsize=6.5, pad=12,
                     linespacing=1.15)
        ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True, color="#e6e6e2", lw=0.5)
        ax.set_axisbelow(True)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "no hardware data", ha="center", va="center", color=MUTED, transform=ax.transAxes)
    for ax_, letter in ((axs[0, 0], "a"), (axs[0, 1], "b"), (axs[0, 2], "c"), (axs[1, 0], "d"),
                        (axs[1, 1], "e"), (axs[1, 2], "f")):
        # letter at the left of the y-tick labels, level with the top of the axis (never collides with titles)
        ax_.text(-0.32, 1.0, letter, transform=ax_.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")
    save(fig, os.path.join(a.out, "fig7_qrl"))
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "fig7_summary.csv"), index=False)


if __name__ == "__main__":
    main()
