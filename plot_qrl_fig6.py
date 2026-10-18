# -*- coding: utf-8 -*-
"""
plot_qrl_fig6.py — Fig. 6: adaptation of all 14 models (7 algorithms x base/light) to two distribution shifts.

a  Video-traffic surge (arrivals x1.5) and  b  channel deterioration (mean gain x0.5): post-shift normalized score of
   every model when frozen (grey), fully fine-tuned (blue) or adapted with PEQA (green); base = circles, light =
   triangles; mean and 95% CI over seeds.  Scores use reference policies of the shifted system (0 random, 1 myopic).
c  Adaptation cost per algorithm (light models, both shifts pooled): energy of the adaptation updates (5 W x
   measured update time), full fine-tuning vs PEQA; numbers = trainable parameters.
d  Adaptation curves of LightQDRL-X (light): score vs adaptation steps for both shifts, full vs PEQA (mean, 95% CI).
Outputs  figures/fig6_qrl.png/.pdf and figures/fig6_summary.csv
Inputs   results_qrl_adapt/mec/per_run.csv and runs/*.json (exp_qrl_adapt.py), results_qrl_validation/selection.json
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_common import PALETTE, INK, MUTED, DOUBLE_COL_IN, mean_ci, panel_labels_auto, save

ALGOS = ["Q-DQN", "Q-DDQN", "Q-DuelingDQN", "Q-A2C", "Q-A3C", "Q-PPO", "Q-TRPO"]
SHORT = {"Q-DuelingDQN": "Q-Dueling"}
MODE_COL = {"frozen": "#9a9a94", "full": PALETTE[0], "peqa": PALETTE[2]}
MODE_LAB = {"frozen": "Frozen", "full": "Full fine-tuning", "peqa": "PEQA"}
VAR_MK = {"base": "o", "light": "^"}
SHIFT_T = {"traffic": "Video-traffic surge (×1.5)", "channel": "Channel deterioration (−3 dB)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapt_dir", default="results_qrl_adapt/mec")
    ap.add_argument("--selection", default="results_qrl_validation/selection.json")
    ap.add_argument("--algo_x", default=None)
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(os.path.join(a.adapt_dir, "per_run.csv"))
    df = df[df.variant.isin(["base", "light"])]
    X = a.algo_x or (json.load(open(a.selection))["chosen"] if os.path.exists(a.selection) else ALGOS[0])

    fig = plt.figure(figsize=(DOUBLE_COL_IN, 5.4))
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.32, width_ratios=[1.35, 1])
    rows = []
    axes_ab = []
    for k, sh in enumerate(["traffic", "channel"]):
        ax = fig.add_subplot(gs[k, 0])
        axes_ab.append(ax)
        d = df[df["shift"] == sh]
        for i, al in enumerate(ALGOS):
            for j, v in enumerate(["base", "light"]):
                for m_i, mode in enumerate(["frozen", "full", "peqa"]):
                    x = i + (j - 0.5) * 0.42 + (m_i - 1) * 0.12
                    s = d[(d.algo == al) & (d.variant == v) & (d["mode"] == mode)].post_score
                    if len(s) == 0:
                        continue
                    mu, h = mean_ci(s.values)
                    ax.errorbar(x, mu, yerr=h, fmt=VAR_MK[v], ms=3.6, color=MODE_COL[mode], mec="white", mew=0.4,
                                elinewidth=0.8, zorder=3)
                    rows.append(dict(shift=sh, algo=al, variant=v, mode=mode, n=len(s), post_score=mu, ci95=h))
            if i:
                ax.axvline(i - 0.5, color="#ececea", lw=0.6, zorder=0)
        ax.axhline(0, color=MUTED, lw=0.6, ls=":")
        ax.axhline(1, color=MUTED, lw=0.6, ls="--")
        ax.set_xticks(range(len(ALGOS)))
        ax.set_xticklabels([SHORT.get(x, x) for x in ALGOS], rotation=30, ha="right")
        ax.set_ylabel("Post-shift score")
        ax.set_title(SHIFT_T[sh], fontsize=7.5, pad=3)
        ax.set_xlim(-0.6, len(ALGOS) - 0.4)
        ax.yaxis.grid(True, color="#e6e6e2", lw=0.5)
        ax.set_axisbelow(True)
    h = [Line2D([], [], marker="s", ls="", color=MODE_COL[m], label=MODE_LAB[m]) for m in MODE_COL]
    h += [Line2D([], [], marker=VAR_MK[v], ls="", color=INK, mfc="white", label=v) for v in VAR_MK]
    axes_ab[0].legend(handles=h, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.12), fontsize=6.2,
                      handletextpad=0.2, columnspacing=0.8)

    # ---------------- c: adaptation cost (light, pooled shifts)
    axc = fig.add_subplot(gs[0, 1])
    d = df[(df.variant == "light") & df["mode"].isin(["full", "peqa"])]
    for i, al in enumerate(ALGOS):
        for j, mode in enumerate(["full", "peqa"]):
            e = d[(d.algo == al) & (d["mode"] == mode)].groupby("seed").adapt_energy_J.mean()
            if len(e) == 0:
                continue
            mu, hw = mean_ci(e.values)
            x = i + (j - 0.5) * 0.36
            axc.bar(x, mu, width=0.34, color=MODE_COL[mode], edgecolor="white", linewidth=0.5, zorder=2)
            axc.errorbar(x, mu, yerr=hw, fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.5, zorder=3)
            tp = d[(d.algo == al) & (d["mode"] == mode)].trainable_params.mean()
            rows.append(dict(shift="pooled", algo=al, variant="light", mode=mode, adapt_energy_J=mu,
                             energy_ci95=hw, trainable_params=tp))
    axc.set_xticks(range(len(ALGOS)))
    axc.set_xticklabels([SHORT.get(x, x) for x in ALGOS], rotation=30, ha="right")
    axc.set_ylabel("Adaptation energy (J)")
    tp_full = d[d["mode"] == "full"].trainable_params.median() if len(d) else np.nan
    tp_peqa = d[d["mode"] == "peqa"].trainable_params.median() if len(d) else np.nan
    axc.set_title(f"Light models: trainable params {tp_full:.0f} (full) vs {tp_peqa:.0f} (PEQA)", fontsize=7, pad=3)
    axc.yaxis.grid(True, color="#e6e6e2", lw=0.5)
    axc.set_axisbelow(True)

    # ---------------- d: adaptation curves of X
    axd = fig.add_subplot(gs[1, 1])
    runs = os.path.join(a.adapt_dir, "runs")
    for sh, ls in (("traffic", "-"), ("channel", "--")):
        for mode in ("full", "peqa"):
            curves = []
            for f in sorted(os.listdir(runs)):
                if f.startswith(f"{X}_light_{sh}_{mode}_seed") and f.endswith(".json"):
                    curves.append(np.array(json.load(open(os.path.join(runs, f)))["curve"]))
            if not curves:
                continue
            ref = json.load(open(os.path.join(a.adapt_dir, f"reference_{sh}.json")))
            L = min(len(c) for c in curves)
            C = np.stack([c[:L] for c in curves])
            t = C[0, :, 0]
            S = (C[:, :, 1] - ref["random"]) / (ref["myopic"] - ref["random"])
            mu = S.mean(0)
            hw = np.array([mean_ci(S[:, k])[1] for k in range(L)])
            axd.plot(t / 1000, mu, ls=ls, color=MODE_COL[mode], lw=1.3, label=f"{MODE_LAB[mode]}, {sh}")
            axd.fill_between(t / 1000, mu - hw, mu + hw, color=MODE_COL[mode], alpha=0.15, lw=0)
    axd.set_xlabel("Adaptation steps (×10³)")
    axd.set_ylabel("Post-shift score")
    axd.set_title(f"LightQDRL-X ({X}) adaptation", fontsize=7.5, pad=3)
    axd.legend(fontsize=6, loc="lower right")
    axd.yaxis.grid(True, color="#e6e6e2", lw=0.5)
    axd.set_axisbelow(True)

    panel_labels_auto(fig, [(axes_ab[0], "a"), (axes_ab[1], "b"), (axc, "c"), (axd, "d")])
    save(fig, os.path.join(a.out, "fig6_qrl"))
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "fig6_summary.csv"), index=False)


if __name__ == "__main__":
    main()
