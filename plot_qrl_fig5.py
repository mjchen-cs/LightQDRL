# -*- coding: utf-8 -*-
"""
plot_qrl_fig5.py — Fig. 5: recommended configuration LightQDRL-X and its 2^3 factorial ablation (DQCP x JCQQ x PEQA).

a  Pre-registered selection on validation seeds: light (and base) normalized score of the 7 algorithms; the chosen
   algorithm X is highlighted; hatched = not admissible (light loses more than delta against base).
b  Deployment outcomes (2^2: DQCP x JCQQ, test seeds): effect of each component on static memory, logical 2Q gates,
   inference latency and nominal score.  Effects from OLS with +-1/2 coding and seed fixed effects, i.e. average
   effect of switching the factor on; costs as % of the uncompressed cell, score in normalized units; 95% CI.
c,d Adaptation outcomes (full 2^3, both shifts pooled with a shift fixed effect): effects of DQCP, JCQQ, PEQA and their
   two-factor interactions on post-shift score (normalized units), adaptation energy and trainable parameters (%).
e  Cell means of the post-shift score for the 8 cells (mean, 95% CI, seeds).

Inputs   results_qrl_validation/selection.json            (select_lightqdrl_x.py)
         results_qrl_validation/mec/per_seed.csv
         results_qrl/mec/per_seed.csv                       (variants base, dqcp, jcqq, light of X)
         results_qrl/mec/latency.csv                        (measure_qrl_latency.py; optional)
         results_qrl_adapt/mec/per_run.csv                  (exp_qrl_adapt.py for X, variants base,dqcp,jcqq,light)
Outputs  figures/fig5_qrl.png/.pdf, figures/fig5_effects.csv, figures/fig5_cells.csv
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

from plot_common import PALETTE, INK, MUTED, DOUBLE_COL_IN, mean_ci, panel_labels_auto, save

ALGOS = ["Q-DQN", "Q-DDQN", "Q-DuelingDQN", "Q-A2C", "Q-A3C", "Q-PPO", "Q-TRPO"]
SHORT = {"Q-DuelingDQN": "Q-Dueling"}
VAR_DJ = {"base": (0, 0), "dqcp": (1, 0), "jcqq": (0, 1), "light": (1, 1)}
TERM = {"D": "DQCP", "J": "JCQQ", "P": "PEQA", "DxJ": "DQCP×JCQQ", "DxP": "DQCP×PEQA", "JxP": "JCQQ×PEQA"}


def ols_effects(df, y, factors, inter, fe=("seed",)):
    """OLS with +-1/2 coded factors, chosen interactions and fixed effects; returns {term: (est, ci95)}."""
    X = [np.ones(len(df))]
    names = ["const"]
    for f in factors:
        X.append(df[f].values - 0.5); names.append(f)
    for a_, b_ in inter:
        X.append((df[a_].values - 0.5) * (df[b_].values - 0.5) * 2); names.append(f"{a_}x{b_}")
    for f in fe:
        for lvl in sorted(df[f].unique())[1:]:
            X.append((df[f].values == lvl).astype(float)); names.append(f"{f}={lvl}")
    X = np.column_stack(X)
    yv = df[y].values.astype(float)
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    res = yv - X @ beta
    dof = max(1, len(yv) - X.shape[1])
    s2 = res @ res / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    tq = stats.t.ppf(0.975, dof)
    out = {}
    for i, n in enumerate(names):
        if n == "const" or "=" in n:
            continue
        se = np.sqrt(max(cov[i, i], 0))
        p = 2 * (1 - stats.t.cdf(abs(beta[i] / se), dof)) if se > 0 else np.nan
        out[n] = (beta[i], tq * se, p)
    return out


def forest(ax, effects, title):
    """effects: list of (label, est, ci, color)."""
    ys = np.arange(len(effects))[::-1]
    for y, (lab, est, ci, col) in zip(ys, effects):
        ax.hlines(y, est - ci, est + ci, color=col, lw=1.6)
        ax.scatter(est, y, s=18, color=col, edgecolor="white", linewidth=0.5, zorder=3)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_yticks(ys)
    ax.set_yticklabels([e[0] for e in effects], fontsize=6.5)
    ax.set_title(title, fontsize=7.5, pad=3)
    ax.xaxis.grid(True, color="#e6e6e2", lw=0.5)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="results_qrl_validation/selection.json")
    ap.add_argument("--val_csv", default="results_qrl_validation/mec/per_seed.csv")
    ap.add_argument("--bench_csv", default="results_qrl/mec/per_seed.csv")
    ap.add_argument("--latency_csv", default="results_qrl/mec/latency.csv")
    ap.add_argument("--adapt_csv", default="results_qrl_adapt/mec/per_run.csv")
    ap.add_argument("--algo", default=None, help="override the selected algorithm")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sel = json.load(open(a.selection))
    X = a.algo or sel["chosen"]
    delta = sel["rule"]["delta"]

    fig = plt.figure(figsize=(DOUBLE_COL_IN, 5.6))
    gs = fig.add_gridspec(2, 1, hspace=0.75)
    top = gs[0].subgridspec(1, 2, width_ratios=[0.8, 1.6], wspace=1.25)
    bot = gs[1].subgridspec(1, 3, width_ratios=[1, 1, 0.9], wspace=1.05)

    # ---------------- a: selection on validation seeds
    ax = ax_a = fig.add_subplot(top[0])
    val = pd.read_csv(a.val_csv)
    val = val[val.core == "vqc"]
    cand = {c["algo"]: c for c in sel["candidates"]}
    for i, al in enumerate(ALGOS):
        d = val[(val.algo == al) & (val.variant == "light")]["norm_score"]
        b = val[(val.algo == al) & (val.variant == "base")]["norm_score"]
        if len(d) == 0:
            continue
        col = PALETTE[1] if al == X else "#c9c9c4"
        adm = cand.get(al, {}).get("admissible", True)
        ax.bar(i, d.mean(), width=0.66, color=col, edgecolor=INK if al == X else "white", linewidth=0.6,
               hatch=None if adm else "////", zorder=2)
        ax.scatter([i] * len(d), d, s=6, color=INK, zorder=3)
        ax.scatter(i, b.mean(), marker="_", s=60, color=PALETTE[0], linewidths=1.6, zorder=4)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xticks(range(len(ALGOS)))
    ax.set_xticklabels([SHORT.get(x, x) for x in ALGOS], rotation=45, ha="right")
    ax.set_ylabel("Validation score")
    ax.set_title(f"LightQDRL-X = {X}", fontsize=7.5, pad=3)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=PALETTE[1], label="light (chosen)"),
                       plt.Rectangle((0, 0), 1, 1, color="#c9c9c4", label="light"),
                       Line2D([], [], marker="_", ls="", color=PALETTE[0], ms=8, mew=1.6, label="base"),
                       plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=INK, hatch="////",
                                     label=f"light < base − {delta:g}")],
              loc="upper center", bbox_to_anchor=(0.5, -0.42), fontsize=5.8, ncol=2, handlelength=1.2)
    ax.yaxis.grid(True, color="#e6e6e2", lw=0.5)
    ax.set_axisbelow(True)

    effects_rows = []
    # ---------------- b: deployment outcomes (2^2)
    ax = ax_b = fig.add_subplot(top[1])
    bench = pd.read_csv(a.bench_csv)
    bench = bench[(bench.algo == X) & (bench.core == "vqc") & bench.variant.isin(VAR_DJ)].copy()
    if os.path.exists(a.latency_csv):
        lat = pd.read_csv(a.latency_csv)[["algo", "variant", "seed", "latency_ms_median"]]
        bench = bench.merge(lat, on=["algo", "variant", "seed"], how="left")
    bench["D"] = bench.variant.map(lambda v: VAR_DJ[v][0])
    bench["J"] = bench.variant.map(lambda v: VAR_DJ[v][1])
    eff = []
    cols = {"D": PALETTE[0], "J": PALETTE[1], "DxJ": MUTED, "P": PALETTE[2], "DxP": MUTED, "JxP": MUTED}
    outcomes = [("static_mem_bytes", "Static memory", True), ("n_2q", "2Q gates", True),
                ("latency_ms_median", "Inference latency", True), ("update_ms_mean", "Update latency", True)]
    have = set(bench.variant)
    if have >= set(VAR_DJ):
        for c, lab, pct in outcomes:
            if c not in bench or bench[c].isna().all():
                continue
            ref = bench[bench.variant == "base"][c].mean()
            d = bench.dropna(subset=[c]).copy()
            d["y"] = 100 * d[c] / ref if pct else d[c]
            e = ols_effects(d, "y", ["D", "J"], [("D", "J")])
            for term in ("D", "J", "DxJ"):
                est, ci, p = e[term]
                eff.append((f"{lab}: {TERM[term]}", est, ci, cols[term]))
                effects_rows.append(dict(panel="b", outcome=c, term=term, estimate=est, ci95=ci, p=p,
                                         unit="% of uncompressed" if pct else "units"))
        forest(ax, eff, f"Deployment costs of {X} (negative = cheaper)")
        ax.set_xlabel("Effect (%-points of the uncompressed cell)")
        sc = ols_effects(bench, "norm_score", ["D", "J"], [("D", "J")])
        txt = "  ".join(f"{TERM[t]}: {sc[t][0]:+.3f}±{sc[t][1]:.3f}"
                        for t in ("D", "J", "DxJ"))
        ax.text(-0.45, -0.30, f"Effect on nominal score (normalized units): {txt}", transform=ax.transAxes, fontsize=6)
        for t in ("D", "J", "DxJ"):
            effects_rows.append(dict(panel="b", outcome="norm_score", term=t, estimate=sc[t][0], ci95=sc[t][1],
                                     p=sc[t][2], unit="normalized score"))
    else:
        ax.text(0.5, 0.5, f"run exp_qrl_bench.py --algos {X} --variants dqcp,jcqq", ha="center", va="center",
                transform=ax.transAxes, color=MUTED)

    # ---------------- c, d: adaptation (2^3)
    axc = fig.add_subplot(bot[0])
    axc2 = fig.add_subplot(bot[1])
    axd = fig.add_subplot(bot[2])
    cells = []
    if os.path.exists(a.adapt_csv):
        ad = pd.read_csv(a.adapt_csv)
        ad = ad[(ad.algo == X) & ad["mode"].isin(["full", "peqa"]) & ad.variant.isin(VAR_DJ)].copy()
        ad["D"] = ad.variant.map(lambda v: VAR_DJ[v][0])
        ad["J"] = ad.variant.map(lambda v: VAR_DJ[v][1])
        ad["P"] = (ad["mode"] == "peqa").astype(int)
        ref_cell = ad[(ad.D == 0) & (ad.J == 0) & (ad.P == 0)]
        eff_s, eff_c = [], []
        for c, lab, pct in (("post_score", "Score", False), ("adapt_energy_J", "Energy", True),
                            ("trainable_params", "Trainable", True)):
            d = ad.copy()
            d["y"] = 100 * d[c] / ref_cell[c].mean() if pct else d[c]
            e = ols_effects(d, "y", ["D", "J", "P"], [("D", "J"), ("D", "P"), ("J", "P")], fe=("seed", "shift"))
            for term in ("D", "J", "P", "DxP", "JxP"):
                est, ci, p = e[term]
                (eff_c if pct else eff_s).append((f"{lab}: {TERM[term]}", est, ci, cols[term]))
                effects_rows.append(dict(panel="c", outcome=c, term=term, estimate=est, ci95=ci, p=p,
                                         unit="% of base/full cell" if pct else "normalized score"))
        forest(axc, eff_s, "Post-shift score effects")
        axc.set_xlabel("Effect (normalized units)")
        forest(axc2, eff_c, "Adaptation cost effects")
        axc2.set_xlabel("Effect (% of base, full fine-tuning)")
        order = [("base", 0), ("dqcp", 0), ("jcqq", 0), ("light", 0), ("base", 1), ("dqcp", 1), ("jcqq", 1), ("light", 1)]
        rng = np.random.default_rng(0)
        for i, (v, p_) in enumerate(order):
            d = ad[(ad.variant == v) & (ad.P == p_)].groupby("seed").post_score.mean()
            if len(d) == 0:
                continue
            m, h = mean_ci(d.values)
            col = PALETTE[2] if p_ else PALETTE[0]
            axd.bar(i, m, width=0.7, color=col, alpha=0.85, edgecolor="white", zorder=2)
            axd.errorbar(i, m, yerr=h, fmt="none", ecolor=INK, elinewidth=0.8, capsize=2, zorder=3)
            axd.scatter(i + rng.uniform(-0.15, 0.15, len(d)), d, s=4, color=INK, zorder=4)
            cells.append(dict(variant=v, peqa=p_, mean=m, ci95=h, n=len(d)))
        axd.set_xticks(range(8))
        axd.set_xticklabels([f"{'D' if VAR_DJ[v][0] else '–'}{'J' if VAR_DJ[v][1] else '–'}{'P' if p_ else '–'}"
                             for v, p_ in order], rotation=45, fontsize=6.5)
        axd.set_ylabel("Post-shift score")
        axd.set_title("Cell means (D=DQCP, J=JCQQ, P=PEQA)", fontsize=7, pad=3)
        axd.axhline(0, color=MUTED, lw=0.6)
        axd.yaxis.grid(True, color="#e6e6e2", lw=0.5)
        axd.set_axisbelow(True)
    else:
        axc2.axis("off")
        axc.text(0.5, 0.5, "run exp_qrl_adapt.py for X with --variants base,dqcp,jcqq,light --modes full,peqa",
                 ha="center", va="center", transform=axc.transAxes, color=MUTED, fontsize=7)
    panel_labels_auto(fig, [(ax_a, "a"), (ax_b, "b"), (axc, "c"), (axc2, "d"), (axd, "e")])
    save(fig, os.path.join(a.out, "fig5_qrl"))
    pd.DataFrame(effects_rows).to_csv(os.path.join(a.out, "fig5_effects.csv"), index=False)
    pd.DataFrame(cells).to_csv(os.path.join(a.out, "fig5_cells.csv"), index=False)
    print(pd.DataFrame(effects_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
