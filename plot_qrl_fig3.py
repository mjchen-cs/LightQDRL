# -*- coding: utf-8 -*-
"""
plot_qrl_fig3.py — Fig. 3: performance of the 7 base/light pairs and paired equivalence tests.

Top row    (a-c)  absolute normalized score per algorithm, base (uncompressed) vs light (LightQDRL), mean and 95% CI
                  across seeds, one panel per environment (MEC, CartPole-v1, multichannel access).
Bottom row (d-f)  paired difference light - base per algorithm (same seed): mean, 90% CI (thick, = TOST) and
                  95% CI (thin), individual seeds, shaded equivalence region [-delta, +delta].
                  Filled marker = equivalent (TOST, Holm-corrected over the 7 algorithms, alpha = 0.05);
                  grey cross = neither variant learned (mean score < --floor for both): equivalence is not claimed.

Normalized scores (higher is better, comparable across panels)
  MEC       (R - R_random) / (R_myopic - R_random)          0 = random, 1 = myopic max-weight (exhaustive search)
  CartPole  R / 500                                          1 = maximum episode return
  Channel   (R - R_random) / (R_myopic - R_random)          0 = random channel, 1 = belief-myopic (Whittle) policy
  (Acrobot-v1 was replaced by the multichannel-access task: no algorithm learned it within the budget; pass
   --envs mec,cartpole,acrobot to plot it anyway)
The equivalence margin delta (default 0.05 normalized units) must be fixed before looking at the data and reported
in the paper; change it with --margin.

Inputs   results_qrl/<env>/per_seed.csv, results_qrl/<env>/reference_policies.json   (written by exp_qrl_bench.py)
Outputs  figures/fig3_qrl.png/.pdf, figures/fig3_stats.csv (all numbers shown in the figure + TOST p-values)
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

from plot_common import PALETTE, INK, MUTED, DOUBLE_COL_IN, mean_ci, panel_label, save

ALGOS = ["Q-DQN", "Q-DDQN", "Q-DuelingDQN", "Q-A2C", "Q-A3C", "Q-PPO", "Q-TRPO"]
LABELS = {"Q-DQN": "Q-DQN", "Q-DDQN": "Q-DDQN", "Q-DuelingDQN": "Q-Dueling", "Q-A2C": "Q-A2C",
          "Q-A3C": "Q-A3C", "Q-PPO": "Q-PPO", "Q-TRPO": "Q-TRPO"}
ENVS = [("mec", "WP-MEC (queue-aware)"), ("cartpole", "CartPole-v1"), ("channel", "Multichannel access")]
C_BASE, C_LIGHT = PALETTE[0], PALETTE[1]


def normalized(df, env, ref):
    R = df["eval_return"].astype(float)
    if env in ("mec", "channel"):
        return (R - ref["random"]) / (ref["myopic"] - ref["random"])
    if env == "cartpole":
        return R / 500.0
    r0 = ref.get("random", -500.0)
    return (R - r0) / (0.0 - r0)


def holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    idx = np.argsort(p[ok])
    m = ok.sum()
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(idx):
        run = max(run, (m - k) * p[ok][i])
        adj[i] = min(1.0, run)
    out[ok] = adj
    return out


def tost_paired(d, delta):
    """Two one-sided paired t-tests of H0: mean(d) <= -delta or >= +delta.  Returns p_TOST = max(p_lower, p_upper)."""
    d = np.asarray(d, float)
    n = len(d)
    if n < 2:
        return np.nan
    se = d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return 0.0 if abs(d.mean()) < delta else 1.0
    t_lo = (d.mean() + delta) / se
    t_hi = (d.mean() - delta) / se
    return float(max(1 - stats.t.cdf(t_lo, n - 1), stats.t.cdf(t_hi, n - 1)))


def ci_half(d, conf):
    d = np.asarray(d, float)
    if len(d) < 2:
        return 0.0
    return stats.t.ppf(0.5 + conf / 2, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results_qrl")
    ap.add_argument("--margin", type=float, default=0.05, help="equivalence margin delta in normalized units")
    ap.add_argument("--floor", type=float, default=0.05,
                    help="if the mean score of both variants is below this (0 = random), the pair is flagged 'not learned' and is "
                         "never counted as equivalent (equivalence of two failed learners is uninformative)")
    ap.add_argument("--envs", default="mec,cartpole,channel")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COL_IN, 4.9), gridspec_kw=dict(hspace=0.75, wspace=0.42))
    rows = []
    rng = np.random.default_rng(0)
    titles = dict(ENVS + [("acrobot", "Acrobot-v1")])
    for col, env in enumerate(a.envs.split(",")[:3]):
        title = titles[env]
        f = os.path.join(a.root, env, "per_seed.csv")
        if not os.path.exists(f):
            for r in range(2):
                axes[r, col].text(0.5, 0.5, f"{env}: no data", ha="center", va="center", color=MUTED,
                                  transform=axes[r, col].transAxes)
            continue
        df = pd.read_csv(f)
        df = df[(df["core"] == "vqc") & df["algo"].isin(ALGOS) & df["variant"].isin(["base", "light"])].copy()
        ref = json.load(open(os.path.join(a.root, env, "reference_policies.json")))
        df["score"] = normalized(df, env, ref)
        piv = df.pivot_table(index=["algo", "seed"], columns="variant", values="score").dropna()
        xs = np.arange(len(ALGOS))

        # ---------------- top: absolute scores
        ax = axes[0, col]
        for i, al in enumerate(ALGOS):
            if al not in piv.index.get_level_values(0):
                continue
            p = piv.loc[al]
            for k, (v, c, dx) in enumerate((("base", C_BASE, -0.17), ("light", C_LIGHT, 0.17))):
                m, h = mean_ci(p[v])
                ax.errorbar(i + dx, m, yerr=h, fmt="o", ms=4, color=c, mfc=c, mec="white", mew=0.6,
                            elinewidth=0.9, capsize=0, zorder=4)
                ax.scatter(i + dx + rng.uniform(-0.05, 0.05, len(p)), p[v], s=3, color=c, alpha=0.35,
                           linewidths=0, zorder=3)
        if env in ("mec", "channel"):
            ax.axhline(0, color=MUTED, lw=0.6, ls=":", zorder=1)
            ax.axhline(1, color=MUTED, lw=0.6, ls="--", zorder=1)
            ax.text(len(ALGOS) - 0.5, 1, "myopic", ha="right", va="bottom", fontsize=6, color=MUTED)
            ax.text(len(ALGOS) - 0.5, 0, "random", ha="right", va="bottom", fontsize=6, color=MUTED)
        ax.set_xticks(xs)
        ax.set_xticklabels([LABELS[x] for x in ALGOS], rotation=45, ha="right")
        ax.set_ylabel("Normalized score")
        ax.set_title(title, fontsize=8, pad=4)
        ax.yaxis.grid(True, color="#e6e6e2", lw=0.5)
        ax.set_axisbelow(True)
        panel_label(ax, "abc"[col])

        # ---------------- bottom: paired differences + equivalence region
        ax = axes[1, col]
        ax.axhspan(-a.margin, a.margin, color="#e9eef7", zorder=0, lw=0)
        ax.axhline(0, color=MUTED, lw=0.6, zorder=1)
        p_list, res = [], []
        for i, al in enumerate(ALGOS):
            if al not in piv.index.get_level_values(0):
                p_list.append(np.nan); res.append(None)
                continue
            p = piv.loc[al]
            d = (p["light"] - p["base"]).values
            p_list.append(tost_paired(d, a.margin))
            res.append(d)
        p_adj = holm(p_list)
        for i, al in enumerate(ALGOS):
            d = res[i]
            if d is None:
                continue
            m = d.mean()
            h90, h95 = ci_half(d, 0.90), ci_half(d, 0.95)
            pb = piv.loc[al]
            floor = max(pb["base"].mean(), pb["light"].mean()) < a.floor     # neither variant better than random
            eq = np.isfinite(p_adj[i]) and p_adj[i] < 0.05 and not floor
            ax.vlines(i, m - h95, m + h95, color=INK, lw=0.7, zorder=3)
            ax.vlines(i, m - h90, m + h90, color=INK, lw=2.2, zorder=3)
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(d)), d, s=4, color=C_LIGHT, alpha=0.45, linewidths=0, zorder=2)
            if floor:
                ax.scatter(i, m, s=22, marker="x", color=MUTED, linewidth=1.0, zorder=5)
            else:
                ax.scatter(i, m, s=22, facecolor=INK if eq else "white", edgecolor=INK, linewidth=0.8, zorder=5)
            w = stats.wilcoxon(d).pvalue if len(d) > 1 and np.any(d != 0) else np.nan
            rows.append(dict(env=env, algo=al, n_seeds=len(d), base_mean=pb["base"].mean(),
                             base_ci95=mean_ci(pb["base"])[1], light_mean=pb["light"].mean(),
                             light_ci95=mean_ci(pb["light"])[1], diff_mean=m, diff_ci90_lo=m - h90,
                             diff_ci90_hi=m + h90, diff_ci95_lo=m - h95, diff_ci95_hi=m + h95, margin=a.margin,
                             p_tost=p_list[i], p_tost_holm=p_adj[i], equivalent=bool(eq), floor_effect=bool(floor),
                             p_wilcoxon=w))
        ax.set_xticks(xs)
        ax.set_xticklabels([LABELS[x] for x in ALGOS], rotation=45, ha="right")
        ax.set_ylabel("Δ score (light − base)")
        ax.yaxis.grid(True, color="#e6e6e2", lw=0.5)
        ax.set_axisbelow(True)
        lim = max(a.margin * 1.6, np.nanmax([np.max(np.abs(r)) for r in res if r is not None] + [0]) * 1.08)
        ax.set_ylim(-lim, lim)
        panel_label(ax, "def"[col])

    handles = [Line2D([], [], marker="o", ls="", color=C_BASE, mec="white", label="Base (uncompressed)"),
               Line2D([], [], marker="o", ls="", color=C_LIGHT, mec="white", label="Light (LightQDRL)"),
               Line2D([], [], marker="o", ls="", mfc=INK, mec=INK, label="Equivalent (TOST, Holm)"),
               Line2D([], [], marker="o", ls="", mfc="white", mec=INK, label="Not shown equivalent"),
               Line2D([], [], marker="x", ls="", color=MUTED, label="Not learned (floor)"),
               plt.Rectangle((0, 0), 1, 1, color="#e9eef7", label=f"Equivalence region ±{a.margin:g}")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07), handletextpad=0.3,
               columnspacing=1.0)
    save(fig, os.path.join(a.out, "fig3_qrl"))
    st = pd.DataFrame(rows)
    st.to_csv(os.path.join(a.out, "fig3_stats.csv"), index=False)
    with pd.option_context("display.width", 200):
        print(st[["env", "algo", "base_mean", "light_mean", "diff_mean", "diff_ci90_lo", "diff_ci90_hi",
                  "p_tost_holm", "equivalent", "floor_effect"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
