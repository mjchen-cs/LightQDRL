# -*- coding: utf-8 -*-
"""Shared Nature-style plotting helpers for plot_fig5.py / plot_fig6.py / plot_fig7.py.
The figures are drawn ONLY from the CSV raw-data files written by the exp_*.py scripts."""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "lines.linewidth": 1.2, "savefig.dpi": 600, "savefig.bbox": "tight", "figure.dpi": 150,
})

# Colour-blind-validated categorical palette (fixed order, never cycled)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#2b2b29"
MUTED = "#8a8a84"
DOUBLE_COL_IN = 7.2   # Nature double-column width (183 mm)


def mean_ci(x, conf=0.95):
    """Mean and half-width of the t-based 95% confidence interval across seeds."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    if len(x) == 1:
        return x[0], 0.0
    return x.mean(), stats.t.ppf(0.5 + conf / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def panel_label(ax, letter):
    ax.text(-0.40, 1.04, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


def seed_bars(ax, groups, values_by_group, colors, ylabel, log=False, na_groups=(), fmt=None, rotate=35):
    """Bar = mean, error bar = 95% CI, dots = individual seeds (jittered)."""
    rng = np.random.default_rng(0)
    xs = np.arange(len(groups))
    for i, g in enumerate(groups):
        if g in na_groups:
            ax.text(i, 0.02, "N/A", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=6.5, color=MUTED)
            continue
        v = np.asarray(values_by_group[g], float)
        m, ci = mean_ci(v)
        ax.bar(i, m, width=0.68, color=colors[i], alpha=0.85, edgecolor="white", linewidth=0.8, zorder=2)
        ax.errorbar(i, m, yerr=ci, fmt="none", ecolor=INK, elinewidth=0.8, capsize=2.2, capthick=0.8, zorder=4)
        ax.scatter(i + rng.uniform(-0.16, 0.16, len(v)), v, s=7, facecolor="white", edgecolor=INK,
                   linewidth=0.5, zorder=5)
        if fmt is not None:
            ax.text(i, 1.0, fmt(m), transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=5.8, color=INK, clip_on=False)
    ax.set_xticks(xs)
    ax.set_xticklabels(groups, rotation=rotate, ha="right" if rotate else "center")
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, color="#e6e6e2", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def save(fig, stem):
    fig.savefig(stem + ".png")
    fig.savefig(stem + ".pdf")
    print("saved", stem + ".png / .pdf")


def panel_labels_auto(fig, axes_letters, dy=0.012):
    """Place bold panel letters at the left edge of each axis' full extent (tick labels included), just above the
    axis — robust for axes with long y tick labels.  Call after all artists are drawn, right before save()."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for ax, letter in axes_letters:
        bb = ax.get_tightbbox(r)
        x0 = inv.transform((bb.x0, bb.y0))[0]
        y1 = ax.get_position().y1
        fig.text(max(0.0, x0 - 0.005), y1 + dy, letter, fontsize=10, fontweight="bold", va="bottom", ha="left")
