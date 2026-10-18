# -*- coding: utf-8 -*-
"""
plot_qrl_fig4.py — Fig. 4: deployment-cost reduction of LightQDRL for each of the 7 quantum RL algorithms.

Heat map, rows = algorithms, columns = cost metrics; cell = mean paired reduction 100 x (1 - light/base) over seeds
(same seed for base and light), text = mean and 95% CI half-width.  Blue = cost reduced, red = cost increased,
neutral grey at 0 (diverging scale, symmetric limits).

Metrics (all from results_qrl/<env>/per_seed.csv unless stated)
  2Q gates        n_2q                  logical CNOT count of the deployed circuit
  Depth           transpiled_depth      depth after Qiskit transpilation on the 6-qubit heavy-hex topology
  SWAPs           swaps                 SWAPs inserted by routing on heavy-hex (base needs SWAPs; reductions of
                                        "x -> 0" are 100 %; a seed pair with base = 0 is skipped)
  Memory          static_mem_bytes      static model memory
  Latency         latency_ms_median     single-decision inference latency, results_qrl/<env>/latency.csv
                                        (python measure_qrl_latency.py --env mec); if that file is missing the
                                        per-update training latency update_ms_mean is used and the column says so
  Update energy   train_update_energy_J total energy of all training updates (5 W x measured update time)
Note: DQCP needs extra no-grad loss evaluations during training (dqcp_extra_circuit_evals); they are not part of
the timed updates and are reported in fig4_costs.csv for transparency.

Outputs figures/fig4_qrl.png/.pdf and figures/fig4_costs.csv (absolute means of every metric + reductions + CIs)
Usage   python plot_qrl_fig4.py --env mec
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from plot_common import INK, DOUBLE_COL_IN, mean_ci, save

ALGOS = ["Q-DQN", "Q-DDQN", "Q-DuelingDQN", "Q-A2C", "Q-A3C", "Q-PPO", "Q-TRPO"]
LABELS = {"Q-DuelingDQN": "Q-Dueling DQN"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="mec")
    ap.add_argument("--root", default="results_qrl")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(os.path.join(a.root, a.env, "per_seed.csv"))
    df = df[(df.core == "vqc") & df.algo.isin(ALGOS) & df.variant.isin(["base", "light"])].copy()

    lat_f = os.path.join(a.root, a.env, "latency.csv")
    if os.path.exists(lat_f):
        lat = pd.read_csv(lat_f)[["algo", "variant", "seed", "latency_ms_median"]]
        df = df.merge(lat, on=["algo", "variant", "seed"], how="left")
        lat_col, lat_name = "latency_ms_median", "Inference\nlatency"
    else:
        lat_col, lat_name = "update_ms_mean", "Update\nlatency*"
        print("latency.csv not found -> using per-update training latency (run measure_qrl_latency.py)")
    metrics = [("n_2q", "2Q gates"), ("transpiled_depth", "Depth\n(heavy-hex)"), ("swaps", "SWAPs\n(heavy-hex)"),
               ("static_mem_bytes", "Static\nmemory"), (lat_col, lat_name), ("train_update_energy_J", "Update\nenergy")]

    M = np.full((len(ALGOS), len(metrics)), np.nan)
    H = np.full_like(M, np.nan)
    rows = []
    for i, al in enumerate(ALGOS):
        d = df[df.algo == al]
        rec = dict(algo=al)
        for j, (c, _) in enumerate(metrics):
            if c not in d:
                continue
            p = d.pivot_table(index="seed", columns="variant", values=c).dropna()
            p = p[p["base"] > 0]
            if len(p) == 0:
                continue
            red = 100.0 * (1.0 - p["light"] / p["base"])
            M[i, j], H[i, j] = mean_ci(red.values)
            rec.update({f"{c}_base": p["base"].mean(), f"{c}_light": p["light"].mean(),
                        f"{c}_reduction_pct": M[i, j], f"{c}_reduction_ci95": H[i, j], f"{c}_n_pairs": len(p)})
        rec["dqcp_extra_circuit_evals_light"] = d[d.variant == "light"]["dqcp_extra_circuit_evals"].mean()
        rows.append(rec)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "fig4_costs.csv"), index=False)

    lim = max(10.0, np.nanmax(np.abs(M)))
    cmap = LinearSegmentedColormap.from_list("div", ["#c8392f", "#f2c9c3", "#efefec", "#bcd3f2", "#1f5fb0"])
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_IN * 0.75, 3.1))
    im = ax.imshow(M, cmap=cmap, norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim), aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isfinite(M[i, j]):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=6.5, color=INK)
                continue
            dark = abs(M[i, j]) > 0.6 * lim
            ax.text(j, i - 0.12, f"{M[i, j]:.0f}%", ha="center", va="center", fontsize=7,
                    color="white" if dark else INK, fontweight="bold")
            ax.text(j, i + 0.24, f"±{H[i, j]:.0f}", ha="center", va="center", fontsize=5.5,
                    color="white" if dark else INK)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m[1] for m in metrics], fontsize=6.5)
    ax.set_yticks(range(len(ALGOS)))
    ax.set_yticklabels([LABELS.get(x, x) for x in ALGOS])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(metrics)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ALGOS)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.xaxis.tick_top()
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Cost reduction (%)  [blue = lower cost]")
    cb.outline.set_visible(False)
    if lat_col == "update_ms_mean":
        fig.text(0.01, -0.02, "*per-update training latency; run measure_qrl_latency.py for inference latency",
                 fontsize=6)
    save(fig, os.path.join(a.out, "fig4_qrl"))
    print(pd.DataFrame(M, index=ALGOS, columns=[m[1].replace("\n", " ") for m in metrics]).round(1).to_string())


if __name__ == "__main__":
    main()
