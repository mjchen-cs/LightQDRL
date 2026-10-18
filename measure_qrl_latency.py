# -*- coding: utf-8 -*-
"""
measure_qrl_latency.py — single-decision inference latency of every trained base / light model (needed for Fig. 4).

For each results_qrl/<env>/runs/<algo>_<variant>_vqc_seed<k>.pt the deployed network is loaded and one decision
(batch 1, exact state-vector simulation of the policy circuit, classical encoder/decoder, argmax) is timed
--reps times after --warmup untimed calls on a fixed observation; the median is stored.
Light models run with their deployed (pruned, quantized) circuit.  Run on an idle machine, single thread.

Output: results_qrl/<env>/latency.csv  (algo, variant, seed, latency_ms_median, latency_ms_p90, energy_mJ = 5 W x median)
Usage:  python measure_qrl_latency.py --env mec
"""
import os
import re
import time
import argparse
import numpy as np
import pandas as pd
import torch

from quafu_hw_validation import load_agent_net
from qrl_envs import make_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="mec", choices=["mec", "cartpole", "channel", "acrobot"])
    ap.add_argument("--root", default="results_qrl")
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--power_w", type=float, default=5.0)
    a = ap.parse_args()
    torch.set_num_threads(1)
    run_dir = os.path.join(a.root, a.env, "runs")
    obs = torch.tensor(make_env(a.env, seed=12345).reset(seed=12345)).unsqueeze(0)
    pat = re.compile(r"(.+)_(base|light|dqcp|jcqq)_vqc_seed(\d+)\.pt$")
    rows = []
    files = sorted(f for f in os.listdir(run_dir) if pat.match(f))
    for k, f in enumerate(files):
        algo, variant, seed = pat.match(f).groups()
        net, _ = load_agent_net(run_dir, algo, variant, int(seed))
        with torch.no_grad():
            for _ in range(a.warmup):
                out = net(obs)
            ts = []
            for _ in range(a.reps):
                t0 = time.perf_counter()
                out = net(obs)
                _ = {h: v.argmax(-1) for h, v in out.items()}
                ts.append(time.perf_counter() - t0)
        ts = np.array(ts) * 1e3
        rows.append(dict(env=a.env, algo=algo, variant=variant, seed=int(seed), latency_ms_median=float(np.median(ts)),
                         latency_ms_p90=float(np.percentile(ts, 90)),
                         energy_mJ=float(a.power_w * np.median(ts)), n_ops=len(net.ops())))
        print(f"[{k + 1}/{len(files)}] {f}: {np.median(ts):.3f} ms", flush=True)
    df = pd.DataFrame(rows)
    out = os.path.join(a.root, a.env, "latency.csv")
    df.to_csv(out, index=False)
    print(df.groupby(["algo", "variant"]).latency_ms_median.mean().unstack().round(3).to_string())
    print("written to", out)


if __name__ == "__main__":
    main()
