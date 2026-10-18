# -*- coding: utf-8 -*-
"""
exp_qrl_robust.py — noise, shot and topology robustness of the 14 trained models (Fig. 7).

For every trained MEC agent (7 algorithms x base/light x seeds, results_qrl/mec/runs) the deployed policy circuit is
re-evaluated on a fixed set of n_states MEC states (collected once with a random policy, identical for all models)
and compared with its own noiseless decisions:
  noise      : exact density-matrix simulation with 2Q depolarizing p2 in --p2_levels, 1Q depolarizing p2/10
  combined   : realistic superconducting-like setting  p2 = 0.01, p1 = 0.001, amplitude damping 0.002, readout 0.02
               (x1 and x3)
  shots      : finite-shot estimation of <Z> (noiseless) with --shots_levels shots
  topology   : the circuit is routed (Qiskit SABRE) onto all-to-all / ring / linear / heavy-hex couplings, SWAPs are
               simulated as 3 noisy CNOTs under the combined noise; compiled 2Q count, depth and SWAPs are recorded
  closed-loop: --episodes MEC episodes acting with the noisy circuit (noiseless, combined x1, combined x3) ->
               normalized score (0 random, 1 myopic)
Metrics: device-decision agreement with the noiseless decisions, exact match of the full offloading vector, Kendall
tau of the per-device preferences, mean |<Z>| error.

Outputs  results_qrl_robust/mec/per_model.csv (one row per model x condition), states.npz, run_config.json
Usage    python exp_qrl_robust.py --workers 4            (resumable: finished models are skipped)
"""
import os
import re
import time
import argparse
import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau

import lqdrl_core as LC
from qrl_core import SV
from qrl_envs import make_env, reference_action
from quafu_hw_validation import load_agent_net, preference
from exp_qrl_bench import EVAL_SEED0, norm_score

COMBINED = dict(p2=0.01, p1=0.001, gamma=0.002, ro=0.02)


def scaled(noise, k):
    return {key: min(0.5, v * k) for key, v in noise.items()}


def decisions(net, e, nb=10):
    out = net.heads_from_features(net.post_circuit(e))
    pref = preference(net, out, nb)
    return pref, (pref > 0).astype(int)


def expz_noisy(dm, ops, ang, th, noise, layout=None):
    nz = {k: v for k, v in noise.items() if k != "ro"}
    pr = dm.probs(ops, ang, th, noise=nz)
    return dm.expz_from_probs(pr, layout=layout, readout=noise.get("ro", 0.0)).to(torch.float32)


def compare(pref0, m0, e0, pref, m, e):
    taus = [kendalltau(pref0[i], pref[i])[0] for i in range(len(m0))]
    return dict(agreement=float((m == m0).mean()), exact_match=float((m == m0).all(1).mean()),
                kendall_tau=float(np.nanmean(taus)), expz_mae=float((e - e0).abs().mean()))


def closed_loop(net, dm, noise, n_ep, ref):
    ops = net.ops()
    rets = []
    for k in range(n_ep):
        env = make_env("mec", seed=EVAL_SEED0 + k)
        o = env.reset(seed=EVAL_SEED0 + k)
        done, R = False, 0.0
        while not done:
            with torch.no_grad():
                ang, th = net.pre_circuit(torch.tensor(o).unsqueeze(0))
                e = SV.run(ops, ang, th) if noise is None else expz_noisy(dm, ops, ang, th, noise)
                _, m = decisions(net, e)
            o, r, done, _ = env.step(m[0])
            R += r
        rets.append(R)
    return norm_score("mec", float(np.mean(rets)), ref)


def run_model(job):
    algo, variant, seed, a = job
    torch.set_num_threads(a.threads)
    out_f = os.path.join(a.out, "models", f"{algo}_{variant}_seed{seed}.csv")
    if os.path.exists(out_f):
        return out_f
    net, _ = load_agent_net(a.run_dir, algo, variant, seed)
    ops = net.ops()
    S = torch.tensor(np.load(os.path.join(a.out, "states.npz"))["S"])
    dm = LC.DensityMatrixSim()
    gen = torch.Generator().manual_seed(seed)
    rows = []
    with torch.no_grad():
        ang, th = net.pre_circuit(S)
        ang, th = ang.detach(), th.detach()
        e0 = SV.run(ops, ang, th)
        pref0, m0 = decisions(net, e0)
        base = dict(algo=algo, variant=variant, seed=seed, n_2q_logical=LC.gate_counts(ops)[1],
                    depth_logical=LC.logical_depth(ops))
        for p in a.p2_levels:
            e = expz_noisy(dm, ops, ang, th, dict(p2=p, p1=p / 10))
            pref, m = decisions(net, e)
            rows.append(dict(base, condition="depolarizing", level=p, **compare(pref0, m0, e0, pref, m, e)))
        for k in (1, 3):
            e = expz_noisy(dm, ops, ang, th, scaled(COMBINED, k))
            pref, m = decisions(net, e)
            rows.append(dict(base, condition="combined", level=k, **compare(pref0, m0, e0, pref, m, e)))
        pr0 = dm.probs(ops, ang, th, noise=None)
        for sh in a.shots_levels:
            e = dm.sample_expz(pr0, sh, gen).to(torch.float32)
            pref, m = decisions(net, e)
            rows.append(dict(base, condition="shots", level=sh, **compare(pref0, m0, e0, pref, m, e)))
        for topo, coupling in LC.TOPOLOGIES.items():
            st = LC.transpile_stats(ops, coupling)
            rops, layout = LC.routed_ops(ops, coupling)
            e = expz_noisy(dm, rops, ang, th, COMBINED, layout=layout)
            pref, m = decisions(net, e)
            rows.append(dict(base, condition="topology", level=topo, compiled_2q=st["transpiled_2q"],
                             compiled_depth=st["transpiled_depth"], swaps=st["swaps"],
                             **compare(pref0, m0, e0, pref, m, e)))
    ref = LC.load_json(os.path.join(a.bench_root, "mec", "reference_policies.json"))
    for lab, noise in (("noiseless", None), ("combined_x1", COMBINED), ("combined_x3", scaled(COMBINED, 3))):
        rows.append(dict(base, condition="closed_loop", level=lab, score=closed_loop(net, dm, noise, a.episodes, ref)))
    pd.DataFrame(rows).to_csv(out_f, index=False)
    print(f"{algo}_{variant}_seed{seed}: done", flush=True)
    return out_f


def collect_states(n, seed=20000):
    env = make_env("mec", seed=seed)
    o = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    S = []
    while len(S) < n:
        S.append(o)
        o, _, done, _ = env.step(reference_action(env, "random", rng))
        if done:
            o = env.reset()
    return np.array(S, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench_root", default="results_qrl")
    ap.add_argument("--algos", default="Q-DQN,Q-DDQN,Q-DuelingDQN,Q-A2C,Q-A3C,Q-PPO,Q-TRPO")
    ap.add_argument("--variants", default="base,light")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n_states", type=int, default=256)
    ap.add_argument("--p2_levels", default="0.001,0.003,0.01,0.03,0.1")
    ap.add_argument("--shots_levels", default="64,128,256,512,1024,4096")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default="results_qrl_robust/mec")
    a = ap.parse_args()
    a.p2_levels = [float(x) for x in a.p2_levels.split(",")]
    a.shots_levels = [int(x) for x in a.shots_levels.split(",")]
    a.run_dir = os.path.join(a.bench_root, "mec", "runs")
    os.makedirs(os.path.join(a.out, "models"), exist_ok=True)
    st_f = os.path.join(a.out, "states.npz")
    if not os.path.exists(st_f):
        np.savez_compressed(st_f, S=collect_states(a.n_states))
    LC.save_json(dict(args=vars(a), combined_noise=COMBINED, env=LC.env_info(),
                      started=time.strftime("%Y-%m-%d %H:%M:%S")), os.path.join(a.out, "run_config.json"))
    jobs = [(al, v, s, a) for s in range(a.seeds) for al in a.algos.split(",") for v in a.variants.split(",")
            if os.path.exists(os.path.join(a.run_dir, f"{al}_{v}_vqc_seed{s}.pt"))]
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.workers) as pool:
            for _ in pool.imap_unordered(run_model, jobs):
                pass
    else:
        for j in jobs:
            run_model(j)
    files = [os.path.join(a.out, "models", f) for f in sorted(os.listdir(os.path.join(a.out, "models")))]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(os.path.join(a.out, "per_model.csv"), index=False)
    print(df[df.condition == "combined"].groupby(["algo", "variant", "level"])[["agreement", "kendall_tau"]]
          .mean().round(3).to_string())


if __name__ == "__main__":
    main()
