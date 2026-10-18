# -*- coding: utf-8 -*-
"""
exp_qrl_bench.py — main benchmark of the plug-and-play compression framework.

For every algorithm in {Q-DQN, Q-DDQN, Q-DuelingDQN, Q-A2C, Q-A3C, Q-PPO, Q-TRPO} and every seed it trains
    variant 'base'  : uncompressed hybrid agent
    variant 'light' : the same agent with the LightQDRL framework (DQCP + JCQQ)
on one environment (mec | cartpole | channel; acrobot kept for reference), evaluates the deterministic policy periodically and at the end on
fixed evaluation seeds, and stores the complete raw data.

Supplementary controls (same agents, no quantum circuit): --cores tanh (quantum-removal) or mlp (classical).

Seeds:  test seeds 0..S-1 (reported);  use --seed_offset 100 for validation/tuning runs (never reported as results).

Outputs (./results_qrl/<env>/):
  runs/<algo>_<variant>_<core>_seed<k>.json  scalar results, eval curve, accounting, DQCP history, hyper-parameters
  runs/<algo>_<variant>_<core>_seed<k>.npz   training episode returns, losses, update times, eval curve
  runs/<algo>_<variant>_<core>_seed<k>.pt    final network (deployed actor / Q-network) for hardware validation
  reference_policies.json                     random / all-local / all-offload / myopic (MEC) on the eval seeds
  per_seed.csv                                one row per run  <- analysis / plotting scripts
Usage:
  python exp_qrl_bench.py --env mec --workers 4
  python exp_qrl_bench.py --env cartpole --algos Q-PPO,Q-DQN --seeds 10
  python exp_qrl_bench.py --env mec --quick
  python exp_qrl_bench.py --env channel --workers 4
  python exp_qrl_bench.py --env mec --algos Q-DuelingDQN --variants dqcp,jcqq      # factorial cells (Fig. 5)
"""
import os
import sys
import json
import time
import argparse
import itertools
import numpy as np
import pandas as pd
import torch

import lqdrl_core as LC
from qrl_core import accounting
from qrl_envs import make_env, reference_action
from qrl_agents import ALGOS, ENV_HP, make_agent

ENVS = ["mec", "cartpole", "channel", "acrobot"]
STEPS = {"mec": 50000, "cartpole": 50000, "channel": 30000, "acrobot": 50000}
EVAL_EP = {"mec": 10, "cartpole": 10, "channel": 20, "acrobot": 10}
EVAL_SEED0 = 10_000
VARIANTS = {"base": False, "light": True, "dqcp": "dqcp", "jcqq": "jcqq"}     # dqcp / jcqq: factorial study
REF_KINDS = {"mec": ["random", "local", "offload", "maxrate", "myopic"], "channel": ["random", "myopic"],
             "cartpole": ["random"], "acrobot": ["random"]}


def norm_score(env_name, R, ref):
    """0 = random, 1 = myopic (MEC: max-weight; channel: belief-myopic); CartPole: R/500; Acrobot: (R+500)/500."""
    if env_name in ("mec", "channel"):
        return (R - ref["random"]) / (ref["myopic"] - ref["random"])
    if env_name == "cartpole":
        return R / 500.0
    return (R - ref.get("random", -500.0)) / (0.0 - ref.get("random", -500.0))


def evaluate(agent, env_name, n_ep, seed0=EVAL_SEED0, env_kw=None):
    rets, served, backlog, dropped, arrived, succ, steps = [], [], [], [], [], [], 0
    for k in range(n_ep):
        env = make_env(env_name, seed=seed0 + k, **(env_kw or {}))
        o = env.reset(seed=seed0 + k)
        done, R = False, 0.0
        while not done:
            a = agent.act(o, deterministic=True)
            o, r, done, info = env.step(a)
            R += r
            steps += 1
            if env_name == "mec":
                served.append(info["served"]); backlog.append(info["backlog"]); dropped.append(info["dropped"])
                arrived.append(info["arrived"])
            if env_name == "channel":
                succ.append(info["success"])
        rets.append(R)
    out = dict(eval_return=float(np.mean(rets)), eval_return_std=float(np.std(rets)), eval_returns=rets)
    if env_name == "mec":
        out.update(served_frac=float(np.sum(served) / (env.C_ref * len(served))),
                   served_over_arrived=float(np.sum(served) / max(1e-9, np.sum(arrived))),
                   mean_backlog_norm=float(np.mean(backlog) / env.Q_ref),
                   drop_rate=float(np.sum(dropped) / max(1e-9, np.sum(served) + np.sum(dropped))))
    if env_name == "channel":
        out.update(success_rate=float(np.mean(succ)))
    return out


def reference_scores(out_dir, env_name, n_ep, env_kw=None, fname="reference_policies.json"):
    f = os.path.join(out_dir, fname) if out_dir else None
    if f and os.path.exists(f):
        return LC.load_json(f)
    res = {}
    for kind in REF_KINDS[env_name]:
        rets = []
        for k in range(n_ep):
            env = make_env(env_name, seed=EVAL_SEED0 + k, **(env_kw or {}))
            o = env.reset(seed=EVAL_SEED0 + k)
            rng = np.random.default_rng(k)
            done, R = False, 0.0
            while not done:
                if env_name in ("mec", "channel"):
                    a = reference_action(env, kind, rng)
                else:
                    a = rng.integers(0, env.branches[0], 1)
                o, r, done, _ = env.step(a)
                R += r
            rets.append(R)
        res[kind] = float(np.mean(rets))
        print(f"  reference {kind:8s}: {res[kind]:.3f}", flush=True)
    if out_dir:
        LC.save_json(res, f)
    return res


def run_one(job):
    algo, variant, core, seed, a = job
    torch.set_num_threads(a.threads)
    tag = f"{algo}_{variant}_{core}_seed{seed}"
    stem = os.path.join(a.out, "runs", tag)
    if os.path.exists(stem + ".json"):
        return stem
    env_fn = lambda s: make_env(a.env, seed=s)
    agent = make_agent(algo, env_fn, core=core, compress=VARIANTS[variant], seed=seed, hp=ENV_HP.get(a.env))
    curve = []

    def cb(t):
        ev = evaluate(agent, a.env, max(2, EVAL_EP[a.env] // 2))
        curve.append((t, ev["eval_return"]))
        print(f"  [{tag}] step {t}: eval return {ev['eval_return']:.3f}", flush=True)

    t0 = time.time()
    episodes = agent.train(a.steps, eval_cb=cb, eval_every=a.eval_every)
    wall = time.time() - t0
    final = evaluate(agent, a.env, EVAL_EP[a.env])
    net = agent.net
    net.eval()
    acc = accounting(net, transpile=(core == "vqc" and not a.no_transpile))
    ut = np.array(agent.update_times) if agent.update_times else np.array([np.nan])
    rec = dict(env=a.env, algo=algo, variant=variant, core=core, seed=seed, steps=a.steps, train_wall_s=wall,
               n_updates=agent.n_updates, update_ms_mean=float(np.nanmean(ut) * 1e3),
               update_energy_J=float(5.0 * np.nanmean(ut)), train_update_energy_J=float(5.0 * np.nansum(ut)),
               **{k: v for k, v in final.items() if k != "eval_returns"}, eval_returns=final["eval_returns"],
               eval_curve=curve, **acc,
               dqcp_history=agent.dqcp.history if agent.dqcp else [],
               dqcp_extra_circuit_evals=agent.dqcp.extra_evals if agent.dqcp else 0,
               rot_mask=net.rot_mask.numpy(), ent_mask=net.ent_mask.numpy(), hp=agent.hp, env_info=LC.env_info())
    ep = np.array(episodes, dtype=np.float64) if episodes else np.zeros((0, 2))
    np.savez_compressed(stem + ".npz", episodes=ep, losses=np.array(agent.losses, np.float32), update_times=ut,
                        eval_curve=np.array(curve))
    torch.save(dict(state=net.state_dict(), algo=algo, variant=variant, core=core, obs_dim=agent.obs_dim,
                    branches=agent.branches, heads={k: h.fc2.out_features for k, h in net.heads.items()},
                    compress=VARIANTS[variant], env=a.env, hp=agent.hp,
                    critic=agent.critic.state_dict() if hasattr(agent, "critic") else None), stem + ".pt")
    LC.save_json(rec, stem + ".json")
    print(f"{tag}: final eval return {final['eval_return']:.3f}  2Q={acc['n_2q']}  mem={acc['static_mem_bytes']}B "
          f"({wall:.0f}s)", flush=True)
    return stem


def collect(a):
    rows = []
    ref = LC.load_json(os.path.join(a.out, "reference_policies.json")) if os.path.exists(os.path.join(a.out, "reference_policies.json")) else {}
    for f in sorted(os.listdir(os.path.join(a.out, "runs"))):
        if not f.endswith(".json"):
            continue
        r = LC.load_json(os.path.join(a.out, "runs", f))
        for k in ("eval_curve", "eval_returns", "dqcp_history", "rot_mask", "ent_mask", "hp", "env_info"):
            r.pop(k, None)
        if ref:
            r["norm_score"] = norm_score(a.env, r["eval_return"], ref)
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "per_seed.csv"), index=False)
    if len(df):
        print(df.groupby(["algo", "variant", "core"])[["eval_return", "n_2q", "static_mem_bytes", "update_ms_mean"]]
              .agg(["mean", "std"]).round(4).to_string())
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="mec", choices=ENVS)
    ap.add_argument("--algos", default=",".join(ALGOS))
    ap.add_argument("--variants", default="base,light", help="base, light, dqcp, jcqq (comma-separated)")
    ap.add_argument("--cores", default="vqc", help="vqc | tanh (quantum-removal) | mlp (classical), comma-separated")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed_offset", type=int, default=0, help="100 for validation / tuning runs")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--eval_every", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--no_transpile", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    a.steps = a.steps or STEPS[a.env]
    if a.quick:
        a.steps, a.seeds, a.eval_every = 4000, 1, 2000
    a.out = a.out or os.path.join("results_qrl" if a.seed_offset == 0 else "results_qrl_validation", a.env)
    os.makedirs(os.path.join(a.out, "runs"), exist_ok=True)
    algos = [x.strip() for x in a.algos.split(",") if x.strip()]
    LC.save_json(dict(args=vars(a), env=LC.env_info(), started=time.strftime("%Y-%m-%d %H:%M:%S")),
                 os.path.join(a.out, "run_config.json"))
    print("reference policies on the evaluation seeds:")
    reference_scores(a.out, a.env, EVAL_EP[a.env])
    jobs = [(al, v, c, a.seed_offset + s, a) for s in range(a.seeds) for al in algos
            for v in a.variants.split(",") for c in a.cores.split(",")]
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.workers) as pool:
            for _ in pool.imap_unordered(run_one, jobs):
                pass
    else:
        for j in jobs:
            run_one(j)
    collect(a)


if __name__ == "__main__":
    main()
