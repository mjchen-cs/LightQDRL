# -*- coding: utf-8 -*-
"""
exp_qrl_adapt.py — adaptation under distribution shift (Fig. 6) and the adaptation factor of the 2^3 study (Fig. 5).

Starting point: the trained agents of exp_qrl_bench.py (results_qrl/<env>/runs/*.pt; variants base, light and, for
the factorial study, dqcp / jcqq).  Each agent is deployed in a shifted MEC system and either
    frozen : no further training (pre-adaptation reference)
    full   : all network parameters fine-tuned (pruning masks fixed)
    peqa   : PEQA only — rank-2 adapters on the angle projection and on the last layer of every head, offsets of the
             retained variational angles, two scales; the backbone is frozen
for --steps environment steps with the algorithm's own update rule (DQN family: eps 0.1 -> 0.02, replay warm-up of
500 steps collected with the deployed policy; policy-gradient: the training critic stored with the checkpoint is
re-used and trained too).

Shifts (arrival rates of the nominal system are kept, so a worse channel raises the effective load)
    traffic : video-traffic surge, every device's arrival rate x1.5            (load_scale = 1.5)
    channel : channel deterioration, mean channel gain of every device x0.5    (channel_scale = 0.5, -3 dB)
Scores are normalized with reference policies evaluated in the shifted system (0 = random, 1 = myopic max-weight).
Energy of adaptation = 5 W x total measured update time (same convention as the benchmark).

Outputs  results_qrl_adapt/<env>/runs/<algo>_<variant>_<shift>_<mode>_seed<k>.json   (+ .npz adaptation curve)
         results_qrl_adapt/<env>/reference_<shift>.json
         results_qrl_adapt/<env>/per_run.csv
Usage
  python exp_qrl_adapt.py --env mec --workers 4                                  # Fig. 6: 7 algos x base/light
  python exp_qrl_adapt.py --env mec --algos Q-DuelingDQN --variants base,dqcp,jcqq,light --modes full,peqa   # Fig. 5
"""
import os
import time
import argparse
import numpy as np
import pandas as pd
import torch

import lqdrl_core as LC
from qrl_core import accounting
from qrl_envs import make_env
from qrl_agents import ALGOS, make_agent
from exp_qrl_bench import evaluate, reference_scores, norm_score, EVAL_EP, EVAL_SEED0

SHIFTS = {"traffic": dict(load_scale=1.5), "channel": dict(channel_scale=0.5)}
ADAPT_HP_DQN = dict(warmup=500, warmup_random=False, eps_start=0.1, eps_end=0.02, eps_frac=0.3)


def run_one(job):
    algo, variant, shift, mode, seed, a = job
    torch.set_num_threads(a.threads)
    tag = f"{algo}_{variant}_{shift}_{mode}_seed{seed}"
    stem = os.path.join(a.out, "runs", tag)
    if os.path.exists(stem + ".json"):
        return stem
    ck_f = os.path.join(a.run_dir, f"{algo}_{variant}_vqc_seed{seed}.pt")
    if not os.path.exists(ck_f):
        print(f"missing checkpoint {ck_f}; skipped", flush=True)
        return None
    ck = torch.load(ck_f, map_location="cpu", weights_only=False)
    kw = SHIFTS[shift]
    env_fn = lambda s: make_env(a.env, seed=s, **kw)
    hp = dict(ck.get("hp") or {})
    if algo in ("Q-DQN", "Q-DDQN", "Q-DuelingDQN"):
        hp.update(ADAPT_HP_DQN)
    agent = make_agent(algo, env_fn, core="vqc", compress=ck["compress"], seed=seed + 500, hp=hp)
    agent.load_net_state(ck["state"])
    if ck.get("critic") is not None and hasattr(agent, "critic"):
        agent.critic.load_state_dict(ck["critic"])
    ref = LC.load_json(os.path.join(a.out, f"reference_{shift}.json"))
    n_ev = EVAL_EP[a.env]
    pre = evaluate(agent, a.env, n_ev, env_kw=kw)
    nominal_pre = evaluate(agent, a.env, n_ev)["eval_return"]
    curve = [(0, pre["eval_return"])]
    n_train, t_wall = 0, 0.0
    if mode != "frozen":
        n_train = agent.setup_adaptation(mode)
        t0 = time.time()

        def cb(t):
            curve.append((t, evaluate(agent, a.env, max(3, n_ev // 2), env_kw=kw)["eval_return"]))
        agent.train(a.steps, eval_cb=cb, eval_every=a.eval_every)
        t_wall = time.time() - t0
    post = evaluate(agent, a.env, n_ev, env_kw=kw)
    nominal_post = evaluate(agent, a.env, n_ev)["eval_return"]
    if mode == "peqa":
        breakdown = agent.net.peqa_breakdown()
        agent.net.merge_peqa()
    else:
        breakdown = {}
    acc = accounting(agent.net)
    ut = np.array(agent.update_times) if (mode != "frozen" and agent.update_times) else np.array([0.0])
    ref_nom = LC.load_json(os.path.join(a.bench_root, a.env, "reference_policies.json"))
    rec = dict(env=a.env, algo=algo, variant=variant, shift=shift, mode=mode, seed=seed, steps=a.steps if mode != "frozen" else 0,
               pre_return=pre["eval_return"], post_return=post["eval_return"],
               pre_score=norm_score(a.env, pre["eval_return"], ref), post_score=norm_score(a.env, post["eval_return"], ref),
               nominal_pre_score=norm_score(a.env, nominal_pre, ref_nom),
               nominal_post_score=norm_score(a.env, nominal_post, ref_nom),
               trainable_params=int(n_train), n_updates=agent.n_updates if mode != "frozen" else 0,
               adapt_update_time_s=float(ut.sum()), adapt_energy_J=float(5.0 * ut.sum()),
               update_ms_mean=float(ut.mean() * 1e3) if mode != "frozen" else 0.0, adapt_wall_s=t_wall,
               served_frac=post.get("served_frac"), mean_backlog_norm=post.get("mean_backlog_norm"),
               drop_rate=post.get("drop_rate"), peqa_breakdown=breakdown, curve=curve,
               static_mem_bytes=acc["static_mem_bytes"], n_2q=acc["n_2q"], hp=agent.hp, env_info=LC.env_info())
    np.savez_compressed(stem + ".npz", curve=np.array(curve), update_times=ut)
    LC.save_json(rec, stem + ".json")
    print(f"{tag}: score {rec['pre_score']:.3f} -> {rec['post_score']:.3f}  trainable={n_train}  "
          f"E={rec['adapt_energy_J']:.2f} J", flush=True)
    return stem


def collect(a):
    rows = []
    for f in sorted(os.listdir(os.path.join(a.out, "runs"))):
        if f.endswith(".json"):
            r = LC.load_json(os.path.join(a.out, "runs", f))
            for k in ("curve", "hp", "env_info", "peqa_breakdown"):
                r.pop(k, None)
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "per_run.csv"), index=False)
    if len(df):
        print(df.groupby(["shift", "algo", "variant", "mode"])[["pre_score", "post_score", "adapt_energy_J",
                                                                 "trainable_params"]].mean().round(3).to_string())
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="mec", choices=["mec"])
    ap.add_argument("--bench_root", default="results_qrl")
    ap.add_argument("--algos", default=",".join(ALGOS))
    ap.add_argument("--variants", default="base,light")
    ap.add_argument("--shifts", default="traffic,channel")
    ap.add_argument("--modes", default="frozen,full,peqa")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--quick", action="store_true", help="1 seed, 2000 steps")
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.steps, a.eval_every = 1, 2000, 1000
    a.run_dir = os.path.join(a.bench_root, a.env, "runs")
    a.out = a.out or os.path.join("results_qrl_adapt", a.env)
    os.makedirs(os.path.join(a.out, "runs"), exist_ok=True)
    LC.save_json(dict(args=vars(a), shifts=SHIFTS, env=LC.env_info(), started=time.strftime("%Y-%m-%d %H:%M:%S")),
                 os.path.join(a.out, "run_config.json"))
    shifts = a.shifts.split(",")
    for s in shifts:
        print(f"reference policies under shift '{s}':")
        reference_scores(a.out, a.env, EVAL_EP[a.env], env_kw=SHIFTS[s], fname=f"reference_{s}.json")
    algos = [x.strip() for x in a.algos.split(",") if x.strip()]
    jobs = [(al, v, sh, m, s, a) for s in range(a.seeds) for al in algos for v in a.variants.split(",")
            for sh in shifts for m in a.modes.split(",")]
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
