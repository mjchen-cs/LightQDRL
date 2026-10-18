# -*- coding: utf-8 -*-
"""
quafu_hw_validation.py — inference-only validation of trained hybrid QRL policies on the Quafu quantum cloud (BAQIS).

What it does
  1. Loads a pair of trained agents produced by exp_qrl_bench.py: the uncompressed ('base') and the LightQDRL
     ('light', pruned + quantized) version of the same algorithm and seed.
  2. Collects n_states evaluation states of the MEC environment (fixed seeds) together with the channel and queue
     weights needed to score a decision.
  3. For every state builds the policy circuit (angle encoding + variational layer with the learned DQCP masks) as a
     Quafu QuantumCircuit, measures all qubits and executes it with `shots` shots.
  4. Converts counts to <Z_q>, runs the classical decoder and compares with the exact-simulation policy:
       - Kendall tau between hardware and exact per-device preference scores
       - per-device decision agreement and exact-match rate of the full offloading vector
       - frame-level weighted computation rate of the executed decision relative to the exact decision
       - mean absolute error of <Z_q>, transpiled two-qubit gate count / depth returned by the Quafu compiler.
  A single-qubit X calibration (one circuit per qubit) fixes the bit order of Quafu's count keys automatically.

Modes
  --mode dryrun   no network access: finite-shot sampling from the exact simulator (+ optional depolarizing noise);
                  use it to check the pipeline and to estimate the shot budget before spending hardware time
  --mode quafu    submit to Quafu.  Requires `pip install pyquafu` and an API token (website -> personal center),
                  passed with --token or the environment variable QUAFU_TOKEN.  Raw counts are written after every
                  circuit, so an interrupted run resumes where it stopped.

Usage
  python quafu_hw_validation.py --run_dir results_qrl/mec/runs --algo Q-PPO --seed 0 --mode dryrun
  python quafu_hw_validation.py --run_dir results_qrl/mec/runs --algo Q-PPO --seed 0 --mode quafu \
         --backend ScQ-P18 --shots 2000 --n_states 100
Outputs (./results_hw/<algo>_seed<k>_<mode>_<backend>/): raw_counts_<variant>.json, per_state.csv, summary.json
Note: backend names and queue policies change; check the Quafu website for the devices currently online.
"""
import os
import re
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau

import lqdrl_core as LC
from qrl_core import HybridNet, SV
from qrl_envs import make_env, per_device_rates
from optimization import bisection


# ----------------------------------------------------------------------------------------- model loading
def load_agent_net(run_dir, algo, variant, seed, core="vqc"):
    ck = torch.load(os.path.join(run_dir, f"{algo}_{variant}_{core}_seed{seed}.pt"), map_location="cpu",
                    weights_only=False)
    net = HybridNet(ck["obs_dim"], ck["heads"], core=core, compress=ck["compress"])
    net.load_state_dict(ck["state"])
    net.refresh_ops()
    net.eval()
    return net, ck


def preference(net, heads_out, nb):
    """Per-branch preference for action 1 (offload): Q(1)-Q(0), advantage difference or logit difference."""
    if "q" in heads_out:
        q = heads_out["q"].view(-1, nb, 2)
    elif "adv" in heads_out:
        q = heads_out["adv"].view(-1, nb, 2)
    else:
        q = heads_out["pi"].view(-1, nb, 2)
    return (q[..., 1] - q[..., 0]).detach().numpy()


# ----------------------------------------------------------------------------------------- states
_ENV_CACHE = {}


def frame_value(env_version, h, w, q, m):
    """Frame-level quality of decision m: v1 = bisection weighted computation rate (earlier environment),
    v2 = queue-weighted service sum_i w_i min(Q_i, r_i) of the current MECQueueEnv."""
    if env_version == "v1":
        return bisection(h, m, list(w))[0]
    if "v2" not in _ENV_CACHE:
        _ENV_CACHE["v2"] = make_env("mec", seed=0)
    e = _ENV_CACHE["v2"]
    e.h = np.asarray(h, dtype=np.float64)
    return float(np.dot(w, np.minimum(q, e.rates(m))))


def collect_states(net, n_states, seed=20_000, env_version="v2"):
    """Fixed evaluation states (S, channel H, weights W, queues Q) visited by the base policy."""
    env = make_env("mec" if env_version == "v2" else "mec_v1", seed=seed)
    o = env.reset(seed=seed)
    S, H, W, Q = [], [], [], []
    while len(S) < n_states:
        S.append(o); H.append(env.h.copy()); W.append(env.weights().copy()); Q.append(env.Q.copy())
        with torch.no_grad():
            pref = preference(net, net(torch.tensor(o).unsqueeze(0)), env.N)[0]
        o, _, done, _ = env.step((pref > 0).astype(int))
        if done:
            o = env.reset()
    return np.array(S, np.float32), np.array(H), np.array(W), np.array(Q)


# ----------------------------------------------------------------------------------------- circuits
def quafu_circuit(ops, ang, th, n=6):
    from quafu import QuantumCircuit
    qc = QuantumCircuit(n)
    for g, wires, p in ops:
        if g == "CNOT":
            qc.cnot(wires[0], wires[1])
            continue
        val = float(ang[p[1]]) if p[0] == "in" else float(th[p[1], p[2], p[3]])
        getattr(qc, g.lower())(wires[0], val)
    qc.measure(list(range(n)), list(range(n)))
    return qc


def counts_to_expz(counts, order, n=6):
    """counts: {bitstring: count}; order[q] = character index of qubit q in the bitstring."""
    tot = sum(counts.values())
    z = np.zeros(n)
    for b, c in counts.items():
        for q in range(n):
            z[q] += c * (1.0 if b[order[q]] == "0" else -1.0)
    return z / tot


class QuafuRunner:
    def __init__(self, token, backend, shots):
        from quafu import Task, User
        token = token or os.environ.get("QUAFU_TOKEN")
        if not token:
            raise SystemExit("Quafu API token missing: use --token or set QUAFU_TOKEN")
        self.user = User(api_token=token)
        try:
            self.user.save_apitoken()
        except Exception:
            pass
        try:
            self.task = Task(user=self.user)
        except TypeError:
            self.task = Task()
        self.task.config(backend=backend, shots=shots, compile=True)
        self.backend = backend

    def run(self, qc, name):
        for attempt in range(5):
            try:
                res = self.task.send(qc, name=name, wait=True)
                counts = getattr(res, "logicalq_res", None) or res.counts
                tq = getattr(res, "transpiled_openqasm", "") or ""
                return dict(counts={str(k): int(v) for k, v in counts.items()},
                            n_2q_transpiled=len(re.findall(r"^\s*(cx|cz|cnot|iswap)\b", tq, re.M)),
                            transpiled_qasm=tq)
            except Exception as e:                                   # network / queue hiccups
                print(f"    retry {attempt + 1}: {e}")
                time.sleep(10 * (attempt + 1))
        raise RuntimeError("Quafu submission failed repeatedly")


class DryRunner:
    """Finite-shot sampling from the exact state (optionally with depolarizing noise via the density-matrix simulator)."""

    def __init__(self, shots, p_noise=0.0, seed=0):
        self.shots, self.p = shots, p_noise
        self.gen = torch.Generator().manual_seed(seed)
        self.dm = LC.DensityMatrixSim()
        self.backend = "dryrun"

    def run_ops(self, ops, ang, th):
        noise = dict(p2=self.p, p1=self.p / 10) if self.p > 0 else None
        pr = self.dm.probs(ops, torch.tensor(ang).unsqueeze(0), torch.tensor(th), noise=noise)[0]
        cnt = torch.distributions.Multinomial(total_count=self.shots, probs=pr).sample().numpy()
        n = 6
        return dict(counts={format(i, f"0{n}b"): int(c) for i, c in enumerate(cnt) if c > 0},
                    n_2q_transpiled=np.nan, transpiled_qasm="")


# ----------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="results_qrl/mec/runs")
    ap.add_argument("--algo", default="Q-PPO")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", default="base,light")
    ap.add_argument("--mode", default="dryrun", choices=["dryrun", "quafu"])
    ap.add_argument("--backend", default="ScQ-P18")
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--n_states", type=int, default=100)
    ap.add_argument("--p_noise", type=float, default=0.0, help="dryrun only: 2Q depolarizing probability")
    ap.add_argument("--token", default=None)
    ap.add_argument("--env_version", default="v2", choices=["v2", "v1"],
                    help="v1 only to reproduce runs made with agents trained in the earlier MEC environment")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join("results_hw", f"{a.algo}_seed{a.seed}_{a.mode}_{a.backend if a.mode == 'quafu' else 'sim'}")
    os.makedirs(out, exist_ok=True)

    base_net, _ = load_agent_net(a.run_dir, a.algo, "base", a.seed)
    S, H, W, Qs = collect_states(base_net, a.n_states, env_version=a.env_version)   # same states for both variants
    np.savez_compressed(os.path.join(out, "states.npz"), S=S, H=H, W=W, Q=Qs)

    runner = QuafuRunner(a.token, a.backend, a.shots) if a.mode == "quafu" else DryRunner(a.shots, a.p_noise, a.seed)

    # ---- bit-order calibration: X on qubit q -> which character becomes '1'
    order = list(range(6))
    if a.mode == "quafu":
        cal_f = os.path.join(out, "bit_order_calibration.json")
        if os.path.exists(cal_f):
            order = json.load(open(cal_f))["order"]
        else:
            from quafu import QuantumCircuit
            order = []
            for q in range(6):
                qc = QuantumCircuit(6); qc.x(q); qc.measure(list(range(6)), list(range(6)))
                c = runner.run(qc, f"cal_q{q}")["counts"]
                top = max(c, key=c.get)
                order.append(int(np.argmax([ch == "1" for ch in top])))
            json.dump(dict(order=order, note="order[q] = character index of qubit q"), open(cal_f, "w"))
        print("bit order:", order)

    rows, summary = [], dict(algo=a.algo, seed=a.seed, mode=a.mode, backend=runner.backend, shots=a.shots,
                             n_states=a.n_states, bit_order=order, started=time.strftime("%Y-%m-%d %H:%M:%S"))
    x = torch.tensor(S)
    for variant in a.variants.split(","):
        net, ck = load_agent_net(a.run_dir, a.algo, variant, a.seed)
        ops = net.ops()
        with torch.no_grad():
            ang, th = net.pre_circuit(x)
            ang, th = ang.detach(), th.detach()
            e_exact = SV.run(ops, ang, th).numpy()
            pref_exact = preference(net, net.heads_from_features(net.post_circuit(torch.tensor(e_exact))), 10)
        raw_f = os.path.join(out, f"raw_counts_{variant}.json")
        raw = json.load(open(raw_f)) if os.path.exists(raw_f) else {}
        for i in range(a.n_states):
            key = str(i)
            if key not in raw:
                if a.mode == "quafu":
                    raw[key] = runner.run(quafu_circuit(ops, ang[i].numpy(), th.numpy()), f"{a.algo}_{variant}_{i}")
                else:
                    raw[key] = runner.run_ops(ops, ang[i].numpy(), th.numpy())
                json.dump(raw, open(raw_f, "w"))
                if a.mode == "quafu":
                    print(f"  {variant} state {i + 1}/{a.n_states} done", flush=True)
        e_hw = np.array([counts_to_expz(raw[str(i)]["counts"], order) for i in range(a.n_states)], np.float32)
        with torch.no_grad():
            pref_hw = preference(net, net.heads_from_features(net.post_circuit(torch.tensor(e_hw))), 10)
        for i in range(a.n_states):
            m_ex, m_hw = (pref_exact[i] > 0).astype(int), (pref_hw[i] > 0).astype(int)
            r_ex = frame_value(a.env_version, H[i], W[i], Qs[i], m_ex)
            r_hw = frame_value(a.env_version, H[i], W[i], Qs[i], m_hw)
            rows.append(dict(variant=variant, state=i, kendall_tau=kendalltau(pref_exact[i], pref_hw[i])[0],
                             device_agreement=float(np.mean(m_ex == m_hw)), exact_match=float(np.all(m_ex == m_hw)),
                             rate_ratio=r_hw / max(r_ex, 1e-12), expz_mae=float(np.abs(e_hw[i] - e_exact[i]).mean()),
                             n_2q_logical=LC.gate_counts(ops)[1], n_2q_transpiled=raw[str(i)].get("n_2q_transpiled", np.nan)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "per_state.csv"), index=False)
    agg = df.groupby("variant")[["kendall_tau", "device_agreement", "exact_match", "rate_ratio", "expz_mae",
                                 "n_2q_logical", "n_2q_transpiled"]].mean()
    summary["results"] = agg.reset_index().to_dict(orient="records")
    summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    LC.save_json(summary, os.path.join(out, "summary.json"))
    print(agg.round(4).to_string())
    print("results written to", os.path.abspath(out))


if __name__ == "__main__":
    main()
