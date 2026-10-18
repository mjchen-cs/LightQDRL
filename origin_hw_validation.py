# -*- coding: utf-8 -*-
"""
origin_hw_validation.py — inference-only validation of trained hybrid QRL policies on the Origin Quantum cloud
(本源量子云, e.g. the 72-qubit Wukong / 悟空 superconducting chip), using pyqpanda3.

Same protocol and metrics as quafu_hw_validation.py (whose helper functions are reused):
  1. load the 'base' (uncompressed) and 'light' (LightQDRL) agents of one algorithm/seed from exp_qrl_bench.py;
  2. collect n_states fixed MEC evaluation states (identical for both variants);
  3. build every policy circuit as a pyqpanda3 QProg (angle encoding + variational layer with the learned DQCP masks),
     submit them in batches, convert counts to <Z_q>, run the classical decoder;
  4. compare with the exact policy: Kendall tau, per-device agreement, exact-match rate, frame-level weighted
     computation-rate ratio, <Z> MAE, logical and compiled two-qubit gate counts.

Robust submission
  * Jobs are submitted asynchronously; every job id is written to jobs_<variant>.json immediately, so an interrupted
    or timed-out run never re-submits work: re-running the same command only polls the stored job ids.
  * Raw counts are written to raw_counts_<variant>.json as soon as a batch finishes.
  * A one-off X-calibration job (one circuit per qubit) determines the bit order of the returned count keys.

Modes
  --mode dryrun   no network: finite-shot sampling from the exact simulator (+ optional depolarizing noise)
  --mode origin   Origin Quantum cloud.  pip install pyqpanda3 ;  API key from the Origin Quantum cloud website
                  (个人中心 / API Key), given with --api_key or the environment variable ORIGIN_API_KEY.
  --list_backends print the backends visible to your account and exit.

Usage (PowerShell)
  $env:ORIGIN_API_KEY = "your-key"
  python origin_hw_validation.py --list_backends
  python origin_hw_validation.py --algo Q-PPO --seed 0 --mode dryrun
  python origin_hw_validation.py --algo Q-PPO --seed 0 --mode origin --backend origin_wukong --shots 2000 --n_states 100

Outputs (./results_hw/<algo>_seed<k>_<mode>_<backend>/):
  states.npz, bit_order_calibration.json, jobs_<variant>.json, raw_counts_<variant>.json, per_state.csv, summary.json
Note: backend names, qubit availability and queue policies change; use --list_backends to see what is online.
"""
import os
import re
import json
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau

import lqdrl_core as LC
from qrl_core import SV
from optimization import bisection
from quafu_hw_validation import load_agent_net, preference, collect_states, counts_to_expz, DryRunner, frame_value

NQ = 6


# ----------------------------------------------------------------------------------------- circuits
def origin_prog(ops, ang, th, n=NQ):
    from pyqpanda3.core import QProg, RX, RY, RZ, CNOT, measure
    gates = dict(RX=RX, RY=RY, RZ=RZ)
    prog = QProg()
    for g, wires, p in ops:
        if g == "CNOT":
            prog << CNOT(int(wires[0]), int(wires[1]))
            continue
        val = float(ang[p[1]]) if p[0] == "in" else float(th[p[1], p[2], p[3]])
        prog << gates[g](int(wires[0]), val)
    prog << measure(list(range(n)), list(range(n)))
    return prog


def calibration_prog(q, n=NQ):
    from pyqpanda3.core import QProg, X, measure
    prog = QProg()
    prog << X(q)
    prog << measure(list(range(n)), list(range(n)))
    return prog


def count_2q(text):
    if not isinstance(text, str) or not text:
        return np.nan
    return len(re.findall(r"^\s*(CNOT|CZ|ISWAP|CX)\b", text, re.M | re.I))


# ----------------------------------------------------------------------------------------- runner
class JobFailed(RuntimeError):
    """The cloud reported status FAILED for a job (server-side error; its circuits can be re-submitted)."""


class OriginRunner:
    def __init__(self, api_key, backend, shots, amend=True, mapping=True, optimization=True, poll=20):
        from pyqpanda3 import qcloud
        self.qc = qcloud
        api_key = api_key or os.environ.get("ORIGIN_API_KEY")
        if not api_key:
            raise SystemExit("Origin Quantum API key missing: use --api_key or set ORIGIN_API_KEY")
        self.service = qcloud.QCloudService(api_key=api_key)
        self.backend_name, self.shots, self.poll = backend, shots, poll
        self.backend = self.service.backend(backend)
        self.is_qpu = "amplitude" not in backend.lower() and "simulator" not in backend.lower()
        self.options = None
        if self.is_qpu:
            o = qcloud.QCloudOptions()
            o.set_amend(amend)               # readout-error mitigation on the server side
            o.set_mapping(mapping)           # qubit mapping / routing to the chip topology
            o.set_optimization(optimization)
            self.options = o
        self.settings = dict(amend=amend, mapping=mapping, optimization=optimization, is_qpu=self.is_qpu)
        try:
            ci = self.backend.chip_info()
            self.settings.update(chip_id=str(ci.chip_id()), qubits_num=int(ci.qubits_num()),
                                 available_qubits=[int(x) for x in ci.available_qubits()])
        except Exception:
            pass

    def submit(self, progs):
        for attempt in range(6):
            try:
                if self.options is not None:
                    job = self.backend.run(progs, self.shots, self.options)
                else:
                    job = self.backend.run(progs, self.shots)
                return job.job_id()
            except Exception as e:
                print(f"    submit retry {attempt + 1}: {e}", flush=True)
                time.sleep(15 * (attempt + 1))
        raise RuntimeError("Origin Quantum submission failed repeatedly")

    def fetch(self, job_id, n_expected):
        """Poll until the job finishes; return a list of n_expected dicts {counts, n_2q_compiled}."""
        job = self.qc.QCloudJob(job_id)
        t0 = time.time()
        while True:
            try:
                st = job.status()
            except Exception as e:
                print(f"    status query failed ({e}); retrying", flush=True)
                time.sleep(self.poll)
                continue
            name = getattr(st, "name", str(st))
            if name == "FINISHED":
                break
            if name == "FAILED":
                try:
                    msg = job.query().error_message()
                except Exception:
                    msg = "?"
                raise JobFailed(f"job {job_id} failed: {msg}")
            print(f"    job {job_id}: {name} ({time.time() - t0:.0f}s)", flush=True)
            time.sleep(self.poll)
        res = job.result()
        try:
            counts = res.get_counts_list()
        except Exception:
            counts = []
        if len(counts) != n_expected:                         # some backends return probabilities only
            try:
                probs = res.get_probs_list()
                counts = [{k: int(round(v * self.shots)) for k, v in p.items()} for p in probs]
            except Exception:
                pass
        if len(counts) != n_expected and n_expected == 1:
            counts = [res.get_counts()]
        if len(counts) != n_expected:
            raise RuntimeError(f"job {job_id}: expected {n_expected} results, got {len(counts)}")
        try:
            mq = res.mapping_qprog()
            mq = json.loads(mq) if isinstance(mq, str) and mq.strip().startswith("[") else mq
        except Exception:
            mq = None
        n2 = [count_2q(mq[i]) if isinstance(mq, list) and i < len(mq) else (count_2q(mq) if n_expected == 1 else np.nan)
              for i in range(n_expected)]
        return [dict(counts={str(k): int(v) for k, v in c.items()}, n_2q_transpiled=n2[i]) for i, c in enumerate(counts)]


def run_batched(runner, progs, keys, raw, raw_f, jobs_f, batch, max_resubmit=2):
    """Submit missing keys in batches (job ids persisted first), then collect all pending jobs.
    Jobs reported FAILED by the cloud are moved to the 'failed' record of jobs_f and their circuits are re-submitted
    (at most max_resubmit rounds); successful jobs are never submitted twice."""
    book = json.load(open(jobs_f)) if os.path.exists(jobs_f) else {}
    if "active" not in book:                                   # old format: {job_id: {...}}
        book = dict(active=book, failed={})
    for rnd in range(max_resubmit + 1):
        jobs = book["active"]
        pending_keys = {k for j in jobs.values() for k in j["keys"]}
        todo = [i for i, k in enumerate(keys) if k not in raw and k not in pending_keys]
        for s in range(0, len(todo), batch):
            idx = todo[s:s + batch]
            jid = runner.submit([progs[i] for i in idx])
            jobs[jid] = dict(keys=[keys[i] for i in idx], submitted=time.strftime("%Y-%m-%d %H:%M:%S"))
            json.dump(book, open(jobs_f, "w"), indent=1)
            print(f"  submitted job {jid} ({len(idx)} circuits)", flush=True)
        n_failed = 0
        for jid in list(jobs):
            j = jobs[jid]
            if all(k in raw for k in j["keys"]):
                continue
            try:
                out = runner.fetch(jid, len(j["keys"]))
            except JobFailed as e:
                print(f"  {e}  -> its {len(j['keys'])} circuits will be re-submitted", flush=True)
                book["failed"][jid] = dict(j, error=str(e), failed_at=time.strftime("%Y-%m-%d %H:%M:%S"))
                del jobs[jid]
                json.dump(book, open(jobs_f, "w"), indent=1)
                n_failed += 1
                continue
            for k, r in zip(j["keys"], out):
                r["job_id"] = jid
                raw[k] = r
            json.dump(raw, open(raw_f, "w"))
            print(f"  job {jid} collected", flush=True)
        if all(k in raw for k in keys):
            return raw
        if n_failed and rnd < max_resubmit:
            print(f"  re-submitting failed circuits (round {rnd + 1}/{max_resubmit}) after 30 s", flush=True)
            time.sleep(30)
    missing = [k for k in keys if k not in raw]
    raise RuntimeError(f"{len(missing)} circuits still missing after {max_resubmit} re-submissions; "
                       f"re-run the same command later (failed jobs are listed in {jobs_f})")


# ----------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="results_qrl/mec/runs")
    ap.add_argument("--algo", default="Q-PPO")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", default="base,light")
    ap.add_argument("--mode", default="dryrun", choices=["dryrun", "origin"])
    ap.add_argument("--backend", default="origin_wukong")
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--n_states", type=int, default=100)
    ap.add_argument("--batch", type=int, default=25, help="circuits per cloud job")
    ap.add_argument("--no_amend", action="store_true", help="disable server-side readout-error mitigation")
    ap.add_argument("--poll", type=int, default=20, help="seconds between status queries")
    ap.add_argument("--max_resubmit", type=int, default=2, help="re-submission rounds for jobs the cloud reports FAILED")
    ap.add_argument("--p_noise", type=float, default=0.0, help="dryrun only: 2Q depolarizing probability")
    ap.add_argument("--api_key", default=None)
    ap.add_argument("--env_version", default="v2", choices=["v2", "v1"],
                    help="v1 only to reproduce the first WK_C180 run (agents trained in the earlier MEC environment)")
    ap.add_argument("--list_backends", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)       # benign overflow inside the bisection solver
    np.seterr(all="ignore")

    if a.list_backends:
        from pyqpanda3 import qcloud
        key = a.api_key or os.environ.get("ORIGIN_API_KEY")
        if not key:
            raise SystemExit("Origin Quantum API key missing: use --api_key or set ORIGIN_API_KEY")
        for name, ok in qcloud.QCloudService(api_key=key).backends().items():
            print(f"  {name:30s} {'available' if ok else 'unavailable'}")
        return

    tag = a.backend if a.mode == "origin" else "sim"
    out = a.out or os.path.join("results_hw", f"{a.algo}_seed{a.seed}_{a.mode}_{tag}")
    os.makedirs(out, exist_ok=True)

    base_net, _ = load_agent_net(a.run_dir, a.algo, "base", a.seed)
    S, H, W, Qs = collect_states(base_net, a.n_states, env_version=a.env_version)   # same states for both variants
    np.savez_compressed(os.path.join(out, "states.npz"), S=S, H=H, W=W, Q=Qs)

    if a.mode == "origin":
        runner = OriginRunner(a.api_key, a.backend, a.shots, amend=not a.no_amend, poll=a.poll)
    else:
        runner = DryRunner(a.shots, a.p_noise, a.seed)

    # ---- bit-order calibration: X on qubit q -> which character of the count key becomes '1'
    order = list(range(NQ))
    if a.mode == "origin":
        cal_f = os.path.join(out, "bit_order_calibration.json")
        if os.path.exists(cal_f) and "order" in json.load(open(cal_f)):
            order = json.load(open(cal_f))["order"]
        else:
            keys = [f"cal_q{q}" for q in range(NQ)]
            cal_raw_f = os.path.join(out, "cal_raw.json")
            cal = run_batched(runner, [calibration_prog(q) for q in range(NQ)], keys,
                              json.load(open(cal_raw_f)) if os.path.exists(cal_raw_f) else {},
                              cal_raw_f, os.path.join(out, "jobs_cal.json"), NQ, a.max_resubmit)
            order = []
            for k in keys:
                top = max(cal[k]["counts"], key=cal[k]["counts"].get)
                order.append(int(np.argmax([ch == "1" for ch in top])))
            if len(set(order)) != NQ:
                print("WARNING: calibration ambiguous (readout errors?); falling back to little-endian order")
                order = [NQ - 1 - q for q in range(NQ)]
            json.dump(dict(order=order, note="order[q] = character index of qubit q", raw=cal), open(cal_f, "w"))
        print("bit order:", order)

    rows = []
    summary = dict(algo=a.algo, seed=a.seed, mode=a.mode, platform="Origin Quantum" if a.mode == "origin" else "dryrun",
                   backend=a.backend if a.mode == "origin" else "dryrun", shots=a.shots, n_states=a.n_states,
                   bit_order=order, settings=getattr(runner, "settings", dict(p_noise=a.p_noise)),
                   started=time.strftime("%Y-%m-%d %H:%M:%S"))
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
        keys = [str(i) for i in range(a.n_states)]
        if a.mode == "origin":
            progs = [origin_prog(ops, ang[i].numpy(), th.numpy()) for i in range(a.n_states)]
            raw = run_batched(runner, progs, keys, raw, raw_f, os.path.join(out, f"jobs_{variant}.json"), a.batch,
                              a.max_resubmit)
        else:
            for k in keys:
                if k not in raw:
                    raw[k] = runner.run_ops(ops, ang[int(k)].numpy(), th.numpy())
            json.dump(raw, open(raw_f, "w"))
        e_hw = np.array([counts_to_expz(raw[k]["counts"], order) for k in keys], np.float32)
        with torch.no_grad():
            pref_hw = preference(net, net.heads_from_features(net.post_circuit(torch.tensor(e_hw))), 10)
        n2_log = LC.gate_counts(ops)[1]
        for i in range(a.n_states):
            m_ex, m_hw = (pref_exact[i] > 0).astype(int), (pref_hw[i] > 0).astype(int)
            r_ex = frame_value(a.env_version, H[i], W[i], Qs[i], m_ex)
            r_hw = frame_value(a.env_version, H[i], W[i], Qs[i], m_hw)
            rows.append(dict(variant=variant, state=i, kendall_tau=kendalltau(pref_exact[i], pref_hw[i])[0],
                             device_agreement=float(np.mean(m_ex == m_hw)), exact_match=float(np.all(m_ex == m_hw)),
                             rate_ratio=r_hw / max(r_ex, 1e-12), expz_mae=float(np.abs(e_hw[i] - e_exact[i]).mean()),
                             n_2q_logical=n2_log, n_2q_transpiled=raw[str(i)].get("n_2q_transpiled", np.nan)))
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
