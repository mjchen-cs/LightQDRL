# -*- coding: utf-8 -*-
"""
measure_edge_device.py — on-device latency, memory and energy of the trained policies on an edge board
(Raspberry Pi 5, NVIDIA Jetson Orin Nano, any Linux/ARM or x86 CPU).

For every selected checkpoint (results_qrl/<env>/runs/<algo>_<variant>_<core>_seed<k>.pt) the script builds a
*deployment* version of the policy that uses ordinary torch.nn modules, so that real integer kernels can be used:
  fp32  : the trained network executed in float32 (fake-quantization switched off)
  int8  : the same network with every Linear layer converted to dynamic INT8 (torch.ao.quantization.quantize_dynamic;
          qnnpack on ARM, fbgemm/x86 on Intel/AMD) and the variational angles stored on the 8-bit phase grid.
          This is how JCQQ is meant to be deployed; it is run for 'light' (and 'jcqq') checkpoints, and also for 'base'
          checkpoints as a post-training-quantization reference.
The quantum circuit (6 qubits) is executed with the exact state-vector simulator on the device CPU; this is the
'emulated co-processor' setting of the paper.  For classical cores (mlp / tanh) no circuit is executed.

Measured per model and mode
  latency_ms_median / p90       single-sample decision (batch 1), --reps repetitions after --warmup calls
  throughput_hz                  decisions per second
  payload_bytes                  arrays a deployment actually stores: Linear weights (fp32, or int8 codes + scale),
                                 biases and LayerNorm in fp32, only the RETAINED variational angles (fp32, or int8
                                 codes on the 8-bit phase grid) and, for pruned circuits, the gate masks as bit fields
  file_bytes                     the same arrays written as an uncompressed .npz file (includes container overhead)
  torch_bytes                    torch.save(state_dict) of the deployed module (dense, pruned angles stored as 0)
  analytic_bytes                 static memory as computed in the paper (qrl_core.static_memory_bytes)
  agreement_vs_trained           fraction of identical per-device decisions vs the trained (fake-quant) network on
                                 the states visited by the trained policy in the --eval_ep evaluation episodes
  return_deployed / norm_deployed  closed-loop return (and normalized score) of the deployed model on the same
                                 evaluation episodes; return_trained / norm_trained for the trained network
  native                         1 for the deployment format a variant is meant for (base: fp32, light/jcqq: int8)
  power_w, energy_mJ_per_decision  if a power reading is available:  energy = (P_active - P_idle) x latency
Power readings (--power auto|rapl|jetson|pi5|none|manual):
  rapl    Intel/AMD Linux:  /sys/class/powercap/intel-rapl:0/energy_uj
  jetson  INA3221 hwmon rails (/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*); the VDD_IN (module input) rail is used
  pi5     `vcgencmd pmic_read_adc` (sum of V x A over the PMIC rails)
  manual  the script runs every model for --burn_s seconds and logs start/end times to burn_log.csv, so that the
          average power shown by a USB power meter can be entered afterwards (column power_w) and energy recomputed
          with --recompute.
Usage (on the board, in a folder that contains qrl_core.py, lqdrl_core.py, optimization.py, qrl_envs.py, data/
and results_qrl/mec/runs + results_qrl/mec/reference_policies.json)
  python3 measure_edge_device.py --device "Jetson Orin Nano Super" --cores vqc,mlp --variants base,light --seeds 0,1,2
  python3 measure_edge_device.py --recompute edge_results/edge_latency.csv      (after filling power_w manually)
Output: edge_results/edge_latency.csv, edge_results/device_info.json, edge_results/burn_log.csv
"""
import os
import io
import re
import glob
import json
import time
import math
import argparse
import platform
import subprocess
import numpy as np
import torch
import torch.nn as nn

import lqdrl_core as LC
from qrl_core import HybridNet, SV, static_memory_bytes


# ----------------------------------------------------------------------------------------- deployment network
class DeployNet(nn.Module):
    """Plain-module copy of a trained HybridNet (no fake quantization), suitable for quantize_dynamic."""

    def __init__(self, net, phase_bits=None):
        super().__init__()
        self.core, self.C = net.core, net.cfg["clip_C"]
        self.enc_fc = nn.Linear(net.enc_fc.in_features, net.enc_fc.out_features)
        self.enc_ln = nn.LayerNorm(net.enc_ln.normalized_shape)
        self.enc_proj = nn.Linear(net.enc_proj.in_features, net.enc_proj.out_features)
        self.enc_fc.load_state_dict(net.enc_fc.state_dict())
        self.enc_ln.load_state_dict(net.enc_ln.state_dict())
        self.enc_proj.load_state_dict(net.enc_proj.state_dict())
        self.phase_bits = phase_bits
        if self.core == "vqc":
            th = net.theta.detach().clone()
            if phase_bits:
                th = LC.fq_phase(th, phase_bits).detach()
            self.register_buffer("theta", th)
            self.ops = net.ops()
        elif self.core == "tanh":
            self.Wq = nn.Linear(net.Wq.in_features, net.Wq.out_features, bias=False)
            self.Wq.load_state_dict(net.Wq.state_dict())
        self.heads = nn.ModuleDict()
        for k, h in net.heads.items():
            m = nn.Sequential(nn.Linear(h.fc1.in_features, h.fc1.out_features), nn.ReLU(),
                              nn.Linear(h.fc2.in_features, h.fc2.out_features))
            m[0].load_state_dict(h.fc1.state_dict())
            m[2].load_state_dict(h.fc2.state_dict())
            self.heads[k] = m

    def forward(self, x):
        z = torch.clamp(x, -self.C, self.C)
        a = torch.relu(self.enc_ln(self.enc_fc(z)))
        ang = torch.tanh(self.enc_proj(a)) * math.pi
        if self.core == "vqc":
            if self.phase_bits:
                ang = LC.fq_phase(ang, self.phase_bits)
            e = SV.run(self.ops, ang, self.theta)
            if self.phase_bits:
                e = torch.round(e * 127.5) / 127.5
        elif self.core == "tanh":
            e = torch.tanh(self.Wq(ang))
        else:
            e = ang / math.pi
        return {k: h(e) for k, h in self.heads.items()}


def actions(out, branches):
    """Greedy per-branch actions exactly as the agents act (Q / dueling advantage / policy logits; ragged branches masked)."""
    key = "q" if "q" in out else ("adv" if "adv" in out else "pi")
    nb, na = len(branches), max(branches)
    v = out[key].reshape(-1, nb, na).clone()
    for b, k in enumerate(branches):
        v[:, b, k:] = -1e9
    return v.argmax(-1)


def torch_bytes(m):
    buf = io.BytesIO()
    torch.save(m.state_dict(), buf)
    return buf.tell()


def compact_payload(m, net, mode):
    """Arrays a deployment of model m has to store (see module docstring)."""
    arrs = {}
    for name, mod in m.named_modules():
        w = getattr(mod, "weight", None)
        if callable(w):                                       # dynamic-quantized Linear (int8 weights)
            qw = mod.weight()
            arrs[name + ".w"] = qw.int_repr().numpy().astype(np.int8)
            if qw.qscheme() in (torch.per_channel_affine, torch.per_channel_symmetric):
                arrs[name + ".s"] = qw.q_per_channel_scales().numpy().astype(np.float32)
            else:
                arrs[name + ".s"] = np.array([qw.q_scale()], np.float32)
            b = mod.bias()
            if b is not None:
                arrs[name + ".b"] = b.detach().numpy().astype(np.float32)
        elif isinstance(mod, (nn.Linear, nn.LayerNorm)):
            arrs[name + ".w"] = mod.weight.detach().numpy().astype(np.float32)
            if mod.bias is not None:
                arrs[name + ".b"] = mod.bias.detach().numpy().astype(np.float32)
    if net.core == "vqc":
        rm = net.rot_mask.bool()
        th = m.theta[rm].detach()
        if mode == "int8":                                    # 256-level phase grid, step pi/128, wrapped to [-pi, pi)
            codes = torch.round(th * 128.0 / math.pi).long()
            arrs["theta"] = (((codes + 128) % 256) - 128).numpy().astype(np.int8)
        else:
            arrs["theta"] = th.numpy().astype(np.float32)
        if bool((net.rot_mask < 0.5).any()) or bool((net.ent_mask < 0.5).any()):
            bits = np.concatenate([net.rot_mask.flatten().numpy(), net.ent_mask.flatten().numpy()]) > 0.5
            arrs["masks"] = np.packbits(bits.astype(np.uint8))
    payload = int(sum(a.nbytes for a in arrs.values()))
    buf = io.BytesIO()
    np.savez(buf, **arrs)
    return payload, buf.tell()


def rollout(policy, env_name, n_ep, seed0=10_000):
    """Deterministic evaluation episodes (same seeds as exp_qrl_bench.evaluate); returns mean return and visited states."""
    from qrl_envs import make_env
    rets, states = [], []
    for k in range(n_ep):
        env = make_env(env_name, seed=seed0 + k)
        o = env.reset(seed=seed0 + k)
        done, R = False, 0.0
        while not done:
            states.append(np.asarray(o, np.float32))
            a = policy(torch.as_tensor(states[-1]).unsqueeze(0))
            o, r, done, _ = env.step(a)
            R += r
        rets.append(R)
    return float(np.mean(rets)), torch.as_tensor(np.stack(states))


# ----------------------------------------------------------------------------------------- power readers
class Power:
    def __init__(self, mode):
        self.mode = mode
        if mode == "auto":
            for cand in ("rapl", "jetson", "pi5"):
                if self._probe(cand):
                    self.mode = cand
                    break
            else:
                self.mode = "none"
        self.rapl_prev = None

    def _probe(self, m):
        try:
            self.mode = m
            return self.read() is not None
        except Exception:
            return False

    def read(self):
        """Instantaneous power in W (rapl: average since previous call)."""
        if self.mode == "rapl":
            f = "/sys/class/powercap/intel-rapl:0/energy_uj"
            e, t = int(open(f).read()), time.perf_counter()
            if self.rapl_prev is None:
                self.rapl_prev = (e, t)
                time.sleep(0.2)
                return self.read()
            (e0, t0), self.rapl_prev = self.rapl_prev, (e, t)
            return max(0.0, (e - e0) * 1e-6 / max(1e-6, t - t0))
        if self.mode == "jetson":
            rails = self.jetson_rails()
            if not rails:
                return None
            tot = [r for r in rails if "VDD_IN" in r[0].upper()]       # whole-module input rail
            use = tot if tot else rails
            return sum(int(open(v).read()) * int(open(c).read()) for _, v, c in use) / 1e6   # mV*mA -> W
        return None if self.mode != "pi5" else self._read_pi5()

    def jetson_rails(self):
        """INA3221 rails as (label, voltage file, current file); paths are resolved once.
        (A recursive glob under /sys follows sysfs symlink cycles and never returns.)"""
        if getattr(self, "_rails", None) is None:
            rails = []
            for d in sorted(glob.glob("/sys/bus/i2c/drivers/ina3221*/*/hwmon/hwmon*")):
                for lab in sorted(glob.glob(os.path.join(d, "in*_label"))):
                    ch = re.search(r"in(\d+)_label$", lab).group(1)
                    v, c = os.path.join(d, f"in{ch}_input"), os.path.join(d, f"curr{ch}_input")
                    if os.path.exists(v) and os.path.exists(c):
                        rails.append((open(lab).read().strip(), v, c))
            self._rails = rails
        return self._rails

    def _read_pi5(self):
        out = subprocess.run(["vcgencmd", "pmic_read_adc"], capture_output=True, text=True, timeout=5).stdout
        cur = {m.group(1): float(m.group(2)) for m in re.finditer(r"(\w+)_A current\(\d+\)=([\d.]+)A", out)}
        vol = {m.group(1): float(m.group(2)) for m in re.finditer(r"(\w+)_V volt\(\d+\)=([\d.]+)V", out)}
        p = sum(cur[k] * vol[k] for k in cur if k in vol)
        return p if p > 0 else None

    def average(self, seconds, busy_fn=None):
        vals, t_end = [], time.perf_counter() + seconds
        while time.perf_counter() < t_end:
            if busy_fn is not None:
                for _ in range(20):
                    busy_fn()
            w = self.read()
            if w is not None:
                vals.append(w)
            if busy_fn is None:
                time.sleep(0.1)
        return float(np.mean(vals)) if vals else float("nan")


# ----------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results_qrl")
    ap.add_argument("--env", default="mec")
    ap.add_argument("--algos", default="Q-DQN,Q-DDQN,Q-DuelingDQN,Q-A2C,Q-A3C,Q-PPO,Q-TRPO")
    ap.add_argument("--cores", default="vqc,mlp")
    ap.add_argument("--variants", default="base,light")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--power", default="auto", choices=["auto", "rapl", "jetson", "pi5", "none", "manual"])
    ap.add_argument("--idle_s", type=float, default=10.0)
    ap.add_argument("--burn_s", type=float, default=10.0)
    ap.add_argument("--device", default=platform.node())
    ap.add_argument("--out", default="edge_results")
    ap.add_argument("--recompute", default=None, help="CSV with a filled power_w column (manual mode)")
    ap.add_argument("--eval_ep", type=int, default=10, help="closed-loop evaluation episodes (0: random probe states only)")
    a = ap.parse_args()
    import pandas as pd
    if a.recompute:
        df = pd.read_csv(a.recompute)
        idle = json.load(open(os.path.join(os.path.dirname(a.recompute), "device_info.json"))).get("idle_w", float("nan"))
        df["energy_mJ_per_decision"] = (df["power_w"] - df.get("idle_w", idle)) * df["latency_ms_median"]
        df.to_csv(a.recompute, index=False)
        print(df.groupby(["core", "variant", "mode"])[["latency_ms_median", "energy_mJ_per_decision"]].mean().round(4))
        return
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(a.threads)
    eng = "qnnpack" if platform.machine().lower() in ("aarch64", "arm64", "armv7l") else "fbgemm"
    try:
        torch.backends.quantized.engine = eng
    except Exception:
        pass
    print("probing power sensors ...", flush=True)
    pw = Power(a.power)
    info = dict(device=a.device, machine=platform.machine(), platform=platform.platform(), torch=torch.__version__,
                quant_engine=torch.backends.quantized.engine, threads=a.threads, power_mode=pw.mode,
                started=time.strftime("%Y-%m-%d %H:%M:%S"))
    if pw.mode == "jetson":
        rails = pw.jetson_rails()
        info["jetson_rails"] = [r[0] for r in rails]
        info["power_rail_used"] = [r[0] for r in rails if "VDD_IN" in r[0].upper()] or "sum of all rails"
    if pw.mode not in ("none", "manual"):
        print("measuring idle power ...", flush=True)
        info["idle_w"] = pw.average(a.idle_s)
    print(json.dumps(info, indent=1))
    json.dump(info, open(os.path.join(a.out, "device_info.json"), "w"), indent=1)
    run_dir = os.path.join(a.root, a.env, "runs")
    ref_f = os.path.join(a.root, a.env, "reference_policies.json")
    ref = json.load(open(ref_f)) if os.path.exists(ref_f) else {}
    if a.eval_ep and not ref:
        print(f"note: {ref_f} not found -> normalized scores are not computed", flush=True)

    def norm(R):
        if a.env in ("mec", "channel") and "random" in ref and "myopic" in ref:
            return (R - ref["random"]) / (ref["myopic"] - ref["random"])
        if a.env == "cartpole":
            return R / 500.0
        return float("nan")

    rows, burn = [], []
    for core in a.cores.split(","):
        for algo in a.algos.split(","):
            for variant in a.variants.split(","):
                for s in [int(v) for v in a.seeds.split(",")]:
                    f = os.path.join(run_dir, f"{algo}_{variant}_{core}_seed{s}.pt")
                    if not os.path.exists(f):
                        print(f"missing {f} (skipped)", flush=True)
                        continue
                    ck = torch.load(f, map_location="cpu", weights_only=False)
                    branches = list(ck.get("branches") or [2] * 10)
                    net = HybridNet(ck["obs_dim"], ck["heads"], core=core, compress=ck["compress"])
                    net.load_state_dict(ck["state"])
                    net.refresh_ops()
                    net.eval()
                    jf = f[:-3] + ".json"
                    json_ret = float("nan")
                    if os.path.exists(jf):
                        try:
                            jr = json.load(open(jf))
                            json_ret = float(jr.get("final", jr).get("eval_return", float("nan")))
                        except Exception:
                            pass
                    with torch.no_grad():
                        if a.eval_ep > 0:
                            R_tr, S = rollout(lambda x: actions(net(x), branches)[0].numpy(), a.env, a.eval_ep)
                        else:
                            g = torch.Generator().manual_seed(0)
                            R_tr, S = float("nan"), torch.randn(256, ck["obs_dim"], generator=g).clamp(-3, 3)
                        ref_act = actions(net(S), branches)
                    x1 = S[:1]
                    modes = {"fp32": DeployNet(net).eval()}
                    q_src = DeployNet(net, phase_bits=8).eval()
                    modes["int8"] = torch.ao.quantization.quantize_dynamic(q_src, {nn.Linear}, dtype=torch.qint8)
                    for mode, m in modes.items():
                        with torch.no_grad():
                            agree = float((actions(m(S), branches) == ref_act).float().mean())
                            R_dep = rollout(lambda x: actions(m(x), branches)[0].numpy(), a.env, a.eval_ep)[0] \
                                if a.eval_ep > 0 else float("nan")
                            for _ in range(a.warmup):
                                m(x1)
                            ts = []
                            for _ in range(a.reps):
                                t0 = time.perf_counter()
                                actions(m(x1), branches)
                                ts.append(time.perf_counter() - t0)
                        ts = np.array(ts) * 1e3
                        payload, fbytes = compact_payload(m, net, mode)
                        native = int((variant == "base" and mode == "fp32") or (variant != "base" and mode == "int8"))
                        row = dict(device=a.device, env=a.env, algo=algo, variant=variant, core=core, seed=s, mode=mode,
                                   native=native,
                                   latency_ms_median=float(np.median(ts)), latency_ms_p90=float(np.percentile(ts, 90)),
                                   throughput_hz=float(1e3 / np.median(ts)), payload_bytes=payload, file_bytes=fbytes,
                                   torch_bytes=torch_bytes(m), analytic_bytes=static_memory_bytes(net),
                                   agreement_vs_trained=agree, n_states=int(S.shape[0]),
                                   return_trained=R_tr, return_json=json_ret, return_deployed=R_dep,
                                   norm_trained=norm(R_tr), norm_deployed=norm(R_dep),
                                   n_2q=len([o for o in net.ops() if o[0] == "CNOT"]))
                        if pw.mode not in ("none", "manual"):
                            with torch.no_grad():
                                p = pw.average(a.burn_s, busy_fn=lambda: m(x1))
                            row.update(power_w=p, idle_w=info["idle_w"],
                                       energy_mJ_per_decision=(p - info["idle_w"]) * row["latency_ms_median"])
                        elif pw.mode == "manual":
                            t_start = time.time()
                            t_end = time.perf_counter() + a.burn_s
                            with torch.no_grad():
                                while time.perf_counter() < t_end:
                                    m(x1)
                            burn.append(dict(algo=algo, variant=variant, core=core, seed=s, mode=mode,
                                             start=time.strftime("%H:%M:%S", time.localtime(t_start)),
                                             end=time.strftime("%H:%M:%S")))
                            row.update(power_w=float("nan"))
                        rows.append(row)
                        print(f"{algo}_{variant}_{core}_seed{s} [{mode}]: {row['latency_ms_median']:.3f} ms, "
                              f"payload {payload} B (analytic {row['analytic_bytes']} B), agree {agree:.3f}, "
                              f"score trained {row['norm_trained']:.3f} -> deployed {row['norm_deployed']:.3f}",
                              flush=True)
                    # incremental save, so an interrupted run keeps what it has measured
                    pd.DataFrame(rows).to_csv(os.path.join(a.out, "edge_latency.csv"), index=False)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "edge_latency.csv"), index=False)
    if burn:
        pd.DataFrame(burn).to_csv(os.path.join(a.out, "burn_log.csv"), index=False)
    cols = [c for c in ("latency_ms_median", "payload_bytes", "analytic_bytes", "agreement_vs_trained",
                        "norm_trained", "norm_deployed", "energy_mJ_per_decision") if c in df]
    pd.set_option("display.width", 200)
    print("\nall deployments:")
    print(df.groupby(["core", "variant", "mode"])[cols].mean().round(4).to_string())
    print("\nnative deployments (base: fp32, light: int8):")
    print(df[df.native == 1].groupby(["core", "variant"])[cols].mean().round(4).to_string())
    print("written to", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
