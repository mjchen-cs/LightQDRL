# -*- coding: utf-8 -*-
"""
origin_transpile_estimate.py — compiled two-qubit gate count / SWAPs / depth of the hardware-validated policy circuits
on the topology of an Origin Quantum chip (e.g. WK_C180), estimated locally with Qiskit.

Why: the Origin cloud does not return the mapped circuit, so n_2q_transpiled is NaN in per_state.csv.  This script
re-compiles exactly the same circuits (structure of the base / light policy circuits, learned DQCP masks) onto the
chip's coupling graph restricted to the qubits that were available, in the chip's native two-qubit basis.

Step 1  (needs the API key, a few seconds, no QPU time):  download and cache the chip information
          $env:ORIGIN_API_KEY = "your-key"
          python origin_transpile_estimate.py --fetch --backend WK_C180
        -> results_hw/chip_WK_C180.json  (edges, available qubits, basic gates, 1Q/2Q fidelities, readout, T1/T2,
                                          download time; cite this snapshot in the paper)
Step 2  (offline):  compile and report
          python origin_transpile_estimate.py --backend WK_C180 --algo Q-PPO --seed 0 \
                 --hw_dir results_hw/Q-PPO_seed0_WK_C180_s1000
        -> <hw_dir>/transpile_estimate.csv     one row per (variant, layout mode, transpiler seed)
           <hw_dir>/transpile_summary.json    mean / sd / min per variant and layout mode
           <hw_dir>/per_state_with_transpile.csv  per_state.csv with n_2q_transpiled filled in (original kept)

Layout modes
  chip   : Qiskit SABRE layout + routing on all available qubits (opt level 1), n_seeds transpiler seeds
  block  : restricted to the 6-qubit block recommended by the Origin cloud (best_qubit_blocks), if it was fetched
The circuits are compiled with symbolic angles, so the counts are identical for all evaluation states.
ESP (estimated success probability) = product of the 2Q fidelities of the physical edges used x 1Q fidelities x
readout fidelities of the measured qubits — reported only when the chip snapshot contains fidelities.
"""
import os
import re
import json
import time
import argparse
import numpy as np
import pandas as pd

import lqdrl_core as LC
from quafu_hw_validation import load_agent_net


# ----------------------------------------------------------------------------------------- fetch
def _call(obj, name, *args):
    try:
        return getattr(obj, name)(*args)
    except Exception:
        return None


def fetch_chip(backend_name, api_key, out_f):
    from pyqpanda3 import qcloud
    api_key = api_key or os.environ.get("ORIGIN_API_KEY")
    if not api_key:
        raise SystemExit("Origin Quantum API key missing: use --api_key or set ORIGIN_API_KEY")
    be = qcloud.QCloudService(api_key=api_key).backend(backend_name)
    ci = be.chip_info()
    avail = [int(q) for q in ci.available_qubits()]
    edges = [[int(a), int(b)] for a, b in ci.get_chip_topology()]
    gates = [str(g) for g in (_call(ci, "get_basic_gates") or [])]
    two_q = []
    for d in (_call(ci, "double_qubits_info") or []):
        qs = _call(d, "get_qubits")
        if qs is not None and len(qs) == 2:
            two_q.append(dict(qubits=[int(qs[0]), int(qs[1])], fidelity=_call(d, "get_fidelity")))
    one_q = []
    for s in (_call(ci, "single_qubit_info") or []):
        one_q.append(dict(qubit=_call(s, "get_qubit_id"), gate_fidelity=_call(s, "get_single_gate_fidelity"),
                          readout_fidelity=_call(s, "get_readout_fidelity"), t1=_call(s, "get_t1"),
                          t2=_call(s, "get_t2"), frequency=_call(s, "get_frequency")))
    block = None
    bb = _call(be, "best_qubit_blocks", 6)
    if bb is not None:
        txt = str(bb)
        for attr in ("qubit_blocks", "blocks", "get_blocks", "qubits"):
            v = getattr(bb, attr, None)
            try:
                v = v() if callable(v) else v
                if v is not None:
                    txt = json.dumps(v) if not isinstance(v, str) else v
                    break
            except Exception:
                pass
        nums = [int(x) for x in re.findall(r"\d+", txt)]
        cand = [q for q in nums if q in set(avail)]
        if len(cand) >= 6:
            block = cand[:6]
        block_raw = txt
    else:
        block_raw = None
    snap = dict(backend=backend_name, chip_id=str(_call(ci, "chip_id")), qubits_num=_call(ci, "qubits_num"),
                available_qubits=avail, edges=edges, basic_gates=gates, double_qubits=two_q, single_qubits=one_q,
                best_block_6=block, best_block_raw=block_raw, fetched=time.strftime("%Y-%m-%d %H:%M:%S"))
    LC.save_json(snap, out_f)
    print(f"chip snapshot written to {out_f}: {len(avail)} available qubits, {len(edges)} edges, "
          f"basic gates {gates}, 2Q fidelities for {len(two_q)} pairs, best 6-qubit block {block}")


# ----------------------------------------------------------------------------------------- compile
def native_2q(gates):
    g = {x.upper() for x in gates}
    if "CZ" in g:
        return "cz"
    if "ISWAP" in g:
        return "iswap"
    return "cx"


def compile_once(qc, cmap, basis2q, seed, initial_layout=None):
    from qiskit import transpile
    native = transpile(qc, basis_gates=["rz", "sx", "x", basis2q], coupling_map=cmap, optimization_level=1,
                       seed_transpiler=seed, initial_layout=initial_layout)
    routed = transpile(qc, basis_gates=["rx", "ry", "rz", "cx", "swap"], coupling_map=cmap, optimization_level=1,
                       seed_transpiler=seed, initial_layout=initial_layout)
    cn = native.count_ops()
    used_edges, used_qubits = [], set()
    for inst in native.data:
        qs = [native.find_bit(q).index for q in inst.qubits]
        if inst.operation.name == basis2q:
            used_edges.append(tuple(sorted(qs)))
        if inst.operation.name in ("rz", "sx", "x", basis2q):
            used_qubits.update(qs)
    layout = native.layout.final_index_layout() if native.layout is not None else list(range(qc.num_qubits))
    return dict(n_2q=int(cn.get(basis2q, 0)), n_1q=int(sum(cn.get(k, 0) for k in ("rz", "sx", "x"))),
                n_sx_x=int(cn.get("sx", 0) + cn.get("x", 0)), depth=int(native.depth()),
                depth_2q=int(native.depth(lambda i: i.operation.num_qubits == 2)),
                swaps=int(routed.count_ops().get("swap", 0)), used_edges=used_edges,
                measured_qubits=[int(q) for q in layout])


def esp(rec, snap):
    f2 = {tuple(sorted(d["qubits"])): d["fidelity"] for d in snap.get("double_qubits", []) if d.get("fidelity")}
    s1 = {d["qubit"]: d for d in snap.get("single_qubits", []) if d.get("qubit") is not None}
    if not f2:
        return np.nan
    p = 1.0
    for e in rec["used_edges"]:
        f = f2.get(e)
        if f is None:
            return np.nan
        p *= f if f <= 1 else f / 100.0
    if s1:
        g1 = np.nanmean([d["gate_fidelity"] for d in s1.values() if d.get("gate_fidelity")] or [np.nan])
        g1 = g1 if g1 <= 1 else g1 / 100.0
        p *= g1 ** rec["n_sx_x"] if np.isfinite(g1) else 1.0
        for q in rec["measured_qubits"]:
            r = s1.get(q, {}).get("readout_fidelity")
            if r:
                p *= r if r <= 1 else r / 100.0
    return float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="WK_C180")
    ap.add_argument("--fetch", action="store_true", help="download and cache the chip snapshot (needs API key)")
    ap.add_argument("--api_key", default=None)
    ap.add_argument("--chip_json", default=None, help="default: results_hw/chip_<backend>.json")
    ap.add_argument("--run_dir", default="results_qrl/mec/runs")
    ap.add_argument("--algo", default="Q-PPO")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", default="base,light")
    ap.add_argument("--hw_dir", default=None, help="hardware result folder whose per_state.csv is completed")
    ap.add_argument("--n_seeds", type=int, default=20, help="transpiler seeds for the SABRE layout")
    a = ap.parse_args()
    chip_f = a.chip_json or os.path.join("results_hw", f"chip_{a.backend}.json")
    if a.fetch:
        os.makedirs(os.path.dirname(chip_f) or ".", exist_ok=True)
        fetch_chip(a.backend, a.api_key, chip_f)
        return
    if not os.path.exists(chip_f):
        raise SystemExit(f"{chip_f} not found; run first:  python origin_transpile_estimate.py --fetch --backend {a.backend}")

    from qiskit.transpiler import CouplingMap
    import networkx as nx
    snap = LC.load_json(chip_f)
    avail = set(snap["available_qubits"])
    G = nx.Graph([tuple(e) for e in snap["edges"] if e[0] in avail and e[1] in avail])
    comp = max(nx.connected_components(G), key=len)          # largest connected set of available qubits
    phys = sorted(comp)
    to_idx = {q: i for i, q in enumerate(phys)}               # compact indices for Qiskit
    edges = [(to_idx[a_], to_idx[b]) for a_, b in G.subgraph(comp).edges()]
    cmap = CouplingMap(couplinglist=[list(e) for e in edges] + [[b, a_] for a_, b in edges])
    snap_idx = dict(snap, double_qubits=[dict(d, qubits=[to_idx[d["qubits"][0]], to_idx[d["qubits"][1]]])
                                         for d in snap.get("double_qubits", [])
                                         if d["qubits"][0] in to_idx and d["qubits"][1] in to_idx],
                    single_qubits=[dict(d, qubit=to_idx[d["qubit"]]) for d in snap.get("single_qubits", [])
                                   if d.get("qubit") in to_idx])
    block = snap.get("best_block_6")
    if block and not (all(q in to_idx for q in block) and nx.is_connected(G.subgraph(block))):
        print(f"best block {block} is not a connected set of available qubits; 'block' mode skipped")
        block = None
    basis2q = native_2q(snap.get("basic_gates", []))
    print(f"{snap['backend']}: {len(avail)} available qubits, largest connected set {len(phys)} qubits / "
          f"{len(edges)} edges, native 2Q gate {basis2q.upper()}")

    out_dir = a.hw_dir or os.path.join("results_hw", f"{a.algo}_seed{a.seed}_{a.backend}_transpile")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for variant in a.variants.split(","):
        net, _ = load_agent_net(a.run_dir, a.algo, variant, a.seed)
        ops = net.ops()
        qc = LC.ops_to_qiskit(ops)
        n2_log, depth_log = LC.gate_counts(ops)[1], LC.logical_depth(ops)
        modes = [("chip", None)]
        if block:
            modes.append(("block", [to_idx[q] for q in block]))
        for mode, layout in modes:
            for s in range(a.n_seeds if mode == "chip" else 1):
                r = compile_once(qc, cmap, basis2q, seed=s, initial_layout=layout)
                rows.append(dict(variant=variant, mode=mode, transpiler_seed=s, n_2q_logical=n2_log,
                                 depth_logical=depth_log, n_2q_compiled=r["n_2q"], n_1q_compiled=r["n_1q"],
                                 swaps=r["swaps"], depth_compiled=r["depth"], depth_2q_compiled=r["depth_2q"],
                                 esp=esp(r, snap_idx),
                                 physical_qubits=json.dumps([phys[q] for q in r["measured_qubits"]])))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "transpile_estimate.csv"), index=False)
    cols = ["n_2q_logical", "n_2q_compiled", "swaps", "depth_logical", "depth_compiled", "depth_2q_compiled", "esp"]
    agg = df.groupby(["mode", "variant"])[cols].agg(["mean", "std", "min"])
    print(agg.round(3).to_string())
    summ = dict(backend=snap["backend"], chip_snapshot=chip_f, chip_fetched=snap.get("fetched"), native_2q=basis2q,
                qiskit_optimization_level=1, n_transpiler_seeds=a.n_seeds,
                results=[dict(mode=m, variant=v, **{f"{c}_{s}": float(agg.loc[(m, v), (c, s)])
                                                     for c in cols for s in ("mean", "std", "min")})
                         for (m, v) in agg.index])
    LC.save_json(summ, os.path.join(out_dir, "transpile_summary.json"))
    ps = os.path.join(out_dir, "per_state.csv")
    if os.path.exists(ps):
        p = pd.read_csv(ps)
        m = df[df["mode"] == "chip"].groupby("variant")
        for c, src in (("n_2q_transpiled", "n_2q_compiled"), ("swaps_transpiled", "swaps"),
                       ("depth_transpiled", "depth_compiled")):
            p[c] = p["variant"].map(m[src].mean())
        p["transpile_source"] = f"Qiskit estimate on {snap['backend']} topology (mean of {a.n_seeds} seeds)"
        p.to_csv(os.path.join(out_dir, "per_state_with_transpile.csv"), index=False)
    print("written to", os.path.abspath(out_dir))


if __name__ == "__main__":
    main()
