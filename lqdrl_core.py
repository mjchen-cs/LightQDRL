# -*- coding: utf-8 -*-
"""
lqdrl_core.py — shared implementation for the Fig. 5 / Fig. 6 / Fig. 7 / Table 3 / Table 6
experiments of the LightQDRL manuscript.

Everything that the experiment scripts measure is computed here from a single source of truth:

  * one gate list (``build_ops``) drives the PennyLane QNode used for training/latency, the
    Qiskit circuit used for transpilation (depth / SWAP counts) and the torch density-matrix
    simulator used for the finite-shot / noise study, so gate counts can never drift apart;
  * DQCP  — dynamic quantum circuit pruning (EMA first-order saliency for rotations,
            loss-removal sensitivity for CNOTs, exponential sparsity schedule);
  * JCQQ  — joint classical–quantum quantization (INT8/INT4 fake-quant with STE on the
            classical weights + activations, INT8 phase quantization of all rotation angles);
  * PEQA  — parameter-efficient quantum adaptation (LoRA adapters on the angle projection and on
            decoder layer 2, sparse offsets on the retained rotation angles, two scale scalars).

The DROO-style online loop (adaptive K, order-preserving quantization, bisection optimizer,
80/20 cyclic split) is identical to ``lightQDRL.py``.

Requirements: numpy, scipy, torch, pennylane (>=0.30), pandas, matplotlib,
              qiskit (>=1.0, only for transpiled depth / SWAP counts), psutil (optional).
"""
import os
import json
import math
import time
import random
import platform
from collections import deque

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F

from optimization import bisection

# =====================================================================================
# 0. Global configuration (edit here; every value is also written into the result files)
# =====================================================================================
N_USERS = 10
N_QUBITS = 6
N_LAYERS = 2
HIDDEN = 64
H_SCALE = 1e6                       # identical to lightQDRL.py (channel * 1e6)
REAL_5G_BASE_RATE_MBPS = 300.0
TASK_SIZE_BITS = 5.0 * 1024 * 1024 * 8
EDGE_POWER_W = 5.0                  # fixed representative edge-processor power (energy = P * dt)
W_DEFAULT = [1.5 if i % 2 == 1 else 1.0 for i in range(N_USERS)]   # weights used to build data_10.mat
W_ALT = [1.0 if i % 2 == 1 else 1.5 for i in range(N_USERS)]       # weights of data_10_WeightsAlternated.mat

DATA_SOURCE = "./data/data_10.mat"
DATA_SHIFT = "./data/data_10_WeightsAlternated.mat"

DEFAULT_CFG = dict(
    lr=0.01, training_interval=10, batch_size=128, memory_size=1024, delta=32,
    clip_C=3.0,                      # JCQQ / input-normalisation clipping bound
    w_bits=8, a_bits=8, phase_bits=8,
    rho_rot=0.5, rho_ent=0.5,        # DQCP final sparsity of rotations / CNOTs
    prune_start_frac=0.2,            # pruning starts after 20 % of the expected updates
    prune_end_frac=0.7,              # final sparsity reached at 70 % of the expected updates
    prune_every=20,                  # pruning event every 20 optimizer updates
    saliency_beta=0.9,               # EMA factor of the saliency scores
    lora_rank=2, lora_alpha=2.0,     # PEQA adapters
    peqa_lr=0.01,
)

# Realistic six-qubit device topology: 6-qubit connected subgraph of the IBM Falcon r5.11H
# 7-qubit "H" coupling map (ibm_lagos / ibm_nairobi / ibm_perth).
TOPOLOGIES = {
    "All-to-all": None,
    "Ring": [(i, (i + 1) % N_QUBITS) for i in range(N_QUBITS)],
    "Linear": [(i, i + 1) for i in range(N_QUBITS - 1)],
    "Heavy-hex (6Q)": [(0, 1), (1, 2), (1, 3), (3, 5), (4, 5)],
}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def env_info():
    info = dict(python=platform.python_version(), platform=platform.platform(),
                processor=platform.processor(), torch=torch.__version__,
                torch_threads=torch.get_num_threads())
    try:
        import pennylane as qml
        info["pennylane"] = qml.__version__
    except Exception:
        pass
    try:
        import qiskit
        info["qiskit"] = qiskit.__version__
    except Exception:
        pass
    return info


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    return o


def save_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=1, ensure_ascii=False)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =====================================================================================
# 1. Data
# =====================================================================================
def load_data(path):
    d = sio.loadmat(path)
    return d["input_h"].astype(np.float64), d["output_obj"].astype(np.float64).ravel()


def normalisation_stats(h_raw, train_frac=0.8):
    """FP32 input statistics (0 trainable parameters) estimated on the training split only."""
    x = h_raw[: int(train_frac * len(h_raw))] * H_SCALE
    return x.mean(0).astype(np.float32), (x.std(0) + 1e-8).astype(np.float32)


class RateCache:
    """Memoises bisection(h, m, w) by (dataset tag, slot index, mode) — pure speed-up."""

    def __init__(self):
        self.d = {}
        self.hits = 0
        self.calls = 0

    def rate(self, tag, idx, h_raw, m, weights):
        key = (tag, int(idx), np.asarray(m, dtype=np.int8).tobytes())
        self.calls += 1
        if key in self.d:
            self.hits += 1
            return self.d[key]
        r = bisection(h_raw, np.asarray(m), weights)[0]
        self.d[key] = r
        return r


def op_candidates(scores, K):
    """Order-preserving quantization of lightQDRL.py: candidate k offloads the top-(k+1) users."""
    idx = np.argsort(scores)
    n = len(scores)
    m_list = np.zeros((K, n))
    for k in range(K):
        if k < n:
            m_list[k, idx[-(k + 1):]] = 1
    return m_list


# =====================================================================================
# 2. Circuit description (single source of truth)
# =====================================================================================
ROT_NAMES = ("RX", "RY", "RZ")


def build_ops(n=N_QUBITS, L=N_LAYERS, rot_mask=None, ent_mask=None):
    """Return the gate list [(name, wires, param_ref)].  param_ref = ('in', q) | ('w', l, q, k) | None.
    With all-ones masks this is exactly the circuit of lightQDRL.py."""
    rot_mask = np.ones((L, n, 3)) if rot_mask is None else np.asarray(rot_mask)
    ent_mask = np.ones((L, n)) if ent_mask is None else np.asarray(ent_mask)
    ops = [("RY", (q,), ("in", q)) for q in range(n)]
    for l in range(L):
        for q in range(n):
            for k, g in enumerate(ROT_NAMES):
                if rot_mask[l, q, k] > 0.5:
                    ops.append((g, (q,), ("w", l, q, k)))
        for q in range(n):
            if ent_mask[l, q] > 0.5:
                ops.append(("CNOT", (q, (q + 1) % n), None))
    return ops


def gate_counts(ops):
    n1 = sum(1 for o in ops if len(o[1]) == 1)
    n2 = sum(1 for o in ops if len(o[1]) == 2)
    return n1, n2


def logical_depth(ops, n=N_QUBITS):
    """ASAP-scheduled depth of the logical (un-routed) circuit; every gate counts one layer."""
    level = [0] * n
    for _, wires, _ in ops:
        t = max(level[w] for w in wires) + 1
        for w in wires:
            level[w] = t
    return max(level) if ops else 0


def make_pennylane_qnode(get_ops, n=N_QUBITS):
    """QNode whose gate list is read from ``get_ops()`` at every call (masks may change)."""
    import pennylane as qml
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        for g, wires, p in get_ops():
            if g == "CNOT":
                qml.CNOT(wires=list(wires))
            else:
                val = inputs[..., p[1]] if p[0] == "in" else weights[p[1], p[2], p[3]]
                getattr(qml, g)(val, wires=wires[0])
        return [qml.expval(qml.PauliZ(i)) for i in range(n)]

    return circuit


def run_qnode(qnode, angles, weights):
    out = qnode(angles, weights)
    if isinstance(out, (list, tuple)):
        out = torch.stack(list(out), dim=-1)
    if out.dim() == 1:
        out = out.unsqueeze(0)
    return out.to(angles.dtype)


# =====================================================================================
# 3. JCQQ quantizers (fake quantization + straight-through estimator)
# =====================================================================================
def _ste(x, xq):
    return x + (xq - x).detach()


def fq_weight_sym(w, bits):
    """Per-tensor symmetric weight quantization to signed `bits` integers."""
    if bits >= 32:
        return w
    qmax = 2 ** (bits - 1) - 1
    s = w.detach().abs().max().clamp_min(1e-8) / qmax
    return _ste(w, torch.clamp(torch.round(w / s), -qmax - 1, qmax) * s)


def fq_fixed_sym(x, bits, scale):
    """Fixed-scale symmetric quantizer, e.g. S_in = 2C/255 or S_exp = 2/255."""
    if bits >= 32:
        return x
    qmax = 2 ** (bits - 1) - 1
    return _ste(x, torch.clamp(torch.round(x / scale), -qmax - 1, qmax) * scale)


def fq_act_unsigned(x, bits):
    """Dynamic per-sample unsigned quantization of non-negative (post-ReLU) activations
    (one scale per row, so a decision never depends on the other samples of a batch)."""
    if bits >= 32:
        return x
    qmax = 2 ** bits - 1
    s = x.detach().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    return _ste(x, torch.clamp(torch.round(x / s), 0, qmax) * s)


def wrap_phase(theta):
    return torch.remainder(theta + math.pi, 2 * math.pi) - math.pi


def fq_phase(theta, bits):
    """Phase quantization of rotation angles on a uniform 2^bits grid over [-pi, pi)."""
    if bits >= 32:
        return theta
    step = 2 * math.pi / (2 ** bits)
    return _ste(theta, torch.round(wrap_phase(theta) / step) * step)


# =====================================================================================
# 4. LightQDRL hybrid policy (factorial switches: dqcp / jcqq / peqa)
# =====================================================================================
class HybridPolicy(nn.Module):
    """Encoder (10->64, BN, 64->6) + 6-qubit, 2-layer VQC (RX,RY,RZ + CNOT ring) + decoder (6->64->10).
    Base trainable parameters: 704 + 128 + 390 + 36 + 448 + 650 = 2,356 (identical to lightQDRL.py)."""

    def __init__(self, h_mean, h_std, dqcp=False, jcqq=False, cfg=None,
                 n_users=N_USERS, n_qubits=N_QUBITS, n_layers=N_LAYERS, hidden=HIDDEN):
        super().__init__()
        self.cfg = dict(DEFAULT_CFG, **(cfg or {}))
        self.dqcp, self.jcqq = bool(dqcp), bool(jcqq)
        self.n, self.L, self.N = n_qubits, n_layers, n_users
        self.register_buffer("h_mean", torch.tensor(h_mean, dtype=torch.float32))
        self.register_buffer("h_std", torch.tensor(h_std, dtype=torch.float32))

        self.enc_fc = nn.Linear(n_users, hidden)
        self.enc_bn = nn.BatchNorm1d(hidden)
        self.enc_proj = nn.Linear(hidden, n_qubits)
        # same initialisation as qml.qnn.TorchLayer: U(0, 2*pi)
        self.theta = nn.Parameter(torch.rand(n_layers, n_qubits, 3) * 2 * math.pi)
        self.dec_fc1 = nn.Linear(n_qubits, hidden)
        self.dec_fc2 = nn.Linear(hidden, n_users)

        self.register_buffer("rot_mask", torch.ones(n_layers, n_qubits, 3))
        self.register_buffer("ent_mask", torch.ones(n_layers, n_qubits))
        self.register_buffer("sal_rot", torch.zeros(n_layers, n_qubits, 3))
        self.register_buffer("sal_ent", torch.zeros(n_layers, n_qubits))
        self.theta.register_hook(lambda g: g * self.rot_mask)   # pruned angles receive no update

        self.peqa = None            # nn.ParameterDict when PEQA is enabled
        self.peqa_merged = False
        self._ops = build_ops(self.n, self.L, self.rot_mask.numpy(), self.ent_mask.numpy())
        self.qnode = make_pennylane_qnode(lambda: self._ops, self.n)

    # ----------------------------------------------------------------------- utilities
    def refresh_ops(self):
        self._ops = build_ops(self.n, self.L, self.rot_mask.cpu().numpy(), self.ent_mask.cpu().numpy())

    def ops(self):
        return self._ops

    def base_parameters(self):
        return [p for n_, p in self.named_parameters() if not n_.startswith("peqa")]

    def n_base_params(self):
        return sum(p.numel() for p in self.base_parameters())

    def n_active_angles(self):
        return int(self.rot_mask.sum().item())

    def __getstate__(self):          # QNode closures are not picklable
        st = self.__dict__.copy()
        st.pop("qnode", None)
        return st

    def __setstate__(self, st):
        self.__dict__.update(st)
        self.qnode = make_pennylane_qnode(lambda: self._ops, self.n)

    # ------------------------------------------------------------------- PEQA (adapters)
    def enable_peqa(self, rank=None, alpha=None):
        r = int(rank or self.cfg["lora_rank"])
        self.lora_scale = float(alpha or self.cfg["lora_alpha"]) / r
        H = self.enc_proj.in_features
        self.peqa = nn.ParameterDict(dict(
            A_enc=nn.Parameter(torch.randn(r, H) / math.sqrt(H)),
            B_enc=nn.Parameter(torch.zeros(self.n, r)),
            A_dec=nn.Parameter(torch.randn(r, H) / math.sqrt(H)),
            B_dec=nn.Parameter(torch.zeros(self.N, r)),
            dtheta=nn.Parameter(torch.zeros(self.L, self.n, 3)),
            s_enc=nn.Parameter(torch.ones(1)),
            s_dec=nn.Parameter(torch.ones(1)),
        ))
        self.peqa["dtheta"].register_hook(lambda g: g * self.rot_mask)  # only retained angles move
        for p in self.base_parameters():
            p.requires_grad_(False)
        self.peqa_merged = False

    def peqa_param_breakdown(self):
        if self.peqa is None:
            return {}
        P = self.peqa
        return {
            "enc_adapter": P["A_enc"].numel() + P["B_enc"].numel(),
            "dec_adapter": P["A_dec"].numel() + P["B_dec"].numel(),
            "active_angles": self.n_active_angles(),
            "scales": 2,
        }

    def n_online_trainable(self):
        if self.peqa is not None and not self.peqa_merged:
            return int(sum(self.peqa_param_breakdown().values()))
        return self.n_base_params() - (self.theta.numel() - self.n_active_angles())

    @torch.no_grad()
    def merge_peqa(self):
        """Fold adapters / angle offsets into the backbone for deployment (scales s_enc, s_dec remain)."""
        if self.peqa is None or self.peqa_merged:
            return
        P = self.peqa
        self.enc_proj.weight += self.lora_scale * P["B_enc"] @ P["A_enc"]
        self.dec_fc2.weight += self.lora_scale * P["B_dec"] @ P["A_dec"]
        self.theta += self.rot_mask * P["dtheta"]
        P["A_enc"].zero_(); P["B_enc"].zero_(); P["A_dec"].zero_(); P["B_dec"].zero_(); P["dtheta"].zero_()
        self.peqa_merged = True

    # ---------------------------------------------------------------- forward pieces
    def _w(self, w):
        return fq_weight_sym(w, self.cfg["w_bits"]) if self.jcqq else w

    def _a(self, x):
        return fq_act_unsigned(x, self.cfg["a_bits"]) if self.jcqq else x

    def encode_angles(self, x):
        C = self.cfg["clip_C"]
        z = torch.clamp((x - self.h_mean) / self.h_std, -C, C)
        if self.jcqq:
            z = fq_fixed_sym(z, 8, 2 * C / 255.0)
        a = F.linear(z, self._w(self.enc_fc.weight), self.enc_fc.bias)
        a = self._a(F.relu(self.enc_bn(a)))
        Wp = self.enc_proj.weight
        if self.peqa is not None and not self.peqa_merged:
            Wp = Wp + self.lora_scale * self.peqa["B_enc"] @ self.peqa["A_enc"]
        ang = torch.tanh(F.linear(a, self._w(Wp), self.enc_proj.bias)) * math.pi
        if self.peqa is not None:
            ang = torch.clamp(ang * self.peqa["s_enc"], -math.pi, math.pi)
        if self.jcqq:
            ang = fq_phase(ang, self.cfg["phase_bits"])
        return ang

    def effective_theta(self):
        th = self.theta
        if self.peqa is not None and not self.peqa_merged:
            th = th + self.rot_mask * self.peqa["dtheta"]
        if self.jcqq:
            th = fq_phase(th, self.cfg["phase_bits"])
        return th

    def decode_from_expz(self, e):
        if self.jcqq:
            e = fq_fixed_sym(e, 8, 2.0 / 255.0)
        if self.peqa is not None:
            e = e * self.peqa["s_dec"]
        h2 = self._a(F.relu(F.linear(e, self._w(self.dec_fc1.weight), self.dec_fc1.bias)))
        W2 = self.dec_fc2.weight
        if self.peqa is not None and not self.peqa_merged:
            W2 = W2 + self.lora_scale * self.peqa["B_dec"] @ self.peqa["A_dec"]
        return F.linear(h2, self._w(W2), self.dec_fc2.bias)

    def forward(self, x):
        ang = self.encode_angles(x)
        e = run_qnode(self.qnode, ang, self.effective_theta())
        return self.decode_from_expz(e)


class ParamMatchedMLP(nn.Module):
    """Quantum-removal control: the VQC is replaced by z_out = tanh(W z_in), W in R^{6x6}, no bias (36 weights)."""

    def __init__(self, h_mean, h_std, n_users=N_USERS, n_qubits=N_QUBITS, hidden=HIDDEN, clip_C=3.0):
        super().__init__()
        self.register_buffer("h_mean", torch.tensor(h_mean, dtype=torch.float32))
        self.register_buffer("h_std", torch.tensor(h_std, dtype=torch.float32))
        self.C = clip_C
        self.enc_fc = nn.Linear(n_users, hidden)
        self.enc_bn = nn.BatchNorm1d(hidden)
        self.enc_proj = nn.Linear(hidden, n_qubits)
        self.Wq = nn.Linear(n_qubits, n_qubits, bias=False)
        self.dec_fc1 = nn.Linear(n_qubits, hidden)
        self.dec_fc2 = nn.Linear(hidden, n_users)
        self.dqcp = self.jcqq = False
        self.peqa = None

    def ops(self):
        return []

    def forward(self, x):
        z = torch.clamp((x - self.h_mean) / self.h_std, -self.C, self.C)
        a = F.relu(self.enc_bn(self.enc_fc(z)))
        ang = torch.tanh(self.enc_proj(a)) * math.pi
        e = torch.tanh(self.Wq(ang))
        return self.dec_fc2(F.relu(self.dec_fc1(e)))


class QuantumDQNRef(nn.Module):
    """Quantum-DQN of quantum_6metrics.py: online + target hybrid networks (same circuit)."""

    def __init__(self, h_mean, h_std):
        super().__init__()
        self.online = HybridPolicy(h_mean, h_std)
        self.target = HybridPolicy(h_mean, h_std)
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.dqcp = self.jcqq = False
        self.peqa = None

    def ops(self):
        return self.online.ops()

    def forward(self, x):
        return self.online(x)

    def train_loss(self, x, m):
        with torch.no_grad():
            soft = 0.7 * m + 0.3 * torch.sigmoid(self.target(x))
        return F.binary_cross_entropy_with_logits(self.online(x), soft)


class QuantumPPORef(nn.Module):
    """Quantum-PPO of quantum_6metrics.py: VQC actor (10->32->6, VQC, 6->10) + classical critic."""

    def __init__(self, h_mean, h_std):
        super().__init__()
        self.register_buffer("h_mean", torch.tensor(h_mean, dtype=torch.float32))
        self.register_buffer("h_std", torch.tensor(h_std, dtype=torch.float32))
        self.actor_encoder = nn.Sequential(nn.Linear(N_USERS, 32), nn.ReLU(), nn.Linear(32, N_QUBITS), nn.Tanh())
        self.theta = nn.Parameter(torch.rand(N_LAYERS, N_QUBITS, 3) * 2 * math.pi)
        self.actor_fc = nn.Linear(N_QUBITS, N_USERS)
        self.critic = nn.Sequential(nn.Linear(N_USERS, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                                    nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self._ops = build_ops()
        self.qnode = make_pennylane_qnode(lambda: self._ops)
        self.dqcp = self.jcqq = False
        self.peqa = None

    def ops(self):
        return self._ops

    def forward(self, x):
        z = (x - self.h_mean) / self.h_std
        e = run_qnode(self.qnode, self.actor_encoder(z) * math.pi, self.theta)
        return self.actor_fc(e)

    def train_loss(self, x, m):
        z = (x - self.h_mean) / self.h_std
        logits = self.forward(x)
        v = self.critic(z).squeeze(-1)
        adv = (m.mean(-1) - v).detach()
        return F.binary_cross_entropy_with_logits(logits, m) + 0.5 * (adv ** 2).mean() + 0.0 * v.sum()


# =====================================================================================
# 5. DQCP scheduler
# =====================================================================================
class DQCPScheduler:
    def __init__(self, model, expected_updates):
        c = model.cfg
        self.m = model
        self.u_start = int(c["prune_start_frac"] * expected_updates)
        self.u_end = max(self.u_start + 1, int(c["prune_end_frac"] * expected_updates))
        self.tau = max(1.0, (self.u_end - self.u_start) / 5.0)
        self.every = c["prune_every"]
        self.beta = c["saliency_beta"]
        self.extra_circuit_evals = 0       # circuits executed only for pruning decisions
        self.history = []

    def target_sparsity(self, u, rho_final):
        if u < self.u_start:
            return 0.0
        if u >= self.u_end:
            return rho_final
        return rho_final * (1.0 - math.exp(-(u - self.u_start) / self.tau)) / (1.0 - math.exp(-5.0))

    @torch.no_grad()
    def update_rotation_saliency(self):
        g = self.m.theta.grad
        if g is None:
            return
        s = (self.m.theta * g).abs() * self.m.rot_mask
        self.m.sal_rot.mul_(self.beta).add_((1 - self.beta) * s)

    @torch.no_grad()
    def update_cnot_sensitivity(self, x, y):
        """Loss-removal sensitivity dL_k = L(without CNOT k) - L(full), evaluated on a replay batch."""
        m = self.m
        was_training = m.training
        m.eval()
        base = F.binary_cross_entropy_with_logits(m(x), y).item()
        self.extra_circuit_evals += x.shape[0]
        for l in range(m.L):
            for q in range(m.n):
                if m.ent_mask[l, q] < 0.5:
                    continue
                m.ent_mask[l, q] = 0.0
                m.refresh_ops()
                lk = F.binary_cross_entropy_with_logits(m(x), y).item()
                self.extra_circuit_evals += x.shape[0]
                m.ent_mask[l, q] = 1.0
                m.sal_ent[l, q] = self.beta * m.sal_ent[l, q] + (1 - self.beta) * (lk - base)
        m.refresh_ops()
        m.train(was_training)

    @torch.no_grad()
    def _prune(self, mask, sal, rho):
        total = mask.numel()
        n_target = int(round(rho * total))
        n_pruned = int(total - mask.sum().item())
        if n_target <= n_pruned:
            return
        flat_s = sal.flatten().clone()
        flat_m = mask.flatten()
        flat_s[flat_m < 0.5] = float("inf")
        order = torch.argsort(flat_s)
        for idx in order[: n_target - n_pruned]:
            flat_m[idx] = 0.0
        mask.copy_(flat_m.view_as(mask))

    def step(self, u, probe_x, probe_y):
        """Call after every optimizer update u (1-based)."""
        if u < self.u_start or (u % self.every != 0 and u != self.u_end):
            return
        if u > self.u_end:
            return
        c = self.m.cfg
        self.update_cnot_sensitivity(probe_x, probe_y)
        self._prune(self.m.rot_mask, self.m.sal_rot, self.target_sparsity(u, c["rho_rot"]))
        self._prune(self.m.ent_mask, self.m.sal_ent, self.target_sparsity(u, c["rho_ent"]))
        self.m.refresh_ops()
        self.history.append(dict(update=u, active_rot=int(self.m.rot_mask.sum()), active_cnot=int(self.m.ent_mask.sum())))


# =====================================================================================
# 6. DROO online agent (identical loop to lightQDRL.py)
# =====================================================================================
class Agent:
    def __init__(self, model, params, lr, cfg, expected_updates=None):
        self.model = model
        self.cfg = cfg
        self.memory = deque(maxlen=cfg["memory_size"])
        self.opt = torch.optim.Adam(params, lr=lr)
        self.params = params
        self.time_step = 0
        self.updates = 0
        self.cost_his = []
        self.update_wall = []
        self.dqcp = DQCPScheduler(model, expected_updates) if (expected_updates and getattr(model, "dqcp", False)) else None

    def scores(self, h_raw):
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(h_raw * H_SCALE, dtype=torch.float32).unsqueeze(0)
            return self.model(x).squeeze(0).numpy()

    def encode(self, h_raw, m):
        self.memory.append((h_raw * H_SCALE, m))
        self.time_step += 1
        if self.time_step % self.cfg["training_interval"] == 0 and len(self.memory) >= self.cfg["batch_size"]:
            self.train_step()

    def train_step(self):
        self.model.train()
        batch = random.sample(self.memory, self.cfg["batch_size"])
        x = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
        y = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32)
        t0 = time.perf_counter()
        loss = F.binary_cross_entropy_with_logits(self.model(x), y)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, 0.5)
        if self.dqcp is not None:
            self.dqcp.update_rotation_saliency()
        self.opt.step()
        self.update_wall.append(time.perf_counter() - t0)
        self.updates += 1
        self.cost_his.append(loss.item())
        if self.dqcp is not None:
            self.dqcp.step(self.updates, x, y)


def droo_loop(agent, h_all, ref_rate, weights, n_slots, K_init, delta, tag, cache=None,
              train_frac=0.8, idx_offset=0, verbose=True):
    """Online loop of lightQDRL.py. Returns per-slot normalised rate, K history and the final K."""
    N = h_all.shape[1]
    split_idx = int(train_frac * len(h_all))
    num_test = min(len(h_all) - split_idx, n_slots - int(train_frac * n_slots))
    K = K_init
    ncr, k_idx_his, K_his = [], [], []
    for i in range(n_slots):
        if verbose and i % max(1, n_slots // 10) == 0:
            print(f"    [{tag}] {i / n_slots:.1f}", flush=True)
        if i > 0 and i % delta == 0:
            max_k = max(k_idx_his[-delta:-1]) + 1 if delta > 1 else k_idx_his[-1] + 1
            K = min(max_k + 1, N)
        i_idx = i % split_idx if i < n_slots - num_test else i - n_slots + num_test + split_idx
        h = h_all[i_idx]
        m_list = agent.candidates(h, K) if hasattr(agent, "candidates") else op_candidates(agent.scores(h), K)
        if cache is not None:
            r_list = [cache.rate(tag, i_idx + idx_offset, h, m, weights) for m in m_list]
        else:
            r_list = [bisection(h, m, weights)[0] for m in m_list]
        best = int(np.argmax(r_list))
        agent.encode(h, m_list[best])
        ncr.append(r_list[best] / ref_rate[i_idx])
        k_idx_his.append(best)
        K_his.append(K)
    return np.array(ncr), np.array(K_his), K, num_test


@torch.no_grad()
def evaluate_ncr(model, h_all, ref_rate, idx, weights, K, cache, tag, score_fn=None):
    """Greedy evaluation on fixed slots (no learning).  Returns dict with per-slot arrays.
    ncr     : best of the K order-preserving candidates (DROO decision rule, K = deployed K)
    ncr_thr : single action m_i = 1[logit_i > 0] (pure policy output, no candidate search)"""
    model.eval()
    x = torch.tensor(h_all[idx] * H_SCALE, dtype=torch.float32)
    scores = (score_fn(x) if score_fn is not None else model(x)).numpy()
    ncr, ncr_thr = [], []
    for j, i in enumerate(idx):
        h = h_all[i]
        cands = op_candidates(scores[j], K)
        r = max(cache.rate(tag, i, h, m, weights) for m in cands)
        ncr.append(r / ref_rate[i])
        m_thr = (scores[j] > 0).astype(float)
        ncr_thr.append(cache.rate(tag, i, h, m_thr, weights) / ref_rate[i])
    return dict(ncr=np.array(ncr), ncr_thr=np.array(ncr_thr), scores=scores)


# =====================================================================================
# 7. Resource accounting
# =====================================================================================
def classical_macs(model):
    """Classical multiply-accumulate operations per inference (batch 1), counted from the layer shapes.
    Linear: in*out; BatchNorm (not folded): 1 per feature; input normalisation: 1 per feature.
    Under JCQQ the BN is folded into the preceding INT8 linear layer at deployment (0 extra MACs)."""
    macs = 0
    for name, mod in model.named_modules():
        if name.startswith("target") or name.startswith("critic") or ".peqa" in name:
            continue
        if isinstance(mod, nn.Linear):
            macs += mod.in_features * mod.out_features
        elif isinstance(mod, nn.BatchNorm1d) and not getattr(model, "jcqq", False):
            macs += mod.num_features
    macs += N_USERS  # input normalisation
    return int(macs)


def static_memory_bytes(model):
    """Deployment storage of the inference model.
    FP32 model: every parameter and BN running statistic at 4 B.
    JCQQ: linear weights at w_bits, biases INT32, BN folded, rotation angles at phase_bits,
          one FP32 scale per quantized tensor and per activation quantizer.
    DQCP: only retained angles are stored, plus one mask bit per prunable gate.
    PEQA (merged): adds the two FP32 scale scalars s_enc, s_dec.
    Quantum-DQN: online + target networks are both stored (as in quantum_6metrics.py)."""
    if isinstance(model, QuantumDQNRef):
        return 2 * static_memory_bytes(model.online)
    if isinstance(model, (ParamMatchedMLP, QuantumPPORef)):
        b = sum(p.numel() * 4 for p in model.parameters())
        b += sum(bf.numel() * 4 for n_, bf in model.named_buffers() if "num_batches" not in n_)
        return int(b)
    m = model
    b = (m.h_mean.numel() + m.h_std.numel()) * 4
    lin = [m.enc_fc, m.enc_proj, m.dec_fc1, m.dec_fc2]
    if m.jcqq:
        wb = m.cfg["w_bits"] / 8.0
        b += sum(math.ceil(l.weight.numel() * wb) + l.bias.numel() * 4 + 4 for l in lin)
        b += 4 * 4                           # input / angle / expectation / hidden activation scales
        ang_b = m.cfg["phase_bits"] / 8.0
    else:
        b += sum((l.weight.numel() + l.bias.numel()) * 4 for l in lin)
        b += (m.enc_bn.weight.numel() + m.enc_bn.bias.numel() + m.enc_bn.running_mean.numel()
              + m.enc_bn.running_var.numel()) * 4
        ang_b = 4.0
    if m.dqcp:
        b += math.ceil(m.n_active_angles() * ang_b) + math.ceil((m.rot_mask.numel() + m.ent_mask.numel()) / 8)
    else:
        b += math.ceil(m.theta.numel() * ang_b)
    if m.peqa is not None:
        b += 2 * 4
    return int(b)


def working_memory_bytes(model):
    """Peak live inference memory (batch 1): static weights + largest pair of consecutive activations
    + one 2^n complex128 state vector held by the simulator."""
    acts = [N_USERS, HIDDEN, N_QUBITS, N_QUBITS, HIDDEN, N_USERS]
    peak_act = max(a + b for a, b in zip(acts[:-1], acts[1:])) * 4
    sv = (2 ** N_QUBITS) * 16 if len(model.ops()) else 0
    return int(static_memory_bytes(model) + peak_act + sv)


def circuits_per_update(model, mode, batch):
    """Circuit executions per optimisation update on hardware with parameter-shift gradients:
    per sample 1 forward + 2 per trainable circuit angle + 2 per encoding angle whenever any parameter
    upstream of the circuit (encoder / adapter / s_enc) is trainable.  Quantum-DQN adds the target pass."""
    if not len(model.ops()):
        return 0
    if isinstance(model, QuantumDQNRef):
        return batch * (1 + 2 * (model.online.n_active_angles() + N_QUBITS)) + batch
    n_ang = model.n_active_angles() if hasattr(model, "n_active_angles") else model.theta.numel()
    return batch * (1 + 2 * (n_ang + N_QUBITS))


def ops_to_qiskit(ops, n=N_QUBITS):
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    qc = QuantumCircuit(n)
    pmap = {}
    for g, wires, p in ops:
        if g == "CNOT":
            qc.cx(wires[0], wires[1])
            continue
        name = f"in_{p[1]}" if p[0] == "in" else f"w_{p[1]}_{p[2]}_{p[3]}"
        par = pmap.setdefault(name, Parameter(name))
        getattr(qc, g.lower())(par, wires[0])
    return qc


def transpile_stats(ops, coupling, seed=42, opt_level=1):
    """Transpiled depth / 2Q count in the native basis {rz, sx, x, cx} and SWAPs inserted by routing."""
    try:
        from qiskit import transpile
        from qiskit.transpiler import CouplingMap
    except ImportError as e:   # fail loudly: silent NaNs would end up as "N/A" in Fig. 5c / Table 3
        raise ImportError("Qiskit is required for transpiled depth / SWAP counts: pip install qiskit") from e
    qc = ops_to_qiskit(ops)
    cmap = None if coupling is None else CouplingMap(couplinglist=[list(e) for e in coupling] + [[b, a] for a, b in coupling])
    native = transpile(qc, basis_gates=["rz", "sx", "x", "cx"], coupling_map=cmap,
                       optimization_level=opt_level, seed_transpiler=seed)
    routed = transpile(qc, basis_gates=["rx", "ry", "rz", "cx", "swap"], coupling_map=cmap,
                       optimization_level=opt_level, seed_transpiler=seed)
    cn = native.count_ops()
    return dict(transpiled_depth=int(native.depth()), transpiled_2q=int(cn.get("cx", 0)),
                transpiled_1q=int(sum(v for k, v in cn.items() if k in ("rz", "sx", "x"))),
                swaps=int(routed.count_ops().get("swap", 0)))


def routed_ops(ops, coupling, seed=42):
    """Routed circuit (rx/ry/rz/cx/swap, parameters untouched) as a gate list + final virtual->physical map."""
    from qiskit import transpile
    from qiskit.transpiler import CouplingMap
    qc = ops_to_qiskit(ops)
    if coupling is None:
        return ops, list(range(N_QUBITS))
    cmap = CouplingMap(couplinglist=[list(e) for e in coupling] + [[b, a] for a, b in coupling])
    t = transpile(qc, basis_gates=["rx", "ry", "rz", "cx", "swap"], coupling_map=cmap, optimization_level=0,
                  layout_method="sabre", routing_method="sabre", seed_transpiler=seed)
    out = []
    for inst in t.data:
        name = inst.operation.name
        qs = tuple(t.find_bit(q).index for q in inst.qubits)
        if name == "cx":
            out.append(("CNOT", qs, None))
        elif name == "swap":
            out.append(("SWAP", qs, None))
        elif name in ("rx", "ry", "rz"):
            pname = str(inst.operation.params[0])
            parts = pname.split("_")
            ref = ("in", int(parts[1])) if parts[0] == "in" else ("w", int(parts[1]), int(parts[2]), int(parts[3]))
            out.append((name.upper(), qs, ref))
        elif name in ("barrier", "measure"):
            continue
        else:
            raise RuntimeError(f"unexpected gate after routing: {name}")
    final = t.layout.final_index_layout() if t.layout is not None else list(range(N_QUBITS))
    return out, list(final)


# =====================================================================================
# 8. Timing (single-sample inference latency, optimisation-step duration)
# =====================================================================================
def measure_latency(model, n_warmup=100, n_runs=1000, seed=0):
    model.eval()
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(1, N_USERS, generator=g) * 5.0
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        lat = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(x)
            lat.append((time.perf_counter() - t0) * 1e3)
    lat = np.array(lat)
    return dict(lat_mean_ms=float(lat.mean()), lat_p50_ms=float(np.percentile(lat, 50)),
                lat_p95_ms=float(np.percentile(lat, 95)), lat_std_ms=float(lat.std()))


def measure_update(model, mode="full", batch=128, n_warmup=5, n_steps=30, seed=0):
    """Wall-clock of one optimisation step: forward + loss.backward() + clip + optimizer.step().
    mode='full' : every (non-pruned) backbone parameter is trainable;  mode='peqa': PEQA parameters only."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(batch, N_USERS, generator=g) * 5.0
    y = (torch.rand(batch, N_USERS, generator=g) > 0.5).float()
    if mode == "peqa":
        params = list(model.peqa.parameters())
    else:
        params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-4)
    model.train()
    ts = []
    for s in range(n_warmup + n_steps):
        t0 = time.perf_counter()
        loss = model.train_loss(x, y) if hasattr(model, "train_loss") else F.binary_cross_entropy_with_logits(model(x), y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 0.5)
        opt.step()
        if s >= n_warmup:
            ts.append(time.perf_counter() - t0)
    ts = np.array(ts)
    return dict(update_ms=float(ts.mean() * 1e3), update_std_ms=float(ts.std() * 1e3),
                energy_per_update_J=float(EDGE_POWER_W * ts.mean()), n_trainable=int(sum(p.numel() for p in params)))


def peak_rss_mb():
    try:
        import psutil
        mi = psutil.Process(os.getpid()).memory_info()
        return float(getattr(mi, "peak_wset", mi.rss)) / 2 ** 20
    except Exception:
        return float("nan")


# =====================================================================================
# 9. Torch density-matrix simulator (finite shots, noise channels, routed circuits)
# =====================================================================================
_PAULI_FLIP = None


def _rot(g, th):
    c, s = torch.cos(th / 2), torch.sin(th / 2)
    z = torch.zeros_like(c)
    if g == "RX":
        return torch.stack([torch.stack([c + 0j, -1j * s], -1), torch.stack([-1j * s, c + 0j], -1)], -2)
    if g == "RY":
        return torch.stack([torch.stack([c + 0j, -s + 0j], -1), torch.stack([s + 0j, c + 0j], -1)], -2)
    if g == "RZ":
        e = torch.exp(-0.5j * th.to(torch.complex128))
        return torch.stack([torch.stack([e, z + 0j], -1), torch.stack([z + 0j, e.conj()], -1)], -2)
    raise ValueError(g)


class DensityMatrixSim:
    """Batched exact density-matrix simulation for n<=8 qubits. Wire 0 is the most significant bit
    (PennyLane convention).  Noise model (applied after every gate on the qubits it acts on):
      p1  : one-qubit depolarizing  rho -> (1-p)rho + p/3 (XrhoX + YrhoY + ZrhoZ)
      p2  : two-qubit depolarizing  rho -> (1-p)rho + p/15 sum_{P != II} P rho P   (after each CNOT/SWAP-CNOT)
      gamma: amplitude damping on every qubit touched by a gate
      ro  : symmetric readout bit-flip probability (applied at measurement)."""

    def __init__(self, n=N_QUBITS):
        self.n = n
        self.D = 2 ** n
        idx = np.arange(self.D)
        self.bits = torch.tensor(np.array([((idx >> (n - 1 - q)) & 1) for q in range(n)]), dtype=torch.float64)  # (n, D)
        self._perm = {}

    def _cnot_perm(self, c, t):
        key = ("c", c, t)
        if key not in self._perm:
            idx = np.arange(self.D)
            cb = (idx >> (self.n - 1 - c)) & 1
            self._perm[key] = torch.tensor(np.where(cb == 1, idx ^ (1 << (self.n - 1 - t)), idx))
        return self._perm[key]

    def _apply_perm(self, rho, P):
        return rho[:, P][:, :, P]

    def _apply_1q(self, rho, U, q):
        B, D = rho.shape[0], self.D
        a, c = 2 ** q, 2 ** (self.n - q - 1)
        if U.dim() == 2:
            U = U.unsqueeze(0).expand(B, 2, 2)
        r = rho.reshape(B, a, 2, c, D)
        r = torch.einsum("bij,bajcd->baicd", U, r).reshape(B, D, D)
        r = r.reshape(B, D, a, 2, c)
        r = torch.einsum("bdajc,bij->bdaic", r, U.conj()).reshape(B, D, D)
        return r

    def _depolarize(self, rho, qubits, lam):
        """rho -> (1-lam) rho + lam * Tr_Q(rho) (x) I_Q / 2^|Q|."""
        if lam <= 0:
            return rho
        B, n = rho.shape[0], self.n
        R = rho.reshape((B,) + (2,) * (2 * n))
        letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVW"
        rows = list(letters[:n]); cols = list(letters[n:2 * n])
        new_r, new_c = list(rows), list(cols)
        extra = []
        k = 2 * n
        for q in qubits:
            cols_in = cols
            cols_in[q] = rows[q]
            xr, yc = letters[k], letters[k + 1]; k += 2
            new_r[q], new_c[q] = xr, yc
            extra.append(xr + yc)
        eq = "Z" + "".join(rows) + "".join(cols_in) + "," + ",".join(extra) + "->Z" + "".join(new_r) + "".join(new_c)
        eye = torch.eye(2, dtype=rho.dtype)
        red = torch.einsum(eq, R, *([eye] * len(qubits))).reshape(B, self.D, self.D) / (2 ** len(qubits))
        return (1 - lam) * rho + lam * red

    def _amp_damp(self, rho, q, gamma):
        if gamma <= 0:
            return rho
        K0 = torch.tensor([[1, 0], [0, math.sqrt(1 - gamma)]], dtype=torch.complex128)
        K1 = torch.tensor([[0, math.sqrt(gamma)], [0, 0]], dtype=torch.complex128)
        return self._apply_1q(rho, K0, q) + self._apply_1q(rho, K1, q)

    def _noise(self, rho, wires, noise):
        if not noise:
            return rho
        if len(wires) == 1:
            rho = self._depolarize(rho, [wires[0]], 4.0 / 3.0 * noise.get("p1", 0.0))
        else:
            rho = self._depolarize(rho, list(wires), 16.0 / 15.0 * noise.get("p2", 0.0))
        for w in wires:
            rho = self._amp_damp(rho, w, noise.get("gamma", 0.0))
        return rho

    @torch.no_grad()
    def probs(self, ops, inputs, weights, noise=None):
        inputs = inputs.to(torch.float64)
        weights = weights.to(torch.float64)
        B = inputs.shape[0]
        rho = torch.zeros(B, self.D, self.D, dtype=torch.complex128)
        rho[:, 0, 0] = 1.0
        for g, wires, p in ops:
            if g == "CNOT":
                rho = self._apply_perm(rho, self._cnot_perm(*wires))
                rho = self._noise(rho, wires, noise)
            elif g == "SWAP":
                a, b = wires
                for c, t in ((a, b), (b, a), (a, b)):
                    rho = self._apply_perm(rho, self._cnot_perm(c, t))
                    rho = self._noise(rho, (c, t), noise)
            else:
                th = inputs[:, p[1]] if p[0] == "in" else weights[p[1], p[2], p[3]].expand(B)
                rho = self._apply_1q(rho, _rot(g, th), wires[0])
                rho = self._noise(rho, wires, noise)
        pr = torch.diagonal(rho, dim1=1, dim2=2).real.clamp_min(0)
        return pr / pr.sum(-1, keepdim=True)

    def expz_from_probs(self, pr, layout=None, readout=0.0):
        z = 1.0 - 2.0 * self.bits                      # (n, D)
        e = pr @ z.T                                   # (B, n) physical qubits
        if layout is not None:
            e = e[:, layout]
        return (1.0 - 2.0 * readout) * e

    def sample_expz(self, pr, shots, gen, layout=None, readout=0.0):
        counts = torch.distributions.Multinomial(total_count=int(shots), probs=pr).sample()
        if readout > 0:
            # exact symmetric readout flips on every shot: E[Z_obs] = (1-2p) Z ; sample flips per qubit
            bits = self.bits.T.unsqueeze(0)                         # (1, D, n)
            ones = (counts.unsqueeze(-1) * bits).sum(1)             # (B, n) number of shots with bit=1
            zeros = shots - ones
            flip_1to0 = torch.distributions.Binomial(ones, torch.full_like(ones, readout)).sample()
            flip_0to1 = torch.distributions.Binomial(zeros, torch.full_like(zeros, readout)).sample()
            ones_obs = ones - flip_1to0 + flip_0to1
            e = 1.0 - 2.0 * ones_obs / shots
        else:
            e = (counts @ (1.0 - 2.0 * self.bits).T) / shots
        if layout is not None:
            e = e[:, layout]
        return e


def checkpoint_paths(root, D, J, seed):
    base = os.path.join(root, "checkpoints", f"D{int(D)}J{int(J)}_seed{seed}")
    return dict(pre=base + "_pretrained.pt", P0=base + "_adaptP0.pt", P1=base + "_adaptP1.pt")


def save_model(model, path, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(dict(state=model.state_dict(), dqcp=model.dqcp, jcqq=model.jcqq, cfg=model.cfg,
                    h_mean=model.h_mean.numpy(), h_std=model.h_std.numpy(),
                    peqa=model.peqa is not None, peqa_merged=model.peqa_merged, extra=extra or {}), path)


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = HybridPolicy(ck["h_mean"], ck["h_std"], dqcp=ck["dqcp"], jcqq=ck["jcqq"], cfg=ck["cfg"])
    if ck["peqa"]:
        m.enable_peqa()
    m.load_state_dict(ck["state"])
    m.peqa_merged = ck["peqa_merged"]
    m.refresh_ops()
    return m, ck.get("extra", {})
