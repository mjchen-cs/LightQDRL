# -*- coding: utf-8 -*-
"""
qrl_core.py — plug-and-play compression framework for hybrid quantum reinforcement-learning agents.

Contents
  * TorchSV            exact batched state-vector simulator in PyTorch (autograd, double backward for TRPO);
                       validated against PennyLane default.qubit at import time of the test script
  * HybridNet          shared hybrid backbone: classical encoder -> n-qubit VQC (RX,RY,RZ + CNOT ring) -> heads.
                       core = 'vqc' (quantum) | 'tanh' (quantum-removal control) | 'mlp' (classical control)
  * Compression        LightQDRL framework, applicable to ANY agent that uses HybridNet:
                         DQCP  dynamic quantum circuit pruning (EMA |theta*grad| for rotations,
                               loss-removal sensitivity for CNOTs, exponential sparsity schedule)
                         JCQQ  joint classical-quantum quantization (INT8 weights/activations with STE,
                               8-bit phase grid for encoding and variational angles, INT8 expectation values)
                         PEQA  parameter-efficient adaptation (rank-r adapters on angle projection and on the last
                               layer of every head, offsets of retained angles, two scale scalars)
  * Resource accounting (parameters, static memory, MACs, gate counts, depth)

Gate list, Qiskit transpilation and noise simulation are shared with lqdrl_core.py.
"""
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lqdrl_core as LC        # build_ops, gate_counts, logical_depth, transpile_stats, quantizers

N_QUBITS = 6
N_LAYERS = 2
HIDDEN = 64

COMPRESS_DEFAULT = dict(
    w_bits=8, a_bits=8, phase_bits=8, clip_C=3.0,
    rho_rot=0.5, rho_ent=0.5, prune_start_frac=0.2, prune_end_frac=0.7, prune_every=20, saliency_beta=0.9,
    lora_rank=2, lora_alpha=2.0,
)


# =====================================================================================
# Exact batched state-vector simulator (differentiable)
# =====================================================================================
def _rot(g, th):
    th = th.to(torch.float64)
    c, s = torch.cos(th / 2), torch.sin(th / 2)
    z = torch.zeros_like(c)
    if g == "RX":
        return torch.stack([torch.stack([torch.complex(c, z), torch.complex(z, -s)], -1),
                            torch.stack([torch.complex(z, -s), torch.complex(c, z)], -1)], -2)
    if g == "RY":
        return torch.stack([torch.stack([torch.complex(c, z), torch.complex(-s, z)], -1),
                            torch.stack([torch.complex(s, z), torch.complex(c, z)], -1)], -2)
    if g == "RZ":
        return torch.stack([torch.stack([torch.complex(c, -s), torch.complex(z, z)], -1),
                            torch.stack([torch.complex(z, z), torch.complex(c, s)], -1)], -2)
    raise ValueError(g)


class TorchSV:
    """Wire 0 = most significant bit (PennyLane convention).  run(ops, inputs (B,n), weights (L,n,3)) -> <Z> (B,n)."""

    def __init__(self, n=N_QUBITS):
        self.n, self.D = n, 2 ** n
        idx = np.arange(self.D)
        self.zsign = torch.tensor(np.array([1 - 2 * ((idx >> (n - 1 - q)) & 1) for q in range(n)]),
                                  dtype=torch.float64)            # (n, D)
        self._perm = {}

    def _cnot(self, c, t):
        k = (c, t)
        if k not in self._perm:
            idx = np.arange(self.D)
            cb = (idx >> (self.n - 1 - c)) & 1
            self._perm[k] = torch.tensor(np.where(cb == 1, idx ^ (1 << (self.n - 1 - t)), idx))
        return self._perm[k]

    def run(self, ops, inputs, weights):
        B = inputs.shape[0]
        n = self.n
        psi = torch.zeros(B, self.D, dtype=torch.complex128)
        psi[:, 0] = 1.0
        for g, wires, p in ops:
            if g == "CNOT":
                psi = psi[:, self._cnot(*wires)]
                continue
            th = inputs[:, p[1]] if p[0] == "in" else weights[p[1], p[2], p[3]].expand(B)
            U = _rot(g, th)                                       # (B,2,2)
            q = wires[0]
            a, c = 2 ** q, 2 ** (n - q - 1)
            psi = torch.einsum("bij,bajc->baic", U, psi.reshape(B, a, 2, c)).reshape(B, self.D)
        prob = psi.real ** 2 + psi.imag ** 2
        return (prob @ self.zsign.T).to(torch.float32)


SV = TorchSV()


# =====================================================================================
# Hybrid backbone with compression switches
# =====================================================================================
class Head(nn.Module):
    def __init__(self, d_in, d_out, hidden=HIDDEN):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden)
        self.fc2 = nn.Linear(hidden, d_out)


class HybridNet(nn.Module):
    """encoder (obs->64, LayerNorm, ReLU, 64->n_q, pi*tanh) -> core -> heads (n_q->64->out).
    heads: dict name -> output size, e.g. {'q': nb*na} (DQN), {'adv': nb*na, 'val': 1} (Dueling), {'pi': nb*na}."""

    def __init__(self, obs_dim, heads, core="vqc", compress=False, cfg=None, n_qubits=N_QUBITS, n_layers=N_LAYERS):
        super().__init__()
        self.cfg = dict(COMPRESS_DEFAULT, **(cfg or {}))
        # compress: False | True (DQCP + JCQQ = LightQDRL) | "dqcp" | "jcqq"  (single components, factorial study)
        self.core = core
        self.compress_spec = compress if isinstance(compress, str) else bool(compress)
        self.compress = bool(compress)
        self.dqcp = (compress is True or compress == "dqcp") and core == "vqc"
        self.jcqq = compress is True or compress == "jcqq"             # quantization applies to every core
        self.n, self.L = n_qubits, n_layers
        self.enc_fc = nn.Linear(obs_dim, HIDDEN)
        self.enc_ln = nn.LayerNorm(HIDDEN)
        self.enc_proj = nn.Linear(HIDDEN, n_qubits)
        if core == "vqc":
            self.theta = nn.Parameter(torch.rand(n_layers, n_qubits, 3) * 2 * math.pi)
        elif core == "tanh":
            self.Wq = nn.Linear(n_qubits, n_qubits, bias=False)
        elif core == "mlp":
            pass
        else:
            raise ValueError(core)
        self.heads = nn.ModuleDict({k: Head(n_qubits, v) for k, v in heads.items()})
        self.register_buffer("rot_mask", torch.ones(n_layers, n_qubits, 3))
        self.register_buffer("ent_mask", torch.ones(n_layers, n_qubits))
        self.register_buffer("sal_rot", torch.zeros(n_layers, n_qubits, 3))
        self.register_buffer("sal_ent", torch.zeros(n_layers, n_qubits))
        if core == "vqc":
            self.theta.register_hook(lambda g: g * self.rot_mask)
        self.peqa = None
        self.peqa_merged = False
        self.refresh_ops()

    # ------------------------------------------------------------------ helpers
    def refresh_ops(self):
        self._ops = LC.build_ops(self.n, self.L, self.rot_mask.cpu().numpy(), self.ent_mask.cpu().numpy()) \
            if self.core == "vqc" else []

    def ops(self):
        return self._ops

    def n_active_angles(self):
        return int(self.rot_mask.sum().item()) if self.core == "vqc" else 0

    def base_parameters(self):
        return [p for n_, p in self.named_parameters() if not n_.startswith("peqa")]

    def n_base_params(self):
        return int(sum(p.numel() for p in self.base_parameters()))

    def _w(self, w):
        return LC.fq_weight_sym(w, self.cfg["w_bits"]) if self.jcqq else w

    def _a(self, x):
        return LC.fq_act_unsigned(x, self.cfg["a_bits"]) if self.jcqq else x

    def _lin(self, layer, x, lora_key=None):
        W = layer.weight
        if lora_key is not None and self.peqa is not None and not self.peqa_merged:
            W = W + self.lora_scale * self.peqa[f"B_{lora_key}"] @ self.peqa[f"A_{lora_key}"]
        return F.linear(x, self._w(W), layer.bias)

    # ------------------------------------------------------------------ PEQA
    def enable_peqa(self, rank=None, alpha=None):
        r = int(rank or self.cfg["lora_rank"])
        self.lora_scale = float(alpha or self.cfg["lora_alpha"]) / r
        P = {"A_enc": nn.Parameter(torch.randn(r, HIDDEN) / math.sqrt(HIDDEN)),
             "B_enc": nn.Parameter(torch.zeros(self.n, r)),
             "s_enc": nn.Parameter(torch.ones(1)), "s_dec": nn.Parameter(torch.ones(1))}
        for k, h in self.heads.items():
            P[f"A_{k}"] = nn.Parameter(torch.randn(r, HIDDEN) / math.sqrt(HIDDEN))
            P[f"B_{k}"] = nn.Parameter(torch.zeros(h.fc2.out_features, r))
        if self.core == "vqc":
            P["dtheta"] = nn.Parameter(torch.zeros(self.L, self.n, 3))
        self.peqa = nn.ParameterDict(P)
        if self.core == "vqc":
            self.peqa["dtheta"].register_hook(lambda g: g * self.rot_mask)
        for p in self.base_parameters():
            p.requires_grad_(False)
        self.peqa_merged = False

    def peqa_breakdown(self):
        if self.peqa is None:
            return {}
        P = self.peqa
        out = {"enc_adapter": P["A_enc"].numel() + P["B_enc"].numel(),
               "head_adapters": sum(P[f"A_{k}"].numel() + P[f"B_{k}"].numel() for k in self.heads),
               "active_angles": self.n_active_angles(), "scales": 2}
        return out

    @torch.no_grad()
    def merge_peqa(self):
        if self.peqa is None or self.peqa_merged:
            return
        P = self.peqa
        self.enc_proj.weight += self.lora_scale * P["B_enc"] @ P["A_enc"]
        for k, h in self.heads.items():
            h.fc2.weight += self.lora_scale * P[f"B_{k}"] @ P[f"A_{k}"]
        if self.core == "vqc":
            self.theta += self.rot_mask * P["dtheta"]
        for k, v in P.items():
            if k.startswith(("A_", "B_")) or k == "dtheta":
                v.zero_()
        self.peqa_merged = True

    # ------------------------------------------------------------------ forward
    def pre_circuit(self, x):
        """Encoder: returns (encoding angles (B,n), effective variational angles (L,n,3)) — the circuit inputs."""
        C = self.cfg["clip_C"]
        z = torch.clamp(x, -C, C)
        if self.jcqq:
            z = LC.fq_fixed_sym(z, 8, 2 * C / 255.0)
        a = self._a(F.relu(self.enc_ln(self._lin(self.enc_fc, z))))
        ang = torch.tanh(self._lin(self.enc_proj, a, "enc")) * math.pi
        if self.peqa is not None:
            ang = torch.clamp(ang * self.peqa["s_enc"], -math.pi, math.pi)
        th = None
        if self.core == "vqc":
            if self.jcqq:
                ang = LC.fq_phase(ang, self.cfg["phase_bits"])
            th = self.theta
            if self.peqa is not None and not self.peqa_merged:
                th = th + self.rot_mask * self.peqa["dtheta"]
            if self.jcqq:
                th = LC.fq_phase(th, self.cfg["phase_bits"])
        return ang, th

    def post_circuit(self, e):
        """Measured <Z> (or classical core output) -> decoder features."""
        if self.core == "vqc" and self.jcqq:
            e = LC.fq_fixed_sym(e, 8, 2.0 / 255.0)
        if self.peqa is not None:
            e = e * self.peqa["s_dec"]
        return e

    def features(self, x, expz_fn=None):
        ang, th = self.pre_circuit(x)
        if self.core == "vqc":
            e = expz_fn(ang, th) if expz_fn is not None else SV.run(self._ops, ang, th)
        elif self.core == "tanh":
            e = torch.tanh(self._lin(self.Wq, ang))
        else:
            e = ang / math.pi
        return self.post_circuit(e)

    def heads_from_features(self, e):
        return {k: self.head(k, e) for k in self.heads}

    def head(self, name, e):
        h = self.heads[name]
        z = self._a(F.relu(self._lin(h.fc1, e)))
        return self._lin(h.fc2, z, name)

    def forward(self, x):
        e = self.features(x)
        return {k: self.head(k, e) for k in self.heads}


# =====================================================================================
# DQCP scheduler (algorithm-agnostic: the agent supplies a no-grad loss closure)
# =====================================================================================
class DQCP:
    def __init__(self, net, expected_updates):
        c = net.cfg
        self.net = net
        self.u0 = int(c["prune_start_frac"] * expected_updates)
        self.u1 = max(self.u0 + 1, int(c["prune_end_frac"] * expected_updates))
        self.tau = max(1.0, (self.u1 - self.u0) / 5.0)
        self.every, self.beta = c["prune_every"], c["saliency_beta"]
        self.u = 0
        self.extra_evals = 0
        self.history = []

    def rho(self, u, rho_f):
        if u < self.u0:
            return 0.0
        if u >= self.u1:
            return rho_f
        return rho_f * (1 - math.exp(-(u - self.u0) / self.tau)) / (1 - math.exp(-5.0))

    @torch.no_grad()
    def record(self, theta_grad=None):
        """Call once per optimizer update, after gradients are available."""
        g = theta_grad if theta_grad is not None else self.net.theta.grad
        if g is not None:
            s = (self.net.theta * g).abs() * self.net.rot_mask
            self.net.sal_rot.mul_(self.beta).add_((1 - self.beta) * s)

    @torch.no_grad()
    def _prune(self, mask, sal, rho):
        total = mask.numel()
        target = int(round(rho * total))
        done = int(total - mask.sum().item())
        if target <= done:
            return
        s = sal.flatten().clone()
        m = mask.flatten()
        s[m < 0.5] = float("inf")
        for i in torch.argsort(s)[: target - done]:
            m[i] = 0.0
        mask.copy_(m.view_as(mask))

    @torch.no_grad()
    def step(self, loss_fn, batch_size):
        """loss_fn(): scalar loss of the current batch with the current masks (no grad)."""
        self.u += 1
        u = self.u
        if u < self.u0 or u > self.u1 or (u % self.every and u != self.u1):
            return
        net = self.net
        base = float(loss_fn())
        self.extra_evals += batch_size
        for l in range(net.L):
            for q in range(net.n):
                if net.ent_mask[l, q] < 0.5:
                    continue
                net.ent_mask[l, q] = 0.0
                net.refresh_ops()
                lk = float(loss_fn())
                self.extra_evals += batch_size
                net.ent_mask[l, q] = 1.0
                net.sal_ent[l, q] = self.beta * net.sal_ent[l, q] + (1 - self.beta) * (lk - base)
        c = net.cfg
        self._prune(net.rot_mask, net.sal_rot, self.rho(u, c["rho_rot"]))
        self._prune(net.ent_mask, net.sal_ent, self.rho(u, c["rho_ent"]))
        net.refresh_ops()
        self.history.append(dict(update=u, active_rot=int(net.rot_mask.sum()), active_cnot=int(net.ent_mask.sum())))


# =====================================================================================
# Resource accounting of a deployed HybridNet
# =====================================================================================
def static_memory_bytes(net):
    """FP32: all parameters + LayerNorm affine at 4 B.  JCQQ: weights at w_bits, biases INT32, LayerNorm affine FP32,
    one FP32 scale per quantized tensor and activation quantizer, angles at phase_bits.  DQCP: retained angles
    + 1 mask bit per prunable gate.  PEQA (merged): + 2 FP32 scales."""
    lin = [m for m in net.modules() if isinstance(m, nn.Linear)]
    b = 0
    if net.jcqq:
        wb = net.cfg["w_bits"] / 8
        b += sum(math.ceil(l.weight.numel() * wb) + (l.bias.numel() * 4 if l.bias is not None else 0) + 4 for l in lin)
        b += 4 * (2 + 2 * len(net.heads))
    else:
        b += sum((l.weight.numel() + (l.bias.numel() if l.bias is not None else 0)) * 4 for l in lin)
    b += (net.enc_ln.weight.numel() + net.enc_ln.bias.numel()) * 4
    if net.core == "vqc":
        ab = net.cfg["phase_bits"] / 8 if net.jcqq else 4
        if net.dqcp:
            b += math.ceil(net.n_active_angles() * ab) + math.ceil((net.rot_mask.numel() + net.ent_mask.numel()) / 8)
        else:
            b += math.ceil(net.theta.numel() * ab)
    if net.peqa is not None:
        b += 8
    return int(b)


def classical_macs(net):
    return int(sum(m.in_features * m.out_features for m in net.modules() if isinstance(m, nn.Linear))
               + net.enc_ln.normalized_shape[0] * 2)


def accounting(net, transpile=False):
    ops = net.ops()
    n1, n2 = LC.gate_counts(ops) if ops else (0, 0)
    out = dict(params=net.n_base_params(),
               trainable_after_pruning=net.n_base_params() - ((net.theta.numel() - net.n_active_angles()) if net.core == "vqc" else 0),
               static_mem_bytes=static_memory_bytes(net), classical_macs=classical_macs(net),
               n_1q=n1, n_2q=n2, logical_depth=LC.logical_depth(ops) if ops else 0,
               active_angles=net.n_active_angles())
    if transpile and ops:
        out.update(LC.transpile_stats(ops, LC.TOPOLOGIES["Heavy-hex (6Q)"]))
    return out
