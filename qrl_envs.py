# -*- coding: utf-8 -*-
"""
qrl_envs.py — environments for the plug-and-play compression benchmark.

MECQueueEnv  (scenario A, main benchmark; version 2)
    Multi-step, queue-aware wireless-powered mobile-edge computing in the spirit of LyDROO (Bi et al., IEEE TWC 2021).
    * N = 10 devices, 1-s frames, episodes of 200 frames.  Channels h_i(t) = g_i |c_i(t)|^2, per-device mean gain g_i
      from data_10.mat, Gauss-Markov fading c(t+1) = rho c(t) + sqrt(1-rho^2) w, rho = 0.95.
    * Frame structure: the AP first transfers energy for a fixed fraction a = 0.3 of the frame; the remaining 1-a is
      shared equally (TDMA) by the devices that offload.  Rates use the same physical constants as optimization.py:
        local    r_i = eta1 (h_i/k)^(1/3) a^(1/3)
        offload  r_i = (B/(v ln2)) tau log(1 + eta2 h_i^2 a / tau),   tau = (1-a)/n_off
      so the decisions interact only through the number of offloading devices (learnable coupling).
    * Arrivals: device i receives a task with probability p = 0.3; sizes are the empirical video-clip sizes
      (data/video_task_sizes.csv), scaled so that device i's mean arrival rate is load x s_i, where s_i is its mean
      service rate under the sum-rate-optimal decision (fixed reference fading draws, nominal channel).  load = 0.5.
    * Queues Q_i <- min(Qmax_i, Q_i - served_i + A_i), Qmax_i = 200 lambda_i; overflow is dropped.
    * Observation (30-dim): [log(1 + h_i/g_i), Q_i/Q_ref, A_i/A_ref].
    * Reward (default 'drift', per-frame drift-plus-penalty surrogate as in LyDROO):
        r = sum_i w_i served_i / C_ref - kappa dropped / C_ref,   w_i = Q_i / mean(Q) clipped to [0.1, 10]
      ('backlog' = served/C_ref - beta mean(Q)/Q_ref - kappa dropped/C_ref is kept as an option).
    * Distribution shifts for adaptation: load_scale (traffic surge) multiplies arrivals, channel_scale multiplies g_i
      (channel deterioration); arrival rates stay those of the nominal system.
    Reference policies: random, all-local, all-offload, max-rate (sum-rate optimal, queue-blind) and myopic
    max-weight (exhaustive search over all 2^10 decisions of sum_i w_i min(Q_i, r_i)), used to normalize scores:
    0 = random, 1 = myopic max-weight.

ChannelAccessEnv (scenario D2): dynamic multichannel access with Gilbert-Elliott channels (replaces Acrobot-v1, on
which no algorithm learned).  GymEnv (scenario D1) wraps gymnasium CartPole-v1 (and Acrobot-v1, kept for reference).
"""
import os
import math
import warnings
import itertools
import numpy as np
import scipy.io as sio

from optimization import bisection

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# constants of optimization.py (needed to recover per-device rates from the bisection solution)
_o, _p, _u, _ki, _B, _Vu = 100, 3, 0.7, 1e-26, 2e6, 1.1
_ETA1 = ((_u * _p) ** (1.0 / 3)) / _o
_ETA2 = _u * _p / 1e-10
_EPS = _B / (_Vu * np.log(2))


def per_device_rates(h, m, w):
    """Solve the frame-level resource allocation for decision m and weights w; return per-device rates (bit/s)."""
    m = np.asarray(m).astype(int)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        _, a, tau = bisection(h, m, list(w))
    r = np.zeros(len(h))
    off = np.where(m == 1)[0]
    loc = np.where(m == 0)[0]
    r[loc] = _ETA1 * (h[loc] / _ki) ** (1.0 / 3) * a ** (1.0 / 3)
    for j, i in enumerate(off):
        t = max(tau[j], 1e-12)
        r[i] = _EPS * t * np.log(1 + _ETA2 * h[i] ** 2 * a / t)
    return r


def load_video_sizes():
    """Video file sizes (bytes -> bits).  The CSV is read explicitly as UTF-8 because file names may contain
    Chinese characters (Windows would otherwise use the GBK locale codec)."""
    import csv
    f = os.path.join(DATA_DIR, "video_task_sizes.csv")
    with open(f, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    sizes = np.array([float(r[-1]) for r in rows[1:] if r and r[-1].strip()], dtype=np.float64)
    if sizes.size == 0:
        raise RuntimeError(f"no video sizes found in {f}; run: python make_video_sizes.py \"task data\"")
    return sizes[sizes > 0] * 8.0          # bits


class MECQueueEnvV1:
    """Earlier version (bisection-optimized WPT/TDMA, load relative to the bisection capacity).  Kept only to reproduce
    results obtained with it (e.g. the first hardware run); its agents did not learn -- use MECQueueEnv."""
    branches = [2] * 10
    obs_dim = 30

    def __init__(self, seed=0, N=10, T_ep=200, load=0.4, p_arrival=0.3, rho=0.95, beta=0.5, kappa=1.0,
                 qmax_factor=50.0, channel_scale=1.0, load_scale=1.0):
        self.N, self.T_ep, self.p = N, T_ep, p_arrival
        self.rho, self.beta, self.kappa = rho, beta, kappa
        self.rng = np.random.default_rng(seed)
        d = sio.loadmat(os.path.join(DATA_DIR, "data_10.mat"))
        self.g = d["input_h"].mean(0) * channel_scale            # per-device mean gain
        self.C_ref = float(d["output_obj"].mean() / np.mean([1.0, 1.5]))   # mean capacity (bit/s), unit weights
        sizes = load_video_sizes()
        lam_dev = load * load_scale * self.C_ref / N             # mean arrival per device (bit/frame)
        self.sizes = sizes / sizes.mean() * (lam_dev / p_arrival)
        self.A_ref = lam_dev / p_arrival
        self.Q_ref = 10.0 * lam_dev
        self.Qmax = qmax_factor * lam_dev
        self.reset()

    # ------------------------------------------------------------------
    def _arrivals(self):
        hit = self.rng.random(self.N) < self.p
        return np.where(hit, self.rng.choice(self.sizes, self.N), 0.0)

    def _obs(self):
        hn = self.h / self.g
        return np.concatenate([np.log1p(hn), self.Q / self.Q_ref, self.A / self.A_ref]).astype(np.float32)

    def weights(self):
        q = self.Q / self.Q_ref
        w = q / q.mean() if q.sum() > 1e-12 else np.ones(self.N)
        return np.clip(w, 0.1, 10.0)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.c = (self.rng.normal(size=self.N) + 1j * self.rng.normal(size=self.N)) / np.sqrt(2)
        self.h = self.g * np.abs(self.c) ** 2
        self.Q = self.rng.uniform(0, 1, self.N) * self.Q_ref * 0.2
        self.A = self._arrivals()
        self.t = 0
        return self._obs()

    def step(self, m):
        m = np.asarray(m).astype(int)
        r = per_device_rates(self.h, m, self.weights())
        served = np.minimum(self.Q, r * 1.0)
        Q = self.Q - served + self.A
        dropped = np.maximum(Q - self.Qmax, 0.0)
        self.Q = np.minimum(Q, self.Qmax)
        info = dict(served=served.sum(), backlog=self.Q.mean(), dropped=dropped.sum())
        rew = served.sum() / self.C_ref - self.beta * self.Q.mean() / self.Q_ref - self.kappa * dropped.sum() / self.C_ref
        w = (self.rng.normal(size=self.N) + 1j * self.rng.normal(size=self.N)) / np.sqrt(2)
        self.c = self.rho * self.c + np.sqrt(1 - self.rho ** 2) * w
        self.h = self.g * np.abs(self.c) ** 2
        self.A = self._arrivals()
        self.t += 1
        done = self.t >= self.T_ep
        return self._obs(), float(rew), done, info


def reference_action_v1(env, kind, rng):
    N = env.N
    if kind == "random":
        return rng.integers(0, 2, N)
    if kind == "local":
        return np.zeros(N, int)
    if kind == "offload":
        return np.ones(N, int)
    if kind == "myopic":
        w = env.weights()
        order = np.argsort(-w * env.h)
        best, best_m = -1, None
        for k in range(1, N + 1):
            m = np.zeros(N, int)
            m[order[:k]] = 1
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                v = bisection(env.h, m, list(w))[0]
            if v > best:
                best, best_m = v, m
        return best_m
    raise ValueError(kind)


_ALL_M = np.array(list(itertools.product([0, 1], repeat=10)), dtype=np.int64)     # all 2^10 decisions


class MECQueueEnv:
    branches = [2] * 10
    obs_dim = 30

    def __init__(self, seed=0, N=10, T_ep=200, load=0.5, p_arrival=0.3, rho=0.95, beta=0.5, kappa=1.0,
                 qmax_factor=200.0, a_wpt=0.3, channel_scale=1.0, load_scale=1.0, reward_mode="drift"):
        assert N == 10
        self.N, self.T_ep, self.p = N, T_ep, p_arrival
        self.rho, self.beta, self.kappa, self.a = rho, beta, kappa, a_wpt
        self.reward_mode = reward_mode
        d = sio.loadmat(os.path.join(DATA_DIR, "data_10.mat"))
        g_nom = d["input_h"].mean(0).astype(np.float64)
        # nominal per-device service shares under the sum-rate-optimal decision (fixed reference draws)
        self.g = g_nom
        gen = np.random.default_rng(999)
        share = np.zeros(N)
        for _ in range(300):
            c = (gen.normal(size=N) + 1j * gen.normal(size=N)) / np.sqrt(2)
            self.h = g_nom * np.abs(c) ** 2
            R = self.rates_all(_ALL_M)
            share += R[np.argmax(R.sum(1))]
        share /= 300
        self.C_ref = float(share.sum())
        self.lam = load * share                                   # nominal mean arrival per device (bit/frame)
        sizes = load_video_sizes()
        self.sizes = sizes / sizes.mean()
        self.load_scale = load_scale
        self.A_ref = float(self.lam.mean() / p_arrival)
        self.Q_ref = 10.0 * float(self.lam.mean())
        self.Qmax = qmax_factor * self.lam
        self.g = g_nom * channel_scale                            # (possibly shifted) channel
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------------ physics
    def rates_all(self, M):
        """Per-device rates (bit/frame) for a batch of decisions M (K, N) in the current channel state."""
        h, a = self.h, self.a
        r_loc = _ETA1 * (h / _ki) ** (1.0 / 3) * a ** (1.0 / 3)
        n = M.sum(-1, keepdims=True)
        tau = np.where(n > 0, (1 - a) / np.maximum(n, 1), 1.0)
        r_off = _EPS * tau * np.log1p(_ETA2 * h ** 2 * a / tau)
        return np.where(M == 1, r_off, r_loc)

    def rates(self, m):
        return self.rates_all(np.asarray(m, dtype=np.int64)[None])[0]

    def _arrivals(self):
        hit = self.rng.random(self.N) < self.p
        return np.where(hit, self.rng.choice(self.sizes, self.N) * self.lam * self.load_scale / self.p, 0.0)

    def _obs(self):
        hn = self.h / self.g
        return np.concatenate([np.log1p(hn), self.Q / self.Q_ref, self.A / self.A_ref]).astype(np.float32)

    def weights(self):
        q = self.Q / self.Q_ref
        w = q / q.mean() if q.sum() > 1e-12 else np.ones(self.N)
        return np.clip(w, 0.1, 10.0)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.c = (self.rng.normal(size=self.N) + 1j * self.rng.normal(size=self.N)) / np.sqrt(2)
        self.h = self.g * np.abs(self.c) ** 2
        self.Q = self.rng.uniform(0, 1, self.N) * self.Q_ref * 0.2
        self.A = self._arrivals()
        self.t = 0
        return self._obs()

    def step(self, m):
        m = np.asarray(m).astype(np.int64)
        w = self.weights()
        r = self.rates(m)
        served = np.minimum(self.Q, r)
        Q = self.Q - served + self.A
        dropped = np.maximum(Q - self.Qmax, 0.0)
        self.Q = np.minimum(Q, self.Qmax)
        if self.reward_mode == "drift":
            rew = float(np.dot(w, served)) / self.C_ref - self.kappa * dropped.sum() / self.C_ref
        else:
            rew = served.sum() / self.C_ref - self.beta * self.Q.mean() / self.Q_ref - self.kappa * dropped.sum() / self.C_ref
        info = dict(served=served.sum(), backlog=self.Q.mean(), dropped=dropped.sum(), arrived=self.A.sum())
        if self.reward_mode == "drift":
            # difference rewards (Wolpert & Tumer 2002) for branch-factored credit assignment:
            # D_b = r(m) - r(m with device b's decision flipped), evaluated on the same frame (channel, queues,
            # arrivals).  Aligned with the global objective (no tragedy of the commons on the shared TDMA time).
            M = np.repeat(m[None], self.N, 0)
            M[np.arange(self.N), np.arange(self.N)] ^= 1
            Rcf = self.rates_all(M)                                        # (N, N) counterfactual rates
            Qprev = Q - self.A + served                                    # queue before service
            s_cf = np.minimum(Qprev[None], Rcf)
            d_cf = np.maximum(Qprev[None] - s_cf + self.A[None] - self.Qmax[None], 0.0)
            r_cf = (s_cf @ w - self.kappa * d_cf.sum(1)) / self.C_ref
            info["reward_vec"] = (rew - r_cf).astype(np.float32)
        z = (self.rng.normal(size=self.N) + 1j * self.rng.normal(size=self.N)) / np.sqrt(2)
        self.c = self.rho * self.c + np.sqrt(1 - self.rho ** 2) * z
        self.h = self.g * np.abs(self.c) ** 2
        self.A = self._arrivals()
        self.t += 1
        return self._obs(), float(rew), self.t >= self.T_ep, info


def reference_action(env, kind, rng):
    if isinstance(env, MECQueueEnvV1):
        return reference_action_v1(env, kind, rng)
    if isinstance(env, ChannelAccessEnv):
        return channel_reference_action(env, kind, rng)
    N = env.N
    if kind == "random":
        return rng.integers(0, 2, N)
    if kind == "local":
        return np.zeros(N, int)
    if kind == "offload":
        return np.ones(N, int)
    R = env.rates_all(_ALL_M)
    if kind == "maxrate":
        return _ALL_M[np.argmax(R.sum(1))]
    if kind == "myopic":
        return _ALL_M[np.argmax((np.minimum(env.Q, R) * env.weights()).sum(1))]
    raise ValueError(kind)


# ======================================================================================
class GymEnv:
    """CartPole-v1 / Acrobot-v1 with fixed observation scaling (no running statistics)."""
    SCALE = {"CartPole-v1": np.array([2.4, 3.0, 0.21, 3.5]),
             "Acrobot-v1": np.array([1, 1, 1, 1, 4 * math.pi, 9 * math.pi])}

    def __init__(self, name, seed=0):
        import gymnasium as gym
        self.e = gym.make(name)
        self.name = name
        self.scale = self.SCALE[name].astype(np.float32)
        self.obs_dim = self.e.observation_space.shape[0]
        self.branches = [int(self.e.action_space.n)]
        self._seed = seed
        self._first = True

    def reset(self, seed=None):
        if seed is None and self._first:
            seed = self._seed
        self._first = False
        o, _ = self.e.reset(seed=seed)
        return (o / self.scale).astype(np.float32)

    def step(self, a):
        o, r, term, trunc, _ = self.e.step(int(np.asarray(a).ravel()[0]))
        return (o / self.scale).astype(np.float32), float(r), bool(term or trunc), dict(terminated=bool(term))


# ======================================================================================
class ChannelAccessEnv:
    """Dynamic multichannel access (scenario D2, replaces Acrobot-v1).

    A transmitter senses/uses one of C = 8 channels per slot.  Channel c evolves as an independent Gilbert-Elliott
    Markov chain (good/bad) with heterogeneous stationary good-probability pi_c and memory lambda_c = 1 - p_gb - p_bg
    (fixed per channel, drawn once from a fixed generator so that every run sees the same channel set).
    Only the chosen channel is observed (partial observability).  Reward +1 if the chosen channel is good, -1 otherwise.
    Observation (16-dim): for every channel, the last observed state (+1 good / -1 bad / 0 never observed) and the
    normalized age of that observation min(k, K)/K with K = 10.  Episodes have T = 100 slots.
    Reference policies: random channel, and the belief-myopic (Whittle-type) policy that picks the channel with the
    highest posterior good-probability b_c = pi_c + (s_c - pi_c) lambda_c^k_c, which is optimal for positively
    correlated channels (Zhao, Krishnamachari & Liu, IEEE TWC 2008).
    """
    C, K_AGE, T_EP = 8, 10, 100

    def __init__(self, seed=0, channel_seed=7):
        g = np.random.default_rng(channel_seed)
        self.pi = np.sort(g.uniform(0.15, 0.85, self.C))[::-1][g.permutation(self.C)]
        self.lam = g.uniform(0.5, 0.9, self.C)                    # positive memory
        self.p_bg = (1 - self.lam) * self.pi                      # bad -> good
        self.p_gb = (1 - self.lam) * (1 - self.pi)                # good -> bad
        self.branches = [self.C]
        self.obs_dim = 2 * self.C
        self.rng = np.random.default_rng(seed)
        self.reset()

    def _obs(self):
        age = np.minimum(self.k, self.K_AGE) / self.K_AGE
        return np.concatenate([self.last, age]).astype(np.float32)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = (self.rng.random(self.C) < self.pi).astype(int)
        self.last = np.zeros(self.C)
        self.k = np.full(self.C, self.K_AGE, dtype=float)
        self.t = 0
        return self._obs()

    def belief(self):
        s = np.where(self.last > 0, 1.0, np.where(self.last < 0, 0.0, self.pi))
        return self.pi + (s - self.pi) * self.lam ** self.k

    def step(self, a):
        c = int(np.asarray(a).ravel()[0])
        good = self.state[c] == 1
        rew = 1.0 if good else -1.0
        self.last[c] = 1.0 if good else -1.0
        self.k += 1
        self.k[c] = 1
        u = self.rng.random(self.C)
        self.state = np.where(self.state == 1, (u >= self.p_gb).astype(int), (u < self.p_bg).astype(int))
        self.t += 1
        done = self.t >= self.T_EP
        return self._obs(), rew, done, dict(success=float(good))


def channel_reference_action(env, kind, rng):
    if kind == "random":
        return np.array([rng.integers(0, env.C)])
    if kind == "myopic":
        return np.array([int(np.argmax(env.belief()))])
    raise ValueError(kind)


def make_env(name, seed=0, **kw):
    if name == "mec":
        return MECQueueEnv(seed=seed, **kw)
    if name == "mec_v1":
        return MECQueueEnvV1(seed=seed, **kw)
    if name == "cartpole":
        return GymEnv("CartPole-v1", seed)
    if name == "acrobot":
        return GymEnv("Acrobot-v1", seed)
    if name == "channel":
        return ChannelAccessEnv(seed=seed, **kw)
    raise ValueError(name)
