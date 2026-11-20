"""Heterogeneous fog-cloud infrastructure, network and energy model, and ground-truth
stochastic performance model (correlated congestion, stragglers, WAN variability).

All parameters are scenario assumptions (see paper Table 3); energy is modelled, not measured.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

FOG, CLOUD = 0, 1


@dataclass
class Infra:
    speed: np.ndarray      # Gop/s
    power: np.ndarray      # W while busy (dynamic + attributable static share)
    tier: np.ndarray       # 0 fog, 1 cloud
    bw: np.ndarray         # MB/s  (K,K), inf on diagonal
    lat: np.ndarray        # s     (K,K), 0 on diagonal
    e_mb: np.ndarray       # J/MB  (K,K)

    @property
    def K(self):
        return len(self.speed)

    @property
    def fog(self):
        return np.where(self.tier == FOG)[0]

    @property
    def cloud(self):
        return np.where(self.tier == CLOUD)[0]


def make_infra(n_fog=10, n_cloud=4, rng=None, fog_speed=(2.0, 6.0), cloud_speed=16.0,
               fog_lan_bw=12.5, fog_lan_lat=0.002, wan_bw=6.25, wan_lat=0.030,
               cloud_bw=125.0, cloud_lat=0.0005):
    rng = np.random.default_rng(12345) if rng is None else rng
    fs = np.linspace(fog_speed[0], fog_speed[1], n_fog) if n_fog > 1 else np.array([np.mean(fog_speed)])
    speed = np.concatenate([fs, np.full(n_cloud, cloud_speed)])
    # fog: low-power SoCs/micro-servers; cloud: VM share of a server incl. PUE overhead
    power = np.concatenate([3.0 + 1.6 * fs, np.full(n_cloud, 80.0)])
    tier = np.concatenate([np.zeros(n_fog, int), np.ones(n_cloud, int)])
    K = len(speed)
    bw = np.empty((K, K)); lat = np.empty((K, K)); e = np.empty((K, K))
    for i in range(K):
        for j in range(K):
            if i == j:
                bw[i, j], lat[i, j], e[i, j] = np.inf, 0.0, 0.0
            elif tier[i] == FOG and tier[j] == FOG:
                bw[i, j], lat[i, j], e[i, j] = fog_lan_bw, fog_lan_lat, 0.05
            elif tier[i] == CLOUD and tier[j] == CLOUD:
                bw[i, j], lat[i, j], e[i, j] = cloud_bw, cloud_lat, 0.01
            else:
                bw[i, j], lat[i, j], e[i, j] = wan_bw, wan_lat, 0.50
    return Infra(speed, power, tier, bw, lat, e)


@dataclass
class TruthParams:
    sigma_task: float = 0.25     # idiosyncratic log-noise per (task,node)
    sigma_glob: float = 0.20     # common congestion (all nodes)
    sigma_node: float = 0.20     # node-specific congestion
    tau: float = 10.0            # OU time constant (s)
    p_strag: float = 0.02        # straggler probability
    strag_scale: float = 2.0     # multiplier = 1 + scale * Lomax(2)
    sigma_wan: float = 0.30      # WAN slowdown log-noise (time-correlated)
    work_est_sigma: float = 0.15 # predictor's work-estimate log error


class Congestion:
    """Discretised Ornstein-Uhlenbeck congestion processes (stationary N(0,1))."""

    def __init__(self, K, horizon, rng, tau=10.0, dt=0.1):
        self.dt = dt
        T = int(np.ceil(horizon / dt)) + 2
        a = np.exp(-dt / tau)
        s = np.sqrt(1 - a * a)
        eps = rng.standard_normal((K + 2, T))
        x = np.empty((K + 2, T))
        x[:, 0] = eps[:, 0]
        for t in range(1, T):
            x[:, t] = a * x[:, t - 1] + s * eps[:, t]
        self.G = x[0]
        self.H = x[1:K + 1]
        self.W = x[K + 1]
        self.T = T

    def slot(self, t):
        return min(self.T - 1, max(0, int(t / self.dt)))
