"""Discrete-event fog-cloud workflow simulator with plan construction, predictive
Monte-Carlo risk scoring and ground-truth execution.

Execution semantics: every node is a single non-preemptive FIFO server that runs
fragments in the order in which they were committed (static per-node order, no
backfilling).  Because jobs are committed in arrival order and tasks inside a job in a
topological order, realised start/finish times can be computed exactly at commit time.
The scheduler never sees realised values before they happen: its node-availability
estimate uses only completions observed up to the decision time (see ``NodeTracker``).
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import time
import numpy as np

from .infra import Infra, TruthParams, Congestion, FOG, CLOUD

COORD = 0.004          # split coordination/join overhead (s)
E2S = 1.0 / 20.0       # converts joules to seconds in the energy-weighted list-scheduling cost


# --------------------------------------------------------------------------- plans
@dataclass
class Plan:
    name: str
    frags: list            # per task: tuple of (node, frac)
    host: np.ndarray       # per task: node holding its output (join node)
    nom_fin: np.ndarray    # per task nominal finish (absolute time)
    frag_nom_fin: list     # per task: tuple of nominal fragment finishes
    nom_makespan: float    # nominal completion - arrival
    energy: float          # predicted energy (J, modelled)
    n_split: int
    cloud_work: float      # fraction of work placed in the cloud


def build_plan(job, inf: Infra, est_free, t0, work_hat, allowed, w_energy=0.0,
               allow_split=True, name=""):
    K = inf.K
    n = job.n
    src = job.src
    fin = np.empty(n)
    host = np.empty(n, int)
    frags = [None] * n
    ffin = [None] * n
    avail = est_free.copy()
    mask = np.where(allowed, 0.0, np.inf)
    energy = 0.0
    nsplit = 0
    cw = 0.0
    lat, bw, emb, spd, pw = inf.lat, inf.bw, inf.e_mb, inf.speed, inf.power
    for v in job.order:
        pv = job.preds[v]
        if len(pv) == 0:
            mb = job.in_mb[v]
            ready = t0 + lat[src] + mb / bw[src]
            ready_h = t0 + lat[src] + 0.5 * mb / bw[src]
            e_in = emb[src] * mb
        else:
            ready = np.full(K, -np.inf); ready_h = np.full(K, -np.inf); e_in = np.zeros(K)
            for u in pv:
                mb = job.edge_mb[(u, v)]
                hu = host[u]
                r = fin[u] + lat[hu] + mb / bw[hu]
                rh = fin[u] + lat[hu] + 0.5 * mb / bw[hu]
                np.maximum(ready, r, out=ready)
                np.maximum(ready_h, rh, out=ready_h)
                e_in = e_in + emb[hu] * mb
        ex = work_hat[v] / spd
        f1 = np.maximum(ready, avail) + ex
        e1 = pw * ex + e_in
        cost = f1 + w_energy * E2S * e1 + mask
        k = int(np.argmin(cost))
        best = cost[k]
        choice = ((k, 1.0),); choice_f = (f1[k],); fv = f1[k]; ev = e1[k]
        if allow_split and job.splittable[v]:
            cand = np.argsort(cost)[:3]
            fh = np.maximum(ready_h, avail) + 0.5 * ex
            eh = 0.5 * pw * ex + 0.5 * e_in
            for i in range(len(cand)):
                for j in range(i + 1, len(cand)):
                    a, b = int(cand[i]), int(cand[j])
                    if not (np.isfinite(cost[a]) and np.isfinite(cost[b])):
                        continue
                    fs = max(fh[a], fh[b]) + COORD + lat[b, a]
                    es = eh[a] + eh[b]
                    cs = fs + w_energy * E2S * es
                    if cs < best - 1e-12:
                        best = cs
                        choice = ((a, 0.5), (b, 0.5)); choice_f = (fh[a], fh[b]); fv = fs; ev = es
        for (kk, fr), ff in zip(choice, choice_f):
            avail[kk] = ff
            if inf.tier[kk] == CLOUD:
                cw += fr * job.work[v]
        if len(choice) > 1:
            nsplit += 1
        frags[v] = choice
        ffin[v] = choice_f
        fin[v] = fv
        host[v] = choice[0][0]
        energy += ev
    comp = t0
    for v in range(n):
        if len(job.succs[v]) == 0:
            hv = host[v]
            mb = job.out_mb[v]
            comp = max(comp, fin[v] + lat[hv, src] + mb / bw[hv, src])
            energy += inf.e_mb[hv, src] * mb
    return Plan(name, frags, host, fin, ffin, comp - t0, energy, nsplit, cw / job.work.sum())


# family of candidate plans: (name, w_energy, allow_split, node-set)
FAMILY = [
    ("heft", 0.0, False, "all"),
    ("heft_split", 0.0, True, "all"),
    ("ew0.5", 0.5, True, "all"),
    ("ew2", 2.0, True, "all"),
    ("ew8", 8.0, True, "all"),
    ("fog_heft", 0.0, True, "fog"),
    ("fog_energy", 1000.0, True, "fog"),
    ("cloud", 0.0, True, "cloud"),
]
FAMILY_NOSPLIT = [(n + "_ns", w, False, s) for (n, w, _, s) in FAMILY if n != "heft"] + [("heft", 0.0, False, "all")]


def build_family(job, inf, est_free, t0, work_hat, family=FAMILY):
    plans = []
    allm = np.ones(inf.K, bool)
    fogm = inf.tier == FOG
    clm = inf.tier == CLOUD
    for (name, w, sp, ns) in family:
        if ns == "all":
            m = allm
        elif ns == "fog":
            m = fogm
        else:
            m = clm.copy()
        plans.append(build_plan(job, inf, est_free, t0, work_hat, m, w, sp, name))
    return plans


# --------------------------------------------------------------------------- predictive MC
def mc_makespan(job, inf, plan, est_free, t0, work_hat, sigma_hat, xi):
    """Predictive makespan samples under an independent log-normal runtime model.
    xi: (M, n, 2) standard normals."""
    M = xi.shape[0]
    lat, bw, spd = inf.lat, inf.bw, inf.speed
    src = job.src
    fin = [None] * job.n
    avail = {}
    for v in job.order:
        pv = job.preds[v]
        outs = []
        for fi, (k, fr) in enumerate(plan.frags[v]):
            if len(pv) == 0:
                ready = t0 + lat[src, k] + fr * job.in_mb[v] / bw[src, k]
            else:
                ready = None
                for u in pv:
                    hu = plan.host[u]
                    r = fin[u] + (lat[hu, k] + fr * job.edge_mb[(u, v)] / bw[hu, k])
                    ready = r if ready is None else np.maximum(ready, r)
            a = avail.get(k, est_free[k])
            st = np.maximum(ready, a)
            f = st + fr * work_hat[v] / spd[k] * np.exp(sigma_hat * xi[:, v, fi])
            avail[k] = f
            outs.append(f)
        if len(outs) == 1:
            fin[v] = outs[0]
        else:
            a0 = plan.frags[v][0][0]; b0 = plan.frags[v][1][0]
            fin[v] = np.maximum(outs[0], outs[1]) + COORD + lat[b0, a0]
    comp = np.full(M, t0)
    for v in range(job.n):
        if len(job.succs[v]) == 0:
            hv = plan.host[v]
            comp = np.maximum(comp, fin[v] + lat[hv, src] + job.out_mb[v] / bw[hv, src])
    return comp - t0


# --------------------------------------------------------------------------- ground truth
@dataclass
class JobNoise:
    Z: np.ndarray       # (n, K) task-node idiosyncratic normals
    U: np.ndarray       # (n, K) uniforms for stragglers
    L: np.ndarray       # (n, K) Lomax(2) draws
    work_hat: np.ndarray


def real_execute(job, inf, plan, real_free, t0, noise: JobNoise, tp: TruthParams, cong: Congestion,
                 commit=False):
    """Execute plan under the ground-truth model.  Returns (makespan, energy, frag_records).
    If commit, real_free is updated in place."""
    lat, bw, spd, pw = inf.lat, inf.bw, inf.speed, inf.power
    src = job.src
    fin = np.empty(job.n)
    avail = real_free if commit else real_free.copy()
    energy = 0.0
    recs = []
    W = cong.W
    for v in job.order:
        pv = job.preds[v]
        outs = []
        for fi, (k, fr) in enumerate(plan.frags[v]):
            if len(pv) == 0:
                tt = lat[src, k] + fr * job.in_mb[v] / bw[src, k]
                if inf.tier[src] != inf.tier[k]:
                    tt *= np.exp(tp.sigma_wan * W[cong.slot(t0)])
                ready = t0 + tt
                energy += inf.e_mb[src, k] * fr * job.in_mb[v]
            else:
                ready = -np.inf
                for u in pv:
                    hu = plan.host[u]
                    mb = fr * job.edge_mb[(u, v)]
                    tt = lat[hu, k] + mb / bw[hu, k]
                    if inf.tier[hu] != inf.tier[k]:
                        tt *= np.exp(tp.sigma_wan * W[cong.slot(fin[u])])
                    ready = max(ready, fin[u] + tt)
                    energy += inf.e_mb[hu, k] * mb
            st = max(ready, avail[k])
            s = cong.slot(st)
            mult = np.exp(tp.sigma_task * noise.Z[v, k] + tp.sigma_glob * cong.G[s] + tp.sigma_node * cong.H[k, s])
            if noise.U[v, k] < tp.p_strag:
                mult *= 1.0 + tp.strag_scale * noise.L[v, k]
            rt = fr * job.work[v] / spd[k] * mult
            f = st + rt
            energy += pw[k] * rt
            avail[k] = f
            outs.append(f)
            recs.append((k, plan.frag_nom_fin[v][fi], f, fr * noise.work_hat[v] / spd[k]))
        if len(outs) == 1:
            fin[v] = outs[0]
        else:
            a0 = plan.frags[v][0][0]; b0 = plan.frags[v][1][0]
            fin[v] = max(outs) + COORD + lat[b0, a0]
    comp = t0
    for v in range(job.n):
        if len(job.succs[v]) == 0:
            hv = plan.host[v]
            tt = lat[hv, src] + job.out_mb[v] / bw[hv, src]
            if inf.tier[hv] != inf.tier[src]:
                tt *= np.exp(tp.sigma_wan * W[cong.slot(fin[v])])
            comp = max(comp, fin[v] + tt)
            energy += inf.e_mb[hv, src] * job.out_mb[v]
    return comp - t0, energy, recs


class NodeTracker:
    """Scheduler-side node-availability estimate built only from information observable at
    decision time: completed fragments (their realised lag) and the nominal durations of
    fragments still queued or running."""

    def __init__(self, K):
        self.q = [deque() for _ in range(K)]
        self.lag = np.zeros(K)
        self.backlog = np.zeros(K)          # sum of nominal durations of queued fragments
        self.real_free = np.zeros(K)        # ground truth (simulator only)

    def estimate(self, t):
        K = len(self.q)
        est = np.empty(K)
        for k in range(K):
            q = self.q[k]
            while q and q[0][1] <= t:
                nom, real, dur = q.popleft()
                self.lag[k] = real - nom
                self.backlog[k] -= dur
            if not q:
                self.backlog[k] = 0.0
                est[k] = t
            else:
                est[k] = max(t + self.backlog[k], q[-1][0] + self.lag[k])
        return est

    def commit(self, recs):
        for (k, nom, real, dur) in recs:
            self.q[k].append((nom, real, dur))
            self.backlog[k] += dur
