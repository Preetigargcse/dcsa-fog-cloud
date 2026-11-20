"""Job streams, scheduling policies, calibration procedures and the closed-loop runner."""
from __future__ import annotations
import heapq
import time
from dataclasses import dataclass, field, replace
import numpy as np
from scipy.stats import binom

from .infra import make_infra, TruthParams, Congestion, FOG
from .workloads import sample_job, DEFAULT_MIX
from .sim import (build_family, build_plan, mc_makespan, real_execute, NodeTracker, JobNoise,
                  FAMILY, FAMILY_NOSPLIT)

LEVELS = (0.80, 0.90, 0.95, 0.98)
S_MAX = 4.0


# --------------------------------------------------------------------------- streams
@dataclass
class StreamCfg:
    n_jobs: int = 2000
    rho: float = 0.6
    mix: dict = field(default_factory=lambda: dict(DEFAULT_MIX))
    slack: tuple = (1.3, 3.0)
    n_fog: int = 10
    n_cloud: int = 4
    layered_size: tuple = (10, 30)
    truth: TruthParams = field(default_factory=TruthParams)
    shift: object = None          # callable(frac in [0,1]) -> TruthParams, or None
    sigma_hat: float = 0.25
    M: int = 32
    family: str = "full"          # full | nosplit | heft
    warmup: int = 100


_MEAN_WORK = {}


def mean_work(mix, layered_size):
    key = (tuple(sorted(mix.items())), layered_size)
    if key not in _MEAN_WORK:
        rng = np.random.default_rng(999)
        _MEAN_WORK[key] = float(np.mean([sample_job(rng, mix, layered_size).work.sum() for _ in range(3000)]))
    return _MEAN_WORK[key]


@dataclass
class Stream:
    cfg: StreamCfg
    inf: object
    jobs: list
    noise: list
    tps: list
    cong: Congestion
    seed: int


def make_stream(cfg: StreamCfg, seed: int, role: int = 0, jobs_override=None):
    """role 0 = test stream, 1 = calibration stream (independent randomness)."""
    ss = np.random.SeedSequence([seed, role, 7])
    r_job, r_noise, r_cong, r_arr = [np.random.default_rng(s) for s in ss.spawn(4)]
    inf = make_infra(cfg.n_fog, cfg.n_cloud)
    K = inf.K
    cap = inf.speed.sum()
    rate = cfg.rho * cap / mean_work(cfg.mix, cfg.layered_size)
    if jobs_override is not None:
        src_jobs = jobs_override[:cfg.n_jobs]
        ta = np.array([getattr(j, "trace_arrival", i) for i, j in enumerate(src_jobs)], float)
        ta = ta - ta[0]
        mw = np.mean([j.work.sum() for j in src_jobs])
        span = len(src_jobs) * mw / (cfg.rho * cap)     # duration giving offered load rho
        arr = ta / max(ta[-1], 1e-9) * span if ta[-1] > 0 else np.cumsum(r_arr.exponential(span / len(src_jobs), len(src_jobs)))
        arr = arr + 1e-6 * np.arange(len(arr))
    else:
        src_jobs = None
        arr = np.cumsum(r_arr.exponential(1.0 / rate, size=cfg.n_jobs))
    fog = inf.fog
    jobs, noise, tps = [], [], []
    zeros = np.zeros(K)
    allm = np.ones(K, bool)
    for i in range(len(arr)):
        j = sample_job(r_job, cfg.mix, cfg.layered_size) if src_jobs is None else src_jobs[i]
        j.src = int(r_job.choice(fog))
        j.arrival = float(arr[i])
        n = j.n
        wh = j.work * np.exp(cfg.truth.work_est_sigma * r_noise.standard_normal(n))
        jn = JobNoise(r_noise.standard_normal((n, K)), r_noise.random((n, K)),
                      r_noise.pareto(2.0, size=(n, K)), wh)
        ref = build_plan(j, inf, zeros, 0.0, wh, allm, 0.0, True)
        j.deadline = float(r_job.uniform(*cfg.slack) * ref.nom_makespan)
        frac = i / max(1, len(arr) - 1)
        tps.append(cfg.shift(frac) if cfg.shift is not None else cfg.truth)
        jobs.append(j)
        noise.append(jn)
    cong = Congestion(K, arr[-1] + 300.0, r_cong, tau=cfg.truth.tau)
    return Stream(cfg, inf, jobs, noise, tps, cong, seed)


# --------------------------------------------------------------------------- per-job context
class Ctx:
    def __init__(self, stream, i, est_free, need_mc=True):
        self.s = stream
        self.i = i
        self.job = stream.jobs[i]
        self.est = est_free
        self._plans = None
        self._scores = None

    @property
    def plans(self):
        if self._plans is None:
            fam = {"full": FAMILY, "nosplit": FAMILY_NOSPLIT,
                   "heft": [("heft_split", 0.0, True, "all")]}[self.s.cfg.family]
            self._plans = build_family(self.job, self.s.inf, self.est, self.job.arrival,
                                       self.s.noise[self.i].work_hat, fam)
        return self._plans

    @property
    def scores(self):
        """(P, len(LEVELS)) predictive quantiles of makespan divided by the deadline."""
        if self._scores is None:
            cfg = self.s.cfg
            rng = np.random.default_rng([self.s.seed, 31, self.i])
            xi = rng.standard_normal((cfg.M, self.job.n, 2))
            out = np.empty((len(self.plans), len(LEVELS)))
            for p, pl in enumerate(self.plans):
                smp = mc_makespan(self.job, self.s.inf, pl, self.est, self.job.arrival,
                                  self.s.noise[self.i].work_hat, cfg.sigma_hat, xi)
                out[p] = np.quantile(smp, LEVELS) / self.job.deadline
            self._scores = np.clip(out, 0.0, S_MAX)
        return self._scores

    def energies(self):
        return np.array([p.energy for p in self.plans])

    def noms(self):
        return np.array([p.nom_makespan for p in self.plans])


def level_index(alpha):
    return int(np.argmin(np.abs(np.array(LEVELS) - (1 - alpha))))


def pick_min_energy(ctx, feasible):
    if not np.any(feasible):
        return None
    e = np.where(feasible, ctx.energies(), np.inf)
    return int(np.argmin(e))


# --------------------------------------------------------------------------- policies
class Policy:
    name = "base"
    def decide(self, ctx):  # -> plan index or None
        raise NotImplementedError
    def feedback(self, admitted, late):
        pass


class FastestMean(Policy):
    name = "FastestMean"
    def decide(self, ctx):
        return int(np.argmin(ctx.noms()))


class EnergyAll(Policy):
    name = "EnergyGreedy"
    def decide(self, ctx):
        return int(np.argmin(ctx.energies()))


class DAEnergy(Policy):
    """Deadline-aware energy minimisation on nominal predictions, no admission control."""
    name = "DA-Energy"
    def decide(self, ctx):
        p = pick_min_energy(ctx, ctx.noms() <= ctx.job.deadline)
        return int(np.argmin(ctx.noms())) if p is None else p


class MeanAdmit(Policy):
    """Deadline-aware energy minimisation with nominal-feasibility admission control."""
    name = "MeanAdmit"
    def decide(self, ctx):
        return pick_min_energy(ctx, ctx.noms() <= ctx.job.deadline)


class Threshold(Policy):
    """Admit iff some plan has score <= lam; choose the min-energy (or min-score) such plan."""
    def __init__(self, alpha, lam, name, select="energy"):
        self.alpha, self.lam, self.name, self.select = alpha, lam, name, select
        self.li = level_index(alpha)
    def decide(self, ctx):
        sc = ctx.scores[:, self.li]
        if self.select == "risk":
            p = int(np.argmin(sc))
            return p if sc[p] <= self.lam else None
        return pick_min_energy(ctx, sc <= self.lam)


class CPMarginal(Policy):
    """Per-plan split-conformal latency multiplier (marginal, ignores selection)."""
    name = "CP-Marginal"
    def __init__(self, ratios):
        self.r = ratios
    def decide(self, ctx):
        return pick_min_energy(ctx, self.r * ctx.noms() <= ctx.job.deadline)


class DCSAOnline(Threshold):
    """Online selective-risk threshold: lam <- lam - eta * a_t (late_t - alpha).
    Optional safeguard (paper Sec. 5.3): after `reset_after` consecutive rejections with no
    outstanding feedback, reset lam to lam0 (disabled by default; never needed in our runs)."""
    def __init__(self, alpha, lam0, eta, name="DCSA-On", select="energy", reset_after=None):
        super().__init__(alpha, lam0, name, select)
        self.eta = eta
        self.lam0 = lam0
        self.trace = []
        self.reset_after = reset_after
        self.rejections = 0
        self.outstanding = 0
        self.resets = 0
    def decide(self, ctx):
        p = super().decide(ctx)
        if p is None:
            self.rejections += 1
            if self.reset_after and self.rejections >= self.reset_after and self.outstanding == 0:
                self.lam = self.lam0; self.rejections = 0; self.resets += 1
        else:
            self.rejections = 0
            self.outstanding += 1
        return p
    def feedback(self, admitted, late):
        if admitted:
            self.outstanding -= 1
            self.lam -= self.eta * (float(late) - self.alpha)
            self.trace.append(self.lam)


# --------------------------------------------------------------------------- runner
def run(stream: Stream, policy: Policy, record_calib=False, record_series=False):
    cfg = stream.cfg
    inf = stream.inf
    tr = NodeTracker(inf.K)
    fb = []          # (time, seq, late)
    seq = 0
    N = len(stream.jobs)
    adm = np.zeros(N, np.int8); late = np.zeros(N, np.int8); ratio = np.full(N, np.nan)
    energy = np.zeros(N); dtime = np.zeros(N); nsplit = np.zeros(N); cloudw = np.zeros(N)
    lam_series = np.full(N, np.nan)
    calib = [] if record_calib else None
    max_pending = 0
    for i, job in enumerate(stream.jobs):
        t = job.arrival
        while fb and fb[0][0] <= t:
            _, _, lt = heapq.heappop(fb)
            policy.feedback(True, lt)
        t_start = time.perf_counter()
        est = tr.estimate(t)
        ctx = Ctx(stream, i, est)
        p = policy.decide(ctx)
        dtime[i] = time.perf_counter() - t_start
        if hasattr(policy, "lam"):
            lam_series[i] = policy.lam
        tp = stream.tps[i]
        if record_calib:
            sc = ctx.scores
            real = np.array([real_execute(job, inf, pl, tr.real_free, t, stream.noise[i], tp, stream.cong)[0]
                             for pl in ctx.plans])
            calib.append((sc.copy(), ctx.noms(), ctx.energies(), real, job.deadline))
        if p is None:
            continue
        pl = ctx.plans[p]
        mk, en, recs = real_execute(job, inf, pl, tr.real_free, t, stream.noise[i], tp, stream.cong, commit=True)
        tr.commit(recs)
        adm[i] = 1
        lt = mk > job.deadline
        late[i] = lt
        ratio[i] = mk / job.deadline
        energy[i] = en
        nsplit[i] = pl.n_split > 0
        cloudw[i] = pl.cloud_work
        heapq.heappush(fb, (t + min(mk, job.deadline), seq, bool(lt)))
        seq += 1
        max_pending = max(max_pending, len(fb))
    res = summarize(stream, adm, late, ratio, energy, dtime, nsplit, cloudw)
    res["policy"] = policy.name
    res["max_pending"] = max_pending
    res["adm_all"] = int(adm.sum()); res["late_all"] = int(late.sum())
    if hasattr(policy, "lam"):
        res["lam_final"] = float(policy.lam)
    if hasattr(policy, "resets"):
        res["resets"] = int(policy.resets)
    if record_series:
        res["series"] = dict(adm=adm, late=late, lam=lam_series, arrival=np.array([j.arrival for j in stream.jobs]))
    if record_calib:
        res["calib"] = calib
    return res


def summarize(stream, adm, late, ratio, energy, dtime, nsplit, cloudw):
    w = stream.cfg.warmup
    a = adm[w:].astype(bool); l = late[w:].astype(bool)
    n = len(a); na = a.sum(); nl = (a & l).sum(); nok = (a & ~l).sum()
    r = ratio[w:][a]
    return dict(offered=int(n), admitted=int(na), late=int(nl), ontime=int(nok),
                admit_rate=na / n, sel_risk=(nl / na) if na else 0.0, ontime_offered=nok / n,
                miss_offered=nl / n,
                energy_per_admit=energy[w:][a].mean() if na else np.nan,
                energy_per_ontime=energy[w:][a].sum() / nok if nok else np.nan,
                energy_total=float(energy[w:].sum()),
                p50_ratio=float(np.quantile(r, 0.5)) if na else np.nan,
                p95_ratio=float(np.quantile(r, 0.95)) if na else np.nan,
                p99_ratio=float(np.quantile(r, 0.99)) if na else np.nan,
                dec_ms_mean=1e3 * dtime[w:].mean(), dec_ms_p99=1e3 * np.quantile(dtime[w:], 0.99),
                split_rate=float(nsplit[w:][a].mean()) if na else np.nan,
                cloud_work=float(cloudw[w:][a].mean()) if na else np.nan)


# --------------------------------------------------------------------------- calibration
LAM_GRID = np.geomspace(0.3, S_MAX, 40)


def calib_arrays(calib, warmup=100):
    c = calib[warmup:]
    S = np.stack([x[0] for x in c])      # (n, P, L)
    NOM = np.stack([x[1] for x in c])    # (n, P)
    E = np.stack([x[2] for x in c])
    R = np.stack([x[3] for x in c])
    D = np.array([x[4] for x in c])
    return S, NOM, E, R, D


def rule_outcomes(S, E, R, D, li, lam, select="energy"):
    feas = S[:, :, li] <= lam
    ok = feas.any(1)
    if select == "risk":
        p = np.argmin(S[:, :, li], 1)
    else:
        e = np.where(feas, E, np.inf)
        p = np.argmin(e, 1)
    lat = R[np.arange(len(D)), p] > D
    return ok, lat & ok


def dcsa_offline(S, E, R, D, alpha, delta=0.1, grid=LAM_GRID, method="bonferroni", select="energy"):
    """Learn-then-Test style certification of the selective (admitted-job) miss risk.
    Returns (lam_hat, table)."""
    li = level_index(alpha)
    G = len(grid)
    rows = []
    for lam in grid:
        ok, lt = rule_outcomes(S, E, R, D, li, lam, select)
        n, k = int(ok.sum()), int(lt.sum())
        pv = binom.cdf(k, n, alpha) if n > 0 else 1.0
        rows.append((lam, n, k, pv))
    rows = np.array(rows)
    if method == "bonferroni":
        cert = rows[:, 3] <= delta / G
        lam_hat = grid[cert].max() if cert.any() else 0.0
    else:   # fixed-sequence from the strictest lambda upward
        lam_hat = 0.0
        for lam, n, k, pv in rows:
            if pv <= delta:
                lam_hat = lam
            else:
                break
    return float(lam_hat), rows


def cp_marginal_ratios(NOM, R, alpha):
    n = NOM.shape[0]
    ratio = R / NOM
    lvl = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return np.quantile(ratio, lvl, axis=0, method="higher")
