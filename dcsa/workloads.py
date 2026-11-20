"""Workflow (DAG) generators for the DCSA fog-cloud study.

Every generator returns a ``Job`` with task work (giga-operations, Gop), edge data
volumes (MB), per-task splittability flags and a topological processing order.
Families:
  * iot      -- sense/pre-process/parallel-analytics/fuse/actuate pipelines
                (inspired by the application models used with iFogSim).
  * layered  -- random layered DAGs following the generator of Topcuoglu et al. (2002).
  * montage, epigenomics, cybershake, ligo -- shape-faithful, down-scaled versions of the
                Pegasus scientific workflows characterised by Bharathi et al. (2008).
Task weights are synthetic (log-normal) and are NOT measured runtimes.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Job:
    family: str
    work: np.ndarray                 # (n,) Gop
    preds: list                      # list of int arrays
    succs: list
    edge_mb: dict                    # (u, v) -> MB
    in_mb: np.ndarray                # (n,) MB from ingress for entry tasks (0 otherwise)
    out_mb: np.ndarray               # (n,) MB returned to ingress for exit tasks
    splittable: np.ndarray           # (n,) bool
    order: np.ndarray = None         # processing order (filled by finalize)
    src: int = 0                     # ingress fog node
    arrival: float = 0.0
    deadline: float = 0.0            # relative deadline (s)

    @property
    def n(self):
        return len(self.work)

    def finalize(self, mean_speed=6.0, mean_bw=12.5):
        n = self.n
        # upward rank (HEFT) with mean speed and mean bandwidth
        rank = np.zeros(n)
        topo = topo_order(self.preds, self.succs)
        for v in topo[::-1]:
            best = 0.0
            for s in self.succs[v]:
                best = max(best, self.edge_mb[(v, s)] / mean_bw + rank[s])
            rank[v] = self.work[v] / mean_speed + best
        # sort by decreasing rank; ties broken by topological position -> valid topo order
        pos = np.empty(n, int)
        pos[topo] = np.arange(n)
        self.order = np.array(sorted(range(n), key=lambda v: (-rank[v], pos[v])))
        # guarantee topological validity (rank is strictly decreasing along edges if work>0)
        seen = set()
        for v in self.order:
            for u in self.preds[v]:
                assert u in seen, "order not topological"
            seen.add(v)
        return self


def topo_order(preds, succs):
    n = len(preds)
    indeg = np.array([len(p) for p in preds])
    stack = [v for v in range(n) if indeg[v] == 0]
    out = []
    while stack:
        v = stack.pop()
        out.append(v)
        for s in succs[v]:
            indeg[s] -= 1
            if indeg[s] == 0:
                stack.append(s)
    assert len(out) == n, "cycle"
    return np.array(out)


def _build(family, n, edges, work, rng, split_mask, in_scale=1.0, out_scale=0.2, ccr=0.3):
    preds = [[] for _ in range(n)]
    succs = [[] for _ in range(n)]
    edge_mb = {}
    for (u, v) in edges:
        preds[v].append(u)
        succs[u].append(v)
        # data volume roughly proportional to producer work and the communication ratio
        edge_mb[(u, v)] = float(max(0.01, ccr * work[u] * rng.lognormal(0.0, 0.4)))
    preds = [np.array(sorted(p), int) for p in preds]
    succs = [np.array(sorted(s), int) for s in succs]
    in_mb = np.array([in_scale * rng.lognormal(0, 0.3) if len(preds[v]) == 0 else 0.0 for v in range(n)])
    out_mb = np.array([out_scale * rng.lognormal(0, 0.3) if len(succs[v]) == 0 else 0.0 for v in range(n)])
    return Job(family, np.asarray(work, float), preds, succs, edge_mb, in_mb, out_mb,
               np.asarray(split_mask, bool)).finalize()


def gen_iot(rng):
    k = rng.integers(2, 5)                      # parallel analytics branches
    # 0 sense, 1 preprocess, 2..k+1 analytics, k+2 fuse, k+3 decide
    n = k + 4
    edges = [(0, 1)] + [(1, 2 + i) for i in range(k)] + [(2 + i, k + 2) for i in range(k)] + [(k + 2, k + 3)]
    work = np.empty(n)
    work[0] = rng.lognormal(np.log(0.05), 0.3)
    work[1] = rng.lognormal(np.log(0.4), 0.4)
    work[2:k + 2] = rng.lognormal(np.log(1.2), 0.5, size=k)
    work[k + 2] = rng.lognormal(np.log(0.3), 0.3)
    work[k + 3] = rng.lognormal(np.log(0.08), 0.3)
    split = np.zeros(n, bool)
    split[2:k + 2] = rng.random(k) < 0.6
    return _build("iot", n, edges, work, rng, split, in_scale=2.0, out_scale=0.05, ccr=0.25)


def gen_layered(rng, vmin=10, vmax=30, shape=1.0):
    v = int(rng.integers(vmin, vmax + 1))
    height = max(2, int(round(np.sqrt(v) / shape)))
    # distribute v tasks over `height` levels
    cuts = np.sort(rng.choice(np.arange(1, v), size=height - 1, replace=False))
    sizes = np.diff(np.concatenate([[0], cuts, [v]]))
    levels, idx = [], 0
    for s in sizes:
        levels.append(list(range(idx, idx + s)))
        idx += s
    edges = set()
    for li in range(1, len(levels)):
        for t in levels[li]:
            # each task gets 1-3 parents from the previous level (guarantees connectivity)
            par = rng.choice(levels[li - 1], size=min(len(levels[li - 1]), int(rng.integers(1, 4))), replace=False)
            for p in par:
                edges.add((int(p), t))
            # occasional skip edge
            if li >= 2 and rng.random() < 0.15:
                edges.add((int(rng.choice(levels[li - 2])), t))
    work = rng.lognormal(np.log(0.35), 0.7, size=v)
    split = (work > np.quantile(work, 0.6)) & (rng.random(v) < 0.5)
    return _build("layered", v, sorted(edges), work, rng, split, in_scale=1.0, ccr=0.4)


def gen_montage(rng):
    m = int(rng.integers(3, 6))           # number of input images
    ids = {}
    def nid(name):
        ids[name] = len(ids)
        return ids[name]
    proj = [nid(f"proj{i}") for i in range(m)]
    diffs = []
    edges = []
    for i in range(m):
        for j in (i + 1,):
            if j < m:
                d = nid(f"diff{i}{j}")
                diffs.append(d)
                edges += [(proj[i], d), (proj[j], d)]
    concat = nid("concat"); edges += [(d, concat) for d in diffs]
    bg = nid("bgmodel"); edges += [(concat, bg)]
    back = []
    for i in range(m):
        b = nid(f"back{i}"); back.append(b)
        edges += [(bg, b), (proj[i], b)]
    tbl = nid("imgtbl"); edges += [(b, tbl) for b in back]
    add = nid("add"); edges += [(tbl, add)]
    shr = nid("shrink"); edges += [(add, shr)]
    n = len(ids)
    work = np.empty(n)
    for name, i in ids.items():
        base = {"proj": 0.9, "diff": 0.25, "conc": 0.15, "bgmo": 0.3, "back": 0.4,
                "imgt": 0.1, "add": 1.2, "shri": 0.3}[name[:4] if not name.startswith("add") else "add"]
        work[i] = rng.lognormal(np.log(base), 0.35)
    split = np.zeros(n, bool)
    split[ids["add"]] = True
    for p in proj:
        split[p] = rng.random() < 0.5
    return _build("montage", n, edges, work, rng, split, in_scale=1.5, ccr=0.5)


def gen_epigenomics(rng):
    k = int(rng.integers(2, 5))
    edges, work, split = [], [], []
    def add(w, s=False):
        work.append(w); split.append(s); return len(work) - 1
    sp = add(rng.lognormal(np.log(0.2), 0.3))
    ends = []
    for _ in range(k):
        prev = sp
        for stage, base in enumerate([0.3, 0.15, 0.2, 1.4]):
            t = add(rng.lognormal(np.log(base), 0.35), s=(stage == 3 and rng.random() < 0.6))
            edges.append((prev, t)); prev = t
        ends.append(prev)
    mm = add(rng.lognormal(np.log(0.3), 0.3)); edges += [(e, mm) for e in ends]
    idx = add(rng.lognormal(np.log(0.2), 0.3)); edges.append((mm, idx))
    pil = add(rng.lognormal(np.log(0.6), 0.3), s=True); edges.append((idx, pil))
    return _build("epigenomics", len(work), edges, np.array(work), rng, np.array(split), in_scale=2.0, ccr=0.35)


def gen_cybershake(rng):
    m = int(rng.integers(3, 7))
    edges, work, split = [], [], []
    def add(w, s=False):
        work.append(w); split.append(s); return len(work) - 1
    sgt = [add(rng.lognormal(np.log(0.8), 0.3)) for _ in range(2)]
    seis, peaks = [], []
    for i in range(m):
        s = add(rng.lognormal(np.log(0.9), 0.4), s=rng.random() < 0.5); edges.append((sgt[i % 2], s)); seis.append(s)
        p = add(rng.lognormal(np.log(0.05), 0.3)); edges.append((s, p)); peaks.append(p)
    z1 = add(rng.lognormal(np.log(0.1), 0.3)); edges += [(s, z1) for s in seis]
    z2 = add(rng.lognormal(np.log(0.05), 0.3)); edges += [(p, z2) for p in peaks]
    return _build("cybershake", len(work), edges, np.array(work), rng, np.array(split), in_scale=2.5, ccr=0.5)


def gen_ligo(rng):
    m = int(rng.integers(3, 6))
    edges, work, split = [], [], []
    def add(w, s=False):
        work.append(w); split.append(s); return len(work) - 1
    tb = [add(rng.lognormal(np.log(0.25), 0.3)) for _ in range(m)]
    ins = []
    for t in tb:
        i = add(rng.lognormal(np.log(1.1), 0.4), s=rng.random() < 0.5); edges.append((t, i)); ins.append(i)
    th = add(rng.lognormal(np.log(0.1), 0.3)); edges += [(i, th) for i in ins]
    trig = []
    for _ in range(m):
        tr = add(rng.lognormal(np.log(0.05), 0.3)); edges.append((th, tr)); trig.append(tr)
    ins2 = []
    for tr in trig:
        i = add(rng.lognormal(np.log(0.9), 0.4), s=rng.random() < 0.5); edges.append((tr, i)); ins2.append(i)
    th2 = add(rng.lognormal(np.log(0.1), 0.3)); edges += [(i, th2) for i in ins2]
    return _build("ligo", len(work), edges, np.array(work), rng, np.array(split), in_scale=1.0, ccr=0.3)


GENERATORS = {"iot": gen_iot, "layered": gen_layered, "montage": gen_montage,
              "epigenomics": gen_epigenomics, "cybershake": gen_cybershake, "ligo": gen_ligo}

DEFAULT_MIX = {"iot": 0.45, "layered": 0.25, "montage": 0.075, "epigenomics": 0.075,
               "cybershake": 0.075, "ligo": 0.075}


def sample_job(rng, mix=DEFAULT_MIX, layered_size=(10, 30)):
    fams = list(mix)
    p = np.array([mix[f] for f in fams]); p = p / p.sum()
    f = fams[rng.choice(len(fams), p=p)]
    if f == "layered":
        return gen_layered(rng, *layered_size)
    return GENERATORS[f](rng)
