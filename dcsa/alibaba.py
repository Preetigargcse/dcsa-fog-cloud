"""Loader for the Alibaba cluster-trace-v2018 ``batch_task.csv`` (DAG-structured batch jobs).

The public file (https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018) has no
header and the columns
    task_name, instance_num, job_name, task_type, status, start_time, end_time, plan_cpu, plan_mem
Task names encode dependencies: ``M5_3_4`` is task 5 and depends on tasks 3 and 4; names that do
not follow this pattern (e.g. ``task_xxx``) are independent single-task jobs.

The loader converts each terminated job with 2..max_tasks tasks into a ``Job`` whose task work is
proportional to the observed CPU-seconds (instance_num * duration * plan_cpu/100), scaled so the
median job matches the synthetic workload scale.  Data volumes, deadlines and the fog/cloud
mapping are NOT in the trace and remain modelled.

Usage (outside this sandbox, after downloading the trace):
    python -m experiments.run_alibaba --batch-task /path/to/batch_task.csv
"""
from __future__ import annotations
import re
import numpy as np
from .workloads import _build

_NAME = re.compile(r"^[A-Za-z]+(\d+)((?:_\d+)*)$")


def parse_batch_task(path, max_jobs=20000, max_tasks=60, min_tasks=2, chunksize=1_000_000):
    import pandas as pd
    cols = ["task_name", "instance_num", "job_name", "task_type", "status", "start_time",
            "end_time", "plan_cpu", "plan_mem"]
    jobs = {}
    for chunk in pd.read_csv(path, header=None, names=cols, chunksize=chunksize):
        chunk = chunk[(chunk.status == "Terminated") & (chunk.end_time > chunk.start_time)]
        for row in chunk.itertuples(index=False):
            m = _NAME.match(str(row.task_name))
            if not m:
                continue
            tid = int(m.group(1))
            parents = [int(x) for x in m.group(2).split("_") if x]
            cpu_s = float(row.instance_num) * float(row.end_time - row.start_time) * float(row.plan_cpu or 100) / 100.0
            jobs.setdefault(row.job_name, []).append((tid, parents, cpu_s, float(row.start_time)))
        if len(jobs) > 4 * max_jobs:
            break
    out = []
    for name, tasks in jobs.items():
        if not (min_tasks <= len(tasks) <= max_tasks):
            continue
        ids = {t[0]: i for i, t in enumerate(tasks)}
        edges, ok = [], True
        for (tid, par, _, _) in tasks:
            for p in par:
                if p not in ids:
                    ok = False
                    break
                edges.append((ids[p], ids[tid]))
        if not ok:
            continue
        out.append(dict(name=name, n=len(tasks), edges=edges,
                        cpu_s=np.array([t[2] for t in tasks]),
                        arrival=min(t[3] for t in tasks)))
        if len(out) >= max_jobs:
            break
    out.sort(key=lambda d: d["arrival"])
    return out


def to_jobs(records, rng, target_median_work=4.0, ccr=0.3, split_frac=0.3):
    """Convert parsed records to simulator Jobs (work rescaled; data volumes modelled)."""
    tot = np.array([r["cpu_s"].sum() for r in records])
    scale = target_median_work / np.median(tot[tot > 0])
    jobs = []
    for r in records:
        w = np.maximum(1e-3, r["cpu_s"] * scale)
        split = (w > np.quantile(w, 0.5)) & (rng.random(r["n"]) < split_frac)
        try:
            j = _build("alibaba", r["n"], sorted(set(r["edges"])), w, rng, split, in_scale=1.0, ccr=ccr)
        except AssertionError:      # cyclic or malformed dependency encoding
            continue
        j.trace_arrival = r["arrival"]
        jobs.append(j)
    return jobs
