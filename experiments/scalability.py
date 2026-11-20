"""E6: decision overhead vs. DAG size, number of nodes and Monte-Carlo samples (single process)."""
import json, os, sys, time
import numpy as np
from dataclasses import replace
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dcsa.stream import StreamCfg, make_stream, run, Threshold, FastestMean
R = os.path.join(os.path.dirname(__file__), "..", "results")
rows = []
base = StreamCfg(n_jobs=300, rho=0.05, warmup=50, mix={"layered": 1.0})
for v in (10, 25, 50, 100):
    cfg = replace(base, layered_size=(v, v))
    s = make_stream(cfg, 0)
    for pol in (FastestMean(), Threshold(0.1, 1.0, "DCSA")):
        r = run(s, pol); rows.append(dict(axis="tasks", value=v, policy=pol.name, ms=r["dec_ms_mean"], p99=r["dec_ms_p99"]))
        print(rows[-1], flush=True)
for nf in (10, 20, 40, 80):
    cfg = replace(base, layered_size=(25, 25), n_fog=nf, n_cloud=max(2, nf // 5))
    s = make_stream(cfg, 0)
    for pol in (FastestMean(), Threshold(0.1, 1.0, "DCSA")):
        r = run(s, pol); rows.append(dict(axis="nodes", value=nf + max(2, nf // 5), policy=pol.name, ms=r["dec_ms_mean"], p99=r["dec_ms_p99"]))
        print(rows[-1], flush=True)
for M in (8, 16, 32, 64, 128):
    cfg = replace(base, layered_size=(25, 25), M=M)
    s = make_stream(cfg, 0)
    r = run(s, Threshold(0.1, 1.0, "DCSA")); rows.append(dict(axis="M", value=M, policy="DCSA", ms=r["dec_ms_mean"], p99=r["dec_ms_p99"]))
    print(rows[-1], flush=True)
json.dump(rows, open(os.path.join(R, "E6_scalability.json"), "w"), indent=1)
