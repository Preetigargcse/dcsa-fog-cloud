"""Real-trace experiment (to be run where the Alibaba v2018 trace is available).

    python experiments/run_alibaba.py --batch-task batch_task.csv [--seeds 10] [--rho 0.15]

Job DAGs, task CPU-seconds and arrival pattern come from the trace; data volumes, deadlines,
infrastructure and the performance model remain the simulator's (paper Section 6.1).
Results are appended to results/ALIBABA.jsonl and summarised by experiments/analyze.py."""
import argparse, os, sys, json
import numpy as np
from dataclasses import replace
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from dcsa.alibaba import parse_batch_task, to_jobs
from dcsa.stream import StreamCfg, make_stream, run, calib_arrays, dcsa_offline, cp_marginal_ratios
from common import make_policy, ETA

ap = argparse.ArgumentParser()
ap.add_argument("--batch-task", required=True)
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--rho", type=float, default=0.15)
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--alpha", type=float, default=0.1)
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results", "ALIBABA.jsonl"))
a = ap.parse_args()

recs = parse_batch_task(a.batch_task, max_jobs=4 * a.n * (a.seeds + 1))
print(f"parsed {len(recs)} DAG jobs")
cfg = StreamCfg(n_jobs=a.n, rho=a.rho)
pols = ["FastestMean", "MeanAdmit", "CC-Indep", "CP-Marginal", "DCSA-Off", "DCSA-On", "DCSA-On-R"]
with open(a.out, "a") as f:
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        # disjoint, chronologically ordered windows: calibration precedes test in trace time
        w = len(recs) // (a.seeds + 1)
        cal_jobs = to_jobs(recs[seed * w // 2: seed * w // 2 + a.n], rng)
        test_jobs = to_jobs(recs[seed * w // 2 + a.n: seed * w // 2 + 2 * a.n], rng)
        cal = make_stream(replace(cfg, n_jobs=len(cal_jobs)), seed, 1, jobs_override=cal_jobs)
        rc = run(cal, make_policy("CC-Indep", a.alpha, {}), record_calib=True)
        S, NOM, E, R, D = calib_arrays(rc["calib"])
        lam, _ = dcsa_offline(S, E, R, D, a.alpha)
        cald = dict(lam0=lam, lam_fp=lam, ratios=cp_marginal_ratios(NOM, R, a.alpha).tolist())
        if lam <= 0:
            print("warning: no threshold certified (calibration set too small?); online policies start at 1.0")
        test = make_stream(replace(cfg, n_jobs=len(test_jobs)), seed, 0, jobs_override=test_jobs)
        for p in pols:
            pol = make_policy(p, a.alpha, cald, ETA)
            if p.startswith("DCSA-On"):
                pol.lam = pol.lam0 = (lam if lam > 0 else 1.0); pol.reset_after = 200
            r = run(test, pol)
            r.update(exp="ALIBABA", tag="alibaba", seed=seed, alpha=a.alpha, policy=p, lam0=lam)
            f.write(json.dumps(r, default=float) + "\n"); f.flush()
            print(seed, p, round(r["admit_rate"], 3), round(r["sel_risk"], 3), round(r["ontime_offered"], 3))
