"""Shared experiment harness: calibration + closed-loop evaluation of a policy set for one
(configuration, seed) pair.  Results are written as JSON lines."""
from __future__ import annotations
import json, os, sys, time
from dataclasses import replace
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dcsa.stream import (StreamCfg, make_stream, run, calib_arrays, dcsa_offline, cp_marginal_ratios,
                         FastestMean, EnergyAll, DAEnergy, MeanAdmit, Threshold, CPMarginal, DCSAOnline)

ETA = 0.02
DELTA = 0.1
ALL = ["FastestMean", "EnergyGreedy", "DA-Energy", "MeanAdmit", "CC-Indep", "CP-Marginal",
       "DCSA-Off", "DCSA-Off-FP", "DCSA-On"]


def calibrate(cfg, seed, alpha, fp_iters=0, select="energy", n_cal=None, delta=DELTA):
    """Offline calibration on an independent stream (role=1) generated under nominal truth."""
    ccfg = replace(cfg, shift=None, n_jobs=n_cal or cfg.n_jobs)
    cal = make_stream(ccfg, seed, role=1)
    t = time.perf_counter()
    rc = run(cal, Threshold(alpha, 1.0, "ref"), record_calib=True)
    S, NOM, E, R, D = calib_arrays(rc["calib"], warmup=ccfg.warmup)
    lam, rows = dcsa_offline(S, E, R, D, alpha, delta=delta, select=select)
    ratios = cp_marginal_ratios(NOM, R, alpha)
    out = dict(lam0=lam, ratios=ratios.tolist(), lams=[lam], calib_s=time.perf_counter() - t)
    for _ in range(fp_iters):
        rc = run(cal, Threshold(alpha, max(lam, 1e-3), "ref", select), record_calib=True)
        S, NOM, E, R, D = calib_arrays(rc["calib"], warmup=ccfg.warmup)
        lam, rows = dcsa_offline(S, E, R, D, alpha, delta=delta, select=select)
        out["lams"].append(lam)
    out["lam_fp"] = lam
    return out


def make_policy(name, alpha, cal, eta=ETA):
    if name == "FastestMean": return FastestMean()
    if name == "EnergyGreedy": return EnergyAll()
    if name == "DA-Energy": return DAEnergy()
    if name == "MeanAdmit": return MeanAdmit()
    if name == "CC-Indep": return Threshold(alpha, 1.0, "CC-Indep")
    if name == "CP-Marginal": return CPMarginal(np.array(cal["ratios"]))
    if name == "DCSA-Off": return Threshold(alpha, cal["lam0"], "DCSA-Off")
    if name == "DCSA-Off-FP": return Threshold(alpha, cal["lam_fp"], "DCSA-Off-FP")
    if name == "DCSA-On": return DCSAOnline(alpha, cal["lam0"], eta, "DCSA-On")
    if name == "DCSA-On-cold": return DCSAOnline(alpha, 1.0, eta, "DCSA-On-cold")
    if name == "DCSA-On-S":   # safeguarded: fall back to lambda_1 = 1 without a certificate; reset rule (Sec. 5.3)
        l1 = cal["lam0"] if cal["lam0"] > 0 else 1.0
        return DCSAOnline(alpha, l1, eta, "DCSA-On-S", reset_after=200)
    if name == "DCSA-On-R": return DCSAOnline(alpha, cal["lam0"], eta, "DCSA-On-R", select="risk")
    if name.startswith("DCSA-On-eta"):
        e = float(name.split("eta")[1]); return DCSAOnline(alpha, cal["lam0"], e, name)
    if name.startswith("Fixed-"):
        lam = float(name.split("-")[1]); return Threshold(alpha, lam, name)
    raise KeyError(name)


def run_one(args):
    exp, tag, cfg, seed, alpha, policies, opts = args
    t0 = time.time()
    need_cal = any(p.startswith("DCSA") or p == "CP-Marginal" for p in policies)
    fp = 2 if "DCSA-Off-FP" in policies else 0
    cal = calibrate(cfg, seed, alpha, fp_iters=fp, n_cal=opts.get("n_cal")) if need_cal else {}
    test = make_stream(cfg, seed, role=0)
    rows = []
    for name in policies:
        pol = make_policy(name, alpha, cal, opts.get("eta", ETA))
        r = run(test, pol, record_series=opts.get("series", False))
        r.update(exp=exp, tag=tag, seed=seed, alpha=alpha, policy=name,
                 lam0=cal.get("lam0"), lam_fp=cal.get("lam_fp"), calib_s=cal.get("calib_s"))
        if "series" in r:
            sr = r.pop("series")
            r["series"] = {k: np.asarray(v).tolist() for k, v in sr.items()}
        rows.append(r)
    for r in rows:
        r["wall_s"] = time.time() - t0
    return rows


def run_batch(jobs, out_path, procs=2):
    from multiprocessing import Pool
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            d = json.loads(line)
            done.add((d["tag"], d["seed"], d["alpha"]))
    todo = [j for j in jobs if (j[1], j[3], j[4]) not in done]
    print(f"{out_path}: {len(todo)} of {len(jobs)} tasks to run", flush=True)
    with Pool(procs) as pool, open(out_path, "a") as f:
        for rows in pool.imap_unordered(run_one, todo):
            for r in rows:
                f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            print(rows[0]["tag"], rows[0]["seed"], f"{rows[0]['wall_s']:.0f}s", flush=True)
