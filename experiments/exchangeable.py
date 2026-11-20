"""E9: validity under exchangeability.  Counterfactual calibration records (all plans executed
against the same state) from long reference streams are split at random into calibration and
held-out halves; this is exactly the setting of Theorem 1, free of closed-loop feedback.
Also: calibration-set-size and delta sensitivity."""
import json, os, sys, pickle
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dcsa.stream import (StreamCfg, make_stream, run, Threshold, calib_arrays, dcsa_offline,
                         cp_marginal_ratios, rule_outcomes, level_index)
from multiprocessing import Pool
R = os.path.join(os.path.dirname(__file__), "..", "results")

def records(seed):
    cfg = StreamCfg(n_jobs=4000, rho=0.15)
    s = make_stream(cfg, seed, role=2)
    rc = run(s, Threshold(0.1, 1.0, "ref"), record_calib=True)
    return calib_arrays(rc["calib"], warmup=100)

def evaluate(S, NOM, E, R_, D, a, b, alpha, delta=0.1):
    li = level_index(alpha)
    lam, _ = dcsa_offline(S[a], E[a], R_[a], D[a], alpha, delta=delta)
    out = {}
    ok, lt = rule_outcomes(S[b], E[b], R_[b], D[b], li, lam); out["DCSA-Off"] = (lt.sum() / max(1, ok.sum()), ok.mean())
    ok, lt = rule_outcomes(S[b], E[b], R_[b], D[b], li, 1.0); out["CC-Indep"] = (lt.sum() / max(1, ok.sum()), ok.mean())
    rat = cp_marginal_ratios(NOM[a], R_[a], alpha)
    feas = rat * NOM[b] <= D[b][:, None]; okc = feas.any(1); p = np.argmin(np.where(feas, E[b], np.inf), 1)
    ltc = (R_[b][np.arange(len(b)), p] > D[b]) & okc
    out["CP-Marginal"] = (ltc.sum() / max(1, okc.sum()), okc.mean())
    # oracle: largest grid lambda whose held-out selective risk <= alpha (not implementable)
    return out, lam

if __name__ == "__main__":
    path = os.path.join(R, "exch_records.pkl")
    if os.path.exists(path):
        recs = pickle.load(open(path, "rb"))
    else:
        with Pool(2) as p:
            recs = p.map(records, range(10))
        pickle.dump(recs, open(path, "wb"))
    S, NOM, E, R_, D = [np.concatenate([r[i] for r in recs]) for i in range(5)]
    n = len(D); print("records", n)
    rng = np.random.default_rng(0)
    rows = []
    for alpha in (0.05, 0.1, 0.2):
        for n_cal in (250, 500, 1000, 2000, 4000):
            for rep in range(200):
                idx = rng.permutation(n); a, b = idx[:n_cal], idx[n_cal:]
                out, lam = evaluate(S, NOM, E, R_, D, a, b, alpha)
                for k, (risk, adm) in out.items():
                    rows.append(dict(alpha=alpha, n_cal=n_cal, rep=rep, method=k, risk=float(risk), admit=float(adm), lam=lam))
        print(alpha, "done", flush=True)
    json.dump(rows, open(os.path.join(R, "E9_exchangeable.json"), "w"))
