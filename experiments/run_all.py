"""All simulation experiments of the paper.  Usage: python experiments/run_all.py E1 [E2 ...]"""
import sys, os
from dataclasses import replace
sys.path.insert(0, os.path.dirname(__file__))
from common import run_batch, ALL
from dcsa.stream import StreamCfg
from dcsa.infra import TruthParams

R = os.path.join(os.path.dirname(__file__), "..", "results")
BASE = StreamCfg(n_jobs=2000, rho=0.15)
SWEEP = ["FastestMean", "MeanAdmit", "CC-Indep", "CP-Marginal", "DCSA-Off", "DCSA-On"]


class shift_fn:
    """Picklable truth-parameter schedule: frac in [0,1] (position in stream) -> TruthParams."""
    def __init__(self, kind):
        self.kind = kind
    def __call__(self, frac):
        T = TruthParams(); kind = self.kind
        if kind == "drift":
            m = 1 + 0.8 * frac
            return replace(T, sigma_task=T.sigma_task * m, sigma_glob=T.sigma_glob * m, sigma_node=T.sigma_node * m)
        if frac < 0.5:
            return T
        if kind == "variance":
            return replace(T, sigma_task=T.sigma_task * 1.6, sigma_glob=T.sigma_glob * 1.6, sigma_node=T.sigma_node * 1.6)
        if kind == "congestion":
            return replace(T, sigma_glob=0.45)
        if kind == "stragglers":
            return replace(T, p_strag=0.08)
        raise KeyError(kind)


def jobs_for(exp):
    J = []
    if exp == "E1":     # main comparison
        for s in range(20):
            J.append(("E1", "main", BASE, s, 0.1, ALL + ["DCSA-On-R"], {}))
    if exp == "E2":     # risk-target sweep (alpha = 0.1 is taken from E1)
        for a in (0.05, 0.2):
            for s in range(10):
                J.append(("E2", f"alpha{a}", BASE, s, a, ["FastestMean", "CC-Indep", "CP-Marginal", "DCSA-Off", "DCSA-Off-FP", "DCSA-On"], {}))
    if exp == "E2S":    # safeguarded online variant at the strictest target
        for s in range(10):
            J.append(("E2S", "alpha0.05-safe", BASE, s, 0.05, ["DCSA-On-S"], {"series": True}))
    if exp == "E3":     # load sweep
        for rho in (0.05, 0.10, 0.20, 0.25, 0.30):
            for s in range(10):
                J.append(("E3", f"rho{rho}", replace(BASE, rho=rho), s, 0.1, SWEEP, {}))
    if exp == "E4":     # distribution shift at mid-stream / gradual drift
        for kind in ("variance", "congestion", "stragglers", "drift"):
            for s in range(10):
                J.append(("E4", f"shift-{kind}", replace(BASE, n_jobs=3000, shift=shift_fn(kind)), s, 0.1, SWEEP, {"series": True}))
    if exp == "E5":     # cross-node correlation at constant total congestion variance
        tot = 0.2 ** 2 + 0.2 ** 2
        for c in (0.0, 0.33, 0.67, 1.0):
            tp = replace(TruthParams(), sigma_glob=(c * tot) ** 0.5, sigma_node=((1 - c) * tot) ** 0.5)
            for s in range(10):
                J.append(("E5", f"corr{c}", replace(BASE, truth=tp), s, 0.1, ["FastestMean", "CC-Indep", "DCSA-Off", "DCSA-On"], {}))
    if exp == "E7":     # ablations
        for s in range(10):
            J.append(("E7", "abl-online", BASE, s, 0.1, ["DCSA-On-cold", "DCSA-On-eta0.005", "DCSA-On-eta0.01", "DCSA-On-eta0.05", "DCSA-On-eta0.1"], {}))
            J.append(("E7", "abl-nosplit", replace(BASE, family="nosplit"), s, 0.1, ["CC-Indep", "DCSA-Off", "DCSA-On"], {}))
            J.append(("E7", "abl-single", replace(BASE, family="heft"), s, 0.1, ["CC-Indep", "DCSA-Off", "DCSA-On"], {}))
    if exp == "E8":     # frontier: fixed thresholds
        for s in range(5):
            J.append(("E8", "frontier", BASE, s, 0.1, [f"Fixed-{l}" for l in (0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.3, 2.0, 4.0)], {}))
    return J


if __name__ == "__main__":
    for exp in sys.argv[1:]:
        run_batch(jobs_for(exp), os.path.join(R, f"{exp}.jsonl"))
