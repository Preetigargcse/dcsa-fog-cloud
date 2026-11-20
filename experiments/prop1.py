"""Numerical illustration of Proposition 1 (post-selection failure of per-plan calibration)
and of decision-level recalibration on the same synthetic model."""
import json, os, numpy as np
from scipy.stats import norm
from scipy.integrate import quad

def miss_after_selection(K, alpha, sigma, tau):
    s = np.sqrt(sigma**2 + tau**2); z = norm.ppf(1 - alpha)
    # density of the minimum of K iid N(0, tau^2)
    f = lambda e: K * norm.pdf(e / tau) / tau * norm.sf(e / tau) ** (K - 1)
    g = lambda e: norm.sf((z * s + e) / sigma)
    return quad(lambda e: f(e) * g(e), -12 * tau, 12 * tau, limit=200)[0]

def mc_check(K, alpha, sigma, tau, n=400000, seed=0):
    rng = np.random.default_rng(seed)
    s = np.sqrt(sigma**2 + tau**2); z = norm.ppf(1 - alpha)
    e = rng.normal(0, tau, (n, K)); xi = rng.normal(0, sigma, (n, K))
    k = e.argmin(1)
    return float(np.mean(xi[np.arange(n), k] - e[np.arange(n), k] > z * s))

def decision_level(K, alpha, sigma, tau, n_cal=2000, n_test=20000, seed=1, reps=100):
    """Calibrate the additive margin c of the *selected* plan (split conformal on the decision)."""
    rng = np.random.default_rng(seed)
    def draw(n):
        e = rng.normal(0, tau, (n, K)); xi = rng.normal(0, sigma, (n, K))
        k = e.argmin(1); i = np.arange(n)
        return xi[i, k] - e[i, k]          # latency minus point estimate of the selected plan
    vals = []
    for _ in range(reps):   # average over calibration draws: the guarantee is in expectation
        r = np.sort(draw(n_cal)); c = r[int(np.ceil((n_cal + 1) * (1 - alpha))) - 1]
        vals.append(np.mean(draw(n_test) > c))
    return float(np.mean(vals))

out = []
for (sigma, tau) in [(1.0, 0.5), (1.0, 1.0)]:
    for K in (1, 2, 4, 8, 16, 32):
        m = miss_after_selection(K, 0.1, sigma, tau)
        out.append(dict(sigma=sigma, tau=tau, K=K, analytic=m, mc=mc_check(K, 0.1, sigma, tau),
                        decision_level=decision_level(K, 0.1, sigma, tau)))
        print(out[-1])
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", "prop1.json"), "w"), indent=1)
