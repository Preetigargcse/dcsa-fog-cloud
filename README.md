# DCSA — Decision-Calibrated Selective Admission for fog–cloud workflows

[![tests](https://github.com/Preetigargcse/dcsa-fog-cloud/actions/workflows/tests.yml/badge.svg)](https://github.com/YOUR-GITHUB-USERNAME/dcsa-fog-cloud/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Code, raw results and analysis scripts for the article

> P. Garg. *Decision-Calibrated Selective Admission for Deadline-Constrained Workflows in Fog–Cloud
> Systems.* Submitted to ACM Transactions on Internet Technology (TOIT), 2026.

DCSA is a calibration layer for schedulers that decide **whether** to admit a deadline-constrained DAG
workflow and **which** execution plan to use. It controls the *selective risk*, i.e. the deadline-miss
rate among admitted workflows:

* **DCSA-Off** (offline, Algorithm 1): certifies an admission threshold so that
  P(selective risk > α) ≤ δ under exchangeability (Theorem 1).
* **DCSA-On** (online, Algorithm 2): updates the threshold from delayed outcome feedback
  (λ ← λ − η(L − α)) and guarantees realised selective risk ≤ α + O(1/N) for **any** sequence
  of workloads and system states (Theorem 2).

> **Scope.** All latencies, deadlines, network delays and energy values are produced by the discrete-event
> simulator in this repository; nothing is measured on real hardware. Workflow structures are synthetic
> or shape-faithful reproductions of published scientific workflows.

## Main result (nominal setting, α = 10 %, 20 seeds)

| Policy | Admitted | Misses among admitted | On-time / offered | J per on-time job |
|---|---:|---:|---:|---:|
| FastestMean (admit all) | 100.0 % | 16.1 % | 83.9 % | 41.0 |
| CC-Indep (Monte-Carlo chance constraint) | 91.2 % | 21.5 % | 71.6 % | 33.4 |
| CP-Marginal (per-plan conformal) | 65.4 % | 4.1 % | 62.6 % | 32.5 |
| DCSA-Off | 72.3 % | 5.8 % | 68.1 % | 32.0 |
| **DCSA-On** (energy-first) | 81.0 % | **9.9 %** | 72.9 % | **31.5** |
| **DCSA-On-R** (reliability-first) | 92.3 % | **9.3 %** | **83.7 %** | 39.8 |

Energy is modelled. See `paper/tables/e1_main.tex` and `results/E1_summary.csv` for all policies with
95 % confidence intervals.

## Repository layout

| Path | Content |
|---|---|
| `dcsa/workloads.py` | DAG generators: IoT pipelines, layered random DAGs (Topcuoglu et al.), Montage / Epigenomics / CyberShake / LIGO shapes |
| `dcsa/infra.py` | fog/cloud nodes, network and energy model, Ornstein–Uhlenbeck congestion processes |
| `dcsa/sim.py` | plan family (HEFT-order list scheduling, energy weights, task splitting), predictive Monte-Carlo scoring, ground-truth execution, scheduler-side backlog estimate |
| `dcsa/stream.py` | job streams, baselines, DCSA-Off (`dcsa_offline`), DCSA-On (`DCSAOnline`), closed-loop runner |
| `dcsa/alibaba.py` | parser for Alibaba cluster-trace-v2018 `batch_task.csv` (DAG jobs) |
| `experiments/` | one script per study (see table below) and `analyze.py` (statistics, tables, figures) |
| `results/` | raw per-(study, configuration, seed, policy) results (`*.jsonl`) and summaries (`*.csv`) |
| `figs/` | all figures of the paper (PDF) |
| `paper/` | LaTeX source (`main.tex` = ACM `acmart` / TOIT; `preview.tex` = article-class preview) |
| `tests/` | parser unit test and an end-to-end smoke test |

## Installation

```bash
git clone https://github.com/YOUR-GITHUB-USERNAME/dcsa-fog-cloud.git
cd dcsa-fog-cloud
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tests/test_smoke.py                              # ~30 s
```

Tested with Python 3.13, NumPy 2.5, SciPy 1.18, pandas 3.0 and Matplotlib 3.11. Python ≥ 3.10 should work.

## Reproducing the paper

| Result in paper | Command | Time (2 cores) |
|---|---|---|
| Fig. 1 (Proposition 1) | `python experiments/prop1.py` | < 1 min |
| Table 4, Fig. 3 (main comparison, E1) | `python experiments/run_all.py E1` | 22 min |
| Fig. 6 (risk target, E2) | `python experiments/run_all.py E2 E2S` | 20 min |
| Fig. 4 (load, E3) | `python experiments/run_all.py E3` | 30 min |
| Table 6, Fig. 5 (distribution shift, E4) | `python experiments/run_all.py E4` | 35 min |
| Fig. 8 (correlation, E5) | `python experiments/run_all.py E5` | 18 min |
| Table 8 (ablations, E7) | `python experiments/run_all.py E7` | 12 min |
| Fig. 9 (frontier, E8) | `python experiments/run_all.py E8` | 4 min |
| Table 7, Fig. 7 (exchangeable validity, E9) | `python experiments/exchangeable.py` | 5 min |
| Fig. 10 (overhead, E6) | `python experiments/scalability.py` (run alone: it measures time) | 3 min |
| Table 5 and every table/figure from raw results | `python experiments/analyze.py` | 1 min |

The raw results are included, so `python experiments/analyze.py` regenerates every table and figure
without re-running any simulation. `run_all.py` skips (study, configuration, seed) entries already present
in `results/`; delete a `results/E*.jsonl` file to recompute that study.

## Using DCSA on real traces (Alibaba cluster-trace-v2018)

1. Download `batch_task.csv` from <https://github.com/alibaba/clusterdata> (cluster-trace-v2018).
2. Run
   ```bash
   python experiments/run_alibaba.py --batch-task /path/to/batch_task.csv --seeds 10
   ```
3. Results are appended to `results/ALIBABA.jsonl`. DAG structure, task CPU-seconds and arrival order
   come from the trace; data volumes, deadlines, infrastructure and the noise model remain modelled.

## Using DCSA with your own scheduler

DCSA only needs, for each arriving workflow, a finite set of candidate plans, a risk score per plan in
`[0, s_max]` and a selection cost (e.g. predicted energy):

```python
from dcsa.stream import dcsa_offline, DCSAOnline

# offline: S (n, P, L) scores, E (n, P) costs, R (n, P) realised makespans of every plan, D (n,) deadlines
lam_hat, table = dcsa_offline(S, E, R, D, alpha=0.1, delta=0.1)

# online: admit iff min_p score <= policy.lam; after each admitted workflow finishes (or its deadline passes)
policy = DCSAOnline(alpha=0.1, lam0=lam_hat if lam_hat > 0 else 1.0, eta=0.02, reset_after=200)
policy.feedback(admitted=True, late=False)
```

## Citation

If you use this code, please cite the article (see `CITATION.cff`). The BibTeX entry will be updated on
acceptance.

## Licence

MIT, see [LICENSE](LICENSE).
