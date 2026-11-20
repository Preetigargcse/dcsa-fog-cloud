"""Aggregate results, run paired statistics and draw all figures/tables of the paper."""
import json, os, sys, glob
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
R = os.path.join(HERE, "..", "results")
F = os.path.join(HERE, "..", "figs")
T = os.path.join(HERE, "..", "paper", "tables")
os.makedirs(F, exist_ok=True); os.makedirs(T, exist_ok=True)

# validated categorical palette (dataviz reference instance), fixed policy->slot mapping
COL = {"FastestMean": "#2a78d6", "MeanAdmit": "#eb6834", "CC-Indep": "#1baf7a", "CP-Marginal": "#eda100",
       "DCSA-Off": "#e87ba4", "DCSA-On": "#008300", "DCSA-Off-FP": "#6250d6", "DCSA-On-R": "#e34948",
       "DA-Energy": "#8a8a85", "EnergyGreedy": "#b5b5ae"}
MK = {"FastestMean": "o", "MeanAdmit": "s", "CC-Indep": "^", "CP-Marginal": "v", "DCSA-Off": "D",
      "DCSA-On": "P", "DCSA-Off-FP": "X", "DCSA-On-R": "p", "DA-Energy": "h", "EnergyGreedy": "<"}
INK, MUTED, GRID = "#1f1f1e", "#6b6b66", "#e6e6e1"
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8.5, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "lines.linewidth": 2, "lines.markersize": 5, "savefig.dpi": 300})


def load(name):
    p = os.path.join(R, f"{name}.jsonl")
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return None
    rows = [json.loads(l) for l in open(p)]
    for r in rows:
        r.pop("series", None)
    return pd.DataFrame(rows)


def load_series(name):
    p = os.path.join(R, f"{name}.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def paired_boot(a, b, B=10000, seed=0):
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    bs = rng.choice(d, size=(B, len(d)), replace=True).mean(1)
    return d.mean(), np.quantile(bs, 0.025), np.quantile(bs, 0.975)


def holm(pvals):
    p = np.asarray(pvals); order = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0
    for i, k in enumerate(order):
        run = max(run, (m - i) * p[k]); adj[k] = min(1.0, run)
    return adj


METRICS = [("admit_rate", "Admit", 100), ("sel_risk", "Miss|adm", 100), ("ontime_offered", "On-time/off.", 100),
           ("energy_per_ontime", "J/on-time", 1), ("p95_ratio", "p95 T/D", 1), ("dec_ms_mean", "ms/dec.", 1)]


def summary_table(df, policies, fname, caption_note=""):
    lines = []
    for p in policies:
        d = df[df.policy == p]
        if d.empty:
            continue
        cells = [p]
        for m, _, sc in METRICS:
            v = d[m].values * sc
            fmt = "{:.1f}" if sc == 100 else ("{:.2f}" if m in ("p95_ratio", "dec_ms_mean") else "{:.1f}")
            cells.append(fmt.format(np.nanmean(v)) + r"\,{\scriptsize$\pm$" + fmt.format(ci(v)) + "}")
        lines.append(" & ".join(cells) + r" \\")
    head = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
            r"Policy & Admit (\%) & Risk (\%) & On-time/offered (\%) & J per on-time & p95 $T/D$ & ms/decision \\", r"\midrule"]
    open(os.path.join(T, fname), "w").write("\n".join(head + lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")
    return lines


def stats_vs(df, ref, others, metrics, fname):
    """Paired (by seed) bootstrap CI of mean difference ref - other and Holm-corrected Wilcoxon."""
    out = []
    for m in metrics:
        pv, recs = [], []
        for o in others:
            a = df[df.policy == ref].set_index("seed")[m]; b = df[df.policy == o].set_index("seed")[m]
            idx = a.index.intersection(b.index)
            a, b = a.loc[idx].values, b.loc[idx].values
            ok = ~(np.isnan(a) | np.isnan(b)); a, b = a[ok], b[ok]
            mean, lo, hi = paired_boot(a, b)
            try:
                p = stats.wilcoxon(a, b).pvalue
            except ValueError:
                p = 1.0
            pv.append(p); recs.append([m, o, mean, lo, hi, p, int(ok.sum())])
        adj = holm(pv)
        for r, q in zip(recs, adj):
            out.append(r + [q])
    o = pd.DataFrame(out, columns=["metric", "other", "diff", "lo", "hi", "p", "n", "p_holm"])
    o.to_csv(os.path.join(R, fname), index=False)
    return o


# ------------------------------------------------------------------ E1 main
def e1():
    df = load("E1")
    if df is None:
        return
    pols = ["FastestMean", "EnergyGreedy", "DA-Energy", "MeanAdmit", "CC-Indep", "CP-Marginal",
            "DCSA-Off", "DCSA-Off-FP", "DCSA-On", "DCSA-On-R"]
    summary_table(df, pols, "e1_main.tex")
    agg = df.groupby("policy")[[m for m, _, _ in METRICS] + ["split_rate", "cloud_work", "lam0", "lam_fp",
                                                         "lam_final", "max_pending", "calib_s"]].mean()
    agg.to_csv(os.path.join(R, "E1_summary.csv"))
    s1 = stats_vs(df, "DCSA-On", ["FastestMean", "DA-Energy", "MeanAdmit", "CC-Indep", "CP-Marginal", "DCSA-Off", "DCSA-Off-FP"],
                  ["sel_risk", "admit_rate", "ontime_offered", "energy_per_ontime"], "E1_stats_DCSA-On.csv")
    s2 = stats_vs(df, "DCSA-On-R", ["FastestMean", "DA-Energy", "MeanAdmit", "CC-Indep", "CP-Marginal", "DCSA-On"],
                  ["sel_risk", "admit_rate", "ontime_offered", "energy_per_ontime"], "E1_stats_DCSA-On-R.csv")
    # Theorem 2 check
    on = df[df.policy.str.startswith("DCSA-On")].copy()
    on["emp_all"] = on.late_all / on.adm_all
    on["bound"] = on.alpha + (on.lam0 / 0.02 + 2 * (1 - on.alpha) * on.max_pending) / on.adm_all
    on.groupby("policy")[["emp_all", "bound", "max_pending", "adm_all"]].agg(["mean", "max"]).to_csv(os.path.join(R, "E1_thm2.csv"))
    # figure: risk-coverage plane + energy
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
    show = ["FastestMean", "MeanAdmit", "CC-Indep", "CP-Marginal", "DCSA-Off", "DCSA-Off-FP", "DCSA-On", "DCSA-On-R"]
    for p in show:
        d = df[df.policy == p]
        ax[0].scatter(100 * d.admit_rate, 100 * d.sel_risk, s=9, color=COL[p], marker=MK[p], alpha=0.35, linewidths=0)
        ax[0].scatter(100 * d.admit_rate.mean(), 100 * d.sel_risk.mean(), s=46, color=COL[p], marker=MK[p],
                      edgecolor="white", linewidth=1.0, label=p, zorder=3)
        ax[1].scatter(100 * d.ontime_offered.mean(), d.energy_per_ontime.mean(), s=46, color=COL[p], marker=MK[p],
                      edgecolor="white", linewidth=1.0, zorder=3)
        ax[1].errorbar(100 * d.ontime_offered.mean(), d.energy_per_ontime.mean(), xerr=100 * ci(d.ontime_offered),
                       yerr=ci(d.energy_per_ontime), color=COL[p], lw=1, capsize=0, zorder=2)
    ax[0].axhline(10, color=INK, lw=1, ls="--"); ax[0].text(56, 11.0, r"target $\alpha$ = 10%", color=INK, fontsize=7.5)
    ax[0].set_xlim(54, 102)
    off = df[(df.policy == "CP-Marginal") & (df.admit_rate < 0.54)]
    if len(off):
        ax[0].annotate(f"CP-Marginal: {len(off)} seed(s) at 0% admitted (off-axis)", (55, 1.0), fontsize=6.5, color=MUTED)
    ax[0].set_xlabel("Admitted (% of offered)"); ax[0].set_ylabel("Deadline misses among admitted (%)")
    ax[1].set_xlabel("On-time completions (% of offered)"); ax[1].set_ylabel("Energy per on-time job (J, modelled)")
    ax[0].set_title("(a) Coverage vs. selective risk (dots = seeds)", fontsize=8.5, color=INK)
    ax[1].set_title("(b) Service vs. energy (mean, 95% CI)", fontsize=8.5, color=INK)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=8, fontsize=7, handletextpad=0.2,
               columnspacing=0.8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(os.path.join(F, "fig_main.pdf")); plt.close(fig)
    return df


# ------------------------------------------------------------------ E2 alpha sweep
def e2():
    d1 = load("E1"); d2 = load("E2")
    if d2 is None:
        return
    d1 = d1[d1.seed < 10]
    df = pd.concat([d1, d2])
    pols = ["CC-Indep", "CP-Marginal", "DCSA-Off", "DCSA-Off-FP", "DCSA-On"]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    alphas = sorted(df.alpha.unique())
    for p in pols + ["FastestMean"]:
        g = df[df.policy == p].groupby("alpha")
        m = g.sel_risk.mean(); e = g.sel_risk.apply(ci)
        ax[0].errorbar(100 * m.index, 100 * m.values, yerr=100 * e.values, color=COL[p], marker=MK[p], label=p, capsize=0, lw=1.6)
        m = g.admit_rate.mean(); e = g.admit_rate.apply(ci)
        ax[1].errorbar(100 * m.index, 100 * m.values, yerr=100 * e.values, color=COL[p], marker=MK[p], capsize=0, lw=1.6)
    ax[0].plot([4, 21], [4, 21], color=INK, lw=1, ls="--"); ax[0].text(14, 11.5, "risk = target", color=INK, fontsize=7.5, rotation=0)
    ax[0].set_xlabel(r"Target $\alpha$ (%)"); ax[0].set_ylabel("Miss among admitted (%)")
    ax[1].set_xlabel(r"Target $\alpha$ (%)"); ax[1].set_ylabel("Admitted (% of offered)")
    ax[0].set_xticks([5, 10, 20]); ax[1].set_xticks([5, 10, 20])
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=6, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(os.path.join(F, "fig_alpha.pdf")); plt.close(fig)
    df.groupby(["alpha", "policy"])[["admit_rate", "sel_risk", "ontime_offered", "energy_per_ontime", "lam0", "lam_fp"]].mean().to_csv(os.path.join(R, "E2_summary.csv"))


# ------------------------------------------------------------------ E3 load
def e3():
    d1 = load("E1"); d3 = load("E3")
    if d3 is None:
        return
    d1 = d1[(d1.seed < 10) & d1.policy.isin(d3.policy.unique())].copy(); d1["tag"] = "rho0.15"
    df = pd.concat([d1, d3]); df["rho"] = df.tag.str.replace("rho", "").astype(float)
    pols = ["FastestMean", "MeanAdmit", "CC-Indep", "CP-Marginal", "DCSA-Off", "DCSA-On"]
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.4))
    for p in pols:
        g = df[df.policy == p].groupby("rho")
        for k, m in enumerate(["sel_risk", "ontime_offered", "energy_per_ontime"]):
            mm = g[m].mean(); sc = 1 if m == "energy_per_ontime" else 100
            ax[k].plot(mm.index, sc * mm.values, color=COL[p], marker=MK[p], label=p, lw=1.6)
    ax[0].axhline(10, color=INK, lw=1, ls="--")
    ax[0].set_ylabel("Miss among admitted (%)"); ax[1].set_ylabel("On-time / offered (%)"); ax[2].set_ylabel("J per on-time job")
    ax[0].set_ylim(0, 65); ax[2].set_ylim(20, 60)
    for a in ax: a.set_xlabel(r"Offered load $\rho$")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=6, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.09, 1, 1)); fig.savefig(os.path.join(F, "fig_load.pdf")); plt.close(fig)
    df.groupby(["rho", "policy"])[["admit_rate", "sel_risk", "ontime_offered", "energy_per_ontime", "p95_ratio", "lam0"]].mean().to_csv(os.path.join(R, "E3_summary.csv"))


# ------------------------------------------------------------------ E4 shift
def e4():
    df = load("E4")
    if df is None:
        return
    df.groupby(["tag", "policy"])[["admit_rate", "sel_risk", "ontime_offered", "energy_per_ontime"]].agg(["mean"]).to_csv(os.path.join(R, "E4_summary.csv"))
    # post-shift metrics from series
    ser = load_series("E4")
    rows = []
    kinds = ["variance", "congestion", "stragglers", "drift"]
    W = 250
    curves = {}
    for r in ser:
        s = r["series"]; adm = np.array(s["adm"]); late = np.array(s["late"]); n = len(adm)
        h = n // 2 if r["tag"] != "shift-drift" else int(0.75 * n)
        post = slice(h, n)
        na = adm[post].sum(); nl = (adm[post] * late[post]).sum()
        rows.append(dict(tag=r["tag"], policy=r["policy"], seed=r["seed"], post_risk=nl / max(1, na),
                         post_admit=adm[post].mean(), post_ontime=((adm[post] == 1) & (late[post] == 0)).mean()))
        # rolling risk over admitted jobs in windows of W offered jobs
        cs_a = np.concatenate([[0], np.cumsum(adm)]); cs_l = np.concatenate([[0], np.cumsum(adm * late)])
        idx = np.arange(W, n + 1, 25)
        rr = (cs_l[idx] - cs_l[idx - W]) / np.maximum(1, cs_a[idx] - cs_a[idx - W])
        ra = (cs_a[idx] - cs_a[idx - W]) / W
        curves.setdefault((r["tag"], r["policy"]), []).append((idx, rr, ra, np.array(s["lam"], float)))
    post = pd.DataFrame(rows)
    post.to_csv(os.path.join(R, "E4_post.csv"), index=False)
    ps = post.groupby(["tag", "policy"])[["post_risk", "post_admit", "post_ontime"]].agg(["mean", ci])
    ps.to_csv(os.path.join(R, "E4_post_summary.csv"))
    pols_t = ["FastestMean", "MeanAdmit", "CC-Indep", "CP-Marginal", "DCSA-Off", "DCSA-On"]
    L = [r"\begin{tabular}{@{}l" + "rr" * 4 + "@{}}", r"\toprule",
         " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{k}}}" for k in kinds) + r" \\",
         " & " + " & ".join(["Risk", "Admit"] * 4) + r" \\", r"\midrule"]
    for p in pols_t:
        cells = [p]
        for k in kinds:
            r_ = ps.loc[(f"shift-{k}", p)]
            cells += [f"{100*r_[('post_risk','mean')]:.1f}", f"{100*r_[('post_admit','mean')]:.1f}"]
        L.append(" & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(T, "e4_post.tex"), "w").write("\n".join(L) + "\n")
    pols = ["FastestMean", "CC-Indep", "DCSA-Off", "DCSA-On"]
    fig, ax = plt.subplots(2, 4, figsize=(7.2, 3.6), sharex=True)
    for j, k in enumerate(kinds):
        for p in pols:
            c = curves.get((f"shift-{k}", p))
            if not c:
                continue
            idx = c[0][0]
            rr = np.mean([x[1] for x in c], 0); ra = np.mean([x[2] for x in c], 0)
            ax[0, j].plot(idx, 100 * rr, color=COL[p], lw=1.4, label=p)
            ax[1, j].plot(idx, 100 * ra, color=COL[p], lw=1.4)
        for i in range(2):
            ax[i, j].axvline(1500 if k != "drift" else 0, color=MUTED, lw=0.8, ls=":")
        ax[0, j].axhline(10, color=INK, lw=0.9, ls="--")
        ax[0, j].set_title(k, fontsize=8.5, color=INK); ax[1, j].set_xlabel("Job index")
        ax[0, j].set_ylim(0, 45); ax[1, j].set_ylim(30, 102)
    ax[0, 0].set_ylabel("Rolling miss | admitted (%)"); ax[1, 0].set_ylabel("Rolling admission (%)"); fig.align_ylabels(ax[:, 0])
    fig.legend(*ax[0, 0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(os.path.join(F, "fig_shift.pdf")); plt.close(fig)


# ------------------------------------------------------------------ E5 correlation
def e5():
    df = load("E5")
    if df is None:
        return
    df["c"] = df.tag.str.replace("corr", "").astype(float)
    df.groupby(["c", "policy"])[["admit_rate", "sel_risk", "ontime_offered", "energy_per_ontime", "lam0"]].mean().to_csv(os.path.join(R, "E5_summary.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(5.6, 2.4))
    for p in ["FastestMean", "CC-Indep", "DCSA-Off", "DCSA-On"]:
        g = df[df.policy == p].groupby("c")
        ax[0].errorbar(g.sel_risk.mean().index, 100 * g.sel_risk.mean(), yerr=100 * g.sel_risk.apply(ci), color=COL[p], marker=MK[p], label=p, lw=1.6)
        ax[1].errorbar(g.admit_rate.mean().index, 100 * g.admit_rate.mean(), yerr=100 * g.admit_rate.apply(ci), color=COL[p], marker=MK[p], lw=1.6)
    ax[0].axhline(10, color=INK, lw=1, ls="--")
    ax[0].set_xlabel("Common share of congestion variance"); ax[1].set_xlabel("Common share of congestion variance")
    ax[0].set_ylabel("Miss among admitted (%)"); ax[1].set_ylabel("Admitted (%)")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.09, 1, 1)); fig.savefig(os.path.join(F, "fig_corr.pdf")); plt.close(fig)


# ------------------------------------------------------------------ E7 ablation, E8 frontier
def e7():
    df = load("E7")
    if df is None:
        return
    df.groupby(["tag", "policy"])[["admit_rate", "sel_risk", "ontime_offered", "energy_per_ontime", "split_rate", "lam_final", "lam0"]].agg(["mean", ci]).to_csv(os.path.join(R, "E7_summary.csv"))


def e8():
    df = load("E8"); d1 = load("E1")
    if df is None:
        return
    df["lam"] = df.policy.str.replace("Fixed-", "").astype(float)
    g = df.groupby("lam")[["admit_rate", "sel_risk", "ontime_offered", "energy_per_ontime"]].mean()
    g.to_csv(os.path.join(R, "E8_summary.csv"))
    g = g[g.index <= 1.3]
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.5))
    ax[0].plot(100 * g.admit_rate, 100 * g.sel_risk, color=MUTED, lw=1.4, marker="o", ms=3, label=r"fixed $\lambda$ (energy-first)")
    for lam, row in g.iterrows():
        if lam in (0.5, 0.7, 1.0, 1.3):
            ax[0].annotate(rf"$\lambda$={lam:g}", (100 * row.admit_rate, 100 * row.sel_risk), fontsize=6.5, color=MUTED,
                           xytext=(3, -8), textcoords="offset points")
    ax[1].plot(100 * g.ontime_offered, g.energy_per_ontime, color=MUTED, lw=1.4, marker="o", ms=3)
    d1 = d1[d1.seed < 5]
    for p in ["FastestMean", "MeanAdmit", "CC-Indep", "CP-Marginal", "DCSA-Off", "DCSA-On", "DCSA-On-R"]:
        d = d1[d1.policy == p]
        ax[0].scatter(100 * d.admit_rate.mean(), 100 * d.sel_risk.mean(), color=COL[p], marker=MK[p], s=40, edgecolor="white", zorder=3, label=p)
        ax[1].scatter(100 * d.ontime_offered.mean(), d.energy_per_ontime.mean(), color=COL[p], marker=MK[p], s=40, edgecolor="white", zorder=3)
    ax[0].axhline(10, color=INK, lw=1, ls="--")
    ax[0].set_xlabel("Admitted (%)"); ax[0].set_ylabel("Miss among admitted (%)")
    ax[1].set_xlabel("On-time / offered (%)"); ax[1].set_ylabel("J per on-time job")
    ax[0].set_xlim(30, 102); ax[0].set_ylim(0, 48); ax[1].set_xlim(30, 90); ax[1].set_ylim(28, 46)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=6.8, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.14, 1, 1)); fig.savefig(os.path.join(F, "fig_frontier.pdf")); plt.close(fig)


# ------------------------------------------------------------------ E9 exchangeable, E6, prop1
def e9():
    p = os.path.join(R, "E9_exchangeable.json")
    if not os.path.exists(p):
        return
    df = pd.DataFrame(json.load(open(p)))
    s = df.groupby(["alpha", "n_cal", "method"]).agg(risk_mean=("risk", "mean"), risk_q90=("risk", lambda x: np.quantile(x, 0.9)),
                                                    viol=("risk", lambda x: 0), admit=("admit", "mean")).reset_index()
    df["violate"] = df.risk > df.alpha
    viol = df.groupby(["alpha", "n_cal", "method"]).violate.mean().rename("viol_rate").reset_index()
    s = s.drop(columns="viol").merge(viol)
    s.to_csv(os.path.join(R, "E9_summary.csv"), index=False)
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.3), sharey=True)
    for j, a in enumerate((0.05, 0.1, 0.2)):
        d = df[(df.alpha == a) & (df.n_cal == 2000)]
        for k, m in enumerate(["CC-Indep", "CP-Marginal", "DCSA-Off"]):
            v = 100 * d[d.method == m].risk.values
            parts = ax[j].violinplot([v], positions=[k], widths=0.7, showextrema=False)
            for b in parts["bodies"]:
                b.set_facecolor(COL[m]); b.set_alpha(0.55); b.set_edgecolor(COL[m])
            ax[j].plot([k - 0.25, k + 0.25], [np.mean(v)] * 2, color=INK, lw=1.2)
        ax[j].axhline(100 * a, color=INK, lw=1, ls="--")
        ax[j].set_xticks([0, 1, 2]); ax[j].set_xticklabels(["CC-Indep", "CP-Marg.", "DCSA-Off"], fontsize=7)
        ax[j].set_title(rf"$\alpha$ = {int(100*a)}%", fontsize=8.5, color=INK)
    ax[0].set_ylabel("Held-out miss among admitted (%)")
    fig.tight_layout(); fig.savefig(os.path.join(F, "fig_exch.pdf")); plt.close(fig)


def e6():
    p = os.path.join(R, "E6_scalability.json")
    if not os.path.exists(p):
        return
    df = pd.DataFrame(json.load(open(p)))
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.2))
    for j, (axis, lab) in enumerate([("tasks", "Tasks per workflow"), ("nodes", "Nodes"), ("M", "Monte-Carlo samples M")]):
        d = df[df.axis == axis]
        for p, c, m in [("FastestMean", COL["FastestMean"], "o"), ("DCSA", COL["DCSA-On"], "P")]:
            dd = d[d.policy == p]
            if len(dd):
                ax[j].plot(dd.value, dd.ms, color=c, marker=m, label="plan family only" if p == "FastestMean" else "family + MC scoring (DCSA)")
        ax[j].set_xlabel(lab); ax[j].set_xscale("log", base=2); ax[j].set_ylim(0, None)
    ax[0].set_ylabel("Decision time (ms/job)")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(os.path.join(F, "fig_scal.pdf")); plt.close(fig)


def prop1():
    p = os.path.join(R, "prop1.json")
    if not os.path.exists(p):
        return
    df = pd.DataFrame(json.load(open(p)))
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    for (tau, c, m) in [(0.5, "#2a78d6", "o"), (1.0, "#eb6834", "s")]:
        d = df[df.tau == tau]
        ax.plot(d.K, 100 * d.analytic, color=c, marker=m, label=rf"per-plan calibrated, $\tau/\sigma$={tau:g}")
        ax.plot(d.K, 100 * d.decision_level, color=c, marker=m, ls=":", lw=1.4, label=rf"decision-level, $\tau/\sigma$={tau:g}")
    ax.axhline(10, color=INK, lw=1, ls="--")
    ax.set_xscale("log", base=2); ax.set_xlabel("Number of candidate plans K"); ax.set_ylabel("Miss probability of selected plan (%)")
    ax.legend(fontsize=6.3, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(F, "fig_prop1.pdf")); plt.close(fig)


if __name__ == "__main__":
    for f in [e1, e2, e3, e4, e5, e7, e8, e9, e6, prop1]:
        try:
            f(); print("ok", f.__name__)
        except Exception as ex:
            import traceback; traceback.print_exc(); print("FAIL", f.__name__)


def thm2_all():
    """Theorem 2 check on every DCSA-On run of every closed-loop study."""
    out = []
    for name in ["E1", "E2", "E2S", "E3", "E4", "E5", "E7"]:
        df = load(name)
        if df is None:
            continue
        d = df[df.policy.str.startswith("DCSA-On") & df.adm_all.gt(0)].copy()
        if d.empty:
            continue
        eta = d.policy.str.extract(r"eta([0-9.]+)")[0].astype(float).fillna(0.02)
        lam1 = np.where((d.policy == "DCSA-On-cold") | ((d.policy == "DCSA-On-S") & (d.lam0 <= 0)), 1.0, d.lam0)
        d["emp"] = d.late_all / d.adm_all
        d["bound"] = d.alpha + (lam1 / eta + 2 * (1 - d.alpha) * d.max_pending) / d.adm_all
        d["study"] = name
        out.append(d[["study", "tag", "policy", "seed", "alpha", "emp", "bound", "max_pending", "adm_all", "lam_final"]])
    o = pd.concat(out)
    o.to_csv(os.path.join(R, "thm2_all.csv"), index=False)
    s = o.groupby("study").agg(runs=("emp", "size"), held=("emp", lambda x: 0), emp_minus_alpha_max=("emp", "max"),
                               d_max=("max_pending", "max"), lam_final_min=("lam_final", "min"))
    s["held"] = o.assign(h=o.emp <= o.bound).groupby("study").h.mean()
    s["max_excess"] = o.assign(e=o.emp - o.alpha).groupby("study").e.max()
    s["mean_excess"] = o.assign(e=o.emp - o.alpha).groupby("study").e.mean()
    s["mean_bound_excess"] = o.assign(e=o.bound - o.alpha).groupby("study").e.mean()
    s.to_csv(os.path.join(R, "thm2_summary.csv"))
    print(s)
    names = {"E1": "E1 main", "E2": "E2 risk target", "E2S": "E2 safeguarded", "E3": "E3 load", "E4": "E4 shift", "E5": "E5 correlation", "E7": "E7 ablations"}
    lines = [r"\begin{table}[t]", r"\caption{Theorem~\ref{thm:online} on every closed-loop DCSA-On run (all variants, step sizes and targets). "
             r"Held = share of runs satisfying the bound; excess = realised selective risk minus $\alpha$ over the whole stream; slack = right-hand side of~\eqref{eq:online} minus $\alpha$ (both in percentage points).}",
             r"\label{tab:thm}", r"\small", r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r"Study & Runs & Held & Mean excess & Max excess & Mean slack & Max $d$ \\", r"\midrule"]
    for st, row in s.iterrows():
        lines.append(f"{names.get(st, st)} & {int(row.runs)} & {100*row.held:.0f}\\% & {100*row.mean_excess:+.2f} & {100*row.max_excess:+.2f} & {100*row.mean_bound_excess:.2f} & {int(row.d_max)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(T, "thm2.tex"), "w").write("\n".join(lines) + "\n")
    open(os.path.join(T, "macros_thm.tex"), "w").write(
        "\\newcommand{\\thmmaxexcess}{%.1f}\n\\newcommand{\\thmdmax}{%d}\n" % (100 * o.assign(e=o.emp - o.alpha).e.max(), o.max_pending.max()))


if __name__ == "__main__":
    thm2_all()


def tables_extra():
    # E7 ablation table (+ reference: DCSA-On / CC-Indep on E1 seeds 0-9)
    d7 = load("E7"); d1 = load("E1")
    ref = d1[(d1.seed < 10) & d1.policy.isin(["DCSA-On", "CC-Indep", "DCSA-Off"])].copy(); ref["tag"] = "full"
    df = pd.concat([ref, d7])
    rows = [("full", "DCSA-On", r"DCSA-On (reference)"),
            ("abl-online", "DCSA-On-cold", r"cold start $\lambda_1=1$"),
            ("abl-online", "DCSA-On-eta0.005", r"$\eta=0.005$"), ("abl-online", "DCSA-On-eta0.01", r"$\eta=0.01$"),
            ("abl-online", "DCSA-On-eta0.05", r"$\eta=0.05$"), ("abl-online", "DCSA-On-eta0.1", r"$\eta=0.1$"),
            ("abl-nosplit", "DCSA-On", "no splitting (7 plans)"), ("abl-nosplit", "CC-Indep", r"\quad CC-Indep, no splitting"),
            ("abl-single", "DCSA-On", "single plan (HEFT with splitting)"), ("abl-single", "CC-Indep", r"\quad CC-Indep, single plan"),
            ("full", "CC-Indep", r"\quad CC-Indep, 8 plans (reference)")]
    L = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
         r"Variant & Admit (\%) & Risk (\%) & On-time/offered (\%) & J per on-time \\", r"\midrule"]
    for tag, pol, lab in rows:
        d = df[(df.tag == tag) & (df.policy == pol)]
        L.append(f"{lab} & {100*d.admit_rate.mean():.1f} & {100*d.sel_risk.mean():.1f}\\,{{\\scriptsize$\\pm${100*ci(d.sel_risk):.1f}}} & "
                 f"{100*d.ontime_offered.mean():.1f} & {d.energy_per_ontime.mean():.1f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(T, "e7.tex"), "w").write("\n".join(L) + "\n")
    # E9 table
    s = pd.read_csv(os.path.join(R, "E9_summary.csv"))
    L = [r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
         r" & & \multicolumn{3}{c}{$n_{\mathrm{cal}}=500$} & \multicolumn{3}{c}{$n_{\mathrm{cal}}=2000$} \\",
         r"$\alpha$ & Method & Risk & Viol. & Admit & Risk & Viol. & Admit \\", r"\midrule"]
    for a in (0.05, 0.1, 0.2):
        for m in ("CC-Indep", "CP-Marginal", "DCSA-Off"):
            cells = [f"{int(100*a)}\\%" if m == "CC-Indep" else "", m]
            for n in (500, 2000):
                r_ = s[(s.alpha == a) & (s.n_cal == n) & (s.method == m)].iloc[0]
                cells += [f"{100*r_.risk_mean:.1f}", f"{100*r_.viol_rate:.1f}", f"{100*r_.admit:.1f}"]
            L.append(" & ".join(cells) + r" \\")
        if a != 0.2:
            L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(T, "e9.tex"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    tables_extra()
