import json, statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SP = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))
FIGDIR = os.environ.get("FIG_DIR", os.path.join(ROOT, "figures"))
INK, GRID = "#1b1b1f", "#d9d9e0"
PAL = ["#2f5fd0", "#c2481e", "#1f8a5c", "#8a5ad0"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK,
    "ytick.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 200,
})

# ---------------------------------------------------------------- Figure 2
rows = json.load(open(f"{SP}/prior.json"))
EPS = [0.0, 0.1, 0.2, 0.45, 0.9, 1.8]
LAMS = [1.0, 5.0, 25.0]
cold = [r["regret"] for r in rows if r["cond"] == "cold"]
cold_m = st.mean(cold); cold_se = st.stdev(cold) / len(cold) ** 0.5

fig, ax = plt.subplots(figsize=(5.6, 3.4))
ax.axhspan(cold_m - cold_se, cold_m + cold_se, color="#999", alpha=0.18)
ax.axhline(cold_m, color="#555", ls="--", lw=1.2, label=f"cold start ({cold_m:.0f})")
for i, lam in enumerate(LAMS):
    m, se = [], []
    for e in EPS:
        v = [r["regret"] for r in rows if r["cond"] == "warm" and r["eps"] == e and r["lam"] == lam]
        m.append(st.mean(v)); se.append(st.stdev(v) / len(v) ** 0.5)
    ax.errorbar(EPS, m, yerr=se, marker="o", ms=4, lw=1.6, capsize=2.5,
                color=PAL[i], label=f"$\\lambda$ = {lam:g}")
ax.axvline(0.45, color=INK, lw=0.9, ls=":")
ax.annotate("prior error = $\\|\\theta^*\\|$", xy=(0.45, 165), xytext=(0.56, 168),
            fontsize=8, color=INK)
ax.set_xlabel("prior error  $\\varepsilon = \\|\\mu_0 - \\theta^*\\|_2$")
ax.set_ylabel("cumulative regret at T = 3000")
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig_prior.png", bbox_inches="tight")

# ---------------------------------------------------------------- Figure 3
ope = json.load(open(f"{SP}/ope3.json"))
truth = st.mean([r["truth"] for r in ope])
names = ["IPS\nfilter-aware", "IPS\nnaive", "SNIPS\nfilter-aware", "SNIPS\nnaive"]
keys = ["ips_aware", "ips_naive", "snips_aware", "snips_naive"]
vals = [st.mean([r[k] for r in ope]) for k in keys]
errs = [st.stdev([r[k] for r in ope]) / len(ope) ** 0.5 for k in keys]
cols = [PAL[2], PAL[1], PAL[2], PAL[1]]

fig, ax = plt.subplots(figsize=(5.2, 3.0))
b = ax.bar(names, vals, yerr=errs, capsize=3, color=cols, width=0.62)
ax.axhline(truth, color=INK, ls="--", lw=1.2)
ax.annotate(f"true policy value = {truth:.3f}", xy=(3.45, truth), xytext=(-0.45, truth + 0.035),
            fontsize=8)
for rect, v in zip(b, vals):
    ax.text(rect.get_x() + rect.get_width() / 2, v + 0.012, f"{v:.3f}",
            ha="center", fontsize=8)
ax.set_ylabel("estimated policy value")
ax.set_ylim(0, max(vals) * 1.22)
fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig_ope.png", bbox_inches="tight")

# ---------------------------------------------------------------- Figure 4
cap = json.load(open(f"{SP}/capacity.json"))
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9))
ax = axes[0]
g = [r["greedy_gap"] * 100 for r in cap]; d = [r["dual_gap"] * 100 for r in cap]
ax.boxplot([g, d], labels=["greedy\n+ hard check", "dual\nshadow prices"],
           widths=0.5, patch_artist=True,
           boxprops=dict(facecolor="#e8ecf7", color=INK), medianprops=dict(color=PAL[1]))
ax.set_ylabel("optimality gap vs LP (%)")
ax.set_ylim(0, max(g) * 1.2)

ax = axes[1]
seeds = [r["seed"] for r in cap]
ax.plot(seeds, [r["offline"] for r in cap], "o-", ms=3.5, color=INK, label="LP optimum")
ax.plot(seeds, [r["dual"] for r in cap], "s-", ms=3.5, color=PAL[0], label="dual")
ax.plot(seeds, [r["greedy"] for r in cap], "^-", ms=3.5, color=PAL[1], label="greedy")
ax.set_xlabel("seed"); ax.set_ylabel("total realised reward")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig_capacity.png", bbox_inches="tight")

# ---------------------------------------------------------------- Figure 5
res = json.load(open(f"{SP}/results2.json"))
tr = [t for t in res["trace"] if t["cid"].startswith("catalog_warm")]
steps = [t["step"] for t in tr]
fig, ax = plt.subplots(figsize=(6.4, 3.0))
ax.plot(steps, [t["entropy_active"] for t in tr], "-o", ms=3, color=PAL[0],
        label="entropy over active arms (shipped)")
ax.plot(steps, [t["entropy_naive"] for t in tr], "--s", ms=3, color=PAL[1],
        label="entropy over all arms (naive)")
for x, lab in [(4500, "pause"), (6000, "launch"), (9000, "reactivate"), (10500, "retire")]:
    ax.axvline(x, color="#888", lw=0.8, ls=":")
    ax.annotate(lab, xy=(x, 1.005), fontsize=7.5, rotation=0, ha="center", color="#555")
ax.set_xlabel("step"); ax.set_ylabel("normalised selection entropy")
ax.set_ylim(0.6, 1.04)
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig_entropy.png", bbox_inches="tight")

# ---------------------------------------------------------------- Figure 6
files = [(f"{SP}/results2.json", 7), (f"{SP}/results_s11.json", 11), (f"{SP}/results_s23.json", 23)]
arms = ["boot_new", "sneaker_new"]
warm_i = {a: [] for a in arms}; cold_i = {a: [] for a in arms}
warm_e = {a: [] for a in arms}; cold_e = {a: [] for a in arms}
for path, _ in files:
    d = json.load(open(path))
    ks = list(d["runs"])
    w = d["runs"][[k for k in ks if k.startswith(("warm", "catalog_warm"))][0]]
    c = d["runs"][[k for k in ks if k.startswith(("cold", "catalog_cold"))][0]]
    for a in arms:
        warm_i[a].append(w["new_arms"][a]["impressions"])
        cold_i[a].append(c["new_arms"][a]["impressions"])
        warm_e[a].append([e["err"] for e in w["theta_err"] if e["arm"] == a and e["since"] == 2000][0])
        cold_e[a].append([e["err"] for e in c["theta_err"] if e["arm"] == a and e["since"] == 2000][0])

fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9))
x = np.arange(2); w_ = 0.34
ax = axes[0]
ax.bar(x - w_/2, [st.mean(warm_i[a]) for a in arms], w_, color=PAL[0], label="warm start",
       yerr=[st.stdev(warm_i[a])/3**0.5 for a in arms], capsize=3)
ax.bar(x + w_/2, [st.mean(cold_i[a]) for a in arms], w_, color="#9aa0ad", label="cold start",
       yerr=[st.stdev(cold_i[a])/3**0.5 for a in arms], capsize=3)
ax.set_xticks(x); ax.set_xticklabels(["boot_new\n(strong group)", "sneaker_new\n(weak group)"])
ax.set_ylabel("impressions after launch"); ax.legend(frameon=False, fontsize=8)

ax = axes[1]
ax.bar(x - w_/2, [st.mean(warm_e[a]) for a in arms], w_, color=PAL[0],
       yerr=[st.stdev(warm_e[a])/3**0.5 for a in arms], capsize=3)
ax.bar(x + w_/2, [st.mean(cold_e[a]) for a in arms], w_, color="#9aa0ad",
       yerr=[st.stdev(cold_e[a])/3**0.5 for a in arms], capsize=3)
ax.set_xticks(x); ax.set_xticklabels(["boot_new", "sneaker_new"])
ax.set_ylabel(r"$\|\hat\theta - \theta^*\|_2$  at  t+2000")
fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig_catalog.png", bbox_inches="tight")

print("figures written")
