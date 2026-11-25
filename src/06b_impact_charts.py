"""Figures 14-15: alpha and sourcing efficiency (out-of-sample years 2016-2019 only)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
INK, MUTED, GREY, GRID = "#0b0b0b", "#898781", "#b4b2a9", "#e1e0d9"
COL = {"Random selection": GREY, "Momentum rule": "#b9d3f2", "Linear regression": "#6fa3e3", "Random forest": "#1f5fae"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7", "axes.grid": True,
                     "grid.color": GRID, "axes.axisbelow": True, "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "xtick.color": "#52514e", "ytick.color": MUTED, "axes.labelcolor": "#52514e", "figure.dpi": 150, "savefig.bbox": "tight", "savefig.facecolor": "white"})
A = pd.read_csv("data/dev/impact_alpha_by_year.csv"); Gn = pd.read_csv("data/dev/impact_sourcing_gains.csv")
labels = ["2016", "2017", "2018", "2019\n(validation)", "Four-year\naverage"]; series = list(COL)
fig, ax = plt.subplots(figsize=(10, 4.4)); w = 0.2
for k, s in enumerate(series):
    v = A[s].values * 100; x = np.arange(len(v)) + (k - 1.5) * w
    ax.bar(x, v, width=w - 0.02, color=COL[s], label=s)
    for xi, vi in zip(x, v): ax.text(xi, vi + 0.06, f"{vi:.1f}", ha="center", fontsize=7.5, color=INK)
ax.axvline(3.5, color="#c3c2b7", lw=1, ls=(0, (3, 3))); ax.grid(axis="x", visible=False)
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels); ax.set_ylim(0, 4.6)
ax.set_ylabel("Return above metro, top-fifth picks (points)")
ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.17))
ax.set_title("Figure 14. Alpha: ZIP codes ranked in the top fifth beat their metro by about 3 points a year\n"
             "On a \\$100 million fund, 1 point = \\$1 million a year, before costs. Each year predicted by a model trained only on earlier years.")
fig.savefig("figures/fig14_alpha_by_year.png"); plt.close(fig)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
cov = Gn.coverage * 100
a1.plot(cov, Gn["Random selection"] * 100, color=GREY, lw=2, ls=(0, (4, 3)), label="Random selection")
for s, ls in [("Momentum rule", (0, (1.5, 2))), ("Random forest", "-")]:
    a1.plot(cov, Gn[s] * 100, color=COL[s] if s == "Random forest" else "#6fa3e3", lw=2, ls=ls, label=s)
a1.axvline(20, color="#c3c2b7", lw=1); r = Gn[np.isclose(Gn.coverage, 0.2)].iloc[0]
a1.annotate(f"Cover 20% of ZIP codes:\nfind {r['Random forest']*100:.0f}% of the true winners", xy=(20, r["Random forest"] * 100), xytext=(32, 40),
            fontsize=8.5, color=INK, arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
a1.set_xlim(0, 100); a1.set_ylim(0, 100); a1.set_xlabel("Share of ZIP codes the team screens (highest-ranked first, %)")
a1.set_ylabel("Share of true outperformers found (%)"); a1.legend(frameon=False, fontsize=8.5, loc="lower right")
a1.set_title("Figure 15a. Where to spend sourcing time")
names = ["Random selection", "Momentum rule", "Linear regression", "Random forest"]
vals = [2.0, r["Momentum rule | winners per 10 screened"], r["Linear regression | winners per 10 screened"], r["Random forest | winners per 10 screened"]]
a2.barh(names[::-1], vals[::-1], color=[COL[n] for n in names[::-1]], height=0.6); a2.grid(axis="y", visible=False)
for i, v in enumerate(vals[::-1]): a2.text(v + 0.08, i, f"{v:.1f}", va="center", fontsize=9, color=INK)
a2.set_xlim(0, 7); a2.set_xlabel("True top-fifth outperformers per 10 ZIP codes screened")
a2.set_title("Figure 15b. Hit rate when the team covers 20%"); a2.tick_params(axis="y", labelsize=9)
fig.tight_layout(); fig.savefig("figures/fig15_sourcing_efficiency.png"); plt.close(fig)
