"""Figure 17: what happened on the unseen years (after the one-time test)."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
A = pd.read_csv("results/3_impact/impact_alpha_by_year.csv").iloc[:4]
T = pd.read_csv("results/4_test/test_results.csv"); Dg = pd.read_csv("results/4_test/diagnosis_relationships_by_year.csv").set_index("target_year")
yrs = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
def t(model, y): return float(T[(T.model == model) & (T.period.str.startswith(str(y)))]["Top-fifth alpha (points)"].iloc[0])
rf = list(A["Random forest"] * 100) + [t("Random forest", y) for y in yrs[4:]]
mo = list(A["Momentum rule"] * 100) + [t("Baseline: scaled momentum", y) for y in yrs[4:]]
INK, MUTED, GRID = "#14213D", "#6B7280", "#E1E0D9"
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#C3C2B7", "axes.grid": True, "grid.color": GRID,
                     "axes.axisbelow": True, "xtick.color": "#4A5568", "ytick.color": MUTED, "axes.labelcolor": "#4A5568", "figure.dpi": 160, "savefig.bbox": "tight", "savefig.facecolor": "#F7F5F0", "axes.facecolor": "#F7F5F0"})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 5.6), facecolor="#F7F5F0")
x = np.arange(len(yrs)); w = 0.38
for ax in (a1, a2): ax.axvspan(3.5, 7.5, color="#F3E3D3", zorder=0)
a1.bar(x - w / 2, mo, w, color="#A9B1BF", label="Momentum rule"); a1.bar(x + w / 2, rf, w, color="#2A6FD6", label="Random forest")
for i, v in enumerate(rf): a1.text(i + w / 2, v + 0.06, f"{v:.1f}", ha="center", fontsize=11, color=INK)
a1.set_xticks(x); a1.set_xticklabels(yrs); a1.set_ylabel("Top-fifth return above metro (points)"); a1.grid(axis="x", visible=False); a1.set_xlim(-0.6, 7.6)
a1.text(1.5, 4.25, "Development years", ha="center", color=MUTED, fontsize=12); a1.text(5.5, 4.25, "Unseen: stress and test", ha="center", color="#9A4A0F", fontsize=12)
a1.set_ylim(0, 4.6); a1.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 0.93), fontsize=12)
a1.set_title("Model alpha by year", loc="left", fontsize=15, fontweight="bold", color=INK)
a2.axhline(0, color=INK, lw=1)
a2.plot(x, Dg.loc[yrs, "momentum"], color="#2A6FD6", lw=3, marker="o", label="Momentum (last year's winners)")
a2.plot(x, Dg.loc[yrs, "price_vs_metro"], color="#D9822B", lw=3, marker="s", ls=(0, (5, 2)), label="Price level vs metro (negative = cheap wins)")
a2.set_xticks(x); a2.set_xticklabels(yrs); a2.set_ylim(-0.7, 0.75); a2.set_xlim(-0.6, 7.6); a2.grid(axis="x", visible=False)
a2.set_ylabel("Link to next year's outperformance\n(rank correlation within metro)"); a2.legend(frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.6), fontsize=11.5)
a2.set_title("The two signals it relies on faded from 2021", loc="left", fontsize=15, fontweight="bold", color=INK)
fig.tight_layout(); fig.savefig("figures/4_test/fig17_unseen_years.png")
