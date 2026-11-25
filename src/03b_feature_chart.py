"""Figure 8: built features versus the target, training rows only."""
import importlib.util, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
spec = importlib.util.spec_from_file_location("f", "src/03_features.py"); F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
rc = pd.read_csv("data/dev/feature_rank_correlations_train.csv", index_col=0).iloc[:, 0].sort_values()
BLUE, RED, INK, MUTED = "#2a78d6", "#e34948", "#0b0b0b", "#898781"
fig, ax = plt.subplots(figsize=(9, 6.6), dpi=150)
ax.barh([F.FEATURES[f] for f in rc.index], rc.values, color=[RED if v < 0 else BLUE for v in rc.values], height=.68)
for i, v in enumerate(rc.values): ax.text(v + (0.008 if v >= 0 else -0.008), i, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=8.5, color=INK)
ax.axvline(0, color=INK, linewidth=1); ax.set_xlim(-0.6, 0.65)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#c3c2b7"); ax.tick_params(axis="x", colors=MUTED); ax.tick_params(axis="y", labelsize=9, colors="#52514e", length=0)
ax.grid(axis="x", color="#e1e0d9", linewidth=.8); ax.set_axisbelow(True)
ax.set_xlabel("Rank correlation with next year's growth versus metro (within metro and year, training years)", color="#52514e")
ax.set_title("Figure 8. The 20 model inputs: blue = higher value goes with outperformance, red = with underperformance", loc="left", fontsize=10.5, fontweight="bold")
fig.savefig("figures/fig8_features_vs_target.png", bbox_inches="tight", facecolor="white")
