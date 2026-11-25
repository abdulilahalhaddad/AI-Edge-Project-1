"""Figure 13: image inputs versus tabular only, validation year 2019."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = pd.read_csv("data/dev/image_experiment.csv")
runs = ["Images only", "Tabular only", "+ colour statistics", "+ all image inputs", "+ shuffled images (placebo)"]
cols = ["#eda100", "#2a78d6", "#7fb0e8", "#b9d3f2", "#b4b2a9"]; models = list(R.model.unique())
fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150); w = .16
for k, (run, c) in enumerate(zip(runs, cols)):
    v = [R[(R.model == m) & (R.inputs == run)].val_rank_corr.iloc[0] for m in models]
    ax.bar(np.arange(3) + (k - 2) * w, v, width=w - .015, color=c, label=run)
    for i, x in enumerate(v): ax.text(i + (k - 2) * w, x + .008, f"{x:.2f}", ha="center", fontsize=8, color="#0b0b0b")
ax.set_xticks(range(3)); ax.set_xticklabels(models); ax.set_ylim(0, .68); ax.set_ylabel("Rank correlation within metro (2019)", color="#52514e")
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#e1e0d9"); ax.set_axisbelow(True); ax.tick_params(colors="#52514e")
ax.legend(frameon=False, fontsize=8.5, ncol=5, loc="upper center", bbox_to_anchor=(.5, -.1))
ax.set_title("Figure 13. Validation 2019: adding image inputs does not improve the ranking", loc="left", fontsize=10.5, fontweight="bold")
fig.savefig("figures/fig13_image_experiment.png", bbox_inches="tight", facecolor="white")
