"""Figures 9-12 for Stage 4. Development data only."""
import pickle, importlib.util, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
spec = importlib.util.spec_from_file_location("f", "src/03_features.py"); F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
BLUE, RED, GREY, INK, MUTED = "#2a78d6", "#e34948", "#b4b2a9", "#0b0b0b", "#898781"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": "#e1e0d9",
                     "axes.axisbelow": True, "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "xtick.color": MUTED, "ytick.color": "#52514e",
                     "axes.labelcolor": "#52514e", "figure.dpi": 150, "savefig.bbox": "tight", "savefig.facecolor": "white"})
R = pd.read_csv("data/dev/model_results.csv"); R = R[R.model != "Baseline: zero"].iloc[::-1]
P = pd.read_csv("data/dev/validation_predictions.csv"); S = pickle.load(open("data/dev/models_stage4.pkl", "rb"))
print(R[["model", "val_rmse_points"]].round(5).to_string(index=False))
# Figure 9
fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
for ax, (col, title, fmt) in zip(axes, [("val_rank_corr", "Rank correlation, predicted vs actual", "{:.2f}"),
        ("val_top_minus_bottom_points", "Top fifth minus bottom fifth (points)", "{:.1f}"), ("val_top_fifth_beat_rate", "Top-fifth picks that beat their metro", "{:.0%}")]):
    cols = [GREY if m.startswith("Baseline") else BLUE for m in R.model]
    ax.barh(R.model, R[col], color=cols, height=.65); ax.set_title(title); ax.grid(axis="y", visible=False)
    for i, v in enumerate(R[col]): ax.text(v, i, " " + fmt.format(v), va="center", fontsize=9, color=INK)
    ax.set_xlim(0, R[col].max() * 1.18)
fig.suptitle("Figure 9. Validation year 2019: every model beats the momentum baselines (grey)", x=0.01, ha="left", fontsize=11.5, fontweight="bold", y=1.03)
fig.savefig("figures/2_models/fig09_model_comparison.png"); plt.close(fig)
# Figure 10: the decision tree, top three levels
tree = S["fitted"]["Decision tree"]; names = [F.FEATURES.get(f, "Two-week figure missing") for f in S["feats"]]
fig, ax = plt.subplots(figsize=(20, 6)); ax.grid(False)
plot_tree(tree, feature_names=names, max_depth=2, filled=False, impurity=False, rounded=True, fontsize=9, precision=3, ax=ax)
ax.set_title("Figure 10. The decision tree (top three of four levels). 'value' is predicted growth versus metro: 0.02 = 2 points. Inputs are measured against the metro average.", fontsize=11)
fig.savefig("figures/2_models/fig10_decision_tree.png"); plt.close(fig)
# Figure 11: drivers
d = pd.read_csv("data/dev/model_drivers.csv").set_index("feature"); d.index = names
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 6.3), sharey=False)
c = d.linear_coef_points_per_sd.sort_values()
a1.barh(c.index, c.values, color=[RED if v < 0 else BLUE for v in c.values], height=.68); a1.axvline(0, color=INK, lw=1); a1.grid(axis="y", visible=False)
a1.set_title("Linear regression: effect of a one-standard-deviation rise (points)"); a1.tick_params(axis="y", labelsize=8.5)
p = d.boosted_permutation_importance_points.clip(lower=0).sort_values()
a2.barh(p.index, p.values, color=BLUE, height=.68); a2.grid(axis="y", visible=False); a2.tick_params(axis="y", labelsize=8.5)
a2.set_title("Boosted trees: error added when the input is scrambled (points, 2019)")
fig.suptitle("Figure 11. What drives the predictions", x=0.01, ha="left", fontsize=11.5, fontweight="bold", y=1.0); fig.tight_layout()
fig.savefig("figures/2_models/fig11_drivers.png"); plt.close(fig)
# Figure 12: predicted fifths vs actual
show = [("Baseline: scaled momentum", GREY), ("Linear regression", "#9ec3ee"), ("Random forest", BLUE)]
fig, ax = plt.subplots(figsize=(8.5, 3.9)); w = .26
for k, (m, col) in enumerate(show):
    q = P.groupby("city")[m].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False)); g = P.groupby(q).actual.mean() * 100
    ax.bar(np.arange(5) + (k - 1) * w, g.values, width=w - .02, color=col, label=m.replace("Baseline: s", "S"))
    for i, v in enumerate(g.values): ax.text(i + (k - 1) * w, v + (0.07 if v >= 0 else -0.07), f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5, color=INK)
ax.axhline(0, color=INK, lw=1); ax.set_xticks(range(5)); ax.set_xticklabels(["Bottom fifth", "2", "3", "4", "Top fifth"]); ax.grid(axis="x", visible=False)
ax.set_xlabel("Model's ranking of ZIP codes within each metro (December 2018)"); ax.set_ylabel("Actual growth vs metro in 2019 (points)")
ax.legend(frameon=False, fontsize=9, loc="upper left"); ax.set_title("Figure 12. Validation 2019: what each fifth of the ranking actually did")
fig.savefig("figures/2_models/fig12_validation_fifths.png"); plt.close(fig)
# per-metro consistency
pm = P.groupby("city").apply(lambda z: pd.Series({m: z[m].corr(z.actual, method="spearman") for m in ["Baseline: scaled momentum", "Linear regression", "Random forest", "Gradient boosted trees"]}), include_groups=False)
print("metros (of 30) where model rank corr > momentum:", {m: int((pm[m] > pm["Baseline: scaled momentum"]).sum()) for m in pm.columns[1:]})
print("metros where forest > linear:", int((pm["Random forest"] > pm["Linear regression"]).sum()))
print(pm.describe().loc[["min", "50%", "max"]].round(2).to_string())
