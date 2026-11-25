"""Figure 16: calibration of the probability that a ZIP code beats its metro (2016-2019, out of sample)."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
C = pd.read_csv("data/dev/probability_calibration.csv"); C = C[C.model == "Forest classifier"]
S = json.load(open("data/dev/probability_summary.json")); f = S["Forest classifier"]
BLUE, INK, MUTED, GREY = "#1f5fae", "#0b0b0b", "#898781", "#b4b2a9"
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.3), dpi=150, gridspec_kw={"width_ratios": [1.15, 1]})
for ax in (a1, a2):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7"); ax.spines["bottom"].set_color("#c3c2b7"); ax.tick_params(colors=MUTED); ax.grid(color="#e1e0d9"); ax.set_axisbelow(True)
a1.plot([0, 100], [0, 100], color=GREY, ls=(0, (4, 3)), lw=1.5, label="Perfectly honest probabilities")
a1.plot(C.predicted * 100, C.actual * 100, color=BLUE, lw=2, marker="o", ms=6, mec="white", mew=1.5, label="Model")
a1.set_xlim(0, 100); a1.set_ylim(0, 100); a1.set_xlabel("Model's probability that the ZIP code beats its metro (%)", color="#52514e")
a1.set_ylabel("Share that actually beat the metro (%)", color="#52514e"); a1.legend(frameon=False, fontsize=8.5, loc="upper left")
a1.set_title("Figure 16a. When the model says 70%, it happens about 70% of the time", loc="left", fontsize=10.5, fontweight="bold")
cats = ["Model says 30% or less", "All ZIP codes", "Model says 70% or more"]
vals = [f["beat_rate_when_p<=0.3"] * 100, S["base_rate"] * 100, f["beat_rate_when_p>=0.7"] * 100]
a2.bar(cats, vals, color=["#b9d3f2", GREY, BLUE], width=0.6); a2.grid(axis="x", visible=False)
for i, v in enumerate(vals): a2.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=10, color=INK)
a2.set_ylim(0, 100); a2.set_ylabel("Share that actually beat the metro (%)", color="#52514e"); a2.tick_params(axis="x", labelsize=8.5, colors="#52514e")
a2.set_title(f"Figure 16b. The top flag ({f['share_with_p>=0.7']*100:.0f}% of ZIP codes) is right {f['beat_rate_when_p>=0.7']*100:.0f}% of the time", loc="left", fontsize=10.5, fontweight="bold")
fig.tight_layout(); fig.savefig("figures/fig16_probability.png", bbox_inches="tight", facecolor="white")
