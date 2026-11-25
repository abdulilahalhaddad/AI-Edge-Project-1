"""Stage 9c (after the test, summary only; no model is changed): pool all four unseen years
(2020-2023) from the saved test predictions, using the same scoring as 08_test.py, and
bootstrap by resampling metros. Feeds the headline numbers in the deck.
"""
import numpy as np, pandas as pd
SEED = 42
d = pd.read_csv("data/test_predictions.csv")
g = d.groupby(["city", "target_year"]); win = g.actual.rank(pct=True, method="first") > 0.8
rows = []
for col in ["Random forest", "Baseline: scaled momentum", "Neural network", "Gradient boosted trees", "Linear regression", "Image network (photos only)"]:
    q = g[col].rank(pct=True, method="first"); top = d[q > 0.8]
    rows.append({"model": col, "Top-fifth alpha (points)": top.actual.mean() * 100,
                 "Top-fifth picks beating metro": (top.actual > 0).mean(),
                 "True winners per 10 screened": 10 * win[q > 0.8].mean()})
rows.append({"model": "Random pick (expected)", "True winners per 10 screened": 2.0})
p = d.probability_beats_metro; beat = d.actual > 0
prob = {"share_flagged_70": (p >= .7).mean(), "flagged_70_beat_metro": beat[p >= .7].mean(), "base_rate": beat.mean()}
rng = np.random.default_rng(SEED); cities = d.city.unique(); by = {c: d[d.city == c] for c in cities}; b = []
for _ in range(1000):
    s = pd.concat([by[c] for c in rng.choice(cities, len(cities))])
    qa = s.groupby(["city", "target_year"])["Random forest"].rank(pct=True, method="first")
    qm = s.groupby(["city", "target_year"])["Baseline: scaled momentum"].rank(pct=True, method="first")
    b.append((s.actual[qa > 0.8].mean(), s.actual[qa > 0.8].mean() - s.actual[qm > 0.8].mean()))
b = np.array(b) * 100
R = pd.DataFrame(rows)
R["forest_alpha_lo95"] = np.nan; R["forest_alpha_hi95"] = np.nan
R.loc[R.model == "Random forest", ["forest_alpha_lo95", "forest_alpha_hi95"]] = np.percentile(b[:, 0], [2.5, 97.5])
R.loc[R.model == "Random forest", "forest_minus_momentum_lo95"] = np.percentile(b[:, 1], 2.5)
R.loc[R.model == "Random forest", "forest_minus_momentum_hi95"] = np.percentile(b[:, 1], 97.5)
for k, v in prob.items(): R.loc[R.model == "Random forest", k] = v
R.round(4).to_csv("results/4_test/four_year_summary.csv", index=False)
print(R.round(3).to_string(index=False))
