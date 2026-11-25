"""Stage 5d: does the image network trained in Colab (05c notebook) add anything?
Reads image_scores.csv from the notebook. The network trained on target years 2014-2017 and
used 2018 to decide when to stop; it never saw an outcome from 2019 onwards.
- Blend weight w for "forest + w x image score" is chosen on 2018 (forest predictions for 2018
  come from a forest trained on 2014-2017 only, see 06_impact.py).
- Everything is then judged once on 2019 validation, against the forest alone and against a
  placebo where image scores are shuffled between ZIP codes of the same metro.
Usage: python src/05d_image_network.py path/to/image_scores.csv
"""
import sys, json
import numpy as np, pandas as pd

SEED = 15071; rng = np.random.default_rng(SEED)
S = pd.read_csv(sys.argv[1])
P = pd.read_csv("data/dev/impact_oos_predictions.csv")            # out-of-sample forest predictions 2016-2019
P["row_year"] = P.target_year - 1
D = P.merge(S, on=["zipcode", "row_year"], how="inner")
D = D[D.target_year.isin([2018, 2019])].reset_index(drop=True)
assert not D.used_in_training[D.target_year == 2019].any() if "used_in_training" in D else True

def rank_corr(d, p):
    d = d.assign(p=np.asarray(p))
    return float(d.groupby("city").apply(lambda z: z.p.corr(z.actual, method="spearman"), include_groups=False).mean())
def top_alpha(d, p):
    d = d.assign(p=np.asarray(p)); q = d.groupby("city").p.rank(pct=True, method="first")
    return float(d.actual[q > 0.8].mean() * 100)

y18, y19 = D[D.target_year == 2018], D[D.target_year == 2019].reset_index(drop=True)
out, cols = [], [c for c in ["score_frozen", "score_fine_tuned", "image_score"] if c in D]
weights = [0, 0.25, 0.5, 1, 1.5, 2, 3]
choice = {}
for c in cols:
    best_w = max(weights, key=lambda w: rank_corr(y18, y18["Random forest"] + w * y18[c]))
    choice[c] = best_w
    shuffled = y19.groupby("city")[c].transform(lambda s: s.sample(frac=1, random_state=SEED).values)
    for name, p in [(f"Image network only ({c})", y19[c]),
                    (f"Forest + {best_w} x {c}", y19["Random forest"] + best_w * y19[c]),
                    (f"Forest + {best_w} x shuffled {c} (placebo)", y19["Random forest"] + best_w * shuffled)]:
        out.append({"model": name, "val2019_rank_corr": rank_corr(y19, p), "val2019_top_fifth_alpha_points": top_alpha(y19, p)})
out.insert(0, {"model": "Random forest alone", "val2019_rank_corr": rank_corr(y19, y19["Random forest"]),
               "val2019_top_fifth_alpha_points": top_alpha(y19, y19["Random forest"])})
R = pd.DataFrame(out)

# 95% interval for (forest + image) minus forest alone, resampling metros
c = "image_score" if "image_score" in cols else cols[0]; w = choice[c]
cities = y19.city.unique(); by = {k: y19[y19.city == k] for k in cities}; diffs = []
for _ in range(1000):
    d = pd.concat([by[k] for k in rng.choice(cities, len(cities))])
    diffs.append(rank_corr(d, d["Random forest"] + w * d[c]) - rank_corr(d, d["Random forest"]))
ci = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
decision = {"score_column": c, "blend_weight_chosen_on_2018": w, "gain_in_rank_corr_2019": float(np.mean(diffs)),
            "gain_95_interval": ci, "adopt": bool(w > 0 and ci[0] > 0), "rows_2019": int(len(y19))}
R.to_csv("data/dev/image_network_results.csv", index=False)
json.dump(decision, open("data/dev/image_network_decision.json", "w"), indent=1)
pd.set_option("display.width", 200); print(R.round(3).to_string(index=False)); print(json.dumps(decision, indent=1))
