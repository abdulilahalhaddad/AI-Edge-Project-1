"""Stage 6a: business impact, framed as alpha (return above the metro) and sourcing efficiency.
Uses out-of-sample predictions only: each target year 2016-2019 is predicted by a model fitted
on earlier target years. 2019 is the validation year. Test and stress-test data stay locked.
"""
import json, importlib.util, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
warnings.filterwarnings("ignore")
spec = importlib.util.spec_from_file_location("f", "src/03_features.py"); F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
SEED = 15071; Y = "excess_growth_12m"; FEATS = list(F.FEATURES) + F.FLAGS; FUND = 100e6
S = json.load(open("data/dev/chosen_settings.json"))["Random forest"]
D = pd.read_csv("data/dev/features_dev.csv")
IMG = pd.read_csv("data/dev/image_features_dev.csv", usecols=["zipcode", "year", "has_image"]).rename(columns={"year": "input_year"})
D = D.merge(IMG, on=["zipcode", "input_year"], how="left"); D = D[D.has_image == 1].reset_index(drop=True)
FOLDS = [([2014, 2015], 2016), ([2014, 2015, 2016], 2017), ([2014, 2015, 2016, 2017], 2018), ([2014, 2015, 2016, 2017, 2018], 2019)]

parts = []
for tr_years, te_year in FOLDS:
    tr, te = D[D.target_year.isin(tr_years)], D[D.target_year == te_year]
    lin = make_pipeline(StandardScaler(), LinearRegression()).fit(tr[FEATS], tr[Y])
    rf = RandomForestRegressor(n_estimators=500, max_depth=S["max_depth"], min_samples_leaf=S["min_samples_leaf"],
                               max_features=S["max_features"], n_jobs=-1, random_state=SEED).fit(tr[FEATS], tr[Y])
    parts.append(pd.DataFrame({"zipcode": te.zipcode, "city": te.city, "target_year": te_year, "actual": te[Y],
                               "Momentum rule": te.mom_12m_unclipped, "Linear regression": lin.predict(te[FEATS]),
                               "Random forest": rf.predict(te[FEATS])}))
    print("fitted for", te_year, flush=True)
P = pd.concat(parts, ignore_index=True); P.to_csv("data/dev/impact_oos_predictions.csv", index=False)
MODELS = ["Momentum rule", "Linear regression", "Random forest"]
G = ["city", "target_year"]
for m in MODELS:   # within-metro rank percentile (0 = lowest score, 1 = highest)
    P[m + "_pct"] = P.groupby(G)[m].rank(pct=True, method="first")
P["winner"] = P.groupby(G).actual.rank(pct=True, method="first") > 0.8   # true top fifth within metro-year

# --- alpha of an equal-weighted portfolio of top-fifth picks, by year ---
rows = []
for y in [2016, 2017, 2018, 2019, "All four years"]:
    d = P if y == "All four years" else P[P.target_year == y]
    r = {"year": y, "Random selection": d.actual.mean()}
    for m in MODELS:
        top = d[d[m + "_pct"] > 0.8]; bot = d[d[m + "_pct"] <= 0.2]
        r[m] = top.actual.mean(); r[m + " | bottom fifth"] = bot.actual.mean()
        r[m + " | hit rate"] = (top.actual > 0).mean(); r[m + " | picks below metro by >2 points"] = (top.actual < -0.02).mean()
        r[m + " | metros where picks beat metro"] = (top.groupby("city").actual.mean() > 0).mean()
    rows.append(r)
A = pd.DataFrame(rows); A.to_csv("data/dev/impact_alpha_by_year.csv", index=False)

# --- bootstrap over metros: uncertainty in the four-year average ---
rng = np.random.default_rng(SEED); cities = P.city.unique(); byc = {c: P[P.city == c] for c in cities}; boot = []
for _ in range(1000):
    d = pd.concat([byc[c] for c in rng.choice(cities, len(cities))])
    boot.append({m: d.loc[d[m + "_pct"] > 0.8, "actual"].mean() for m in MODELS})
B = pd.DataFrame(boot); B["Forest minus momentum"] = B["Random forest"] - B["Momentum rule"]; B["Forest minus linear"] = B["Random forest"] - B["Linear regression"]
CI = B.quantile([.025, .975]).T.rename(columns={.025: "lo95", .975: "hi95"}); CI.to_csv("data/dev/impact_alpha_intervals.csv")

# --- sourcing efficiency: cover the top k% of ZIP codes in each metro, as ranked by each method ---
gains = []
for k in np.arange(0.05, 1.0001, 0.05):
    r = {"coverage": round(k, 2), "Random selection": k}
    for m in MODELS:
        sel = P[m + "_pct"] > 1 - k - 1e-9
        r[m] = (sel & P.winner).sum() / P.winner.sum()
        r[m + " | winners per 10 screened"] = 10 * P.loc[sel, "winner"].mean()
        r[m + " | alpha of covered ZIPs"] = P.loc[sel, "actual"].mean()
    gains.append(r)
Gn = pd.DataFrame(gains); Gn.to_csv("data/dev/impact_sourcing_gains.csv", index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
show = A[["year", "Random selection"] + MODELS].copy(); show[["Random selection"] + MODELS] *= 100
print("ALPHA of top-fifth picks (percentage points vs metro):"); print(show.round(2).to_string(index=False))
print("$ per year on a $100m fund (all four years):", {m: round(A.iloc[-1][m] * FUND / 1e6, 2) for m in ["Random selection"] + MODELS})
print((CI * 100).round(2).to_string())
print(A.iloc[-1].filter(like="|").apply(lambda v: round(v * 100, 1)).to_string())
print(A.set_index("year").filter(like="Random forest |").map(lambda v: round(v * 100, 1)).to_string())
print(Gn[Gn.coverage.isin([0.1, 0.2, 0.3, 0.5])].round(3).to_string(index=False))
