"""Stage 7: robustness checks and probability output, training and validation years only (test data stays locked).
 a) Neural network (multilayer perceptron) added to the model ladder
 b) Probability output: chance a ZIP code beats its metro (logistic regression vs forest classifier)
 c) Ablation: momentum + price level only, versus all 21 inputs
 d) Robustness: census inputs lagged one extra year (guards against publication delay)
Out-of-sample years 2016-2019: each year predicted by a model fitted on earlier years only.
"""
import json, importlib.util, warnings, pickle
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
warnings.filterwarnings("ignore")
def load(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
F = load("f", "src/03_features.py")
SEED = 15071; Y = "excess_growth_12m"; FEATS = list(F.FEATURES) + F.FLAGS; TWO = ["mom_12m", "price_level"]
S = json.load(open("data/dev/chosen_settings.json"))["Random forest"]
IMG = pd.read_csv("data/dev/image_features_dev.csv", usecols=["zipcode", "year", "has_image"]).rename(columns={"year": "input_year"})
def universe(D): D = D.merge(IMG, on=["zipcode", "input_year"], how="left"); return D[D.has_image == 1].reset_index(drop=True)
D = universe(pd.read_csv("data/dev/features_dev.csv"))
FOLDS = [([2014, 2015], 2016), ([2014, 2015, 2016], 2017), ([2014, 2015, 2016, 2017], 2018), ([2014, 2015, 2016, 2017, 2018], 2019)]
def forest(): return RandomForestRegressor(n_estimators=300, max_depth=S["max_depth"], min_samples_leaf=S["min_samples_leaf"], max_features=S["max_features"], n_jobs=-1, random_state=SEED)
def mlp(h, a): return TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=h, alpha=a, batch_size=256, max_iter=200, random_state=SEED)), transformer=StandardScaler())
def rank_corr(df, p):
    d = pd.DataFrame({"g": df.city.astype(str).values + df.target_year.astype(str).values, "y": df[Y].values, "p": np.asarray(p)})
    return float(d.groupby("g").apply(lambda z: z.p.corr(z.y, method="spearman"), include_groups=False).mean())
def top_alpha(df, p):
    d = pd.DataFrame({"c": df.city.values, "t": df.target_year.values, "y": df[Y].values, "p": np.asarray(p)})
    q = d.groupby(["c", "t"]).p.rank(pct=True, method="first"); return float(d.y[q > 0.8].mean() * 100)
def oos(make, cols, data=D):
    out = []
    for tr_y, te_y in FOLDS:
        tr, te = data[data.target_year.isin(tr_y)], data[data.target_year == te_y]
        out.append(pd.Series(make().fit(tr[cols], tr[Y]).predict(te[cols]), index=te.index))
    return pd.concat(out)
res = {}
O = D[D.target_year >= 2016]
def score(name, p, df=None):
    df = O if df is None else df; p = p.loc[df.index]; v = df[df.target_year == 2019]
    res[name] = {"oos_rank_corr_2016_19": rank_corr(df, p), "oos_top_fifth_alpha_points": top_alpha(df, p),
                 "val2019_rank_corr": rank_corr(v, p.loc[v.index]), "val2019_r2": float(1 - ((v[Y] - p.loc[v.index]) ** 2).sum() / (v[Y] ** 2).sum())}
    print(name, {k: round(x, 3) for k, x in res[name].items()}, flush=True)

# baselines and existing models (recomputed here so every row uses identical folds)
score("Momentum rule", O.mom_12m_unclipped)
score("Linear: momentum + price level", oos(lambda: make_pipeline(StandardScaler(), LinearRegression()), TWO))
score("Forest: momentum + price level", oos(forest, TWO))
score("Linear regression: all inputs", oos(lambda: make_pipeline(StandardScaler(), LinearRegression()), FEATS))
P_rf = oos(forest, FEATS); score("Random forest: all inputs", P_rf)

# a) neural network: settings chosen on rolling years 2016-2018 (RMSE), then scored
best = None
for h in [(32,), (64,), (64, 32)]:
    for a in [0.1, 1.0, 10.0]:
        e = []
        for tr_y, te_y in FOLDS[:3]:
            tr, te = D[D.target_year.isin(tr_y)], D[D.target_year == te_y]
            e.append(np.sqrt(((mlp(h, a).fit(tr[FEATS], tr[Y]).predict(te[FEATS]) - te[Y]) ** 2).mean()))
        if best is None or np.mean(e) < best[0]: best = (np.mean(e), h, a)
print("neural network settings:", best, flush=True)
score(f"Neural network {best[1]}, penalty {best[2]}", oos(lambda: mlp(best[1], best[2]), FEATS))

# d) census lagged one extra year
clean = pd.read_csv("data/dev/monthly_dev_clean.csv", parse_dates=["date"]).sort_values(["zipcode", "date"])
CEN = ['Total Population', 'Median Age', 'Per Capita Income', 'Total Housing Units', 'Median Rent', 'Median Home Value', 'Total Labor Force', 'Unemployed Population']
lag = clean.copy(); lag[CEN] = clean.groupby("zipcode")[CEN].shift(12)
for c in CEN: lag[c] = lag[c].fillna(clean[c])
raw = pd.read_csv("data/dev/monthly_dev.csv", usecols=["zipcode", "city", "date", "price"], parse_dates=["date"])
t = pd.read_csv("data/dev/targets_dev.csv")
XL = F.build(lag, raw, sorted(t.input_year.unique())).merge(t[["zipcode", "input_year", "target_year", "set", Y]], on=["zipcode", "input_year"], how="inner")
trl = XL[XL.set == "train"]; bounds = {f: [float(trl[f].quantile(.01)), float(trl[f].quantile(.99))] for f in F.FEATURES}
XL = universe(F.finish(XL, bounds)); OL = XL[XL.target_year >= 2016]
score("Random forest: census lagged a year", oos(forest, FEATS, XL), OL)

R = pd.DataFrame(res).T; R.to_csv("data/dev/robustness_checks.csv")

# b) probability of beating the metro
Dc = D.assign(beat=(D[Y] > 0).astype(int)); probs = {}
for name, make in {"Logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
                   "Forest classifier": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=50, max_features=0.5, n_jobs=-1, random_state=SEED)}.items():
    out = []
    for tr_y, te_y in FOLDS:
        tr, te = Dc[Dc.target_year.isin(tr_y)], Dc[Dc.target_year == te_y]
        out.append(pd.Series(make().fit(tr[FEATS], tr.beat).predict_proba(te[FEATS])[:, 1], index=te.index))
    probs[name] = pd.concat(out)
O2 = Dc[Dc.target_year >= 2016].copy(); cal = []
base = O2.groupby("target_year").beat.transform("mean")   # naive forecast: every ZIP gets that year's base rate
summary = {"base_rate": float(O2.beat.mean()), "brier_naive": float(brier_score_loss(O2.beat, np.full(len(O2), O2.beat.mean())))}
for name, p in probs.items():
    p = p.loc[O2.index]; summary[name] = {"auc": float(roc_auc_score(O2.beat, p)), "brier": float(brier_score_loss(O2.beat, p)),
        "share_with_p>=0.7": float((p >= .7).mean()), "beat_rate_when_p>=0.7": float(O2.beat[p >= .7].mean()), "beat_rate_when_p<=0.3": float(O2.beat[p <= .3].mean())}
    b = pd.cut(p, np.linspace(0, 1, 11), include_lowest=True)
    cal.append(O2.assign(p=p, bin=b).groupby("bin", observed=True).agg(predicted=("p", "mean"), actual=("beat", "mean"), n=("beat", "size")).assign(model=name).reset_index())
pd.concat(cal).to_csv("data/dev/probability_calibration.csv", index=False)
O2.assign(**{k: v.loc[O2.index] for k, v in probs.items()})[["zipcode", "city", "target_year", Y, "beat"] + list(probs)].to_csv("data/dev/probability_oos.csv", index=False)
json.dump(summary, open("data/dev/probability_summary.json", "w"), indent=1)
pd.set_option("display.width", 220); print(R.round(3).to_string()); print(json.dumps(summary, indent=1))
