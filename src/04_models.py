"""Stage 4: models, simple to complex. Fit on training years, settings chosen by
rolling through the training years, final comparison on 2019 validation.
Reads data/dev/ only.
"""
import json, itertools, importlib.util, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
warnings.filterwarnings("ignore")
spec = importlib.util.spec_from_file_location("f", "src/03_features.py"); F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
SEED = 15071
FEATS = list(F.FEATURES) + F.FLAGS
Y = "excess_growth_12m"
D = pd.read_csv("data/dev/features_dev.csv")
train, val = D[D.set == "train"], D[D.set == "validation"]
FOLDS = [([2014, 2015], 2016), ([2014, 2015, 2016], 2017), ([2014, 2015, 2016, 2017], 2018)]   # rolling: always predict a later year

def metrics(df, pred):
    d = df[["city", Y]].copy(); d["p"] = np.asarray(pred); y = d[Y]
    out = {"rmse_points": float(np.sqrt(((y - d.p) ** 2).mean()) * 100), "r2_vs_zero": float(1 - ((y - d.p) ** 2).sum() / (y ** 2).sum())}
    if d.p.nunique() < 2: return {**out, "rank_corr": np.nan, "top_fifth_beat_rate": np.nan, "top_minus_bottom_points": np.nan}
    out["rank_corr"] = float(d.groupby("city").apply(lambda z: z.p.corr(z[Y], method="spearman"), include_groups=False).mean())
    d["q"] = d.groupby("city").p.transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
    out["top_fifth_beat_rate"] = float((d.loc[d.q == 4, Y] > 0).mean())
    out["top_minus_bottom_points"] = float((d.loc[d.q == 4, Y].mean() - d.loc[d.q == 0, Y].mean()) * 100)
    return out

def grid(g): return [dict(zip(g, v)) for v in itertools.product(*g.values())]
MODELS = {
 "Linear regression": (lambda: make_pipeline(StandardScaler(), LinearRegression()), [{}]),
 "Ridge regression": (lambda alpha: make_pipeline(StandardScaler(), Ridge(alpha=alpha)), grid({"alpha": [1, 10, 100, 1000, 3000, 10000]})),
 "Lasso regression": (lambda alpha: make_pipeline(StandardScaler(), Lasso(alpha=alpha, max_iter=20000)), grid({"alpha": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]})),
 "Decision tree": (lambda max_depth, min_samples_leaf: DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=SEED),
                   grid({"max_depth": [2, 3, 4, 5, 6], "min_samples_leaf": [100, 300]})),
 "Random forest": (lambda max_depth, min_samples_leaf: RandomForestRegressor(n_estimators=300, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                   max_features=0.5, n_jobs=-1, random_state=SEED), grid({"max_depth": [6, 10, None], "min_samples_leaf": [20, 50]})),
 "Gradient boosted trees": (lambda max_depth, learning_rate, max_iter: HistGradientBoostingRegressor(max_depth=max_depth, learning_rate=learning_rate,
                   max_iter=max_iter, min_samples_leaf=50, random_state=SEED), grid({"max_depth": [2, 3, 4], "learning_rate": [0.03, 0.1], "max_iter": [100, 300]})),
}
def fit_pred(make, params, tr, te):
    m = make(**params).fit(tr[FEATS], tr[Y]); return m, m.predict(te[FEATS])
def baselines(tr, te):
    b = np.polyfit(tr.mom_12m, tr[Y], 1)
    return {"Baseline: zero": np.zeros(len(te)), "Baseline: raw momentum": te.mom_12m_unclipped.values, "Baseline: scaled momentum": np.polyval(b, te.mom_12m)}

rows, chosen, fitted, val_pred = [], {}, {}, {}
# rolling-year results for baselines
for name in ["Baseline: zero", "Baseline: raw momentum", "Baseline: scaled momentum"]:
    fm = [metrics(train[train.target_year == te], baselines(train[train.target_year.isin(tr)], train[train.target_year == te])[name]) for tr, te in FOLDS]
    p = baselines(train, val)[name]; val_pred[name] = p
    rows.append({"model": name, "settings": "", **{"rolling_" + k: np.nanmean([f[k] for f in fm]) for k in fm[0]}, **{"val_" + k: v for k, v in metrics(val, p).items()}})
for name, (make, params_list) in MODELS.items():
    best = None
    for params in params_list:
        fm = [metrics(train[train.target_year == te], fit_pred(make, params, train[train.target_year.isin(tr)], train[train.target_year == te])[1]) for tr, te in FOLDS]
        avg = {k: float(np.mean([f[k] for f in fm])) for k in fm[0]}
        if best is None or avg["rmse_points"] < best[1]["rmse_points"]: best = (params, avg)
    params, avg = best; chosen[name] = params
    m, p = fit_pred(make, params, train, val); fitted[name] = m; val_pred[name] = p
    rows.append({"model": name, "settings": json.dumps(params), **{"rolling_" + k: v for k, v in avg.items()}, **{"val_" + k: v for k, v in metrics(val, p).items()}})
    print("done", name, params, flush=True)
R = pd.DataFrame(rows); R.to_csv("data/dev/model_results.csv", index=False)
pd.set_option("display.width", 250); print(R.drop(columns="settings").round(3).to_string(index=False))
pd.DataFrame(val_pred).assign(zipcode=val.zipcode.values, city=val.city.values, actual=val[Y].values).to_csv("data/dev/validation_predictions.csv", index=False)

# drivers
lin = fitted["Linear regression"][-1].coef_ * 100; las = fitted["Lasso regression"][-1].coef_ * 100; rid = fitted["Ridge regression"][-1].coef_ * 100
pi = permutation_importance(fitted["Gradient boosted trees"], val[FEATS], val[Y], n_repeats=5, random_state=SEED, scoring="neg_root_mean_squared_error")
drv = pd.DataFrame({"feature": FEATS, "linear_coef_points_per_sd": lin, "ridge_coef": rid, "lasso_coef": las, "boosted_permutation_importance_points": pi.importances_mean * 100})
drv.to_csv("data/dev/model_drivers.csv", index=False); print(drv.round(3).to_string(index=False))
print("lasso dropped:", [f for f, c in zip(FEATS, las) if c == 0])
import pickle; pickle.dump({"fitted": fitted, "chosen": chosen, "feats": FEATS}, open("data/dev/models_stage4.pkl", "wb"))
