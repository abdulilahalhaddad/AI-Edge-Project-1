"""Stage 4b: do the image features add anything to the tabular models?
Training and validation years only (data/dev/). Image features come from the
image workstream's image_features.parquet, restricted to row years <= 2018.
Every run uses the same rows: ZIP-years that have an image.
"""
import json, importlib.util, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
warnings.filterwarnings("ignore")
spec = importlib.util.spec_from_file_location("f", "src/03_features.py"); F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
SEED = 15071; Y = "excess_growth_12m"; rng = np.random.default_rng(SEED)
TAB = list(F.FEATURES) + F.FLAGS
COLOUR = ["valid_frac", "mean_r", "mean_g", "mean_b", "green_idx", "texture", "chg_green_idx", "chg_texture"]
EMB = [f"pc_{i:02d}" for i in range(1, 17)] + [f"chg_pc_{i:02d}" for i in range(1, 9)] + ["chg_cos", "img_age"]
IMG = COLOUR + EMB                      # the 34 image inputs
D = pd.read_csv("data/dev/features_dev.csv").merge(pd.read_csv("data/dev/image_features_dev.csv"),
        left_on=["zipcode", "input_year"], right_on=["zipcode", "year"], how="left")
D = D[D.has_image == 1].reset_index(drop=True)
P = D.copy()                            # placebo: image block shuffled between ZIP codes of the same metro and year
for _, idx in D.groupby(["city", "input_year"]).groups.items():
    idx = np.asarray(idx); P.loc[idx, IMG + ["has_change"]] = D.loc[rng.permutation(idx), IMG + ["has_change"]].values
RUNS = {"Tabular only": (D, TAB), "+ colour statistics": (D, TAB + COLOUR + ["has_change"]), "+ all image inputs": (D, TAB + IMG + ["has_change"]),
        "+ shuffled images (placebo)": (P, TAB + IMG + ["has_change"]), "Images only": (D, IMG + ["has_change"])}
FOLDS = [([2014, 2015], 2016), ([2014, 2015, 2016], 2017), ([2014, 2015, 2016, 2017], 2018)]
ALPHAS = [1, 10, 100, 1000, 3000]

def rank_corr(df, p):
    d = pd.DataFrame({"c": df.city.values, "y": df[Y].values, "p": p}); return float(d.groupby("c").apply(lambda z: z.p.corr(z.y, method="spearman"), include_groups=False).mean())
def spread(df, p):
    d = pd.DataFrame({"c": df.city.values, "y": df[Y].values, "p": p}); q = d.groupby("c").p.transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
    return float((d.y[q == 4].mean() - d.y[q == 0].mean()) * 100)
def prep(tr, te, cols):
    med = tr[cols].median(); return tr[cols].fillna(med), te[cols].fillna(med)
def fit(model, tr, te, cols):
    a, b = prep(tr, te, cols)
    if model == "Ridge regression":
        def one(al): return make_pipeline(StandardScaler(), Ridge(alpha=al))
        if tr.target_year.nunique() > 2 and tr.target_year.max() == 2018:      # full training set: pick penalty on rolling years
            sc = {al: np.mean([np.sqrt(((one(al).fit(*[prep(tr[tr.target_year.isin(f)], tr[tr.target_year == t], cols)[0], tr[tr.target_year.isin(f)][Y]][:2]).predict(
                  prep(tr[tr.target_year.isin(f)], tr[tr.target_year == t], cols)[1]) - tr[tr.target_year == t][Y]) ** 2).mean()) for f, t in FOLDS]) for al in ALPHAS}
            al = min(sc, key=sc.get)
        else: al = 100
        return one(al).fit(a, tr[Y]).predict(b)
    if model == "Random forest":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=20, max_features=0.5, n_jobs=-1, random_state=SEED).fit(a, tr[Y]).predict(b)
    return np.mean([HistGradientBoostingRegressor(max_depth=4, learning_rate=0.1, max_iter=100, min_samples_leaf=50, random_state=SEED + s).fit(a, tr[Y]).predict(b) for s in range(3)], axis=0)

rows, preds = [], {}
for model in ["Ridge regression", "Random forest", "Gradient boosted trees"]:
    for run, (X, cols) in RUNS.items():
        tr, va = X[X.set == "train"], X[X.set == "validation"]
        roll = [rank_corr(tr[tr.target_year == t], fit(model, tr[tr.target_year.isin(f)], tr[tr.target_year == t], cols)) for f, t in FOLDS]
        p = fit(model, tr, va, cols); preds[(model, run)] = p
        rows.append({"model": model, "inputs": run, "n_inputs": len(cols), "rolling_rank_corr": np.mean(roll), "val_rank_corr": rank_corr(va, p),
                     "val_top_minus_bottom_points": spread(va, p), "val_rmse_points": float(np.sqrt(((va[Y] - p) ** 2).mean()) * 100)})
        print(rows[-1], flush=True)
R = pd.DataFrame(rows); R.to_csv("data/dev/image_experiment.csv", index=False)
va = D[D.set == "validation"].reset_index(drop=True); n = len(va); boots = []
for model in ["Ridge regression", "Random forest", "Gradient boosted trees"]:
    for run in ["+ colour statistics", "+ all image inputs", "+ shuffled images (placebo)"]:
        d = []
        for b in range(300):
            i = rng.integers(0, n, n); s = va.iloc[i]; d.append(rank_corr(s, preds[(model, run)][i]) - rank_corr(s, preds[(model, "Tabular only")][i]))
        boots.append({"model": model, "comparison": run + " minus tabular only", "delta": float(np.mean(d)), "lo95": float(np.percentile(d, 2.5)), "hi95": float(np.percentile(d, 97.5))})
B = pd.DataFrame(boots); B.to_csv("data/dev/image_experiment_intervals.csv", index=False)
pd.set_option("display.width", 220); print(R.round(4).to_string(index=False)); print(B.round(4).to_string(index=False))
print("rows: train", int((D.set == "train").sum()), "validation", n, "| ZIP codes", D.zipcode.nunique())
