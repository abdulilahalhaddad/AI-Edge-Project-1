"""Stage 8: the final test. Opens the locked years ONCE.

Every model listed in docs/decision_record.md is refit on target years 2014-2019 (train +
validation) with the settings chosen on training years, then scored on:
  - main test: target year 2023
  - stress test: target years 2020, 2021, 2022 (each separately, and pooled)
Cleaning rules and clipping bounds are the ones learned on training years; nothing is re-tuned.
The script refuses to run twice unless --force is given, so the test is opened once.

Usage:
  python src/08_test.py --rehearsal                    # dry run on development data only (2019 plays the test)
  python src/08_test.py path/to/image_features.parquet [path/to/image_scores.csv]
"""
import sys, os, json, importlib.util, warnings, datetime, subprocess
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
def load(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
C = load("c", "src/02_clean.py"); F = load("f", "src/03_features.py")
SEED = 15071; Y = "excess_growth_12m"; FEATS = list(F.FEATURES) + F.FLAGS
COLOUR = ["valid_frac", "mean_r", "mean_g", "mean_b", "green_idx", "texture", "chg_green_idx", "chg_texture"]
IMG = COLOUR + [f"pc_{i:02d}" for i in range(1, 17)] + [f"chg_pc_{i:02d}" for i in range(1, 9)] + ["chg_cos", "img_age", "has_change"]
REHEARSAL = "--rehearsal" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
OUT = "/tmp/rehearsal" if REHEARSAL else "results/4_test"
os.makedirs(OUT, exist_ok=True)
if not REHEARSAL and os.path.exists(f"{OUT}/test_results.csv") and "--force" not in sys.argv:
    sys.exit("The test has already been run once. Results are in results/4_test/. Use --force only to reproduce them.")
S = json.load(open("data/dev/chosen_settings.json"))

# ---------------- data: development rows for fitting, locked rows for scoring ----------------
dev = pd.read_csv("data/dev/features_dev.csv")
if REHEARSAL:
    fit_years, eval_years, img_src = list(range(2014, 2019)), {2019: "rehearsal"}, pd.read_csv("data/dev/image_features_dev.csv")
    m = pd.read_csv("data/dev/monthly_dev.csv", parse_dates=["date"]); t = pd.read_csv("data/dev/targets_dev.csv")
    t = t[t.target_year == 2019]
else:
    fit_years, eval_years = list(range(2014, 2020)), {2020: "stress", 2021: "stress", 2022: "stress", 2023: "test"}
    img_src = pd.read_parquet(args[0])
    for p in ["data/locked/targets_locked.csv", "data/locked/monthly_locked.csv"]: os.chmod(p, 0o400)
    m = pd.concat([pd.read_csv("data/dev/monthly_dev.csv", parse_dates=["date"]),
                   pd.read_csv("data/locked/monthly_locked.csv", parse_dates=["date"])], ignore_index=True)
    t = pd.read_csv("data/locked/targets_locked.csv")
rules = json.load(open("data/dev/cleaning_rules.json")); bounds = json.load(open("data/dev/feature_clip_bounds.json"))
clean, _ = C.apply_rules(m, rules)
raw = m[["zipcode", "city", "date", "price"]]
X = F.build(clean, raw, sorted(t.input_year.unique())).merge(t[["zipcode", "input_year", "target_year", Y]], on=["zipcode", "input_year"], how="inner")
X = F.finish(X, bounds)
img = img_src.rename(columns={"year": "input_year"})
def with_images(d):
    d = d.merge(img[["zipcode", "input_year"] + IMG + ["has_image"]], on=["zipcode", "input_year"], how="left")
    return d[d.has_image == 1].reset_index(drop=True)
tr = with_images(dev[dev.target_year.isin(fit_years)]); te = with_images(X)
if REHEARSAL:   # the rebuilt 2019 rows must match the development features exactly
    chk = te.merge(dev[["zipcode", "input_year"] + FEATS], on=["zipcode", "input_year"], suffixes=("", "_dev"))
    gap = max(float((chk[f] - chk[f + "_dev"]).abs().max()) for f in FEATS)
    print(f"rebuilt rows match development features: max difference {gap:.2e}"); assert gap < 1e-9
med = tr[IMG].median()
print(f"fit rows {len(tr):,} (target years {min(fit_years)}-{max(fit_years)}); scored rows {len(te):,}: {te.target_year.value_counts().sort_index().to_dict()}")

# ---------------- models, exactly as fixed in the decision record ----------------
def mlp(): return TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32,), alpha=1.0, batch_size=256, max_iter=200, random_state=SEED)), transformer=StandardScaler())
rf = S["Random forest"]; gb = S["Gradient boosted trees"]; dt = S["Decision tree"]
MODELS = {
 "Linear regression": (lambda: make_pipeline(StandardScaler(), LinearRegression()), FEATS),
 "Ridge regression": (lambda: make_pipeline(StandardScaler(), Ridge(alpha=S["Ridge regression"]["alpha"])), FEATS),
 "Lasso regression": (lambda: make_pipeline(StandardScaler(), Lasso(alpha=S["Lasso regression"]["alpha"], max_iter=20000)), FEATS),
 "Decision tree": (lambda: DecisionTreeRegressor(max_depth=dt["max_depth"], min_samples_leaf=dt["min_samples_leaf"], random_state=SEED), FEATS),
 "Random forest": (lambda: RandomForestRegressor(n_estimators=500, max_depth=rf["max_depth"], min_samples_leaf=rf["min_samples_leaf"], max_features=rf["max_features"], n_jobs=-1, random_state=SEED), FEATS),
 "Gradient boosted trees": (lambda: HistGradientBoostingRegressor(max_depth=gb["max_depth"], learning_rate=gb["learning_rate"], max_iter=gb["max_iter"], min_samples_leaf=50, random_state=SEED), FEATS),
 "Neural network": (mlp, FEATS),
 "Random forest + image inputs": (lambda: RandomForestRegressor(n_estimators=500, max_depth=rf["max_depth"], min_samples_leaf=rf["min_samples_leaf"], max_features=rf["max_features"], n_jobs=-1, random_state=SEED), FEATS + IMG),
}
pred = pd.DataFrame({"zipcode": te.zipcode, "city": te.city, "target_year": te.target_year, "actual": te[Y]})
pred["Baseline: zero"] = 0.0
pred["Baseline: raw momentum"] = te.mom_12m_unclipped.values
pred["Baseline: scaled momentum"] = np.polyval(np.polyfit(tr.mom_12m, tr[Y], 1), te.mom_12m)
for name, (make, cols) in MODELS.items():
    pred[name] = make().fit(tr[cols].fillna(med), tr[Y]).predict(te[cols].fillna(med)); print("scored", name, flush=True)
dec_path = "data/dev/image_network_decision.json"
if len(args) > 1 and os.path.exists(dec_path) and not REHEARSAL:
    dec = json.load(open(dec_path))
    sc = pd.read_csv(args[1])[["zipcode", "row_year", dec["score_column"]]].rename(columns={"row_year": "input_year"})
    s = te[["zipcode", "input_year"]].merge(sc, on=["zipcode", "input_year"], how="left")[dec["score_column"]].fillna(0).values
    pred["Random forest + image network"] = pred["Random forest"] + dec["blend_weight_chosen_on_2018"] * s
clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=50, max_features=0.5, n_jobs=-1, random_state=SEED).fit(tr[FEATS], (tr[Y] > 0).astype(int))
pred["probability_beats_metro"] = clf.predict_proba(te[FEATS])[:, 1]
pred.to_csv("data/test_predictions.csv" if not REHEARSAL else f"{OUT}/predictions.csv", index=False)

# ---------------- scoring ----------------
models = [c for c in pred.columns if c not in ("zipcode", "city", "target_year", "actual", "probability_beats_metro")]
def score(d, col):
    y, p = d.actual, d[col]; r = {"R-squared": float(1 - ((y - p) ** 2).sum() / (y ** 2).sum()), "RMSE (points)": float(np.sqrt(((y - p) ** 2).mean()) * 100)}
    if p.nunique() < 2: return r
    g = d.groupby(["city", "target_year"])
    r["Rank correlation"] = float(g.apply(lambda z: z[col].corr(z.actual, method="spearman"), include_groups=False).mean())
    q = g[col].rank(pct=True, method="first"); top, bot = d[q > 0.8], d[q <= 0.2]
    win = g.actual.rank(pct=True, method="first") > 0.8
    r.update({"Top-fifth alpha (points)": float(top.actual.mean() * 100), "Top minus bottom fifth (points)": float((top.actual.mean() - bot.actual.mean()) * 100),
              "Top-fifth picks beating metro": float((top.actual > 0).mean()), "True winners per 10 screened (top 20%)": float(10 * win[q > 0.8].mean())})
    return r
rows = []
groups = {f"{y} ({k})": pred[pred.target_year == y] for y, k in eval_years.items()}
if not REHEARSAL: groups["2020-2022 stress, pooled"] = pred[pred.target_year.isin([2020, 2021, 2022])]
for gname, d in groups.items():
    for c in models: rows.append({"period": gname, "model": c, **score(d, c)})
    p = d.probability_beats_metro; beat = (d.actual > 0).astype(int)
    rows.append({"period": gname, "model": "Probability (forest classifier)", "AUC": float(roc_auc_score(beat, p)),
                 "Share flagged 70%+": float((p >= .7).mean()), "Flagged 70%+ that beat metro": float(beat[p >= .7].mean()), "Base rate": float(beat.mean())})
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/test_results.csv", index=False)
rng = np.random.default_rng(SEED); iv = []
for gname, d in groups.items():
    cities = d.city.unique(); by = {c: d[d.city == c] for c in cities}; b = []
    for _ in range(1000):
        s = pd.concat([by[c] for c in rng.choice(cities, len(cities))])
        qa = s.groupby(["city", "target_year"])["Random forest"].rank(pct=True, method="first")
        qm = s.groupby(["city", "target_year"])["Baseline: scaled momentum"].rank(pct=True, method="first")
        b.append((s.actual[qa > 0.8].mean(), s.actual[qa > 0.8].mean() - s.actual[qm > 0.8].mean()))
    b = np.array(b) * 100
    iv.append({"period": gname, "forest_alpha_lo95": np.percentile(b[:, 0], 2.5), "forest_alpha_hi95": np.percentile(b[:, 0], 97.5),
               "forest_minus_momentum_lo95": np.percentile(b[:, 1], 2.5), "forest_minus_momentum_hi95": np.percentile(b[:, 1], 97.5)})
pd.DataFrame(iv).to_csv(f"{OUT}/test_intervals.csv", index=False)
rec = {"run_at": datetime.datetime.now().isoformat(timespec="seconds"), "rehearsal": REHEARSAL, "fit_target_years": fit_years,
       "scored": {k: int(len(v)) for k, v in groups.items()},
       "git_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()}
json.dump(rec, open(f"{OUT}/run_record.json", "w"), indent=1)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print(R.round(3).to_string(index=False)); print(pd.DataFrame(iv).round(2).to_string(index=False))
