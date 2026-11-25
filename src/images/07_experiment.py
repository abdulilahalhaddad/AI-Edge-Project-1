"""Full experiment: do the image columns help? (PRD section 7)

Only the input columns change between runs; rows, split, tuning procedure and scoring stay
fixed. Every run uses rows with has_image == 1. Each (model, run) is tuned on the validation
year, refit on train + validation, and scored once on test.

  R0 tabular only            R3 tabular + all image columns (headline)
  R1 tabular + colour stats  R4 image columns only
  R2 tabular + emb. levels   R5 tabular + image columns shuffled within metro-year (placebo)

Gradient boosting runs R0-R5; ridge and a neural network run R0 and R3. Boosting and the
network are repeated over five seeds. If Stage B's image_score.parquet exists, the best
tabular R0 model is also blended with it (weight chosen on validation, applied once to test).
"""
from __future__ import annotations

import itertools
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, load_config, path  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)

SEEDS = [0, 1, 2, 3, 4]
N_BOOT = 500
MIN_GROUP = 10

GRIDS = {
    "gb": [dict(learning_rate=lr, max_leaf_nodes=nl, l2_regularization=l2)
           for lr, nl, l2 in itertools.product([0.05, 0.1], [15, 63], [0.0, 1.0])],
    "ridge": [dict(alpha=a) for a in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]],
    "nn": [dict(hidden_layer_sizes=h, alpha=a)
           for h, a in itertools.product([(64,), (128, 64)], [1e-4, 1e-2])],
}
FAMILY_RUNS = {"gb": ["R0", "R1", "R2", "R3", "R4", "R5"], "ridge": ["R0", "R3"], "nn": ["R0", "R3"]}
FAMILY_SEEDS = {"gb": SEEDS, "ridge": [0], "nn": SEEDS}
FAMILY_NAME = {"gb": "Gradient boosting", "ridge": "Ridge", "nn": "Neural network (MLP)"}


def make_model(family: str, params: dict, seed: int):
    if family == "gb":
        return HistGradientBoostingRegressor(random_state=seed, max_iter=500, **params)
    if family == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(**params))
    mlp = MLPRegressor(random_state=seed, early_stopping=True, max_iter=200, batch_size=256,
                       learning_rate_init=1e-3, n_iter_no_change=10, **params)
    return TransformedTargetRegressor(
        regressor=make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), mlp),
        transformer=StandardScaler())


def group_spearman(df: pd.DataFrame, pred_col: str, target: str = "target") -> pd.Series:
    """Spearman per (metro, year) group with at least MIN_GROUP rows."""
    out = {}
    for k, g in df.groupby(["city", "year"]):
        if len(g) >= MIN_GROUP:
            out[k] = g[pred_col].rank().corr(g[target].rank())
    return pd.Series(out, dtype=float)


def score(df: pd.DataFrame, pred: np.ndarray) -> dict:
    d = df.assign(pred=pred)
    tops = [g.nlargest(max(1, round(0.1 * len(g))), "pred").target.mean()
            for _, g in d.groupby(["city", "year"]) if len(g) >= MIN_GROUP]
    return {"rmse": float(np.sqrt(np.mean((d.pred - d.target) ** 2))),
            "spearman": float(group_spearman(d, "pred").mean()),
            "top_decile": float(np.mean(tops))}


def main() -> None:
    t_start = time.time()
    cfg = load_config()
    sp = cfg["split"]
    processed, reports = path(cfg, "processed"), path(cfg, "reports")
    panel = pd.read_parquet(REPO / cfg["paths"]["panel"])
    feat = pd.read_parquet(processed / "image_features.parquet")
    groups = json.loads((processed / "image_features_columns.json").read_text())

    df = panel.merge(feat, on=["zipcode", "year"], how="inner", validate="one_to_one")
    df = df[(df.has_image == 1) & df.target.notna()].sort_values(["zipcode", "year"]).reset_index(drop=True)

    tab = [c for c in panel.columns if c.startswith("tab_")]
    img = groups["model_inputs"]
    colour = groups["colour_level"] + groups["colour_change"]

    rng = np.random.default_rng(cfg["seed"])
    shuffled = df[img].to_numpy().copy()
    for _, ix in df.groupby(["city", "year"]).indices.items():
        shuffled[ix] = df[img].to_numpy()[rng.permutation(ix)]
    plc = [f"plc_{c}" for c in img]
    df[plc] = shuffled

    # Linear models and networks get has_change as an input next to the filled image columns
    runs = {
        "R0": tab,
        "R1": tab + colour + ["has_change"],
        "R2": tab + groups["embedding_level"],
        "R3": tab + img + ["has_change"],
        "R4": img + ["has_change"],
        "R5": tab + plc + ["has_change"],
    }
    run_label = {"R0": "Tabular only", "R1": "+ colour stats", "R2": "+ embedding levels",
                 "R3": "+ all image columns", "R4": "Image columns only", "R5": "Placebo (shuffled)"}

    tr = df.year.between(*sp["train_years"]).to_numpy()
    va = df.year.between(*sp["valid_years"]).to_numpy()
    te = df.year.between(*sp["test_years"]).to_numpy()
    trva = tr | va
    print(f"rows with images: train {tr.sum():,}, valid {va.sum():,}, test {te.sum():,}; "
          f"{len(tab)} tabular + {len(img)} image inputs")

    results, tuning_log = [], []
    test_pred = df.loc[te, ["zipcode", "year", "city", "target"]].copy()
    valid_pred = df.loc[va, ["zipcode", "year", "city", "target"]].copy()

    for family, run_ids in FAMILY_RUNS.items():
        for rid in run_ids:
            cols = runs[rid]
            t0 = time.time()
            # 1) tune on validation (seed 0)
            best, best_rho = None, -np.inf
            for params in GRIDS[family]:
                m = make_model(family, params, 0).fit(df.loc[tr, cols], df.loc[tr, "target"])
                rho = score(df.loc[va], m.predict(df.loc[va, cols]))["spearman"]
                tuning_log.append({"family": family, "run": rid, **{k: str(v) for k, v in params.items()},
                                   "valid_spearman": rho})
                if rho > best_rho:
                    best, best_rho = params, rho
            # 2) validation predictions with the chosen params (train only), for blending
            vp = np.mean([make_model(family, best, s).fit(df.loc[tr, cols], df.loc[tr, "target"])
                          .predict(df.loc[va, cols]) for s in FAMILY_SEEDS[family]], axis=0)
            valid_pred[f"{family}_{rid}"] = vp
            # 3) refit on train + validation, score once on test, per seed
            per_seed, preds = [], []
            for s in FAMILY_SEEDS[family]:
                m = make_model(family, best, s).fit(df.loc[trva, cols], df.loc[trva, "target"])
                p = m.predict(df.loc[te, cols])
                preds.append(p)
                per_seed.append(score(df.loc[te], p))
            ps = pd.DataFrame(per_seed)
            test_pred[f"{family}_{rid}"] = np.mean(preds, axis=0)
            results.append({
                "family": family, "run": rid, "label": run_label[rid], "params": json.dumps(best),
                "valid_spearman": best_rho, "seeds": len(per_seed),
                **{f"test_{k}": v for k, v in ps.mean().items()},
                **{f"test_{k}_sd": v for k, v in ps.std(ddof=1).fillna(0).items()},
            })
            r = results[-1]
            print(f"{family:5s} {rid} {run_label[rid]:22s} valid rho {best_rho:.4f} | test rho "
                  f"{r['test_spearman']:.4f} (sd {r['test_spearman_sd']:.4f}) RMSE {r['test_rmse']:.4f} "
                  f"top10 {r['test_top_decile']:+.4f} [{time.time() - t0:.0f}s]")

    res = pd.DataFrame(results)

    # Stage B blend: best R0 family on validation + w * img_score
    blend = None
    score_p = processed / "image_score.parquet"
    if score_p.exists():
        sb = pd.read_parquet(score_p)
        best_fam = res[res.run == "R0"].sort_values("valid_spearman").family.iloc[-1]
        v = valid_pred.merge(sb, on=["zipcode", "year"], how="left")
        t = test_pred.merge(sb, on=["zipcode", "year"], how="left")
        assert v.img_score.notna().all() and t.img_score.notna().all(), "img_score missing rows"
        sd = float(df.loc[tr, "target"].std())
        weights = np.round(np.arange(0, 1.01, 0.05), 2)
        vrho = {w: score(v, v[f"{best_fam}_R0"] + w * sd * v.img_score)["spearman"] for w in weights}
        w = max(vrho, key=vrho.get)
        # .to_numpy(): t has a fresh index after the merge; assign by position, not label
        test_pred["blend_R0_stageB"] = (t[f"{best_fam}_R0"] + w * sd * t.img_score).to_numpy()
        blend = {"family": best_fam, "weight": float(w), "valid_spearman": vrho[w],
                 "valid_spearman_w0": vrho[0.0], **{f"test_{k}": v_ for k, v_ in
                                                     score(df.loc[te], test_pred.blend_R0_stageB.to_numpy()).items()}}
        print(f"Stage B blend on {best_fam} R0: w={w}, valid rho {vrho[w]:.4f} (w=0: {vrho[0.0]:.4f}), "
              f"test rho {blend['test_spearman']:.4f}")

    # Bootstrap 95% interval for R3 - R0 (within-metro Spearman), resampling ZIPs
    tp = test_pred
    zips = tp.zipcode.unique()
    rows_by_zip = tp.groupby("zipcode").indices
    boot_rng = np.random.default_rng(cfg["seed"])
    # Headline R3 - R0 for every family; R1 and the placebo R5 for boosting as secondary reads
    comparisons = [(f, "R3") for f in FAMILY_RUNS] + [("gb", "R1"), ("gb", "R5")]
    picks = [boot_rng.choice(zips, size=len(zips), replace=True) for _ in range(N_BOOT)]
    samples = [tp.iloc[np.concatenate([rows_by_zip[z] for z in pick])] for pick in picks]
    boot = {}
    for family, rid in comparisons:
        a, b = f"{family}_{rid}", f"{family}_R0"
        diffs = [group_spearman(s, a).mean() - group_spearman(s, b).mean() for s in samples]
        point = group_spearman(tp, a).mean() - group_spearman(tp, b).mean()
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        key = family if rid == "R3" else f"{family}_{rid}"
        boot[key] = {"family": family, "run": rid, "delta": float(point),
                     "ci_low": float(lo), "ci_high": float(hi)}
        print(f"bootstrap {family}: {rid}-R0 = {point:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}]")

    # Breakdowns for R3 vs R0 (gradient boosting, seed-averaged predictions)
    tdf = df.loc[te].reset_index(drop=True)
    tp = tp.reset_index(drop=True)
    tp["img_age"] = tdf.img_age.values
    tp["chg_cos"] = tdf.chg_cos.values
    tp["units"] = tdf.tab_total_housing_units.values
    q75 = tp.chg_cos.quantile(0.75)
    med_units = tp.units.median()
    slices = {
        "Test year: target 2022 (row 2021)": tp.year == 2021,
        "Test year: target 2023 (row 2022)": tp.year == 2022,
        "Fresh image (img_age 0)": tp.img_age == 0,
        "Older image (img_age >= 1)": tp.img_age >= 1,
        "Visible change: top quarter of chg_cos": tp.chg_cos >= q75,
        "Visible change: rest": tp.chg_cos < q75,
        "Smaller ZIPs (housing units below median)": tp.units < med_units,
        "Larger ZIPs (housing units at/above median)": tp.units >= med_units,
    }
    brk = []
    for name, mask in slices.items():
        s = tp[mask.fillna(False).to_numpy()]
        r0, r3 = group_spearman(s, "gb_R0").mean(), group_spearman(s, "gb_R3").mean()
        brk.append({"slice": name, "rows": len(s), "R0": r0, "R3": r3, "delta": r3 - r0})
    metro = []
    for c, g in tp.groupby("city"):
        r0, r3 = group_spearman(g, "gb_R0").mean(), group_spearman(g, "gb_R3").mean()
        metro.append({"metro": c, "rows": len(g), "R0": r0, "R3": r3, "delta": r3 - r0})
    metro = pd.DataFrame(metro).sort_values("delta", ascending=False)

    # Save
    res.to_csv(reports / "experiment_runs.csv", index=False)
    pd.DataFrame(tuning_log).to_csv(reports / "experiment_tuning.csv", index=False)
    test_pred.to_parquet(processed / "test_predictions.parquet", index=False)
    pd.DataFrame(brk).to_csv(reports / "experiment_breakdowns.csv", index=False)
    metro.to_csv(reports / "experiment_by_metro.csv", index=False)
    summary = {"bootstrap_R3_minus_R0": boot, "blend": blend, "n_boot": N_BOOT,
               "rows": {"train": int(tr.sum()), "valid": int(va.sum()), "test": int(te.sum())},
               "panel": cfg["paths"]["panel"], "runtime_s": round(time.time() - t_start)}
    (reports / "experiment_summary.json").write_text(json.dumps(summary, indent=2))

    # Markdown report
    g = res.set_index(["family", "run"])
    gb = res[res.family == "gb"].set_index("run")
    L = ["# Experiment: do the image columns help?", "",
         f"Panel: `{cfg['paths']['panel']}`. Rows with an image only: {tr.sum():,} train "
         f"(row years {sp['train_years'][0]}–{sp['train_years'][1]}), {va.sum():,} validation "
         f"({sp['valid_years'][0]}), {te.sum():,} test ({sp['test_years'][0]}–{sp['test_years'][1]}, "
         f"target years {sp['test_years'][0] + 1}–{sp['test_years'][1] + 1}). Each model is tuned on "
         "validation, refit on train + validation, and scored once on test. Boosting and the neural "
         f"network are averaged over {len(SEEDS)} seeds.", "",
         "Metrics on test: within-metro Spearman = rank correlation of predicted and actual relative "
         "growth inside each metro and year, averaged; top decile = actual relative growth of the 10% "
         "of ZIPs ranked highest in each metro and year.", "",
         "## All runs", "",
         "| Model | Run | Inputs | Valid Spearman | Test Spearman (sd) | Δ vs R0 | Test RMSE | Top-decile growth |",
         "|---|---|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        d = r.test_spearman - g.loc[(r.family, "R0"), "test_spearman"]
        L.append(f"| {FAMILY_NAME[r.family]} | {r.run} | {r.label} | {r.valid_spearman:.4f} | "
                 f"{r.test_spearman:.4f} ({r.test_spearman_sd:.4f}) | {d:+.4f} | {r.test_rmse:.4f} | "
                 f"{r.test_top_decile:+.4f} |")
    L += ["", "## Headline: R3 minus R0, 95% bootstrap interval", "",
          f"Resampling ZIPs with replacement, {N_BOOT} times, on seed-averaged test predictions "
          "(so point estimates can differ slightly from the per-seed means above).", "",
          "| Model | Comparison | Δ Spearman | 95% interval | Excludes zero |", "|---|---|---|---|---|"]
    for b in boot.values():
        L.append(f"| {FAMILY_NAME[b['family']]} | {b['run']} − R0 | {b['delta']:+.4f} | "
                 f"[{b['ci_low']:+.4f}, {b['ci_high']:+.4f}] | "
                 f"{'yes' if b['ci_low'] > 0 or b['ci_high'] < 0 else 'no'} |")
    hb, pb, cb = boot["gb"], boot["gb_R5"], boot["gb_R1"]
    helps = hb["ci_low"] > 0 and not pb["ci_low"] > 0
    if helps:
        reading = "images help: R3 beats R0, the interval excludes zero and the placebo shows no gain."
    else:
        reading = ("the rule is not met. Adding all image columns (R3) does not beat tabular-only (R0) "
                   "for any model. Following the PRD, the finding is that ZIP-level imagery, as encoded "
                   "here, adds no forecast value beyond market and census data.")
    colour_note = (f" Colour statistics alone (R1) move rank correlation by {cb['delta']:+.4f} "
                   f"[{cb['ci_low']:+.4f}, {cb['ci_high']:+.4f}]"
                   + (", an interval that excludes zero." if cb["ci_low"] > 0 or cb["ci_high"] < 0
                      else ", an interval that includes zero.")
                   + f" Placebo R5 − R0: {pb['delta']:+.4f} [{pb['ci_low']:+.4f}, {pb['ci_high']:+.4f}].")
    L += ["", "**Reading (PRD rule: R3 beats R0, interval excludes zero, placebo shows no gain):** "
          + reading + colour_note, ""]
    if blend:
        L += ["## Stage B blend", "",
              f"Best R0 model on validation: {FAMILY_NAME[blend['family']]}. Blend = R0 prediction + "
              f"w × target sd × img_score, w chosen on validation: **w = {blend['weight']}** "
              f"(validation Spearman {blend['valid_spearman']:.4f} vs {blend['valid_spearman_w0']:.4f} at w = 0). "
              f"Test: Spearman {blend['test_spearman']:.4f}, RMSE {blend['test_rmse']:.4f}, top decile "
              f"{blend['test_top_decile']:+.4f}. The network was trained once on targets up to 2020 "
              "and not retrained before testing.", ""]
    L += ["## Breakdowns: R3 vs R0, gradient boosting", "",
          "| Slice | Test rows | R0 Spearman | R3 Spearman | Δ |", "|---|---|---|---|---|"]
    L += [f"| {b['slice']} | {b['rows']:,} | {b['R0']:.4f} | {b['R3']:.4f} | {b['delta']:+.4f} |" for b in brk]
    L += ["", f"chg_cos top-quarter threshold: {q75:.4f}. Housing-units median: {med_units:,.0f}.", "",
          "### By metro", "", "| Metro | Test rows | R0 | R3 | Δ |", "|---|---|---|---|---|"]
    L += [f"| {m.metro} | {m.rows:,} | {m.R0:.4f} | {m.R3:.4f} | {m.delta:+.4f} |" for m in metro.itertuples()]
    L += ["", f"Runtime: {summary['runtime_s'] // 60} min."]
    (reports / "experiment_results.md").write_text("\n".join(L) + "\n")
    print(f"done in {summary['runtime_s'] // 60} min -> {reports / 'experiment_results.md'}")


if __name__ == "__main__":
    main()
