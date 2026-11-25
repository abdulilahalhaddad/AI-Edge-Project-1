"""Early read: do the image columns carry signal? (PRD section 7, "Early read")

Gradient boosting at default settings, trained on train row years, scored on the
VALIDATION year only. Test rows are dropped on load and never touched here.
Every run uses the same rows (has_image == 1); only the input columns change.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, load_config, path  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]


def metrics(df: pd.DataFrame, pred: np.ndarray) -> dict:
    df = df.assign(pred=pred)
    rmse = float(np.sqrt(np.mean((df.pred - df.target) ** 2)))
    rhos, tops = [], []
    for _, g in df.groupby(["city", "year"]):
        if len(g) >= 10:
            rhos.append(spearmanr(g.pred, g.target).statistic)
            k = max(1, int(round(0.1 * len(g))))
            tops.append(g.nlargest(k, "pred").target.mean())
    return {"rmse": rmse, "spearman_within_metro": float(np.mean(rhos)),
            "top_decile_rel_growth": float(np.mean(tops))}


def main() -> None:
    cfg = load_config()
    sp = cfg["split"]
    processed = path(cfg, "processed")
    panel = pd.read_parquet(REPO / cfg["paths"]["panel"])
    feat = pd.read_parquet(processed / "image_features.parquet")
    groups = json.loads((processed / "image_features_columns.json").read_text())

    last_valid = sp["valid_years"][1]
    df = panel.merge(feat, on=["zipcode", "year"], how="inner", validate="one_to_one")
    df = df[(df.year <= last_valid) & (df.has_image == 1)].reset_index(drop=True)

    tab = [c for c in panel.columns if c.startswith("tab_")]
    img_all = groups["model_inputs"]
    colour = groups["colour_level"] + groups["colour_change"]
    emb_lvl = groups["embedding_level"]

    # Placebo: image columns shuffled between ZIPs within the same metro and year
    rng = np.random.default_rng(cfg["seed"])
    shuffled = df[img_all].copy()
    for _, ix in df.groupby(["city", "year"]).indices.items():
        shuffled.iloc[ix] = df[img_all].iloc[rng.permutation(ix)].values
    plc = [f"plc_{c}" for c in img_all]
    df[plc] = shuffled.values

    runs = {
        "R0 tabular only": tab,
        "R1 + colour stats": tab + colour,
        "R2 + embedding levels": tab + emb_lvl,
        "R3 + all image columns": tab + img_all,
        "R4 image columns only": img_all,
        "R5 placebo (shuffled images)": tab + plc,
    }

    tr = df.year.between(*sp["train_years"])
    va = df.year.between(*sp["valid_years"])
    print(f"rows: train {tr.sum():,}, validation {va.sum():,} "
          f"(row year {sp['valid_years'][0]}, target year {sp['valid_years'][0] + 1}); "
          f"{len(tab)} tabular, {len(img_all)} image inputs")

    rows = []
    for name, cols in runs.items():
        per_seed = []
        for s in SEEDS:
            m = HistGradientBoostingRegressor(random_state=s)
            m.fit(df.loc[tr, cols], df.loc[tr, "target"])
            per_seed.append(metrics(df.loc[va], m.predict(df.loc[va, cols])))
        ps = pd.DataFrame(per_seed)
        rows.append({"run": name, **{f"{k}_mean": v for k, v in ps.mean().items()},
                     **{f"{k}_sd": v for k, v in ps.std().items()}})
        r = rows[-1]
        print(f"{name:32s} RMSE {r['rmse_mean']:.4f}  rho {r['spearman_within_metro_mean']:.4f} "
              f"(sd {r['spearman_within_metro_sd']:.4f})  top10% {r['top_decile_rel_growth_mean']:+.4f}")

    res = pd.DataFrame(rows)
    out = path(cfg, "reports") / "early_read_validation.csv"
    res.to_csv(out, index=False)

    base = res.set_index("run").loc["R0 tabular only"]
    lines = [
        "# Early read: image columns on the validation year",
        "",
        f"Gradient boosting (scikit-learn defaults), mean of {len(SEEDS)} seeds. Trained on row years "
        f"{sp['train_years'][0]}–{sp['train_years'][1]}, scored on row year {sp['valid_years'][0]} "
        f"(target year {sp['valid_years'][0] + 1}). Test years untouched. Same rows in every run "
        f"(has_image = 1): {tr.sum():,} train, {va.sum():,} validation.",
        "",
        f"Panel: `{cfg['paths']['panel']}`.",
        "",
        "| Run | RMSE | Within-metro Spearman (sd) | Δ vs R0 | Top-decile relative growth |",
        "|---|---|---|---|---|",
    ]
    for _, r in res.iterrows():
        d = r.spearman_within_metro_mean - base.spearman_within_metro_mean
        lines.append(f"| {r.run} | {r.rmse_mean:.4f} | {r.spearman_within_metro_mean:.4f} "
                     f"({r.spearman_within_metro_sd:.4f}) | {d:+.4f} | {r.top_decile_rel_growth_mean:+.4f} |")
    (path(cfg, "reports") / "early_read_validation.md").write_text("\n".join(lines) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
