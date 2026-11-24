"""Steps 4-5: compress embeddings (scaler + PCA fitted on early images only), centre within
(city, image_year), and join each ZIP-year row to its latest image at or before that year.

Output: data/processed/image_features.parquet (+ image_features_columns.json).
Reads no outcome column: only zipcode and city from the tabular CSV.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import COLOUR_COLS, load_config, path, seed_everything, write_json  # noqa: E402


def column_groups(n_pc: int, n_chg: int) -> dict:
    pcs = [f"pc_{i:02d}" for i in range(1, n_pc + 1)]
    groups = {
        "key": ["zipcode", "year"],
        "flags": ["has_image", "has_change"],
        "bookkeeping": ["img_year", "img_age", "img_gap", "img_coverage"],
        "embedding_level": pcs,
        "colour_level": list(COLOUR_COLS),
        "embedding_change": [f"chg_{c}" for c in pcs[:n_chg]] + ["chg_cos"],
        "colour_change": ["chg_green_idx", "chg_texture"],
    }
    groups["model_inputs"] = (
        groups["embedding_level"] + groups["colour_level"]
        + groups["embedding_change"] + groups["colour_change"] + ["img_age"]
    )
    return groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None,
                    help="write outputs here instead (used by 06_checks for the rerun test)")
    args = ap.parse_args()

    cfg = load_config()
    fcfg = cfg["features"]
    seed_everything(cfg["seed"])
    interim = path(cfg, "interim")
    out_dir = Path(args.out_dir) if args.out_dir else path(cfg, "processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    main_run = args.out_dir is None

    idx = pd.read_csv(interim / "image_index.csv")
    emb = np.load(interim / "embeddings_resnet50.npy").astype(np.float32)
    col = pd.read_csv(interim / "colour_stats.csv")
    assert len(idx) == len(emb) == len(col)
    assert (col.zipcode.values == idx.zipcode.values).all()
    assert (col.image_year.values == idx.image_year.values).all()

    n_pc, n_chg = int(fcfg["n_components"]), int(fcfg["n_change_components"])
    groups = column_groups(n_pc, n_chg)
    pc_cols = groups["embedding_level"]

    # Partial captures: an image covering much less of the frame than the same ZIP's
    # fullest image is a clipped tile, not a different ZIP. Treat it as missing.
    coverage = (col.valid_frac / col.groupby("zipcode").valid_frac.transform("max")).values
    keep = coverage >= float(fcfg["min_coverage"])
    idx, emb, col = idx[keep].reset_index(drop=True), emb[keep], col[keep].reset_index(drop=True)
    coverage = coverage[keep]

    # Step 4a: scaler + PCA, fitted only on images from early years
    fit = (idx.image_year <= int(fcfg["pca_fit_max_year"])).values
    scaler = StandardScaler().fit(emb[fit])
    pca = PCA(n_components=n_pc, svd_solver="full", random_state=cfg["seed"])
    pca.fit(scaler.transform(emb[fit]))
    pcs = pca.transform(scaler.transform(emb))

    img = idx[["zipcode", "image_year", "city"]].copy()
    img["img_coverage"] = coverage
    img[pc_cols] = pcs
    img[COLOUR_COLS] = col[COLOUR_COLS].values
    vals = pc_cols + list(COLOUR_COLS)

    # Step 4b: centre within (city, image_year); small groups use the image_year mean
    g = img.groupby(["city", "image_year"])
    small = (g["zipcode"].transform("size") < int(fcfg["centring_min_group"])).values[:, None]
    mean_cy = g[vals].transform("mean").values
    mean_y = img.groupby("image_year")[vals].transform("mean").values
    img[vals] = img[vals].values - np.where(small, mean_y, mean_cy)

    # Change from each image to the one immediately before it for the same ZIP
    same_zip = np.r_[False, img.zipcode.values[1:] == img.zipcode.values[:-1]]
    prev = np.r_[0, np.arange(len(img) - 1)]
    gap = np.where(same_zip, img.image_year.values - img.image_year.values[prev], np.nan)
    img["img_gap"] = gap

    def per_year(cur: np.ndarray, before: np.ndarray) -> np.ndarray:
        return np.where(same_zip, (cur - before) / gap, np.nan)

    for c in pc_cols[:n_chg]:
        v = img[c].values
        img[f"chg_{c}"] = per_year(v, v[prev])
    unit = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    cos_dist = 1.0 - (unit * unit[prev]).sum(axis=1)
    img["chg_cos"] = per_year(cos_dist, 0.0)
    for c in ("green_idx", "texture"):
        v = img[c].values
        img[f"chg_{c}"] = per_year(v, v[prev])

    # Step 5: ZIP-year grid, each row joined to its latest image with image_year <= year
    zips = np.sort(pd.read_csv(path(cfg, "tabular_csv"), usecols=["zipcode"]).zipcode.unique())
    years = np.arange(int(fcfg["row_year_min"]), int(fcfg["row_year_max"]) + 1)
    grid = pd.DataFrame({"zipcode": np.repeat(zips, len(years)), "year": np.tile(years, len(zips))})

    right = img.drop(columns="city").rename(columns={"image_year": "img_year"})
    right["img_year_key"] = right.img_year
    rows = pd.merge_asof(
        grid.sort_values("year"), right.sort_values("img_year_key"),
        left_on="year", right_on="img_year_key", by="zipcode", direction="backward",
    ).drop(columns="img_year_key")

    rows["has_image"] = rows.img_year.notna().astype(np.int8)
    rows["has_change"] = rows.img_gap.notna().astype(np.int8)
    rows["img_age"] = rows.year - rows.img_year

    ordered = groups["key"] + groups["flags"] + groups["bookkeeping"] + [
        c for k in ("embedding_level", "colour_level", "embedding_change", "colour_change")
        for c in groups[k]
    ]
    rows = rows[ordered].sort_values(["zipcode", "year"]).reset_index(drop=True)
    rows["zipcode"] = rows.zipcode.astype(np.int64)
    rows["year"] = rows.year.astype(np.int64)
    float_cols = [c for c in ordered if c not in groups["key"] + groups["flags"]]
    rows[float_cols] = rows[float_cols].astype(np.float32)

    rows.to_parquet(out_dir / "image_features.parquet", index=False)
    write_json(groups, out_dir / "image_features_columns.json")

    evr = float(pca.explained_variance_ratio_.sum())
    if main_run:
        models = path(cfg, "models")
        joblib.dump(scaler, models / "image_scaler.joblib")
        joblib.dump(pca, models / "image_pca.joblib")
        write_json(
            {
                "pca_fit_max_year": int(fcfg["pca_fit_max_year"]),
                "pca_fit_rows": int(fit.sum()),
                "pca_fit_max_image_year_seen": int(idx.image_year[fit].max()),
                "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
                "explained_variance_total": evr,
                "centring_small_groups": int(small.sum()),
                "min_coverage": float(fcfg["min_coverage"]),
                "images_kept": int(keep.sum()),
                "images_dropped_partial": int((~keep).sum()),
            },
            interim / "features_log.json",
        )
    print(f"{len(rows):,} rows x {rows.shape[1]} cols -> {out_dir / 'image_features.parquet'}; "
          f"PCA fit on {int(fit.sum()):,} images, {n_pc} comps explain {evr:.1%}")


if __name__ == "__main__":
    main()
