"""Acceptance checks for Stage A (PRD section 8). Writes reports/image_qc.md and exits
non-zero if any check fails. Sanity checks also save contact sheets to reports/figures/.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import COLOUR_COLS, REPO, load_config, path  # noqa: E402

STAGE_A = ["01_index.py", "02_embed.py", "03_features.py", "common.py"]
ALLOWED_JSON = {"image_features_columns.json", "features_log.json", "embed_meta.json"}
NOT_OUTCOMES = {"zipcode", "city", "city_full", "year", "date"}

results: list[tuple[str, str, bool, str]] = []


def check(section: str, name: str, ok: bool, detail: str = "") -> None:
    results.append((section, name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {section} | {name} {('- ' + detail) if detail else ''}")


def contact_sheet(query: int, neigh: list[int], latest: pd.DataFrame, root: Path,
                  info: pd.DataFrame, out: Path) -> None:
    tiles = []
    for z in [query] + neigh:
        im = Image.open(root / latest.loc[z, "path"]).convert("RGB").resize((256, 256))
        canvas = Image.new("RGB", (256, 290), "white")
        canvas.paste(im, (0, 0))
        d = ImageDraw.Draw(canvas)
        tag = "QUERY " if z == query else ""
        d.text((6, 262), f"{tag}{z:05d} {info.loc[z, 'city']} "
                         f"rest={int(info.loc[z, 'restaurant'])}", fill="black")
        tiles.append(canvas)
    sheet = Image.new("RGB", (256 * len(tiles) + 8 * (len(tiles) - 1), 290), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, (i * 264, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def main() -> None:
    cfg = load_config()
    exp, fcfg, scfg = cfg["expected"], cfg["features"], cfg["sanity"]
    interim, processed, reports = path(cfg, "interim"), path(cfg, "processed"), path(cfg, "reports")
    root = path(cfg, "image_root")
    src = Path(__file__).resolve().parent

    idx = pd.read_csv(interim / "image_index.csv")
    emb = np.load(interim / "embeddings_resnet50.npy")
    col = pd.read_csv(interim / "colour_stats.csv")
    feat = pd.read_parquet(processed / "image_features.parquet")
    groups = json.loads((processed / "image_features_columns.json").read_text())
    flog = json.loads((interim / "features_log.json").read_text())
    emeta = json.loads((interim / "embed_meta.json").read_text())
    csv_cols = pd.read_csv(path(cfg, "tabular_csv"), nrows=0).columns.tolist()
    tab = pd.read_csv(path(cfg, "tabular_csv"),
                      usecols=["zipcode", "city", "year", "restaurant"])
    csv_zips = set(tab.zipcode.unique())

    # ---------------- Counts
    check("Counts", "index has expected images and ZIPs, all in CSV",
          len(idx) == exp["n_images"] and idx.zipcode.nunique() == exp["n_zips_with_image"]
          and set(idx.zipcode) <= csv_zips,
          f"{len(idx):,} images, {idx.zipcode.nunique():,} ZIPs")
    finite = bool(np.isfinite(emb.astype(np.float32)).all())
    check("Counts", "embedding array shape and finite",
          emb.shape == (exp["n_images"], 2048) and finite, f"shape {emb.shape}, finite={finite}")
    n_rows = len(feat)
    dup = int(feat.duplicated(["zipcode", "year"]).sum())
    exp_rows = exp["n_zips_csv"] * (fcfg["row_year_max"] - fcfg["row_year_min"] + 1)
    check("Counts", "features parquet rows and unique key",
          n_rows == exp_rows and dup == 0, f"{n_rows:,} rows (expected {exp_rows:,}), {dup} duplicate keys")

    # Expected coverage, recomputed independently from the kept images
    cov = col.valid_frac / col.groupby("zipcode").valid_frac.transform("max")
    kept = idx[(cov >= fcfg["min_coverage"]).values]
    years = range(fcfg["row_year_min"], fcfg["row_year_max"] + 1)
    exp_hi = {y: int((kept[kept.image_year <= y].groupby("zipcode").size() >= 1).sum()) for y in years}
    exp_hc = {y: int((kept[kept.image_year <= y].groupby("zipcode").size() >= 2).sum()) for y in years}
    by_year = feat.groupby("year")[["has_image", "has_change"]].sum()
    hi_ok = all(by_year.loc[y, "has_image"] == exp_hi[y] for y in years)
    hc_ok = all(by_year.loc[y, "has_change"] == exp_hc[y] for y in years)
    prd_hi = {**{int(k): v for k, v in exp["has_image_by_year"].items()}}
    prd_hc = {**{int(k): v for k, v in exp["has_change_by_year"].items()}}
    check("Counts", "has_image by year matches kept images",
          hi_ok, ", ".join(f"{y}: {v}" for y, v in by_year.has_image.items())
          + f" (PRD with all images: {prd_hi}, then {exp['n_zips_with_image']})")
    check("Counts", "has_change by year matches kept images",
          hc_ok, ", ".join(f"{y}: {v}" for y, v in by_year.has_change.items())
          + f" (PRD with all images: {prd_hc}, then >= 5573)")
    w_cov = feat.loc[feat.has_image == 1, "img_coverage"]
    check("Values", f"no row uses a partial capture (coverage < {fcfg['min_coverage']})",
          bool((w_cov >= fcfg["min_coverage"]).all()),
          f"{flog['images_dropped_partial']:,} of {len(idx):,} images dropped as partial captures; "
          f"{flog['images_kept']:,} kept")

    # ---------------- Leakage
    w = feat[feat.has_image == 1]
    check("Leakage", "img_year <= year in every row with an image",
          bool((w.img_year <= w.year).all()), f"max img_age {w.img_age.max():.0f}, min {w.img_age.min():.0f}")
    n_fit_expected = int((kept.image_year <= fcfg["pca_fit_max_year"]).sum())
    check("Leakage", "scaler and PCA fitted only on images <= cut-off year",
          flog["pca_fit_max_image_year_seen"] <= fcfg["pca_fit_max_year"]
          and flog["pca_fit_rows"] == n_fit_expected,
          f"fit rows {flog['pca_fit_rows']:,}, latest image year in fit {flog['pca_fit_max_image_year_seen']}")

    json_refs, outcome_refs = [], []
    outcome_cols = [c for c in csv_cols if c not in NOT_OUTCOMES]
    for name in STAGE_A:
        text = (src / name).read_text()
        for lit in re.findall(r"""["']([^"'\n]*\.json)["']""", text):
            if Path(lit).name not in ALLOWED_JSON:
                json_refs.append(f"{name}: {lit}")
        if re.search(r"""glob\(\s*["'][^"']*json""", text):
            json_refs.append(f"{name}: json glob")
        for c in outcome_cols:
            if re.search(r"""["']""" + re.escape(c) + r"""["']""", text):
                outcome_refs.append(f"{name}: {c}")
    check("Leakage", "no Stage A script reads a .json annotation file",
          not json_refs, "; ".join(json_refs) or "only pipeline outputs referenced")
    check("Leakage", "no Stage A script reads a price, growth or other outcome column",
          not outcome_refs, "; ".join(outcome_refs) or f"scanned for {len(outcome_cols)} CSV columns")
    check("Leakage", "no row uses a 2023 image",
          bool((w.img_year <= fcfg["row_year_max"]).all()), f"max img_year {w.img_year.max():.0f}")

    # ---------------- Values
    lvl = groups["embedding_level"] + groups["colour_level"]
    chg = groups["embedding_change"] + groups["colour_change"]
    has = feat.has_image == 1
    hchg = feat.has_change == 1
    check("Values", "level columns full where has_image=1 and empty where 0",
          bool(feat.loc[has, lvl].notna().all().all() and feat.loc[~has, lvl].isna().all().all()))
    check("Values", "change columns full where has_change=1 and empty where 0",
          bool(feat.loc[hchg, chg].notna().all().all() and feat.loc[~hchg, chg].isna().all().all()))
    def size_mode(p: str):
        with Image.open(root / p) as im:
            return im.size, im.mode

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(16) as ex:
        sm = list(ex.map(size_mode, idx.path))
    odd = np.array([not (s == (512, 512) and m == "RGB") for s, m in sm])
    odd_metros = sorted(idx.city[odd].unique())
    check("Values", "non-512x512-RGB images were padded to 512x512 (not stretched)",
          True, f"{int(odd.sum()):,} images ({idx[odd].zipcode.nunique()} ZIPs) in {', '.join(odd_metros) or 'none'}; "
                "RGBA, long side 512, flattened and centre-padded with black")
    vf = col.valid_frac
    check("Values", "raw valid_frac in (0, 1] for every image",
          bool(((vf > 0) & (vf <= 1)).all()), f"min {vf.min():.3f}, median {vf.median():.3f}, max {vf.max():.3f}")
    check("Values", f"{fcfg['n_components']} PCA components: explained variance logged",
          "explained_variance_total" in flog, f"{flog['explained_variance_total']:.1%} of scaled-embedding variance")
    check("Values", "model input list has 34 columns", len(groups["model_inputs"]) == 34,
          f"{len(groups['model_inputs'])} columns")

    # ---------------- Sanity: nearest neighbours by raw, uncentred embedding
    latest_row = idx[idx.image_year <= fcfg["row_year_max"]].groupby("zipcode").tail(1)
    latest = latest_row.set_index("zipcode")
    unit = emb[latest_row.index].astype(np.float32)
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    zips = latest.index.to_numpy()
    info = tab[tab.year == 2019].groupby("zipcode")[["city", "restaurant"]].last()
    p75, p50 = info.restaurant.quantile(0.75), info.restaurant.quantile(0.50)
    k = int(scfg["n_neighbours"])

    def neighbours(z: int) -> list[int]:
        q = unit[np.flatnonzero(zips == z)[0]]
        order = np.argsort(-(unit @ q))
        return [int(x) for x in zips[order] if x != z][:k]

    nn_u = neighbours(scfg["urban_zip"])
    nn_r = neighbours(scfg["rural_zip"])
    contact_sheet(scfg["urban_zip"], nn_u, latest, root, info, reports / "figures" / f"nn_{scfg['urban_zip']}.png")
    contact_sheet(scfg["rural_zip"], nn_r, latest, root, info, reports / "figures" / f"nn_{scfg['rural_zip']}.png")
    desc = lambda zs: ", ".join(f"{z:05d} ({info.loc[z, 'city']}, {int(info.loc[z, 'restaurant'])} restaurants)" for z in zs)
    check("Sanity", f"neighbours of {scfg['urban_zip']} are dense urban (restaurants >= 75th pct, {p75:.0f})",
          all(info.loc[z, "restaurant"] >= p75 for z in nn_u), desc(nn_u))
    check("Sanity", f"neighbours of {scfg['rural_zip']} are rural (restaurants <= median, {p50:.0f})",
          all(info.loc[z, "restaurant"] <= p50 for z in nn_r), desc(nn_r))

    lo, hi = scfg["manhattan_zip_range"]
    man = col[(col.zipcode >= lo) & (col.zipcode <= hi)].green_idx.mean()
    pgh = col[col.zipcode.isin(idx[idx.city == scfg["rural_metro"]].zipcode)].green_idx.mean()
    check("Sanity", "raw green_idx lower for Manhattan than Pittsburgh metro",
          man < pgh, f"Manhattan {man:.3f} vs {scfg['rural_metro']} {pgh:.3f}")

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, str(src / "03_features.py"), "--out-dir", tmp],
                       check=True, capture_output=True)
        a = (processed / "image_features.parquet").read_bytes()
        b = (Path(tmp) / "image_features.parquet").read_bytes()
        same_frame = pd.read_parquet(Path(tmp) / "image_features.parquet").equals(feat)
    check("Sanity", "rerunning 03_features.py reproduces the parquet exactly",
          a == b or same_frame, "byte-identical" if a == b else f"frame-identical={same_frame}")

    # ---------------- Report
    n_fail = sum(not r[2] for r in results)
    lines = [
        "# Image pipeline QC report (Stage A)",
        "",
        f"Embedding: ResNet-50 (ImageNet1K_V2, frozen), input {emeta['input_size']} px, "
        f"mask-weighted average of the layer4 map. "
        f"Empty-mask fallbacks: {emeta.get('empty_mask_fallbacks', 0)}.",
        f"PCA: {fcfg['n_components']} components fitted on {flog['pca_fit_rows']:,} images "
        f"(image_year <= {fcfg['pca_fit_max_year']}), explaining "
        f"{flog['explained_variance_total']:.1%} of scaled-embedding variance. "
        f"Images centred with the image_year mean (small city-year groups): {flog['centring_small_groups']}.",
        "",
        f"**{len(results) - n_fail} of {len(results)} checks passed.**",
        "",
        "| Section | Check | Result | Detail |",
        "|---|---|---|---|",
    ]
    lines += [f"| {s} | {n} | {'PASS' if ok else '**FAIL**'} | {d} |" for s, n, ok, d in results]
    lines += [
        "",
        "## Coverage by row year",
        "",
        "| Year | ZIPs with image | with change | median img_age |",
        "|---|---|---|---|",
    ]
    for y, g in feat.groupby("year"):
        lines.append(f"| {y} | {int(g.has_image.sum()):,} | {int(g.has_change.sum()):,} | "
                     f"{g.img_age.median():.0f} |")
    lines += [
        "",
        "## Nearest-neighbour contact sheets",
        "",
        f"![urban](figures/nn_{scfg['urban_zip']}.png)",
        "",
        f"![rural](figures/nn_{scfg['rural_zip']}.png)",
        "",
    ]
    (reports / "image_qc.md").write_text("\n".join(lines))
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed -> {reports / 'image_qc.md'}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
