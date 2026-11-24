"""Step 1: index every aerial image as (zipcode, image_year, path, city).

Reads only image file names and the zipcode/city columns of the tabular CSV.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, path  # noqa: E402

PNG = re.compile(r"^(\d+)_(\d{4})\.png$")


def main() -> None:
    cfg = load_config()
    root = path(cfg, "image_root")

    records = []
    for p in sorted(root.glob("*/*/*.png")):
        m = PNG.match(p.name)
        if not m:
            continue
        zip_str, year = m.groups()
        assert p.parent.name == zip_str, f"folder/file ZIP mismatch: {p}"
        records.append(
            {"zipcode": int(zip_str), "image_year": int(year), "path": str(p.relative_to(root))}
        )
    idx = pd.DataFrame(records)

    zips = pd.read_csv(path(cfg, "tabular_csv"), usecols=["zipcode", "city"]).drop_duplicates()
    assert zips.zipcode.is_unique, "a ZIP maps to more than one metro"
    idx = idx.merge(zips, on="zipcode", how="left", validate="many_to_one")

    idx = idx.sort_values(["zipcode", "image_year"]).reset_index(drop=True)
    assert not idx.duplicated(["zipcode", "image_year"]).any(), "duplicate ZIP-year image"

    exp = cfg["expected"]
    assert len(idx) == exp["n_images"], f"{len(idx)} images, expected {exp['n_images']}"
    assert idx.zipcode.nunique() == exp["n_zips_with_image"], idx.zipcode.nunique()
    assert idx.city.notna().all(), "image ZIP missing from CSV"

    out = path(cfg, "interim") / "image_index.csv"
    idx[["zipcode", "image_year", "city", "path"]].to_csv(out, index=False)
    print(f"{len(idx):,} images, {idx.zipcode.nunique():,} ZIPs, "
          f"years {idx.image_year.min()}-{idx.image_year.max()} -> {out}")


if __name__ == "__main__":
    main()
