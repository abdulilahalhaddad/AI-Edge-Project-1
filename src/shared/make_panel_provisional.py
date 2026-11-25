"""Provisional ZIP-year panel: target + simple tabular inputs, for the image early read.

This is a stand-in for the tabular workstream's panel, not a replacement for it.

Row (zipcode, year t): inputs are what is known in December of year t.
Target: ZIP's 12-month growth of `price` (Dec t -> Dec t+1) minus the metro's mean
growth over the same 12 months.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "images"))
from common import load_config, path  # noqa: E402

DROP = {"date", "city", "city_full", "zipcode", "year"}


def main() -> None:
    cfg = load_config()
    fcfg = cfg["features"]
    d = pd.read_csv(path(cfg, "tabular_csv"), parse_dates=["date"])
    dec = d[d.date.dt.month == 12].copy()
    dec["year"] = dec.date.dt.year
    dec = dec.sort_values(["zipcode", "year"]).reset_index(drop=True)

    g = dec.groupby("zipcode")["price"]
    dec["growth_next"] = g.shift(-1) / dec.price - 1          # Dec t -> Dec t+1 (outcome)
    dec["growth_1y"] = dec.price / g.shift(1) - 1             # Dec t-1 -> Dec t (known)
    dec["growth_3y"] = dec.price / g.shift(3) - 1
    metro_next = dec.groupby(["city", "year"]).growth_next.transform("mean")
    metro_1y = dec.groupby(["city", "year"]).growth_1y.transform("mean")
    dec["target"] = dec.growth_next - metro_next
    dec["rel_growth_1y"] = dec.growth_1y - metro_1y           # last year's target, known at t
    dec["log_price"] = np.log(dec.price)

    numeric = [c for c in d.columns if c not in DROP and c != "price"
               and pd.api.types.is_numeric_dtype(d[c])]
    feats = numeric + ["log_price", "growth_1y", "growth_3y", "rel_growth_1y"]

    lo, hi = int(fcfg["row_year_min"]), int(fcfg["row_year_max"])
    panel = dec[(dec.year >= lo) & (dec.year <= hi)][["zipcode", "year", "city", "target"] + feats]
    panel = panel.rename(columns=lambda c: c if c in ("zipcode", "year", "city", "target")
                         else "tab_" + c.lower().replace(" ", "_"))
    panel = panel.sort_values(["zipcode", "year"]).reset_index(drop=True)

    assert len(panel) == cfg["expected"]["n_zips_csv"] * (hi - lo + 1), len(panel)
    assert not panel.duplicated(["zipcode", "year"]).any()
    out = path(cfg, "processed") / Path(cfg["paths"]["panel"]).name
    panel.to_parquet(out, index=False)
    n_tab = sum(c.startswith("tab_") for c in panel.columns)
    print(f"{len(panel):,} rows, {n_tab} tabular inputs, target missing in "
          f"{panel.target.isna().sum():,} rows -> {out}")


if __name__ == "__main__":
    main()
