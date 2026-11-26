"""Stage 9 (after the test, diagnosis only; no model is changed): did the two relationships the
model relies on hold in every year? Within-metro rank correlation between next-year growth versus
metro and (a) last year's growth versus metro (momentum), (b) price level versus metro (cheapness).
"""
import numpy as np, pandas as pd
t = pd.concat([pd.read_csv("data/dev/targets_dev.csv"), pd.read_csv("data/locked/targets_locked.csv")])
m = pd.concat([pd.read_csv("data/dev/monthly_dev.csv", usecols=["zipcode", "city", "date", "price"], parse_dates=["date"]),
               pd.read_csv("data/locked/monthly_locked.csv", usecols=["zipcode", "city", "date", "price"], parse_dates=["date"])])
d = m[m.date.dt.month == 12].copy(); d["input_year"] = d.date.dt.year; d = d.sort_values(["zipcode", "input_year"])
d["lp"] = np.log(d.price); d["g"] = d.lp - d.groupby("zipcode").lp.shift(1)
d["momentum"] = d.g - d.groupby(["city", "input_year"]).g.transform("mean")
d["price_vs_metro"] = d.lp - d.groupby(["city", "input_year"]).lp.transform("mean")
import sys
img = pd.read_parquet(sys.argv[1], columns=["zipcode", "year", "has_image"])
x = t.merge(d[["zipcode", "input_year", "momentum", "price_vs_metro"]], on=["zipcode", "input_year"]).merge(img.rename(columns={"year": "input_year"}), on=["zipcode", "input_year"])
x = x[x.has_image == 1]
out = x.groupby("target_year").apply(lambda y: pd.Series({c: y.groupby("city").apply(lambda z: z[c].corr(z.excess_growth_12m, method="spearman"), include_groups=False).mean()
                                                       for c in ["momentum", "price_vs_metro"]}), include_groups=False)
out["metro_growth_median_%"] = (t.groupby("target_year").metro_growth_12m.median() * 100).reindex(out.index)
out.round(3).to_csv("results/4_test/diagnosis_relationships_by_year.csv"); print(out.round(3).to_string())
