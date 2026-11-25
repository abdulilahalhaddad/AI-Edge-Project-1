"""Stage 3: features. One row per ZIP code per December, using only
information dated on or before that December. Every feature is expressed
relative to the ZIP code's own metro that December, to match the target.
Clipping bounds are learned from training rows only and saved.
"""
import json
import numpy as np, pandas as pd

AMEN = ['bank','bus','hospital','mall','park','restaurant','school','station','supermarket']
FEATURES = {
 "mom_12m": "Growth over the past 12 months vs metro (momentum)",
 "mom_3m": "Growth over the past 3 months vs metro",
 "price_level": "Home value level",
 "price_to_income": "Home value relative to income per person",
 "price_to_rent": "Home value relative to a year of median rent",
 "turnover_12m": "Homes sold in 12 months per housing unit",
 "days_on_market_12m": "Typical days on market, past 12 months",
 "sale_to_list_12m": "Sale price relative to asking price, past 12 months",
 "sold_above_list_12m": "Share of homes sold above asking, past 12 months",
 "off_market_fast_12m": "Share of homes off the market within two weeks, past 12 months",
 "months_of_supply": "Homes for sale relative to monthly sales",
 "inventory_change": "Change in homes for sale vs a year earlier",
 "days_on_market_change": "Change in days on market vs a year earlier",
 "amenities_per_1000": "Amenities per 1,000 residents",
 "amenity_change": "Change in amenity count over 12 months",
 "population": "Population",
 "population_growth": "Population growth over the past year",
 "median_age": "Median age",
 "unemployment_rate": "Unemployed share of the labour force",
 "stale_data_share": "Share of the past 12 months with carried-forward market data",
}
FLAGS = ["flag_off_market_missing"]

def wavg(d, col, w="homes_sold"):
    return (d[col] * d[w]).sum() / d[w].sum() if d[w].sum() > 0 else np.nan

def build(clean, raw_prices, input_years):
    """clean: cleaned monthly rows. raw_prices: zipcode, city, date, price for ALL ZIP codes (metro benchmark)."""
    rp = raw_prices.sort_values(["zipcode", "date"]).copy(); rp["lp"] = np.log(rp.price)
    for k in (3, 12):
        rp[f"g{k}"] = rp.lp - rp.groupby("zipcode").lp.shift(k)
        rp[f"mom_{k}m"] = rp[f"g{k}"] - rp.groupby(["city", "date"])[f"g{k}"].transform("mean")
    c = clean.sort_values(["zipcode", "date"]).copy()
    c["amen"] = c[AMEN].sum(axis=1)
    for col in ["median_dom", "avg_sale_to_list", "sold_above_list", "off_market_in_two_weeks"]:
        c["w_" + col] = c[col] * c.homes_sold
    out = []
    for y in input_years:
        end = pd.Timestamp(f"{y}-12-31"); w12 = c[(c.date > end - pd.DateOffset(months=12)) & (c.date <= end)]
        q_now = c[(c.date > end - pd.DateOffset(months=3)) & (c.date <= end)]
        e1 = end - pd.DateOffset(years=1); q_prev = c[(c.date > e1 - pd.DateOffset(months=3)) & (c.date <= e1 + pd.offsets.MonthEnd(0))]
        g = w12.groupby("zipcode")
        f = pd.DataFrame({"sales": g.homes_sold.sum(), "inv": g.inventory.mean(), "stale_data_share": g.flag_market_carried_forward.mean()})
        for col, name in [("median_dom", "days_on_market_12m"), ("avg_sale_to_list", "sale_to_list_12m"),
                          ("sold_above_list", "sold_above_list_12m"), ("off_market_in_two_weeks", "off_market_fast_12m")]:
            f[name] = g["w_" + col].sum(min_count=1) / f.sales.replace(0, np.nan)
        def q(d):
            gg = d.groupby("zipcode"); return pd.DataFrame({"inv": gg.inventory.mean(), "dom": gg.w_median_dom.sum() / gg.homes_sold.sum().replace(0, np.nan)})
        a, b = q(q_now), q(q_prev)
        f["inventory_change"] = np.log1p(a.inv) - np.log1p(b.inv)
        f["days_on_market_change"] = np.log1p(a.dom) - np.log1p(b.dom)
        d = c[c.date == end].set_index("zipcode"); dp = c[c.date == e1 + pd.offsets.MonthEnd(0)].set_index("zipcode")
        f["city"] = d.city; f["price_level"] = np.log(d.price)
        f["price_to_income"] = np.log(d.price / d["Per Capita Income"]); f["price_to_rent"] = np.log(d.price / (12 * d["Median Rent"]))
        f["turnover_12m"] = np.log1p(f.sales / d["Total Housing Units"] * 100)
        f["months_of_supply"] = np.log1p(f.inv / (f.sales.replace(0, np.nan) / 12))
        f["amenities_per_1000"] = np.log1p(d.amen / d["Total Population"] * 1000); f["amenity_change"] = d.amen - dp.amen
        f["population"] = np.log(d["Total Population"]); f["population_growth"] = np.log(d["Total Population"] / dp["Total Population"])
        f["median_age"] = d["Median Age"]; f["unemployment_rate"] = d["Unemployed Population"] / d["Total Labor Force"]
        m = rp[rp.date == end].set_index("zipcode"); f["mom_12m"] = m.mom_12m; f["mom_3m"] = m.mom_3m
        f["input_year"] = y; out.append(f.reset_index())
    X = pd.concat(out, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    X["flag_off_market_missing"] = X.off_market_fast_12m.isna().astype(int)
    return X[["zipcode", "city", "input_year"] + list(FEATURES) + FLAGS]

def finish(X, bounds):
    """clip at training 1st/99th percentiles, express vs metro, fill gaps with the metro average (0)."""
    X = X.copy(); raw_mom = X[["mom_12m", "mom_3m"]].copy()
    for f in FEATURES:
        X[f] = X[f].clip(*bounds[f])
        if f not in ("mom_12m", "mom_3m"):      # momentum is already measured against the all-ZIP metro benchmark
            X[f] = X[f] - X.groupby(["city", "input_year"])[f].transform("mean")
        X[f] = X[f].fillna(0.0)
    X["mom_12m_unclipped"] = raw_mom.mom_12m      # kept for the naive baseline only
    return X

if __name__ == "__main__":
    clean = pd.read_csv("data/dev/monthly_dev_clean.csv", parse_dates=["date"])
    raw = pd.read_csv("data/dev/monthly_dev.csv", usecols=["zipcode", "city", "date", "price"], parse_dates=["date"])
    t = pd.read_csv("data/dev/targets_dev.csv")
    X = build(clean, raw, sorted(t.input_year.unique()))
    X = X.merge(t[["zipcode", "input_year", "target_year", "set", "excess_growth_12m"]], on=["zipcode", "input_year"], how="inner")
    tr = X[X.set == "train"]
    print("rows by set:", X.set.value_counts().to_dict())
    print("missing before filling (training rows):", tr[list(FEATURES)].isna().sum().loc[lambda s: s > 0].to_dict())
    bounds = {f: [float(tr[f].quantile(.01)), float(tr[f].quantile(.99))] for f in FEATURES}
    json.dump(bounds, open("data/dev/feature_clip_bounds.json", "w"), indent=1)
    X = finish(X, bounds); X.to_csv("data/dev/features_dev.csv", index=False)
    tr = X[X.set == "train"]
    rc = pd.Series({f: tr.groupby(["city", "target_year"]).apply(lambda z: z[f].corr(z.excess_growth_12m, method="spearman"), include_groups=False).mean() for f in FEATURES}).sort_values()
    rc.to_csv("data/dev/feature_rank_correlations_train.csv")
    print(rc.round(3).to_string())
    C = tr[list(FEATURES)].corr(method="spearman"); hi = C.where(np.triu(np.ones(C.shape), 1).astype(bool)).stack()
    print("feature pairs with |corr| > 0.7:"); print(hi[hi.abs() > .7].round(2).to_string())
