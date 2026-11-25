"""Stage 2: cleaning. Rules are LEARNED from the training period only
(input dates up to December 2017) and saved to data/dev/cleaning_rules.json.
The same saved rules are applied unchanged to validation now and to the
locked data at the final test stage.
"""
import json, sys
import numpy as np, pandas as pd

TRAIN_LAST_INPUT = "2017-12-31"
CENSUS = ['Total Population','Median Age','Per Capita Income','Total Families Below Poverty','Total Housing Units','Median Rent',
          'Median Home Value','Total Labor Force','Unemployed Population','Total School Age Population','Total School Enrollment','Median Commute Time']
MISLABELLED = ['Total Families Below Poverty','Total School Age Population','Total School Enrollment','Median Commute Time']
MARKET = ['median_sale_price','median_list_price','median_ppsf','median_list_ppsf','homes_sold','pending_sales','new_listings',
          'inventory','median_dom','avg_sale_to_list','sold_above_list','off_market_in_two_weeks']
CAP_COLS = ['median_sale_price','median_list_price','median_ppsf','median_list_ppsf','median_dom','avg_sale_to_list']
CENSUS_KEEP = [c for c in CENSUS if c not in MISLABELLED]
ZERO_IS_MISSING = [c for c in CENSUS_KEEP if c != 'Unemployed Population']   # zero unemployed can be real in tiny ZIP codes

def learn_rules(train):
    g = train.groupby("zipcode")
    r = {}
    r["zips_empty_census"] = sorted(g[CENSUS].max().eq(0).all(axis=1).loc[lambda s: s].index.tolist())
    r["zips_no_market_data"] = sorted(g["homes_sold"].max().eq(0).loc[lambda s: s].index.tolist())
    r["zips_pending_never_reported"] = sorted(g["pending_sales"].max().eq(0).loc[lambda s: s].index.tolist())
    r["zips_off_market_never_reported"] = sorted(g["off_market_in_two_weeks"].max().eq(0).loc[lambda s: s].index.tolist())
    r["caps_99_9th_percentile"] = {c: float(train.loc[train[c] > 0, c].quantile(0.999)) for c in CAP_COLS}
    r["home_value_ceiling"] = 2000001; r["rent_ceiling"] = 3501
    return r

def apply_rules(m, r):
    log = {}
    excl = set(r["zips_empty_census"]) | set(r["zips_no_market_data"])
    log["rows_dropped_excluded_zips"] = int(m.zipcode.isin(excl).sum())
    m = m[~m.zipcode.isin(excl)].sort_values(["zipcode", "date"]).drop(columns=MISLABELLED).copy()
    # flags first, on raw values
    m["flag_home_value_capped"] = (m["Median Home Value"] >= r["home_value_ceiling"]).astype(int)
    m["flag_rent_capped"] = (m["Median Rent"] >= r["rent_ceiling"]).astype(int)
    m["flag_market_carried_forward"] = (m.groupby("zipcode")[MARKET].diff() == 0).all(axis=1).astype(int)
    # census zeros -> missing -> previous year's value for the same ZIP code -> else metro median that month
    z = m[ZERO_IS_MISSING].eq(0)
    log["census_zero_cells_set_missing"] = int(z.values.sum())
    m[ZERO_IS_MISSING] = m[ZERO_IS_MISSING].mask(z)
    m["flag_census_filled"] = m[ZERO_IS_MISSING].isna().any(axis=1).astype(int)
    m[ZERO_IS_MISSING] = m.groupby("zipcode")[ZERO_IS_MISSING].ffill()
    left = int(m[ZERO_IS_MISSING].isna().values.sum()); log["census_cells_filled_from_metro_median"] = left
    for c in ZERO_IS_MISSING:
        m[c] = m[c].fillna(m.groupby(["city", "date"])[c].transform("median"))
    # market: never-reported fields -> missing
    p = m.zipcode.isin(r["zips_pending_never_reported"]); o = m.zipcode.isin(r["zips_off_market_never_reported"])
    m.loc[p, "pending_sales"] = np.nan; m.loc[o, "off_market_in_two_weeks"] = np.nan
    log["rows_pending_set_missing"] = int(p.sum()); log["rows_off_market_set_missing"] = int(o.sum())
    # market: cap placeholder / extreme values at the training 99.9th percentile
    for c, cap in r["caps_99_9th_percentile"].items():
        log[f"capped_{c}"] = int((m[c] > cap).sum()); m[c] = m[c].clip(upper=cap)
    return m, log

if __name__ == "__main__":
    m = pd.read_csv("data/dev/monthly_dev.csv", parse_dates=["date"])
    rules = learn_rules(m[m.date <= TRAIN_LAST_INPUT])
    json.dump(rules, open("data/dev/cleaning_rules.json", "w"), indent=1)
    clean, log = apply_rules(m, rules)
    clean.to_csv("data/dev/monthly_dev_clean.csv", index=False)
    print({k: (len(v) if isinstance(v, list) else v) for k, v in rules.items()})
    for k, v in log.items(): print(f"{k:45s}{v:>10,}")
    print("ZIP codes kept:", clean.zipcode.nunique(), "| rows:", len(clean), "| columns:", clean.shape[1])
    print("remaining blanks by column:", clean.isna().sum().loc[lambda s: s > 0].to_dict())
    print("flag shares (%):", (clean.filter(like="flag_").mean() * 100).round(2).to_dict())
    k = clean.drop_duplicates("zipcode").city.value_counts(); a = m.drop_duplicates("zipcode").city.value_counts()
    d = pd.DataFrame({"kept": k, "all": a}); d["pct"] = (d.kept / d["all"] * 100).round(0); print(d[d.pct < 100].to_string())
    ov = set(rules["zips_pending_never_reported"]) & set(rules["zips_off_market_never_reported"]); print("off-market never reported overlap with pending:", len(ov))
    print("off-market never-reported metros:", m[m.zipcode.isin(rules["zips_off_market_never_reported"])].drop_duplicates("zipcode").city.value_counts().head(6).to_dict())
