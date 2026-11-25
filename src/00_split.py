"""Stage 0: build targets, split by time, lock away test data.
This is the ONLY script allowed to read the full raw file. Every later
script reads data/dev/ only, until the final test stage.
Usage: python src/00_split.py <path to HouseTS.csv>
"""
import sys, os, stat
import numpy as np, pandas as pd

RAW = sys.argv[1]
TRAIN_YEARS = [2014, 2015, 2016, 2017, 2018]   # target year = year the 12-month window ends
VAL_YEARS = [2019]
STRESS_YEARS = [2020, 2021, 2022]
TEST_YEARS = [2023]
DEV_LAST_INPUT = pd.Timestamp("2018-12-31")     # latest input date used by train/validation

df = pd.read_csv(RAW, parse_dates=["date"]).sort_values(["zipcode", "date"])
assert df.groupby("zipcode").size().eq(142).all()

# --- target: 12-month forward log growth minus metro average (all ZIP codes in the metro) ---
dec = df[df.date.dt.month == 12][["zipcode", "city", "date", "price"]].copy()
dec["input_year"] = dec.date.dt.year
dec["target_year"] = dec.input_year + 1
nxt = dec[["zipcode", "input_year", "price"]].rename(columns={"price": "price_next", "input_year": "target_year"})
t = dec.merge(nxt, on=["zipcode", "target_year"], how="inner")
t["growth_12m"] = np.log(t.price_next / t.price)
t["metro_growth_12m"] = t.groupby(["city", "target_year"]).growth_12m.transform("mean")
t["excess_growth_12m"] = t.growth_12m - t.metro_growth_12m
t["n_zips_in_metro"] = t.groupby(["city", "target_year"]).zipcode.transform("size")

def which(y):
    if y in TRAIN_YEARS: return "train"
    if y in VAL_YEARS: return "validation"
    if y in STRESS_YEARS: return "stress_test"
    if y in TEST_YEARS: return "test"
    return "unused"          # target year 2013: no 12 months of history for momentum
t["set"] = t.target_year.map(which)
keep = ["zipcode", "city", "input_year", "target_year", "set", "growth_12m", "metro_growth_12m", "excess_growth_12m"]

dev_t = t[t.set.isin(["train", "validation"])][keep]
lock_t = t[t.set.isin(["stress_test", "test"])][keep]
dev_m = df[df.date <= DEV_LAST_INPUT]
lock_m = df[df.date > DEV_LAST_INPUT]

os.makedirs("data/dev", exist_ok=True); os.makedirs("data/locked", exist_ok=True)
for p in ["data/locked/targets_locked.csv", "data/locked/monthly_locked.csv"]:
    if os.path.exists(p): os.chmod(p, stat.S_IWUSR | stat.S_IRUSR)
dev_t.to_csv("data/dev/targets_dev.csv", index=False)
dev_m.to_csv("data/dev/monthly_dev.csv", index=False)
lock_t.to_csv("data/locked/targets_locked.csv", index=False)
lock_m.to_csv("data/locked/monthly_locked.csv", index=False)
for p in ["data/locked/targets_locked.csv", "data/locked/monthly_locked.csv"]:
    os.chmod(p, 0)   # no read permission until the final test stage

# --- checks (counts only; no statistics of locked targets are printed) ---
print("rows per set:"); print(t.groupby(["set", "target_year"]).size().to_string())
print("dev monthly rows:", len(dev_m), "| last date", dev_m.date.max().date())
print("locked monthly rows:", len(lock_m), "| first date", lock_m.date.min().date())
last_price_used_dev = pd.Timestamp(f"{max(VAL_YEARS)}-12-31")
first_locked_window_start = pd.Timestamp(f"{min(STRESS_YEARS)-1}-12-31")
print("dev targets use prices up to", last_print := last_price_used_dev.date(),
      "| first locked window starts", first_locked_window_start.date(),
      "| overlap:", last_price_used_dev > first_locked_window_start)
print("metro mean of excess growth is zero (max abs):", float(t.groupby(['city','target_year']).excess_growth_12m.mean().abs().max()))
