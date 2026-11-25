"""Stage 1: visualise the TRAINING data only (target years 2014-2018).
Reads data/dev/ only. Never reads data/locked/.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BLUE, ORANGE, RED, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#e34948", "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": "#52514e", "figure.dpi": 150,
                     "savefig.bbox": "tight", "savefig.facecolor": "white"})
CENSUS = ['Total Population','Median Age','Per Capita Income','Total Families Below Poverty','Total Housing Units','Median Rent',
          'Median Home Value','Total Labor Force','Unemployed Population','Total School Age Population','Total School Enrollment','Median Commute Time']
MARKET = ['median_sale_price','median_list_price','median_ppsf','median_list_ppsf','homes_sold','pending_sales','new_listings',
          'inventory','median_dom','avg_sale_to_list','sold_above_list','off_market_in_two_weeks']
POI = ['bank','bus','hospital','mall','park','restaurant','school','station','supermarket']

m = pd.read_csv("data/dev/monthly_dev.csv", parse_dates=["date"])
t = pd.read_csv("data/dev/targets_dev.csv")
t = t[t.set == "train"]

# ZIP codes whose census block is entirely zero in every development year -> missing, not real
zero_all = m.groupby("zipcode")[CENSUS].max().eq(0).all(axis=1)
broken = set(zero_all[zero_all].index)
print("ZIP codes with all-zero census in every year:", len(broken))

dec = m[(m.date.dt.month == 12)].copy(); dec["input_year"] = dec.date.dt.year
# momentum: excess growth over the previous 12 months (metro average over all ZIP codes)
dec = dec.sort_values(["zipcode", "date"]); dec["lp"] = np.log(dec.price)
dec["past_growth"] = dec.lp - dec.groupby("zipcode").lp.shift(1)
dec["past_excess_growth"] = dec.past_growth - dec.groupby(["city", "input_year"]).past_growth.transform("mean")
dec["price_vs_metro"] = dec.lp - dec.groupby(["city", "input_year"]).lp.transform("mean")
tr = t.merge(dec, on=["zipcode", "city", "input_year"], how="left")
tr_all = tr.copy(); tr = tr[~tr.zipcode.isin(broken)]
print("training rows:", len(tr_all), "-> after dropping broken-census ZIP codes:", len(tr), "| ZIP codes:", tr.zipcode.nunique())
print("target (train): ", (tr.excess_growth_12m*100).describe().round(2).to_dict(), "skew", round(tr.excess_growth_12m.skew(), 2))
print("share of ZIP-years beating metro:", round((tr.excess_growth_12m > 0).mean(), 3))

# ---------- Figure 1: split timeline ----------
fig, ax = plt.subplots(figsize=(9, 2.6)); ax.grid(False)
sets = [("Not used", 2013, 2013, "#f0efec", MUTED), ("Train", 2014, 2018, "#cfe0f5", INK), ("Validation", 2019, 2019, "#9ec3ee", INK),
        ("Stress test (locked)", 2020, 2022, "#f6c9b6", INK), ("Test (locked)", 2023, 2023, "#eb9b7a", INK)]
for name, a, b, col, tc in sets:
    ax.add_patch(Rectangle((a - .5, 0), b - a + 1, 1, facecolor=col, edgecolor="white", linewidth=2))
    ax.text((a + b) / 2, .62, name if b > a else {"Not used": "Not\nused", "Validation": "Valid-\nation", "Test (locked)": "Test\n(locked)"}[name], ha="center", va="center", fontsize=8.5 if b == a else 10, color=tc, fontweight="bold")
    n = 6226 * (b - a + 1)
    ax.text((a + b) / 2, .22, f"{n:,} rows" if b > a else f"{n:,}\nrows", ha="center", va="center", fontsize=8.5, color="#52514e")
ax.set_xlim(2012.5, 2023.5); ax.set_ylim(0, 1); ax.set_yticks([]); ax.set_xticks(range(2013, 2024))
for s in ax.spines.values(): s.set_visible(False)
ax.set_xlabel("Target year: the year in which the 12-month window ends (inputs are known the December before)")
ax.set_title("Figure 1. How the data is split by time (rows before any ZIP codes are excluded)")
fig.savefig("figures/1_explore/fig01_split_timeline.png"); plt.close(fig)

# ---------- Figure 2: price paths by metro (to December 2018) ----------
mm = m[~m.zipcode.isin(broken)]
q = mm.groupby(["city_full", "date"]).price.quantile([.25, .5, .75]).unstack() / 1000
order = q.xs(pd.Timestamp("2017-12-31"), level="date")[0.5].sort_values(ascending=False).index
fig, axes = plt.subplots(5, 6, figsize=(13, 9), sharex=True, sharey=True)
for ax, c in zip(axes.ravel(), order):
    d = q.loc[c]
    ax.fill_between(d.index, d[0.25], d[0.75], color=BLUE, alpha=.18, linewidth=0)
    ax.plot(d.index, d[0.5], color=BLUE, linewidth=1.8)
    ax.set_yscale("log"); ax.set_title(c.split("-")[0].replace("_Metro", "").replace("DC", "Washington DC"), fontsize=9, fontweight="normal")
    ax.set_yticks([100, 200, 400, 800]); ax.set_yticklabels(["$100k", "$200k", "$400k", "$800k"], fontsize=8)
    ax.set_xticks(pd.to_datetime(["2013-01-01", "2016-01-01", "2019-01-01"])); ax.set_xticklabels(["2013", "2016", "2019"], fontsize=8)
    ax.minorticks_off()
fig.suptitle("Figure 2. Home values by metro, March 2012 to December 2018 - line = typical ZIP code, band = middle half of ZIP codes",
             x=0.01, ha="left", fontsize=11, fontweight="bold")
fig.tight_layout(); fig.savefig("figures/1_explore/fig02_metro_prices.png"); plt.close(fig)

# ---------- Figure 3: the target ----------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.2, 1]})
x = tr.excess_growth_12m * 100
a1.hist(x.clip(-12, 12), bins=48, color=BLUE, edgecolor="white", linewidth=.6)
a1.axvline(0, color=INK, linewidth=1); a1.set_xlabel("Growth over next 12 months minus metro average (percentage points; beyond 12 grouped at the edges)")
a1.set_ylabel("ZIP-years"); a1.set_title("Figure 3a. The target: most ZIP codes land within 3 points of their metro")
p10, p90 = np.percentile(x, [10, 90])
a1.text(0.98, 0.95, f"Middle 80% of ZIP-years:\n{p10:+.1f} to {p90:+.1f} points", transform=a1.transAxes, ha="right", va="top", fontsize=9, color="#52514e")
by = [tr.loc[tr.target_year == y, "excess_growth_12m"] * 100 for y in sorted(tr.target_year.unique())]
bp = a2.boxplot(by, tick_labels=sorted(tr.target_year.unique()), whis=(10, 90), showfliers=False, patch_artist=True, widths=.55,
                medianprops={"color": INK}, boxprops={"facecolor": "#cfe0f5", "edgecolor": BLUE}, whiskerprops={"color": BLUE}, capprops={"color": BLUE})
a2.axhline(0, color=INK, linewidth=1); a2.set_xlabel("Target year"); a2.set_ylabel("Percentage points")
a2.set_title("Figure 3b. Its spread is stable from year to year")
fig.tight_layout(); fig.savefig("figures/1_explore/fig03_target.png"); plt.close(fig)

# ---------- Figure 4: momentum ----------
d = tr.dropna(subset=["past_excess_growth"]).copy()
d["decile"] = d.groupby(["city", "target_year"]).past_excess_growth.transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False)) + 1
g = d.groupby("decile").agg(nxt=("excess_growth_12m", "mean"), beat=("excess_growth_12m", lambda s: (s > 0).mean()))
fig, ax = plt.subplots(figsize=(8, 3.8))
cols = [RED if v < 0 else BLUE for v in g.nxt]
bars = ax.bar(g.index, g.nxt * 100, color=cols, width=.7)
for xi, v, b in zip(g.index, g.nxt * 100, g.beat):
    ax.text(xi, v + (0.08 if v >= 0 else -0.08), f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9, color=INK)
ax.axhline(0, color=INK, linewidth=1); ax.set_xticks(range(1, 11))
ax.set_xticklabels(["1\nweakest", "2", "3", "4", "5", "6", "7", "8", "9", "10\nstrongest"])
ax.set_xlabel("Rank within metro on LAST year's growth versus metro (tenths)"); ax.set_ylabel("Average excess growth NEXT year (points)")
ax.set_title("Figure 4. Momentum: last year's leaders within a metro tend to lead again"); ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig("figures/1_explore/fig04_momentum.png"); plt.close(fig)
print("decile table:"); print((g * 100).round(2).to_string())

# ---------- Figure 5: each column versus the target (rank correlation within metro and year) ----------
feats = ["past_excess_growth", "price_vs_metro"] + MARKET + POI + CENSUS
def within_corr(f):
    r = tr.groupby(["city", "target_year"]).apply(lambda z: z[f].corr(z.excess_growth_12m, method="spearman"), include_groups=False)
    return r.mean()
rc = pd.Series({f: within_corr(f) for f in feats}).sort_values()
group = {**{f: "Housing market" for f in MARKET}, **{f: "Amenity counts" for f in POI}, **{f: "Census" for f in CENSUS},
         "past_excess_growth": "Built by us", "price_vs_metro": "Built by us"}
gcol = {"Housing market": BLUE, "Amenity counts": "#1baf7a", "Census": "#eda100", "Built by us": "#6250d6"}
nice = {"past_excess_growth": "Last year's growth vs metro (momentum)", "price_vs_metro": "Price level vs metro"}
fig, ax = plt.subplots(figsize=(8.5, 8.2))
ax.barh([nice.get(f, f) for f in rc.index], rc.values, color=[gcol[group[f]] for f in rc.index], height=.7)
ax.axvline(0, color=INK, linewidth=1); ax.grid(axis="y", visible=False)
ax.set_xlabel("Rank correlation with next year's excess growth (average within metro and year; 0 = no relationship)")
ax.set_title("Figure 5. Which columns line up with future outperformance (training years only)")
ax.legend(handles=[Rectangle((0, 0), 1, 1, color=c) for c in gcol.values()], labels=list(gcol), frameon=False, loc="lower right", fontsize=9)
ax.tick_params(axis="y", labelsize=8.5, colors="#52514e")
fig.tight_layout(); fig.savefig("figures/1_explore/fig05_feature_vs_target.png"); plt.close(fig)
print("rank correlations:"); print(rc.round(3).to_string())

# ---------- Figure 6: correlation between columns ----------
cols6 = MARKET + POI + CENSUS
C = tr[cols6].corr(method="spearman")
fig, ax = plt.subplots(figsize=(10, 8.6)); ax.grid(False)
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("div", [RED, "#f0efec", BLUE])
im = ax.imshow(C.values, cmap=cmap, vmin=-1, vmax=1)
ax.set_xticks(range(len(cols6))); ax.set_yticks(range(len(cols6)))
ax.set_xticklabels(cols6, rotation=90, fontsize=7.5); ax.set_yticklabels(cols6, fontsize=7.5)
for k in [len(MARKET) - .5, len(MARKET) + len(POI) - .5]:
    ax.axhline(k, color="white", linewidth=2.5); ax.axvline(k, color="white", linewidth=2.5)
for s in ax.spines.values(): s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=.035, pad=.02); cb.set_label("Rank correlation (blue = move together, red = move opposite)"); cb.outline.set_visible(False)
ax.set_title("Figure 6. How the columns relate to each other - solid blue blocks are near-duplicates")
fig.tight_layout(); fig.savefig("figures/1_explore/fig06_feature_correlations.png"); plt.close(fig)
hi = C.where(np.triu(np.ones(C.shape), 1).astype(bool)).stack(); print("pairs with |corr|>0.95:"); print(hi[hi.abs() > .95].round(3).to_string())

# ---------- Figure 7: missing and suspect values ----------
allc = MARKET + POI + CENSUS
z_before = (tr_all[allc] == 0).mean() * 100; z_after = (tr[allc] == 0).mean() * 100
fig, ax = plt.subplots(figsize=(8.5, 8)); y = np.arange(len(allc))
ax.barh(y + .2, z_before.values, height=.38, color="#c3c2b7", label="All 6,226 ZIP codes")
ax.barh(y - .2, z_after.values, height=.38, color=BLUE, label=f"After removing {len(broken)} ZIP codes with an empty census block")
ax.set_yticks(y); ax.set_yticklabels(allc, fontsize=8.5, color="#52514e"); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.set_xlabel("Share of training rows where the value is exactly 0 (%)"); ax.legend(frameon=False, fontsize=9, loc="lower right")
ax.set_title("Figure 7. Zeros by column: there are no blank cells, so missing values hide as zeros")
fig.tight_layout(); fig.savefig("figures/1_explore/fig07_zeros_by_column.png"); plt.close(fig)
print("blank cells in training rows:", int(tr_all[allc].isna().sum().sum()))
print("zero share after (top):"); print(z_after.sort_values(ascending=False).head(12).round(2).to_string())
print("by-metro training ZIP codes kept:"); k = tr.groupby("city").zipcode.nunique(); a = tr_all.groupby("city").zipcode.nunique()
print(pd.DataFrame({"kept": k, "all": a}).assign(pct=lambda d: (d.kept / d["all"] * 100).round(0)).query("pct<100").to_string())
