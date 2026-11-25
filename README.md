# Which ZIP codes will beat their metro?

MIT 15.071 *The AI Edge*, Project 1 (Predictive AI).

**Status:** model development is complete on training and validation years. The test and
stress-test years are **locked and have not been opened**. The headline model was fixed in
advance in [`docs/decision_record.md`](docs/decision_record.md).

## The question

A real estate investor earns fees for **alpha** (return above the market), not **beta** (the
market return itself), and has limited time to source deals. This project ranks ZIP codes within each
metro by their chance of beating the metro over the next 12 months, so the team knows where
to look first.

## Data

[HouseTS](https://www.kaggle.com/datasets/shengkunwang/housets-dataset) (licence CC BY-NC-SA 4.0):
6,226 ZIP codes in 30 United States metros, monthly from March 2012 to December 2023.
It holds Zillow home values, Redfin market activity, OpenStreetMap amenity counts, census
variables and annual aerial images. The data is not stored here. Download it from Kaggle.

## Design

- **One row** per ZIP code per December. Every input uses only information dated on or before that December.
- **Target:** log growth of the home value over the following 12 months, minus the average of that growth
  across all ZIP codes in the same metro.
- **Rows used:** 5,596 ZIP codes that have an aerial image, so results with and without image
  inputs cover identical rows. Excluded: 473 ZIP codes with an empty census block (mostly New York and
  Philadelphia), 16 with no market data, and 143 without images.
- **Split by time** (target year = the year the 12-month window ends):

| Set | Target years | Use |
|---|---|---|
| Train | 2014-2018 | Fit models; tune settings by rolling forward (fit to year t-1, predict year t) |
| Validation | 2019 | Compare models and choose one |
| Stress test (locked) | 2020-2022 | Pandemic and rate shock, reported separately, once |
| Test (locked) | 2023 | Headline result, once |

`src/00_split.py` is the only script that reads the raw file. It writes the locked years to
`data/locked/` and removes read permission. All cleaning rules, clipping bounds and settings are
learned on training years only.

## How to run

```
pip install -r requirements.txt
bash run_all.sh path/to/HouseTS.csv path/to/image_features.parquet [path/to/image_scores.csv]
```

`image_scores.csv` comes from `src/05c_image_network_colab.ipynb`, which retrains a MobileNetV2 image network on
the aerial photos. Run it in Google Colab on a T4 GPU (about 40 minutes); it needs `cnn_targets.csv`, which holds
outcomes for target years 2014-2018 only.

`image_features.parquet` holds the aerial-image inputs: ResNet-50 embeddings (pretrained, PCA fitted on
images up to 2017) and colour statistics, one row per ZIP code and year. Total run time is about 20
minutes on a laptop. Random seed: 15071.

## Repository layout

```
run_all.sh                 runs every step below in order
src/
  00_split.py              target, time split, lock test data
  01_visualise.py          exploratory charts (training years only)
  02_clean.py              cleaning rules
  03_features.py           20 inputs + 1 flag;  03b chart;  03c image inputs
  04_models.py             baselines and six models;  04b charts
  05_images.py             do image inputs help? (with placebo);  05b chart
  05c_image_network_colab.ipynb   retrain an image network in Colab;  05d checks it on 2019
  06_impact.py             alpha and sourcing efficiency;  06b charts
  07_robustness_and_probability.py   neural network, ablation, census lag, probability;  07b chart
figures/  1_explore/  2_models/  3_impact/
results/  1_data/  2_models/  3_impact/     small result tables behind every chart
  08_test.py               the one-time final test (--rehearsal runs it on development data)
docs/decision_record.md    model choice, fixed before any test run
```

## Results on development years (2016-2019)

**Model comparison, validation year 2019** (within-metro rank correlation of predicted and actual):

| Model | Rank correlation | Top-fifth picks beating metro | R-squared |
|---|---|---|---|
| Momentum rule (baseline) | 0.44 | 73% | 0.05 |
| Linear / ridge / lasso regression | 0.54 | 81% | 0.22 |
| Decision tree | 0.51 | 80% | 0.19 |
| Gradient boosted trees | 0.56 | 81% | 0.26 |
| Neural network (one layer, 32 units) | 0.57 | – | 0.27 |
| **Random forest (headline)** | **0.56** | **82%** | **0.27** |

**Business impact, 2016-2019** (each year predicted by a model trained only on earlier years):

- **Alpha:** each metro's top-fifth ZIP codes beat the metro by 3.3 points a year, against 2.7 for the momentum
  rule. The gain over momentum is 0.4 to 0.8 points (95% interval, resampling metros). That is about $3.3 million a year on
  a $100 million fund, gross of costs and for price appreciation only.
- **Sourcing:** screening the top 20% of ZIP codes finds 58% of the true top-fifth outperformers (20% at random).
- **Probability:** ZIP codes given a 70% or higher chance beat their metro 86% of the time.
- **Drivers:** momentum and cheapness relative to the metro carry most of the signal.
- **Images:** aerial-image inputs did not improve any model, and real images scored no better than shuffled ones.
  A MobileNetV2 network retrained on the photos ranks ZIP codes on its own (0.24 in 2019), but adding it to
  the forest lowered accuracy at every weight, so the headline model uses no images.
- **Robustness:** lagging census inputs by an extra year changes nothing.

## Final test: unseen years 2020-2023 (run once)

The model and decision rule were fixed in `docs/decision_record.md` before the locked years were opened. Every model was
refit on target years 2014-2019 and scored once (`src/08_test.py`, record in `results/4_test/run_record.json`).

| Unseen year | Forest top-fifth alpha (points) | Momentum rule | Forest minus momentum (95% interval) | Forest picks beating metro |
|---|---|---|---|---|
| 2020 (pandemic) | 2.54 | 2.03 | +0.51 (0.19 to 0.82) | 83% |
| 2021 (boom) | 0.86 | 0.90 | -0.05 (-0.41 to 0.34) | 57% |
| 2022 (rate shock) | 0.97 | 1.07 | -0.09 (-0.55 to 0.43) | 65% |
| **2023 (main test)** | **0.15 (-0.17 to 0.45)** | 0.45 | -0.29 (-0.82 to 0.16) | 60% |
| All four years | 1.13 (0.83 to 1.45) | 1.11 | +0.02 (-0.22 to 0.24) | 66% |

**What worked.** A data-driven screen still beat the metro in every unseen year: about 1.1 points a year
(~$1.1 million on $100 million, gross). It found 3.4 of each metro's actual top-fifth ZIP codes per 10 screened,
against 2 at random. ZIP codes given 70%+ odds beat their metro 69% of the time against a 52% base rate (2023: 65% against 54%).

**What did not.** The edge fell from 3.3 points (2016-2019) to 1.1, and after 2020 the forest no longer beat the
simple momentum rule; on 2023 its alpha cannot be told apart from zero. Linear models, boosted trees and the neural
network tied with it. The photo-only image network was negative in 2021-2023, confirming it was right to leave it out.
Under the pre-registered rule the random forest remains the reported model; switching to momentum on these results
would be choosing on the test.

**Why** (`src/09_diagnosis.py`, after the test, no model changed). Within-metro rank correlation with next-year excess growth:

| Signal | 2016-2020 | 2021-2023 |
|---|---|---|
| Cheapness (price level vs metro) | about -0.55 | about 0 |
| Momentum (last year's excess growth) | about 0.53 | about 0.18 |

The "cheap neighbourhoods catch up" pattern disappeared after the pandemic migration and rate shock, and momentum
weakened. The model learned a relationship that was stable for seven years and then changed: a regime change,
not overfitting to the training years. Four-year pooled numbers: `src/09c_four_year_summary.py`; chart: `figures/4_test/fig17_unseen_years.png`.

## Limits

- **Index, not assets.** Alpha is measured on a smoothed neighbourhood index, before costs and excluding rent. Part of the momentum signal comes from that smoothing.
- **Regime change.** The signals were learned in a rising, low-rate market and weakened sharply from 2021 (see above).
- **Few market years.** Five training years and a single validation year; the 2023 test is one year of about 5,600 ZIP codes in 30 metros.
- **Coverage.** New York keeps 54% of its ZIP codes and Philadelphia 72%.
- **Earlier exploration.** Exploratory charts made before the split was fixed covered all years. No model was fitted or tuned on the locked years.

## Next steps

- Prefer the simple momentum screen in practice, and confirm that on a fresh year (2024) rather than on the used test.
- Re-check each year whether momentum and cheapness still predict; retrain or stop when they do not.
- Add inputs for the new regime (rate sensitivity, migration, remote work) and deal-level rents and costs.
- Check whether top picks are simply higher-risk (sensitivity to the metro).
