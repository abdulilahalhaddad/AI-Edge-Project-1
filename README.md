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
bash run_all.sh path/to/HouseTS.csv path/to/image_features.parquet
```

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
  06_impact.py             alpha and sourcing efficiency;  06b charts
  07_robustness_and_probability.py   neural network, ablation, census lag, probability;  07b chart
figures/  1_explore/  2_models/  3_impact/
results/  1_data/  2_models/  3_impact/     small result tables behind every chart
docs/decision_record.md    model choice, fixed before any test run
```

## Results (training and validation years only)

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
- **Robustness:** lagging census inputs by an extra year changes nothing.

## Limits

- **Index, not assets.** Alpha is measured on a smoothed neighbourhood index, before costs and excluding rent. Part of the momentum signal comes from that smoothing.
- **Few market years.** Results rest on five training years, all in a rising market. Validation is a single year.
- **Coverage.** New York keeps 54% of its ZIP codes and Philadelphia 72%.
- **Earlier exploration.** Exploratory charts made before the split was fixed covered all years. No model was fitted or tuned on the locked years.

## Next steps

Open the test and stress-test years once, then finish the deck, the video and the final one-page README.
