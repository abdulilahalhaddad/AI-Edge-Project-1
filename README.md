# AI Edge Project 1: which ZIP codes will outperform their metro?

MIT 15.071 The AI Edge, Project 1 (Predictive AI). Status: **work in progress**.
The tabular models are built and compared on validation. The satellite-image
features are not joined yet, and **the test data has not been opened**.

## Question

For a real estate investment committee: is a given ZIP code likely to
outperform its surrounding metro area over the next 12 months?

## Data

[HouseTS](https://www.kaggle.com/datasets/shengkunwang/housets-dataset):
6,226 ZIP codes in 30 United States metros, monthly from March 2012 to
December 2023 (licence: CC BY-NC-SA 4.0). The data is not stored here.
Download `HouseTS.csv` and pass its path to the first script.

## Design

- **One row** per ZIP code per December. Inputs use only information dated on
  or before that December.
- **Target:** log growth in `price` over the following 12 months, minus the
  simple average of that growth across all 6,226 ZIP codes in the same metro.
- **Split by time** ("target year" = the year the 12-month window ends):

| Set | Target years | Use |
|---|---|---|
| Train | 2014 to 2018 | Fit models; tune settings by rolling forward through these years |
| Validation | 2019 | Compare models |
| Stress test (locked) | 2020 to 2022 | Pandemic and rate-shock years, reported separately at the end |
| Test (locked) | 2023 | Headline result, opened once at the end |

`00_split.py` is the only script that reads the raw file. It writes the
stress-test and test rows to `data/locked/` and removes read permission.
Every other script reads `data/dev/` only.

## How to run

```
pip install -r requirements.txt
python src/00_split.py path/to/HouseTS.csv   # targets, split, lock test data
python src/01_visualise.py                    # figures 1-7, training years only
python src/02_clean.py                        # cleaning rules learned on training years
python src/03_features.py                     # 20 inputs + 1 flag
python src/03b_feature_chart.py               # figure 8
python src/04_models.py                       # baselines and seven models (seed 15071)
python src/04b_model_charts.py                # figures 9-12
```

## Cleaning (rules in `results/cleaning_rules.json`)

- The file has no blank cells; missing values appear as zeros.
- Excluded: 473 ZIP codes whose census columns are zero in every year (mostly
  New York and Philadelphia) and 16 with no market data. 5,739 remain (92%).
  New York keeps 54% of its ZIP codes and Philadelphia 72%.
- Dropped four census columns that are near-copies of population despite their
  names (`Total Families Below Poverty`, `Total School Age Population`,
  `Total School Enrollment`, `Median Commute Time`), and `pending_sales`
  (never reported for two whole metros).
- Remaining census zeros are treated as missing. Extreme market values are
  capped at the training 99.9th percentile.
- About 9% of months have market data carried forward from the prior month;
  these are flagged, and market inputs use 12-month figures.

## Results so far (validation year 2019)

| Model | Rank correlation | Top fifth minus bottom fifth (points) | Top-fifth picks beating metro | R² vs zero |
|---|---|---|---|---|
| Baseline: scaled momentum | 0.421 | 3.1 | 72% | 0.008 |
| Linear regression | 0.532 | 4.3 | 81% | 0.183 |
| Ridge / lasso regression | 0.532 / 0.531 | 4.3 | 81% / 80% | 0.183 |
| Decision tree | 0.483 | 4.0 | 79% | 0.183 |
| Random forest | 0.554 | 4.5 | 82% | 0.239 |
| Gradient boosted trees | 0.549 | 4.5 | 81% | 0.223 |
| Neural network (MLP) | 0.556 | 4.5 | 82% | 0.230 |

Rank correlation and the fifths are computed within each metro. Full numbers,
including the rolling training-year results, are in `results/model_results.csv`.

The neural network is a feedforward MLP with one hidden layer and L2
regularization. The rolling training folds select the layer width (32 or 64)
and regularization strength by RMSE. Each fit scales features and the target
using its training rows, then converts predictions back to the original units
for evaluation. The model uses the same inputs and 2019 validation as the trees.
The selected MLP had 64 hidden units and `alpha=1.0`. The random forest had
lower RMSE on the rolling folds (3.054 vs. 3.080 points) and in 2019 validation
(2.917 vs. 2.933 points). The MLP ranked ZIP codes slightly better in 2019.

## Known limits and disclosures

- Exploratory charts made before the split was fixed covered all years,
  including 2022 and 2023, and informed the choice of target. No model was
  fitted or tuned on those years.
- Part of the momentum signal comes from smoothing in the underlying home
  value index.
- Validation is a single year. The effect of price level (cheaper ZIP codes
  outperforming) may be specific to the training period.
- Several tuned settings sit at the edge of the ranges tried.

## Still to do

Join image features on `zipcode` and `year`, restrict to ZIP codes with
images, choose the final model, then open the test and stress-test data once.
