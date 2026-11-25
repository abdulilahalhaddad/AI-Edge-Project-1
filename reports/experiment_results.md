# Experiment: do the image columns help?

Panel: `data/processed/panel_provisional.parquet`. Rows with an image only: 39,257 train (row years 2013–2019), 5,610 validation (2020), 11,221 test (2021–2022, target years 2022–2023). Each model is tuned on validation, refit on train + validation, and scored once on test. Boosting and the neural network are averaged over 5 seeds.

Metrics on test: within-metro Spearman = rank correlation of predicted and actual relative growth inside each metro and year, averaged; top decile = actual relative growth of the 10% of ZIPs ranked highest in each metro and year.

## All runs

| Model | Run | Inputs | Valid Spearman | Test Spearman (sd) | Δ vs R0 | Test RMSE | Top-decile growth |
|---|---|---|---|---|---|---|---|
| Gradient boosting | R0 | Tabular only | 0.1115 | 0.1985 (0.0101) | +0.0000 | 0.0377 | +0.0026 |
| Gradient boosting | R1 | + colour stats | 0.1296 | 0.2089 (0.0095) | +0.0105 | 0.0376 | +0.0034 |
| Gradient boosting | R2 | + embedding levels | 0.1114 | 0.1979 (0.0049) | -0.0006 | 0.0371 | +0.0034 |
| Gradient boosting | R3 | + all image columns | 0.1162 | 0.1962 (0.0095) | -0.0023 | 0.0375 | +0.0033 |
| Gradient boosting | R4 | Image columns only | 0.0065 | 0.0389 (0.0065) | -0.1595 | 0.0385 | -0.0024 |
| Gradient boosting | R5 | Placebo (shuffled) | 0.1102 | 0.2017 (0.0034) | +0.0033 | 0.0369 | +0.0042 |
| Ridge | R0 | Tabular only | 0.1458 | 0.2042 (0.0000) | +0.0000 | 0.0352 | +0.0066 |
| Ridge | R3 | + all image columns | 0.1591 | 0.1844 (0.0000) | -0.0199 | 0.0353 | +0.0055 |
| Neural network (MLP) | R0 | Tabular only | 0.1342 | 0.2004 (0.0040) | +0.0000 | 0.0382 | +0.0054 |
| Neural network (MLP) | R3 | + all image columns | 0.1522 | 0.1713 (0.0212) | -0.0291 | 0.0416 | +0.0040 |

## Headline: R3 minus R0, 95% bootstrap interval

Resampling ZIPs with replacement, 500 times, on seed-averaged test predictions (so point estimates can differ slightly from the per-seed means above).

| Model | Comparison | Δ Spearman | 95% interval | Excludes zero |
|---|---|---|---|---|
| Gradient boosting | R3 − R0 | -0.0033 | [-0.0128, +0.0058] | no |
| Ridge | R3 − R0 | -0.0199 | [-0.0257, -0.0138] | yes |
| Neural network (MLP) | R3 − R0 | -0.0194 | [-0.0346, -0.0035] | yes |
| Gradient boosting | R1 − R0 | +0.0091 | [+0.0017, +0.0178] | yes |
| Gradient boosting | R5 − R0 | +0.0001 | [-0.0073, +0.0090] | no |

**Reading (PRD rule: R3 beats R0, interval excludes zero, placebo shows no gain):** the rule is not met. Adding all image columns (R3) does not beat tabular-only (R0) for any model. Following the PRD, the finding is that ZIP-level imagery, as encoded here, adds no forecast value beyond market and census data. Colour statistics alone (R1) move rank correlation by +0.0091 [+0.0017, +0.0178], an interval that excludes zero. Placebo R5 − R0: +0.0001 [-0.0073, +0.0090].

## Stage B blend

Best R0 model on validation: Ridge. Blend = R0 prediction + w × target sd × img_score, w chosen on validation: **w = 0.0** (validation Spearman 0.1458 vs 0.1458 at w = 0). Test: Spearman 0.2042, RMSE 0.0352, top decile +0.0066. The network was trained once on targets up to 2020 and not retrained before testing.

## Breakdowns: R3 vs R0, gradient boosting

| Slice | Test rows | R0 Spearman | R3 Spearman | Δ |
|---|---|---|---|---|
| Test year: target 2022 (row 2021) | 5,610 | 0.2805 | 0.2831 | +0.0026 |
| Test year: target 2023 (row 2022) | 5,611 | 0.1306 | 0.1214 | -0.0092 |
| Fresh image (img_age 0) | 5,989 | 0.1940 | 0.1902 | -0.0038 |
| Older image (img_age >= 1) | 5,232 | 0.1619 | 0.1857 | +0.0237 |
| Visible change: top quarter of chg_cos | 2,805 | 0.1638 | 0.1719 | +0.0081 |
| Visible change: rest | 8,414 | 0.2042 | 0.2042 | -0.0000 |
| Smaller ZIPs (housing units below median) | 5,610 | 0.1375 | 0.1380 | +0.0005 |
| Larger ZIPs (housing units at/above median) | 5,611 | 0.2226 | 0.2231 | +0.0005 |

chg_cos top-quarter threshold: 0.0624. Housing-units median: 8,835.

### By metro

| Metro | Test rows | R0 | R3 | Δ |
|---|---|---|---|---|
| PHL | 480 | 0.2125 | 0.2648 | +0.0523 |
| LV | 135 | 0.1866 | 0.2386 | +0.0520 |
| SD | 108 | 0.0399 | 0.0880 | +0.0481 |
| LA | 720 | 0.1654 | 0.2111 | +0.0456 |
| DET | 426 | -0.0841 | -0.0472 | +0.0369 |
| TPA | 258 | 0.1181 | 0.1396 | +0.0215 |
| ATL | 406 | 0.2937 | 0.3059 | +0.0122 |
| CHI | 750 | 0.3935 | 0.4034 | +0.0098 |
| CIN | 298 | 0.2297 | 0.2392 | +0.0096 |
| DC | 616 | 0.4564 | 0.4649 | +0.0085 |
| BOS | 496 | 0.5587 | 0.5631 | +0.0043 |
| MIA | 362 | 0.1635 | 0.1671 | +0.0037 |
| MSP | 434 | 0.3179 | 0.3198 | +0.0020 |
| NY | 900 | 0.6809 | 0.6764 | -0.0045 |
| ORL | 182 | -0.0632 | -0.0686 | -0.0055 |
| SF | 330 | 0.5920 | 0.5843 | -0.0077 |
| STL | 340 | 0.0079 | -0.0014 | -0.0093 |
| RIV | 288 | 0.0388 | 0.0262 | -0.0126 |
| DAL | 494 | -0.0788 | -0.0920 | -0.0132 |
| HOU | 438 | 0.1197 | 0.1031 | -0.0166 |
| SEA | 310 | 0.4523 | 0.4315 | -0.0208 |
| DEN | 248 | 0.1118 | 0.0889 | -0.0228 |
| ATX | 172 | 0.0890 | 0.0655 | -0.0235 |
| PHX | 296 | 0.0046 | -0.0191 | -0.0237 |
| BWI | 264 | 0.1187 | 0.0929 | -0.0258 |
| PGH | 592 | 0.0529 | 0.0237 | -0.0293 |
| PDX | 232 | 0.5222 | 0.4912 | -0.0310 |
| SA | 208 | 0.1687 | 0.1216 | -0.0471 |
| SAC | 228 | 0.1173 | 0.0615 | -0.0558 |
| CLT | 210 | 0.1788 | 0.1222 | -0.0566 |

Runtime: 7 min.
