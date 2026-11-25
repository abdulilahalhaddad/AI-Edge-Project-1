# Early read: image columns on the validation year

Gradient boosting (scikit-learn defaults), mean of 5 seeds. Trained on row years 2013–2019, scored on row year 2020 (target year 2021). Test years untouched. Same rows in every run (has_image = 1): 39,257 train, 5,610 validation.

Panel: `data/processed/panel_provisional.parquet`.

| Run | RMSE | Within-metro Spearman (sd) | Δ vs R0 | Top-decile relative growth |
|---|---|---|---|---|
| R0 tabular only | 0.0490 | 0.0869 (0.0100) | +0.0000 | +0.0035 |
| R1 + colour stats | 0.0488 | 0.0987 (0.0091) | +0.0119 | +0.0048 |
| R2 + embedding levels | 0.0488 | 0.0937 (0.0036) | +0.0068 | +0.0036 |
| R3 + all image columns | 0.0487 | 0.0971 (0.0073) | +0.0103 | +0.0046 |
| R4 image columns only | 0.0510 | -0.0721 (0.0108) | -0.1589 | -0.0086 |
| R5 placebo (shuffled images) | 0.0490 | 0.0859 (0.0036) | -0.0010 | +0.0031 |
