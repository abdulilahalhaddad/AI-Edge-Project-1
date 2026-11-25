# Stage 5 decision record (written before any test data is opened)

Date: 25 November 2026. Status: test and stress-test data NOT opened.

## Fixed choices
- **Headline model:** random forest, tabular inputs only.
- **Simpler comparison:** linear regression, tabular inputs only.
- **Rows:** ZIP-years with an aerial image (27,967 training rows, 5,596 validation rows), so results with and without image inputs cover identical rows.
- **Settings:** chosen by lowest average error over the rolling training years (predict 2016, 2017, 2018), with widened ranges. See `chosen_settings.json` and `tuning_log.csv`.
  Random forest: 500 trees, minimum 10 rows per leaf, half the inputs considered per split, no depth limit.
- **Image inputs:** not used in the headline model. On 2019 validation they did not improve any model and did no better than shuffled images.

## What will be scored on the locked data, once
All of: zero, raw momentum and scaled momentum baselines; linear, ridge and lasso regression; decision tree; random forest; gradient boosted trees; neural network (one hidden layer of 32 units, penalty 1.0); random forest plus all image inputs; and, if the Colab image network passes its 2019 check (`src/05d_image_network.py`), random forest plus the image network's score with the blend weight chosen on 2018. A forest classifier gives the probability of beating the metro.
The image network is not retrained before the test: it keeps the version trained on target years up to 2017.
The whole procedure was rehearsed on development data first (`python src/08_test.py --rehearsal`, 2019 playing the test year); it reproduced the validation results exactly.
Each is refit on training plus validation years with the settings above, then scored on target year 2023 (main test) and separately on 2020, 2021 and 2022 (stress test).

## Rule
The headline model stays the random forest whatever the test shows. If another model scores higher on the test data, that is reported as a finding, not used to change the choice.

## Reasons
The random forest had the lowest error on the rolling training years (3.05 points) and on 2019 validation (2.80 points), and the highest rank correlation (0.56). Linear regression is close (0.54) and is kept as the transparent comparison.
