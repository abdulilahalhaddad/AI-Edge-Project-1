# Which ZIP codes will beat their metro?

MIT 15.071 *The AI Edge*, Project 1 (Predictive AI). Repository: https://github.com/abdulilahalhaddad/AI-Edge-Project-1

**Question.** Investors are paid for alpha (return above the market), not beta, and have limited time to source deals.
We rank ZIP codes within each metro by their chance of beating the metro over the next 12 months, so a team knows where to look first.

**Data.** [HouseTS](https://www.kaggle.com/datasets/shengkunwang/housets-dataset) (CC BY-NC-SA 4.0, not stored here):
30 US metros, monthly 2012-2023. It has Zillow values, Redfin market activity, OpenStreetMap amenities, census data and one aerial image per ZIP code per year.
We use the 5,596 ZIP codes with complete data and an image.

**Design.**
- One row per ZIP code per December, using only information available then.
- **Target:** next-12-month log growth minus the metro average.
- **Split by time:** train 2014-2018 (tuned by rolling forward), validate 2019, stress test 2020-2022, test 2023.
  The test years were locked (`src/00_split.py`) until the model was recorded in [`docs/decision_record.md`](docs/decision_record.md), then scored once.
- **Metrics:** within-metro rank correlation and top-fifth alpha, with 95% intervals from resampling metros.
- **Models:** momentum baseline, linear/ridge/lasso, decision tree, random forest (headline), boosted trees, neural network.
  Images were tested two ways: pretrained ResNet-50 features, and MobileNetV2 fine-tuned in Colab. Each image test had a shuffled-photo placebo.

## Results

| Top-fifth return above metro (points a year) | Random forest | Momentum rule | Forest minus momentum (95%) |
|---|---|---|---|
| Development years 2016-2019 | 3.3 | 2.7 | +0.4 to +0.8 |
| 2020 (pandemic) | 2.5 | 2.0 | +0.2 to +0.8 |
| 2021 (boom) | 0.9 | 0.9 | -0.4 to +0.3 |
| 2022 (rate shock) | 1.0 | 1.1 | -0.5 to +0.4 |
| **2023 (test)** | **0.2** (-0.2 to 0.5) | 0.4 | -0.8 to +0.2 |
| All four unseen years | 1.1 (0.8 to 1.4) | 1.1 | -0.2 to +0.2 |

**What worked.**
- The screen beat the metro in every unseen year: about $1.1 million a year on $100 million, gross.
- It found 3.4 of each metro's actual top-fifth ZIP codes per 10 screened, against 2 at random.
- ZIP codes given 70%+ odds beat their metro 69% of the time, against a 52% base rate.

**What did not.**
- After 2020 the forest no longer beat the simple momentum rule. Its 2023 alpha cannot be told apart from zero.
- All the machine learning models tied with one another.
- Images added nothing: real photos scored no better than shuffled ones, and the image network alone lost money on 2021-2023.
- The forest stays the reported model because it was chosen before the test. Switching models now would be choosing on the test.

**Why.** We measured how strongly each input ranked next year's winners within a metro (rank correlation; `src/09_diagnosis.py`):

| Input | 2016-2020 | 2021-2023 |
|---|---|---|
| Cheapness (price level vs metro) | -0.55 | about 0 |
| Momentum (last year's growth vs metro) | 0.53 | 0.18 |

The relationship the model learned held for seven years and then changed with the pandemic and rate shock. This is a change in the market, not overfitting.

**Limits.**
- The target is a smoothed index, before costs and rent.
- Few market years.
- New York and Philadelphia are only partly covered.
- Early exploratory charts covered all years; no model was fitted or tuned on them.

**Next steps.**
- Use the momentum screen, and confirm it on 2024.
- Re-check the inputs every year.
- Add rate, migration and deal-level rent data.
- Test whether top picks are just higher-risk.

## Run

```
pip install -r requirements.txt
bash run_all.sh HouseTS.csv image_features.parquet [image_scores.csv]   # stages 00-07, about 20 min, seed 15071
python src/08_test.py HouseTS.csv image_scores.csv                          # one-time test (--rehearsal: development data)
python src/09_diagnosis.py image_features.parquet && python src/09c_four_year_summary.py
```

`image_scores.csv` comes from `src/05c_image_network_colab.ipynb` (Colab, T4 GPU, about 40 minutes).
Code is in `src/` (numbered stages), charts in `figures/`, and the tables behind every chart in `results/`.
