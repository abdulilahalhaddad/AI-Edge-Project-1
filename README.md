# AI-Edge-Project-1: Do aerial images improve ZIP-level home-value forecasts?

MIT 15.071 *The AI Edge*, Project 1 (predictive AI). The task is to predict a ZIP code's
home-value growth over the next 12 months, relative to its metro's average that year. We
then test whether features from aerial imagery improve on market and census data alone.

- **Data:** [HouseTS](https://arxiv.org/abs/2506.00765) covers 6,226 ZIPs in 30 US metros,
  monthly from 2012 to 2023, plus 35,099 aerial images of 5,611 of those ZIPs (2011–2023).
- **Unit:** one row per ZIP per December, 2013–2022. The target covers the following 12 months.
- **Split by row year:** train 2013–2019, validation 2020, test 2021–2022 (target years 2022–2023).

## Result

On test, image features **do not** improve the forecast. Scores are within-metro rank
correlation of predicted vs actual relative growth, on the 11,221 test rows that have an
image. 95% intervals come from resampling ZIPs (500 times).

| | Gradient boosting | Ridge | Neural net |
|---|---|---|---|
| R0 tabular only | 0.199 | **0.204** | 0.200 |
| R3 + all 34 image columns | 0.196 | 0.184 | 0.171 |
| R3 − R0 | −0.003 [−0.013, +0.006] | −0.020 [−0.026, −0.014] | −0.019 [−0.035, −0.004] |

- **Simple colour statistics (R1) give a small, real gain** with boosting: +0.009
  [+0.002, +0.018]. The greenness and built-up texture of a ZIP carry a little signal.
  This was one of five comparisons, so treat it as suggestive.
- **The placebo (R5, image columns shuffled within metro-year) shows nothing:** +0.000.
- **Validation and test disagree.** On the validation year R3 beat R0 for every model, but
  the gain didn't survive to later years.
- **Stage B fine-tuned network:** image-only rank correlation of 0.06 on test. Validation
  chose a blend weight of 0, so it adds nothing to the tabular model.

Full tables, the breakdowns (by test year, image freshness, visible change, ZIP size and
metro) and every tuned setting are in [reports/experiment_results.md](reports/experiment_results.md).

> **Caveat:** the tabular side uses a *provisional* panel (`src/shared/make_panel_provisional.py`):
> December market, amenity and census columns plus price momentum. When the tabular
> workstream's panel is ready, set `paths.panel` in `config.yaml` and rerun steps 4–7.

## Setup

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
ln -s ~/Downloads/archive data/raw   # Kaggle HouseTS download: HouseTS.csv + Housets_Image/
```

All paths, cut-off years, splits and seeds are set in [config.yaml](config.yaml).

## Reproduce

| Step | Command | Output | Time (M4 Pro) |
|---|---|---|---|
| 1 | `python src/images/01_index.py` | `data/interim/image_index.csv` | 1 s |
| 2 | `python src/images/02_embed.py` | ResNet-50 embeddings + colour stats | ≈ 7 min |
| 3 | `python src/images/03_features.py` | `data/processed/image_features.parquet` | 3 s |
| 4 | `python src/shared/make_panel_provisional.py` | `data/processed/panel_provisional.parquet` | 1 s |
| 5 | `python src/images/05_quick_experiment.py` | Validation-only early read | 35 s |
| 6 | `python src/images/04_finetune.py` | Stage B: `models/resnet18_finetuned.pt`, `image_score.parquet` | ≈ 12 min |
| 7 | `python src/images/07_experiment.py` | Runs R0–R5, bootstrap, breakdowns, blend | ≈ 7 min |
| 8 | `python src/images/06_checks.py` | `reports/image_qc.md` (25 checks) | 6 s |

Step 2 can be resumed (`--limit N` for a timing run). Steps 2 and 6 use MPS, then CUDA, then CPU.

## Image features (Stage A)

Image features come from a **ResNet-50 pretrained on ImageNet, used without retraining**.

1. Each 512 px frame is resized to 448 px. The 14×14 layer4 map is averaged only over
   pixels inside the ZIP boundary, so the black padding doesn't dominate.
2. Six colour statistics are computed over pixels inside the ZIP: coverage, mean R/G/B,
   greenness `(2G−R−B)/(R+G+B)`, and texture (standard deviation of grey levels).
3. StandardScaler + PCA(16) are fitted **only on images up to 2017**. The 16 components
   explain 36.4% of the variance.
4. Components and colour statistics are centred within (metro, image year), which removes
   season and camera differences.
5. Each ZIP-year row takes its latest image with `image_year ≤ year`. Change columns
   compare that image with the previous one, per year of gap.

**`image_features.parquet`:** 62,260 rows keyed on `zipcode, year`.
- 34 model inputs: `pc_01–16`, six colour levels, `chg_pc_01–08`, `chg_cos`, `chg_green_idx`,
  `chg_texture`, `img_age`.
- Flags: `has_image`, `has_change`. Columns are NaN when there is no image. Fill with the
  training median for linear models and nets and keep the flags as inputs; boosting reads NaN directly.
- Column groups are listed in `image_features_columns.json`.

## Stage B and experiment protocol

- **Stage B:** ResNet-18 (ImageNet) with layer4 and the head trained, at 224 px, with flip
  and 90° rotation augmentation. AdamW (1e-4 / 1e-3), Huber loss, up to 8 epochs, early
  stopping on validation within-metro Spearman, seed 0. It trains on row years ≤ 2019 only.
  `img_score` exists only for validation and test rows and is used only for blending.
- **Experiment:** every run is scored on the same rows (`has_image == 1`). Each model and
  run is tuned on validation, refit on train + validation, and scored once on test.
  Boosting and the network are averaged over 5 seeds.

## Leakage safeguards (checked in `06_checks.py`)

- A row never uses an image taken after its year, and no row uses a 2023 image.
- The scaler and PCA see only images up to 2017. Centring uses same-year images only.
- No Stage A script reads an outcome column; a static source scan enforces this.
- The per-ZIP `.json` annotation files are never read: each summarises future images.

## Data issues found, and how they were handled

- **DC images are a different format.** All 1,991 DC images are RGBA with the long side
  512 px. They are centre-padded with black to 512×512, not stretched.
- **Partial captures.** 1,103 images (3.1%) cover less than 50% of their ZIP's fullest frame
  because the imagery tile clipped the ZIP. They are treated as missing, and the row uses the
  previous good image. `min_coverage: 0` in `config.yaml` turns this off.
- **Coverage:** 615 of 6,226 ZIPs (10%) have no images. They are excluded from every
  experiment run.

## Repo layout

```
config.yaml           paths, cut-offs, split, seeds
src/images/           01_index · 02_embed · 03_features · 04_finetune (Stage B)
                      05_quick_experiment · 06_checks · 07_experiment · common
src/shared/           make_panel_provisional (stand-in tabular panel)
data/raw -> download  git-ignored, 12 GB
data/interim/         index, embeddings, colour stats (git-ignored, rebuildable)
data/processed/       image features, panel, Stage B scores, test predictions
models/               scaler + PCA (joblib); fine-tuned weights (.pt, git-ignored)
reports/              image_qc.md, experiment_results.md, early read, Stage B metrics
```
