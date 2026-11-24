# AI-Edge-Project-1: Do aerial images improve ZIP-level home-value forecasts?

MIT 15.071 *The AI Edge*, Project 1 (predictive AI). The task is to predict a ZIP code's
home-value growth over the next 12 months, relative to its metro's average that year. We
then test whether features from aerial imagery improve on market and census data alone.

- **Data:** [HouseTS](https://arxiv.org/abs/2506.00765) covers 6,226 ZIPs in 30 US metros,
  monthly from 2012 to 2023, plus 35,099 aerial images of 5,611 of those ZIPs (2011–2023).
- **Unit:** one row per ZIP per December, 2013–2022 (62,260 rows). The target covers the following 12 months.
- **Split by row year:** train 2013–2019, validation 2020, test 2021–2022.

## Status

| Part | Status |
|---|---|
| Image pipeline, Stage A (frozen ResNet-50 features) | Done. 21 of 21 acceptance checks pass ([reports/image_qc.md](reports/image_qc.md)) |
| Tabular models (ridge, gradient boosting, neural net) | Separate workstream |
| Experiment R0–R5 (do image columns help?) | Not started |
| Stage B (fine-tuned ResNet-18), stretch goal | Not started |

## Setup

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
ln -s ~/Downloads/archive data/raw   # Kaggle HouseTS download: HouseTS.csv + Housets_Image/
```

All paths, cut-off years and seeds are set in [config.yaml](config.yaml). Nothing is hard-coded in the scripts.

## Reproduce the image features

| Step | Command | Output | Time (M4 Pro, MPS) |
|---|---|---|---|
| 1 | `python src/images/01_index.py` | `data/interim/image_index.csv` | 1 s |
| 2 | `python src/images/02_embed.py` | `embeddings_resnet50.npy` (35,099 × 2,048), `colour_stats.csv` | ≈ 7 min (≈ 82 img/s) |
| 3 | `python src/images/03_features.py` | `data/processed/image_features.parquet`, scaler and PCA in `models/` | 3 s |
| 4 | `python src/images/06_checks.py` | `reports/image_qc.md`, contact sheets in `reports/figures/` | 6 s |

Step 2 can be resumed (`--limit N` for a timing run, `--reset` to start over). It picks the
device in the order MPS, CUDA, CPU.

## How the features are built

1. **Embed.** ResNet-50 with ImageNet1K_V2 weights, frozen (no training). Each 512 px frame
   is resized to 448 px. The 14×14 layer4 map is averaged using only pixels inside the ZIP
   boundary (any channel > 8), so the black padding doesn't swamp the embedding.
2. **Colour statistics** over pixels inside the ZIP: `valid_frac`, mean R/G/B, a greenness
   index `(2G−R−B)/(R+G+B)`, and `texture` (standard deviation of grey levels).
3. **Compress.** StandardScaler + PCA(16) fitted **only on images up to 2017**. The 16
   components explain 36.4% of the scaled variance.
4. **Centre** each component and colour statistic within (metro, image year), which removes
   season and camera differences.
5. **Join.** Each ZIP-year row takes its latest image with `image_year ≤ year`. Change
   columns compare that image with the ZIP's previous one, per year of gap.

## Output: `data/processed/image_features.parquet`

62,260 rows, keyed on `zipcode, year` (year is the December the inputs are known at).
Column groups are listed in `image_features_columns.json`.

- **34 model inputs:** `pc_01–16`, six colour levels, `chg_pc_01–08`, `chg_cos`,
  `chg_green_idx`, `chg_texture`, and `img_age`.
- **Flags:** `has_image`, `has_change`. Columns are NaN when there is no image, never zero.
  Fill with the training-set median for linear models and neural nets, and keep the flags as
  inputs. Gradient boosting reads NaN directly.
- **Bookkeeping, not inputs:** `img_year`, `img_gap`, `img_coverage`.
- **Coverage:** 615 ZIPs have no images. Score every experiment run on `has_image == 1` rows only.

## Leakage safeguards (all checked in `06_checks.py`)

- A row never uses an image taken after its year, and no row uses a 2023 image.
- The scaler and PCA see only images up to 2017. Centring uses same-year images only.
- No Stage A script reads a price, growth or other outcome column. A static scan of the
  source enforces this.
- The per-ZIP `.json` annotation files are never read: each summarises all of the ZIP's
  images, including future ones.

## Data issues found, and how they were handled

- **DC images are a different format.** All 1,991 DC images (308 ZIPs) are RGBA with the
  long side 512 px. They are flattened to RGB and centre-padded with black to 512×512, not
  stretched, so the ground scale is unchanged.
- **Partial captures.** Some frames show only a sliver of the ZIP: the imagery tile clipped
  it. Images covering less than 50% of the ZIP's fullest frame are treated as missing
  (1,103 images, 3.1%), and the row falls back to the previous good image.
  `min_coverage: 0` in `config.yaml` turns this off. This rule compares against the ZIP's
  fullest frame across all years. That is image-capture metadata, not outcome data.

## Repo layout

```
config.yaml            paths, cut-offs, seeds
src/images/            01_index, 02_embed, 03_features, 06_checks, common
data/raw -> download   (git-ignored, 12 GB)
data/interim/          index, embeddings, colour stats (git-ignored, rebuildable)
data/processed/        image_features.parquet + column groups (committed)
models/                fitted scaler and PCA (joblib)
reports/               image_qc.md, figures/
```
