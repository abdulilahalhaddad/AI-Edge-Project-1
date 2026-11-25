#!/usr/bin/env bash
# Reproduce every development result (training and validation years only).
# Usage: bash run_all.sh path/to/HouseTS.csv path/to/image_features.parquet
# The test and stress-test years are locked by step 0 and are not used here.
set -euo pipefail
RAW="$1"; IMAGES="$2"
mkdir -p figures/1_explore figures/2_models figures/3_impact results/1_data results/2_models results/3_impact
python src/00_split.py "$RAW"                  # build target, split by time, lock test data
python src/01_visualise.py                     # figures 1-7 (training years only)
python src/02_clean.py                         # cleaning rules learned on training years
python src/03_features.py                      # 20 inputs + 1 flag per ZIP code per December
python src/03b_feature_chart.py                # figure 8
python src/03c_image_inputs.py "$IMAGES"       # aerial-image inputs, row years <= 2018
python src/04_models.py                        # baselines + six models, tuned on rolling years
python src/04b_model_charts.py                 # figures 9-12
python src/05_images.py                        # do image inputs help? (with placebo)
python src/05b_image_chart.py                  # figure 13
python src/06_impact.py                        # alpha and sourcing efficiency, 2016-2019
python src/06b_impact_charts.py                # figures 14-15
python src/07_robustness_and_probability.py    # neural network, ablation, census lag, probability
python src/07b_probability_chart.py            # figure 16
# publish the small result tables into results/
cp data/dev/{cleaning_rules.json,feature_clip_bounds.json,feature_rank_correlations_train.csv} results/1_data/
cp data/dev/{model_results.csv,chosen_settings.json,tuning_log.csv,model_drivers.csv,image_experiment.csv,image_experiment_intervals.csv,robustness_checks.csv} results/2_models/
cp data/dev/{impact_alpha_by_year.csv,impact_alpha_intervals.csv,impact_sourcing_gains.csv,probability_calibration.csv,probability_summary.json} results/3_impact/
echo "Done. Figures in figures/, tables in results/."
