"""Stage 3c: bring in the image inputs built by the image workstream.
Reads image_features.parquet (one row per ZIP code and year; image network and PCA fitted
on images up to 2017) and keeps row years up to 2018 only, so nothing from the locked
years enters development. Usage: python src/03c_image_inputs.py <path to image_features.parquet>
"""
import sys, pandas as pd
f = pd.read_parquet(sys.argv[1])
f[f.year <= 2018].to_csv("data/dev/image_features_dev.csv", index=False)
print(f[f.year <= 2018].groupby("year").has_image.sum().to_dict())
