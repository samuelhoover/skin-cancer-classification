#!/bin/bash

# Unzip data and move in `data` directory
unzip -d data archive.zip

# For simplicity, moving images from `Benign` and `Malignant` directories into
# `train` directory.
# Renaming image filenames to reflect the tumor type (e.g.,
# `train/Benign/1433.jpg` --> `train/Benign_1433.jpg`).
# Delete `data/train/Benign` and `data/train/Malignant` directories afterward
for f in data/train/*/*.jpg; do
  fp=$(basename "$(dirname "$f")")
  fl=$(basename "$f")
  mv "$f" data/train/"${fp}_${fl}"
done && rm -r data/train/{Benign,Malignant}
