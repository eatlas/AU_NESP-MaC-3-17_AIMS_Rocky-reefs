#!/bin/bash

# This script generates multiple versions of the dataset, each with a different
# weighting of the rocky reefs, and with the binary classifier or multi-class 
# classifier. This is part of the hyperparameter search associated with determining
# the best settings for the mapping, to find the best trade off between precision
# and recall.

# Define the dataset path as a variable
DATASET_PATH=~/Documents/2025/GIS/AU_AIMS_S2-comp

GEOTIFF_DIR=working/rocky-reefs-geotiffs

PARALLEL=15

weights=(1.0 1.5 2.0 2.5 3.0)

# Loop through each weight
for weight in "${weights[@]}"; do
  echo "-------------- MULTI CLASS WEIGHT $weight ------------------"
  python 06-classify-rocky-reefs.py \
    --dataset-path "$DATASET_PATH" \
    --geotiff-dir "$GEOTIFF_DIR" \
    --regions NorthernAU,GBR \
    --parallel "$PARALLEL" \
    --weight "$weight"

  python 07-merge-scenes.py --weight "$weight"

  # Clear out the geotiff predictions, so they are not accidentally reused
  rm -rf "$GEOTIFF_DIR"
done

for weight in "${weights[@]}"; do
  echo "-------------- BINARY CLASS WEIGHT $weight ------------------"
  python 06-classify-rocky-reefs.py \
    --dataset-path "$DATASET_PATH" \
    --geotiff-dir "$GEOTIFF_DIR" \
    --regions NorthernAU,GBR \
    --parallel "$PARALLEL" \
    --weight "$weight" \
    --binary-classifier

  python 07-merge-scenes.py --weight "$weight" --binary-classifier
  rm -rf "$GEOTIFF_DIR"
done