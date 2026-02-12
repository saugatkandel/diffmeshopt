#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status


# 2. Configuration
PROJECT_ROOT="/workspace/diffmeshopt"
DATA_DIR="${PROJECT_ROOT}/data"

DATASET="$DATA_DIR/2d_training_data_perturbations.pkl"
OUTPUT_DIR="${PROJECT_ROOT}/output/hp_search"

# 4. Run Hyperparameter Search
echo "Starting Hyperparameter Search..."
python -m diffmeshopt.opt2d.hyperparameter_search \
    --dataset "$DATASET" \
    --output "$OUTPUT_DIR" \
    --n-trials 500 \
    --n-jobs 5 \
    --device cuda
