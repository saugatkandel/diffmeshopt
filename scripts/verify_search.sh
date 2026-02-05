#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

# This script runs a single verification trial of the hyperparameter search
# to ensure the environment and code are working correctly.

# 1. Setup Environment2. Configuration
PROJECT_ROOT="/workspace/diffmeshopt"

# 2. Configuration
DATA_DIR="${PROJECT_ROOT}/data"
DATASET="$DATA_DIR/2d_training_data_perturbations.pkl"
OUTPUT_DIR="${PROJECT_ROOT}/output/verify_run"
# 3. Generate Data (if it doesn't exist)
if [ ! -f "$DATASET" ]; then
    echo "Dataset not found at $DATASET. Generating synthetic data..."
    python -m diffmeshopt.opt2d.generate_2d_data --output "$DATASET" --synthetic
fi

# 4. Run Verification
echo "Starting verification run..."
python -m diffmeshopt.opt2d.hyperparameter_search \
    --dataset "$DATASET" \
    --output "$OUTPUT_DIR" \
    --verify \
    --device cuda