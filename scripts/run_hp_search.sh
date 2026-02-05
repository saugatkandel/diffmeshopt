#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status


# 2. Configuration
PROJECT_ROOT="/workspace/diffmeshopt"
DATA_DIR="${PROJECT_ROOT}/data"

DATASET="$DATA_DIR/2d_training_data_perturbations.pkl"
OUTPUT_DIR="${PROJECT_ROOT}/output/hp_search"

# 3. Generate Data (if it doesn't exist)
if [ ! -f "$DATASET" ]; then
    echo "Dataset not found at $DATASET. Generating synthetic data..."
    python -m diffmeshopt.opt2d.generate_2d_data \
        --output "$DATASET" \
        --synthetic \
        --visualize
fi

# 4. Run Hyperparameter Search
echo "Starting Hyperparameter Search..."
python -m diffmeshopt.opt2d.hyperparameter_search \
    --dataset "$DATASET" \
    --output "$OUTPUT_DIR" \
    --n-trials 500 \
    --n-jobs 10 \
    --device cuda
