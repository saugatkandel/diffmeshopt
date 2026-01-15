# Project Status: 2D Optimization Prototype

**Last Updated:** [Current Date]

## Overview
We have implemented a 2D prototype for refining organelle segmentations using a bi-Gaussian intensity prior. The core logic involves sampling image intensity along vertex normals (using a rectangular average) and optimizing vertex positions to maximize correlation with a template.

## File Structure
*   `src/generate_2d_data.py`: Generates synthetic data or loads real data slices.
*   `src/loss_2d.py`: Defines `BiGaussianLoss` (cross-correlation).
*   `src/optimize_2d.py`: Core geometric functions (`compute_normals`, `sample_profiles` with rectangular averaging, `smooth_contour`).
*   `notebooks/optimize_2d.ipynb`: Interactive optimization loop with visualization.
*   `src/run_2d_optimization.py`: Script version of the optimization loop.
*   `src/model_2d.py`: (Pending Update) Neural network for predicting deformations.
*   `src/train_2d.py`: (Pending Update) Training loop for the network.

## Current State
1.  **Optimization Logic**: The `optimize_2d.py` module correctly implements spline smoothing, normal computation, and rectangular sampling (width=3).
2.  **Validation**: The notebook `optimize_2d.ipynb` and script `run_2d_optimization.py` successfully run the optimization loop, showing the contour snapping to the membrane center.

## Next Steps (Resume Plan)
1.  **Update Neural Model**: Modify `src/model_2d.py` to use the updated `sample_profiles` function from `src/optimize_2d.py` (or replicate the rectangular sampling logic) to ensure the NN sees the same features as the direct optimization.
2.  **Port to 3D**:
    *   Review `plan_3d.md`.
    *   Implement 3D mesh loading/saving.
    *   Implement 3D vertex normal computation.
    *   Implement 3D `grid_sample` logic (sampling prisms/cylinders along normals).

## How to Run
```bash
# Run the 2D optimization script
python src/run_2d_optimization.py
```