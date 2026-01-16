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
- **Core Modules Refactored**:
    - Geometric losses (`LaplacianSmoothingLoss`, `EdgeLengthConsistencyLoss`) have been implemented and moved to `src/loss_2d.py`.
    - The optimization loop has been encapsulated into a `ContourRefiner` class within `src/optimize_2d.py`.
    - The `ContourRefiner` class now correctly uses stochastic sampling for the data term and computes geometric losses on the full contour.
- **Stochastic Sampling Implemented**:
    - A stratified random sampling strategy (`_get_stratified_indices`) has been implemented in `src/optimize_2d.py` to select mini-batches of vertices for the data loss calculation.
    - This approach provides stable normals by calculating them on the coarser, subsampled contour.
- **B-Spline Parameterization**:
    - A `BSplineContourRefiner` class has been added to `src/optimize_2d.py` to optimize control points instead of raw vertices, enforcing smoothness by construction.
    - **Note:** This B-spline refiner has not yet been tested.

## In Progress
- **2D Optimization Validation**:
    - The `ContourRefiner` class is implemented but the end-to-end optimization loop has **not yet been validated**.
    - The next step is to use a Jupyter notebook to run the `ContourRefiner` on synthetic data, visualize the results, and confirm that the contour correctly converges to the target.

## Pending Features & Limitations
- **Template Optimization**: The bi-Gaussian template parameters are currently fixed. Optimization of these parameters (e.g., peak distance, sigma) during the refinement process is not yet implemented.
- **Self-Supervised Learning**: The eventual goal of using this as a self-supervised learning signal for a neural network has not been implemented.

## Next Steps (Resume Plan)
1.  **Update Neural Model**: Modify `src/model_2d.py` to use the updated `sample_profiles` function from `src/optimize_2d.py` (or replicate the rectangular sampling logic) to ensure the NN sees the same features as the direct optimization.
2.  **Port to 3D**:
    *   Review `plan_3d.md`.
    *   Implement 3D mesh loading/saving.
    *   Implement 3D vertex normal computation.
    *   Implement 3D `grid_sample` logic (sampling prisms/cylinders along normals).

## How to Run
Use `notebooks/optimize_2d.ipynb`.
```