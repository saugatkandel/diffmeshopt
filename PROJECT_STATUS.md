# Project Status: 2D Optimization Prototype

**Last Updated:** [Current Date]

## Overview
We have implemented a 2D prototype for refining organelle segmentations using a bi-Gaussian intensity prior. The core logic involves sampling image intensity along vertex normals (using a rectangular average) and optimizing vertex positions to maximize correlation with a template.

## File Structure
*   `diffmeshopt/opt2d/generate_2d_data.py`: Generates synthetic data or loads real data slices. Includes trimming logic.
*   `diffmeshopt/opt2d/loss.py`: Defines `BiGaussianLoss` (cross-correlation) and geometric losses.
*   `diffmeshopt/opt2d/optimize.py`: Core geometric functions and `ContourRefiner` classes (`ContourRefiner`, `BSplineContourRefiner`).
*   `diffmeshopt/opt2d/template.py`: Defines template parameterization models (`Fixed`, `Global`, `PerPoint`, `BSpline`, `NeuralField`).
*   `diffmeshopt/opt2d/sampling.py`: Differentiable sampling logic.
*   `diffmeshopt/opt2d/geometry.py`: Geometric utility functions.
*   `diffmeshopt/opt2d/props.py`: Configuration data structures.
*   `diffmeshopt/opt2d/vis.py`: Visualization utilities.
*   `diffmeshopt/opt2d/trainer.py`: Encapsulates the optimization loop, logging (TensorBoard), and checkpointing.
*   `diffmeshopt/opt2d/evaluation.py`: Geometric evaluation metrics.
*   `notebooks/optimize_2d.ipynb`: Interactive optimization loop with visualization.
*   `notebooks/train_2d.ipynb`: Training loop with TensorBoard visualization.
*   `notebooks/compare_combinations.ipynb`: Systematic testing of different refiner/template combinations.
*   `tests/opt2d/`: Unit and integration tests.

## Current State
- **Core Modules Refactored**:
    - Geometric losses (`LaplacianSmoothingLoss`, `EdgeLengthConsistencyLoss`) have been implemented and moved to `diffmeshopt/opt2d/loss.py`.
    - The optimization loop has been encapsulated into a `ContourRefiner` class within `diffmeshopt/opt2d/optimize.py`.
    - The `ContourRefiner` class now correctly uses stochastic sampling for the data term and computes geometric losses on the full contour.
- **Stochastic Sampling Implemented**:
    - A stratified random sampling strategy (`_get_stratified_indices`) has been implemented in `diffmeshopt/opt2d/sampling.py` to select mini-batches of vertices for the data loss calculation.
    - This approach provides stable normals by calculating them on the coarser, subsampled contour.
- **B-Spline Parameterization**:
    - A `BSplineContourRefiner` class has been added to `diffmeshopt/opt2d/optimize.py` to optimize control points instead of raw vertices, enforcing smoothness by construction.
    - **Note:** This B-spline refiner has not yet been tested.
    - **Update**: Now uses analytical derivatives for exact normal calculation (vectorized).
- **Bug Fixes**:
    - Fixed `make_splprep` usage (it returns a (BSpline, u) tuple).
    - **Robustness**: Added validity masking to handle profiles that extend outside image boundaries.
- **Template Optimization**:
    - Implemented multiple strategies for spatially varying template parameters (`sigma`, `amplitude`, `peak_dist`):
        - `GlobalOptimizableTemplateModel`: Single learnable set of parameters.
        - `PerPointTemplateModel`: Independent parameters per vertex (with smoothness regularization).
        - `BSplineTemplateModel`: Parameters defined by a B-Spline curve along the contour index.
        - `NeuralFieldTemplateModel`: Parameters predicted by an MLP based on spatial coordinates $(x, y)$.
        - **Update**: `BSplineTemplateModel` now uses vectorized calculations for efficiency.
        - `GridTemplateModel`: Parameters defined by a learnable 2D grid (bilinear interpolation).
        - `GaussianSplatTemplateModel`: Parameters defined by a set of Gaussian RBFs (splats) in the image domain.
- **Data Handling**:
    - Added `trim_data` to `generate_2d_data.py` to crop images around the segmentation.
- **Refactoring**:
    - Implemented `TemplateModelFactory` to clean up model instantiation.

## Validation Status
- **Synthetic Data**: Basic functionality works, but recent changes (template models, masking) have not been rigorously verified.
- **Real Data**: The new template models and B-Spline refiner have **not yet been tested** on real data.
- **Tests**: Fixed 6 failing tests. The root causes were an incorrect gradient assertion in `test_loss`, a path type mismatch in `test_trainer`, a missing attribute access guard in `PerPointTemplateModel`, and an unstable learning rate for the `NeuralField` optimization test. All tests now pass.

## Design Decisions
- **No Explicit Remeshing**: We avoid remeshing during the optimization loop to maintain differentiability. Instead, we rely on `EdgeLengthConsistencyLoss` and `LaplacianSmoothingLoss` to maintain mesh quality.
- **B-Spline Regularization**: For the B-spline refiner, we apply Laplacian and Edge Length regularization to the **control points** to ensure a uniform parameterization and prevent control point bunching, even though the spline curve itself is inherently smooth.
- **Implicit Regularization**: We prefer B-Spline or Neural Field parameterizations for template parameters to enforce smoothness implicitly, rather than relying solely on explicit smoothness losses.
- **Template Model Philosophy**: We favor explicit 1D models like `BSplineTemplateModel` for parameterizing template variations. This provides a stronger and more appropriate inductive bias for properties varying along a 1D manifold (the contour) compared to more general but less efficient ambient 2D field models (`NeuralField`, `Grid`).

## In Progress / Next Steps
1.  **2D Optimization Validation**: Run `train_2d.ipynb` and `compare_combinations.ipynb` notebooks to verify the full training loop and systematically test various refiner and template combinations on synthetic data.
2.  **Real Data Validation**: Run `BSplineContourRefiner` with `BSplineTemplateModel` on the real data slice (`data/20289/...`).
3.  **Visualization**: Use the new visualization tools to inspect the learned template parameters on real data.
4.  **Port to 3D**: (Paused until 2D is fully robust).

## Limitations
- **Self-Supervised Learning**: The eventual goal of using this as a self-supervised learning signal for a neural network has not been implemented.

## Next Steps (Resume Plan)
1.  **Update Neural Model**: Modify neural model to use the updated `sample_profiles` function from `diffmeshopt/opt2d/sampling.py` (or replicate the rectangular sampling logic) to ensure the NN sees the same features as the direct optimization.
2.  **Port to 3D**:
    *   Review `plan_3d.md`.
    *   Implement 3D mesh loading/saving.
    *   Implement 3D vertex normal computation.
    *   Implement 3D `grid_sample` logic (sampling prisms/cylinders along normals).

## How to Run
Use `notebooks/optimize_2d.ipynb`.
```