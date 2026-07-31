# Project Status: 2D Optimization Prototype

**Last Updated:** 2026-07-22

## Overview
We have implemented a 2D prototype for refining organelle segmentations using a bi-Gaussian intensity prior. The core logic involves sampling image intensity along vertex normals (using a rectangular average) and optimizing vertex positions to maximize alignment with a profile template.


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
*   `examples/`: Standalone analysis scripts (`analyze_contour_anchor.py`, `compare_refiners.py`, etc.).

## Current State
- **Core Modules Refactored**:
    - Geometric losses (`LaplacianSmoothingLoss`, `EdgeLengthConsistencyLoss`) have been implemented and moved to `diffmeshopt/opt2d/loss.py`.
    - The optimization loop has been encapsulated into a `ContourRefiner` class.
    - Stochastic data-term sampling is used while geometric losses are computed on full contour.
- **Data Loss Framework (New)**:
    - Added `DataLossType` in `opt2d/config.py`:
        - `BIGAUSSIAN_CORRELATION`
        - `BIGAUSSIAN_WASSERSTEIN`
    - Cross-correlation remains baseline; Wasserstein is now selectable for experiments.
- **Config/API Cleanup (New)**:
    - Renamed `initial_loss_weights` → `initial_regularization_weights`.
    - `ContourRefinerProps.profile_width` default updated from `1` to `5` for better SNR in profile averaging.
        - **Open Question**: The original default of `profile_width=1` (i.e., no tangential averaging) is not clearly justified in earlier design notes.
          Possible historical reasons:
          1. **Simplicity during initial bring-up**: A width of 1 avoids any ambiguity about how averaging interacts with normal direction, curvature, or boundary masking — useful when first validating differentiable sampling.
          2. **Avoiding tangential blur across curved regions**: Averaging perpendicular-to-normal strips assumes the membrane is locally straight; on high-curvature contour segments, a wide strip can blend intensities from different physical membrane locations, biasing the profile.
          3. **Untuned placeholder**: It's also plausible this was simply a default that was never revisited after synthetic-data testing (where noise levels were low enough that averaging wasn't necessary).
        - **Why `5` works better in practice**: Real cryo-ET slices have lower SNR than synthetic test data. Averaging across a small tangential window (5 px) reduces per-pixel noise in the sampled profile before cross-correlation/Wasserstein matching, at the cost of some localized blurring on curved regions. Empirically this trade-off has improved fit stability.
        - **Action Item**: If curvature-induced blurring becomes an issue, consider an adaptive width (e.g., shrinking width in high-curvature regions) rather than a single global constant.
- **Platform Robustness (New)**:
    - `evaluation.py` now uses MPS-safe grid sampling fallback:
        - `padding_mode="zeros"` + clamped grid on MPS.
        - `padding_mode="border"` elsewhere.
- **Data Handling Improvements (New)**:
    - `generate_2d_data.py` now stores:
        - `row_start`, `col_start`, `untrimmed_shape`
    - Enables correct coordinate reconciliation between trimmed and original frames during evaluation.

- **New Refinement Strategies**:
    - **Tangential Smoothing**: Implemented `TangentialSmoothingContourRefiner` and updated losses to support a "Tangential Laplacian" and "Normal Consistency" (Fairing). This decouples smoothing from shrinking, allowing vertices to slide along the surface without collapsing the volume.
    - **RBF Deformation**: Implemented `RBFContourRefiner`, which uses Radial Basis Functions to deform the mesh via sparse control points. This is inherently smooth and 3D-ready.
    - **Unified Regularization**: Implemented `RegularizationStrategy` enum and `regularizer_recipes.py` to centralize weight configurations. Weights are now derived from a physics-based "Force Balance" heuristic.
- **3D Readiness**:
    - **Documentation**: Created `plan_3d.md` outlining the porting strategy.
    - Loss functions (`LaplacianSmoothingLoss`, `EdgeLengthConsistencyLoss`, `NormalConsistencyLoss`) updated to support explicit edge connectivity (Graph Laplacian), enabling support for 3D meshes.
    - Template models (`NeuralField`, `Grid`, `Splat`) updated to handle 3D spatial dimensions.
    - `PerPointTemplateModel` updated to use `LaplacianSmoothingLoss` for regularization and support explicit topology.
- **Stochastic Sampling Implemented**:
    - A stratified random sampling strategy (`_get_stratified_indices`) has been implemented in `diffmeshopt/opt2d/sampling.py` to select mini-batches of vertices for the data loss calculation.
    - **Design Choice**: We currently pass full-resolution normals to the sampler rather than recomputing them on the subsampled contour. This avoids geometric inaccuracies (chord vs tangent mismatch) while still benefiting from the efficiency of stochastic evaluation.
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
    - **Intensity Inversion**: Updated `load_real_data` to invert image intensities for real cryo-ET data. Since membranes are dark in cryo-ET but the model expects bright peaks (Gaussian), we multiply the normalized image by -1.
- **Refactoring**:
    - Implemented `TemplateModelFactory` to clean up model instantiation.
    - Enhanced `compare_combinations.ipynb` with detailed parameter visualization (line plots, bar charts) and quantitative metrics tables.
    - **Regularizer Architecture**: Implemented dynamic weight registry (`RegularizerType` enum -> auto-registered buffers) to eliminate manual synchronization and ensure type safety.
    - **Documentation**: Consolidated regularization docs into `REGULARIZATION.md`.
    - **Shape Loss Refactoring**: Reclassified Template Shape Loss as part of the Data Term (summed with Correlation Loss) rather than a Regularizer. This ensures it contributes to the signal magnitude for adaptive regularization balancing.
    - **Analysis Suite**: Added comprehensive example scripts in `examples/` to validate individual components (force dropoff, anchor weights, B-spline parameters). These scripts are currently being refined to ensure experimental correctness (e.g., fixing initialization basins, tuning regularization recipes).

## Validation Status
- **Synthetic Data**: Basic functionality works, but recent changes (template models, masking) have not been rigorously verified.
- **New Features**: `TangentialSmoothingContourRefiner`, `RBFContourRefiner`, and `BSplineContourRefiner` have been verified via convergence tests (`tests/opt2d/test_convergence.py`).
- **Real Data**: The new template models, B-Spline refiner, and RBF refiner have **not yet been tested** on real data.
- **Tests**: All tests pass, including new convergence tests for all refiner types.

## Experimental Results & Observations
- **BSpline Refiner**:
    - Works but is fragile across seeds/settings on perturbed real slices.
    - Increasing control points alone (e.g., 60) did not consistently fix fit quality.
- **RBF Refiner (Correlation)**:
    - Often more robust than BSpline, but can overfit/noise-follow in flexible regimes.
- **RBF Refiner (Wasserstein)**:
    - Promising direction; behavior improved in some runs.
    - Remaining failure mode: local underfit region (notably lower-left area in current test slice).
- **Seam bump note**: After RBF-based refinement, I noticed a bump in the top left and assumed it might be caused by a mismatch between the start/end of the cyclic contour. I tried a cyclic/ghost-contour variant to address it, but that did not fix the issue. I did **not** explicitly visualize the seam to confirm that the bump aligned with the start/end point, so this remains unverified. I am setting this aside for now and focusing first on the alignment issue.
- **Template Models**:
    - Symmetric bi-Gaussian generally outperforms asymmetric in current setup.
    - Neural template model showed high variation/instability in current tests.
- **Optimization Sensitivity**:
    - Very high LR (e.g., `5e-1`) is unstable for joint contour+template updates.
    - Shape/data loss balance remains critical; fixed shape weight can dominate early if not scheduled.
- **BSpline Refiner Shrinking**: The `BSplineContourRefiner` exhibits a tendency to shrink the contour more than the vertex-based refiner.
    - *Update*: This is mitigated by using Tangential Smoothing weights (disabling Laplacian/Edge Length on control points).
- **Symmetric vs. Asymmetry**: Symmetric template models (`sigma1=sigma2`, `amp1=amp2`) consistently perform better than asymmetric ones.
    - *Speculation*: Asymmetric models introduce parameter ambiguity. The optimization can satisfy the loss by skewing the profile shape (e.g., making one side steeper) rather than moving the contour vertex to the true center. Symmetric models enforce a geometric centering constraint, providing a stronger gradient for positional alignment.

## Design Decisions
- **No Explicit Remeshing**: We avoid remeshing during the optimization loop to maintain differentiability. Instead, we rely on `EdgeLengthConsistencyLoss` and `LaplacianSmoothingLoss` to maintain mesh quality.
- **B-Spline Regularization**: For the B-spline refiner, we apply Laplacian and Edge Length regularization to the **control points** to ensure a uniform parameterization and prevent control point bunching, even though the spline curve itself is inherently smooth.
- **Implicit Regularization**: We prefer B-Spline or Neural Field parameterizations for template parameters to enforce smoothness implicitly, rather than relying solely on explicit smoothness losses.
- **Static Geometric Constraints**: Geometric regularizers (Tangential Laplacian, Normal Consistency) are now configured as **static constraints** (`target_ratio=0.0`) by default. They do not adapt during optimization, ensuring mesh quality is maintained even if the data term fluctuates.
- **Force Balance Heuristic**: Default regularization weights are derived analytically based on the template width and a target maximum displacement (e.g., 5 pixels), rather than arbitrary tuning.
- **Template Model Philosophy**: We favor explicit 1D models like `BSplineTemplateModel` for parameterizing template variations. This provides a stronger and more appropriate inductive bias for properties varying along a 1D manifold (the contour) compared to more general but less efficient ambient 2D field models (`NeuralField`, `Grid`).

## In Progress / Next Steps
1. **Stability pass for Wasserstein runs**:
   - Lower LR baseline sweep.
   - Add shape-loss warmup schedule.
   - Evaluate mild contour anchor in early iterations.
2. **Code hygiene**:
   - Remove temporary debug-only behavior before release commits.
   - Keep `debug.py` warnings, but avoid debug overrides in production training path.
3. **Real-data validation matrix**:
   - Compare correlation vs Wasserstein under matched seeds/hyperparameters.
4. **Port to 3D**:
   - Continue after 2D protocol is stable/reproducible.

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