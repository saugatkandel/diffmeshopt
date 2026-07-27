# 2D Prototype Testing Plan

This document outlines the strategy for implementing and testing the differentiable mesh optimization framework in a simplified 2D setting.


## 1. Objectives
1.  **Validate Loss Landscape**: Ensure profile-based losses provide correct gradients to center contours on membrane bilayers.
2.  **Debug Differentiable Sampling**: Verify non-integer coordinate sampling + backprop to vertices.
3.  **Test Regularization Balance**: Data term vs geometric constraints vs template-shape priors.
4.  **Compare Data Terms**: Cross-correlation vs Wasserstein under identical perturbation conditions/seeds.
5.  **Compare Template Models**: Fixed vs B-Spline vs Neural Field parameterization.


## 2. Synthetic Data Generation (`diffmeshopt/opt2d/generate_2d_data.py`) (Done)
*   **Signal**: Bi-Gaussian membrane profile.
*   **Noise/Blur**: Additive Gaussian noise and PSF convolution.
*   **Trimming**: Logic to crop images to the region of interest to test boundary handling.

## 3. Architecture (`diffmeshopt/opt2d/optimize.py` & `diffmeshopt/opt2d/template.py`)

### A. Inputs
*   **Image**: $I \in \mathbb{R}^{H \times W}$
*   **Contour**: $V \in \mathbb{R}^{N \times 2}$ (List of $(x, y)$ coordinates).

### B. Components
1.  **Contour Refiner**:
    *   **Standard**: Optimizes vertices $V$ directly.
    *   **B-Spline**: Optimizes control points $P$, where $V = M \cdot P$. Uses analytical normals.
    *   **Tangential Smoothing**: Optimizes vertices with tangential smoothing and normal consistency to prevent shrinking.
    *   **RBF**: Optimizes sparse control points driving a global deformation field (Mesh-free, 3D-ready).
2.  **Differentiable Sampler**:
    *   Computes normals (finite difference or analytical).
    *   **Stochastic Sampling**: Evaluates data loss on a random subset of vertices per step for efficiency. Uses full-resolution normals to maintain geometric accuracy.
    *   Samples profiles using `grid_sample`.
    *   **Masking**: Ignores profiles that cross image boundaries.
    *   **Profile Width**: Rectangular averaging across the tangent direction to increase SNR before template matching.
        *   Default changed from `1` px → `5` px.
        *   No prior documentation justified the original `1` px default; it is assumed to have been an untuned placeholder from early synthetic-data bring-up, where noise levels were low enough that tangential averaging wasn't necessary.
        *   `5` px empirically improves stability on lower-SNR real slices, at the cost of some blurring on high-curvature contour segments.
        *   Open item: consider adaptive width (narrower in high-curvature regions) if blurring becomes problematic.

3.  **Template Model**:
    *   Predicts template parameters $\theta_i = (\sigma_i, d_i, A_i)$ for each point.
    *   **Models**: Fixed, Global, Per-Point, B-Spline (1D along contour), Neural Field (MLP on $x,y$), Grid (2D learnable map), Gaussian Splat (RBFs).
        *   **1D Contour Models (e.g., B-Spline, Per-Point)**: Model parameters as a function of the contour's arc-length. This provides a strong inductive bias for properties that vary smoothly *along the membrane*.
        *   **2D Ambient Field Models (e.g., Neural Field, Grid, Splat)**: Model parameters as a field in the image's coordinate space. The contour samples this field at its vertex locations. More general but less parameter-efficient and lacks an explicit contour-connectivity bias.
    *   **Factory**: `TemplateModelFactory` handles instantiation to decouple `optimize.py` from specific model classes.

## 4. Loss Functions (`diffmeshopt/opt2d/loss.py`)

### A. Data Term
* **Selectable Loss Type (`DataLossType`)**:
  * `BIGAUSSIAN_CORRELATION` (baseline)
  * `BIGAUSSIAN_WASSERSTEIN` (new experimental path)
* **Template Matching**:
  * Correlation: maximize profile-template alignment (implemented as minimizing `1-corr`).
  * Wasserstein: optimize transport-based mismatch to improve robustness to local shifts/shape mismatch.
* **Shape Coupling**:
  * Shape loss remains part of data term stack; tune carefully against main data loss.
  * Prefer scheduled/annealed shape weighting in future runs.


### B. Geometric Regularization
* Existing terms unchanged (Laplacian, edge, tangential, normal, anchor, RBF decay).
* Config rename:
  * `initial_loss_weights` → `initial_regularization_weights`.

*   **Laplacian Smoothness**: Penalize the distance of a vertex from the centroid of its neighbors. $L_{lap} = \sum ||v_i - \frac{1}{2}(v_{i-1} + v_{i+1})||^2$.
*   **Edge Length Consistency**: Penalize variance in edge lengths to prevent vertex bunching.
*   **Tangential Smoothing (Spacing)**: Penalizes only the tangential component of the Laplacian to distribute vertices evenly without shrinking.
*   **Normal Consistency (Fairing)**: Penalizes the angle between adjacent normals to ensure smoothness.
*   **Contour Anchor**: Penalizes deviation from initialization ($L2$) to prevent drift in ambiguous regions.
*   **RBF Weight Decay**: Penalizes the magnitude of RBF weights to minimize deformation energy.
*   **Weighting Strategy**: Defaults derived via **Force Balance Heuristic** ($\lambda \approx 1/(2 D \sigma)$).
*   **Template Regularization**: Smoothness priors for Per-Point template models.

## 5. Evaluation & Training
*   **Metrics** (`diffmeshopt/opt2d/evaluation.py`):
    *   Mean Distance to Ground Truth.
    *   **Confidence Map**: Visualizing the per-point Cross-Correlation score to identify broken membranes.
    *   Hausdorff Distance.
*   **Trainer** (`diffmeshopt/opt2d/trainer.py`):
    *   `OptimizationTrainer` class to manage the loop.
    *   `TensorBoard` logging (local alternative).
    *   Checkpointing via `joblib`.
*   **Storage**:
    *   `ContourRefiner.export_state()` to save contour and parameters.
*   **Combinatorial Testing**: Use `notebooks/compare_combinations.ipynb` to systematically evaluate different refiner and template model combinations.

## 6. Execution Plan

### Step 1: Setup (Done)
*   Implemented data generation, loss functions, refiners, and template models.

### Step 2: Optimization Loop (Current Focus)
* Run matched sweeps on perturbed real-slice setup:
  * Refiner: BSpline vs RBF
  * Data loss: Correlation vs Wasserstein
  * Template: Symmetric BSpline baseline; optional Neural only as stress test
* Track:
  * Chamfer/mean distance from initialization and GT-adjusted contour
  * Template parameter drift (`peak_dist`, `sigma1/2`)
  * Failure regions (e.g., local corner underfit)
* Practical defaults (current direction):
  * Moderate LR (avoid very high LR for joint optimization)
  * `profile_width=5`
  * Explicit logging of data-vs-shape magnitude ratio

### Step 3: Real Data Evaluation (Ongoing)
* Continue using trimmed-slice metadata (`row_start`, `col_start`) for correct GT frame mapping.
* Validate MPS-specific sampling behavior on Apple Silicon runs.
*   Visual check: Does the contour snap to the membrane? Do the learned parameters (width) make physical sense?

## 6. Next Steps
*   Once 2D is stable and tested on real data, port the logic to 3D.