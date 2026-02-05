# 2D Prototype Testing Plan

This document outlines the strategy for implementing and testing the differentiable mesh optimization framework in a simplified 2D setting.

## 1. Objectives
1.  **Validate Loss Landscape**: Ensure the bi-Gaussian cross-correlation loss provides correct gradients to "snap" vertices to the membrane center.
2.  **Debug Differentiable Sampling**: Verify that sampling image features at non-integer coordinates and backpropagating to vertex positions works correctly.
3.  **Test Regularization**: Determine the balance between the data term (membrane fit) and geometric regularization (smoothness, edge length).
4.  **Compare Template Models**: Evaluate the robustness of Fixed vs. B-Spline vs. Neural Field template parameterization. (Ongoing in `compare_combinations.ipynb`)
    *   Evaluate the trade-offs between explicit 1D contour models (B-Spline) and ambient 2D field models (Neural Field, etc.).

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
3.  **Template Model**:
    *   Predicts template parameters $\theta_i = (\sigma_i, d_i, A_i)$ for each point.
    *   **Models**: Fixed, Global, Per-Point, B-Spline (1D along contour), Neural Field (MLP on $x,y$), Grid (2D learnable map), Gaussian Splat (RBFs).
        *   **1D Contour Models (e.g., B-Spline, Per-Point)**: Model parameters as a function of the contour's arc-length. This provides a strong inductive bias for properties that vary smoothly *along the membrane*.
        *   **2D Ambient Field Models (e.g., Neural Field, Grid, Splat)**: Model parameters as a field in the image's coordinate space. The contour samples this field at its vertex locations. More general but less parameter-efficient and lacks an explicit contour-connectivity bias.
    *   **Factory**: `TemplateModelFactory` handles instantiation to decouple `optimize.py` from specific model classes.

## 4. Loss Functions (`diffmeshopt/opt2d/loss.py`)

### A. Data Term
*   **Template Matching**:
    *   Loss = $1 - \text{CrossCorrelation}(P_i, T(\theta_i))$.
    *   Includes a **Shape Loss** (L1 preferred) to ground the template to the mean data profile.

### B. Geometric Regularization
*   **Laplacian Smoothness**: Penalize the distance of a vertex from the centroid of its neighbors. $L_{lap} = \sum ||v_i - \frac{1}{2}(v_{i-1} + v_{i+1})||^2$.
*   **Edge Length Consistency**: Penalize variance in edge lengths to prevent vertex bunching.
*   **Tangential Smoothing (Spacing)**: Penalizes only the tangential component of the Laplacian to distribute vertices evenly without shrinking.
*   **Normal Consistency (Fairing)**: Penalizes the angle between adjacent normals to ensure smoothness.
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
*   Run `ContourRefiner` on synthetic data.
*   **Validation**:
    *   **New Implementations (Untested)**:
        *   Verify `TangentialSmoothingContourRefiner` prevents shrinking compared to standard Laplacian.
        *   Verify `RBFContourRefiner` moves vertices coherently.
        *   Verify `BSplineContourRefiner` produces smooth curves without explicit Laplacian loss on vertices.
    *   **Status**: All refiners verified in `tests/opt2d/test_convergence.py`.
    *   **Template Verification**: `NeuralFieldTemplateModel` and others tested in `test_template.py`.
    *   **Observations**:
        *   `BSplineContourRefiner` tends to shrink contours due to regularization on control points.
        *   Symmetric templates outperform asymmetric ones due to better centering constraints.

### Step 3: Real Data Evaluation (Pending)
*   Load `data/20289/denoised/data_slice123.pkl`.
    *   **Note**: Ensure intensity inversion is applied (membranes are dark in cryo-ET, model expects bright peaks).
*   Run optimization with `BSplineContourRefiner` + `BSplineTemplateModel`.
*   Visual check: Does the contour snap to the membrane? Do the learned parameters (width) make physical sense?

## 6. Next Steps
*   Once 2D is stable and tested on real data, port the logic to 3D.