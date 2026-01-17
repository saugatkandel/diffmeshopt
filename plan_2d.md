# 2D Prototype Testing Plan

This document outlines the strategy for implementing and testing the differentiable mesh optimization framework in a simplified 2D setting.

## 1. Objectives
1.  **Validate Loss Landscape**: Ensure the bi-Gaussian cross-correlation loss provides correct gradients to "snap" vertices to the membrane center.
2.  **Debug Differentiable Sampling**: Verify that sampling image features at non-integer coordinates and backpropagating to vertex positions works correctly.
3.  **Test Regularization**: Determine the balance between the data term (membrane fit) and geometric regularization (smoothness, edge length).

## 2. Synthetic Data Generation (`src/generate_2d_data.py`)
We will generate synthetic 2D images that mimic cryo-ET slices.

*   **Signal**: A "membrane" modeled as a curve (circle or perturbed ellipse) with a cross-sectional intensity profile of two Gaussians (peaks) separated by a valley.
*   **Noise**: Additive Gaussian noise to simulate low SNR.
*   **Blur**: Convolve with a Point Spread Function (PSF) to simulate microscope optics.
*   **Initialization**: Generate a "rough segmentation" contour that is close to but not perfectly aligned with the membrane (e.g., eroded/dilated or noisy version of the ground truth).

## 3. Architecture (`src/model_2d.py`)
The 2D model will mirror the proposed 3D architecture but operate on 2D tensors.

### A. Inputs
*   **Image**: $I \in \mathbb{R}^{H \times W}$
*   **Contour**: $V \in \mathbb{R}^{N \times 2}$ (List of $(x, y)$ coordinates).

### B. Components
1.  **Feature Extractor (Optional for Phase 1)**:
    *   A small 2D CNN to extract texture features, or simply use raw image intensity.
2.  **Differentiable Sampler**:
    *   For each vertex $v_i$, compute normal $n_i$.
    *   Sample points along the normal: $p_{i,k} = v_i + k \cdot \delta \cdot n_i$, where $k \in [-K, K]$.
    *   Use `grid_sample` (bilinear interpolation) to get intensities at $p_{i,k}$.
3.  **Deformation Module**:
    *   **Input**: Sampled intensity profile vector of size $2K+1$.
    *   **Network**: A 1D Convolutional network or MLP that processes the profile and predicts a scalar shift $s_i$.
    *   **Update**: $v_i^{new} = v_i + s_i \cdot n_i$.

## 4. Loss Functions (`src/loss_2d.py`)

### A. Data Term (Self-Supervised)
*   **Template Matching**:
    *   Define a learnable or fixed 1D template $T$ (bi-Gaussian).
    *   Extract intensity profile $P_i$ at vertex $v_i$.
    *   Loss = $1 - \text{CrossCorrelation}(P_i, T)$.

### B. Geometric Regularization
*   **Laplacian Smoothness**: Penalize the distance of a vertex from the centroid of its neighbors. $L_{lap} = \sum ||v_i - \frac{1}{2}(v_{i-1} + v_{i+1})||^2$.
*   **Edge Length Consistency**: Penalize variance in edge lengths to prevent vertex bunching.

## 5. Execution Plan

### Step 1: Setup
*   Create `src/generate_2d_data.py` to produce `data/2d_synthetic.npz`.
*   Create `src/model_2d.py` defining the `ContourRefiner` class.

### Step 2: Optimization Loop (Script: `src/train_2d.py`)
1.  Load synthetic image and initial contour.
2.  **Forward**:
    *   Compute normals.
    *   Sample profiles.
    *   Predict shifts.
    *   Deform contour.
3.  **Loss**: Compute Data Loss + $\lambda$ * Reg Loss.
4.  **Backward**: Update network weights (or vertex positions directly if doing direct optimization).

### Step 3: Evaluation
*   Visual check: Does the contour align with the "bilayer" signal?
*   Quantitative: Compute Chamfer distance between refined contour and ground truth skeleton.

## 6. Next Steps
*   Once 2D is stable, port the `grid_sample` logic to 3D.
*   Replace 1D contour logic with 3D mesh half-edge data structure.