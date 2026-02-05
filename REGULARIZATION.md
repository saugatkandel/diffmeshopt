# Regularization Strategy

This document outlines the regularization techniques used in `diffmeshopt` to ensure mesh quality and prevent degenerate solutions during optimization.

## 1. Geometric Regularization
These losses act on the contour vertices (or control points) to enforce smoothness and uniform sampling.

### A. Laplacian Smoothing (`CONTOUR_LAPLACIAN`)
*   **Formulation**: $L = \sum ||v_i - \frac{1}{N}\sum_{j \in N(i)} v_j||^2$
*   **Effect**: Moves vertices toward the centroid of their neighbors.
*   **Side Effect**: Causes shrinkage (contours collapse to a point) and smoothing.
*   **Usage**: Used cautiously. Often replaced by Tangential Smoothing to avoid shrinkage.

### B. Tangential Smoothing (`TANGENTIAL_LAPLACIAN`)
*   **Formulation**: Projects the Laplacian vector onto the tangent plane (or line in 2D).
    *   $L_{tan} = L - (L \cdot n)n$
*   **Effect**: Redistributes vertices along the contour to ensure uniform spacing *without* altering the shape (no shrinkage).
*   **Usage**: Primary regularizer for vertex-based refinement (`ContourRefiner`).

### C. Normal Consistency / Fairing (`NORMAL_CONSISTENCY`)
*   **Formulation**: Penalizes the angle between adjacent normals.
    *   $L = \sum (1 - n_i \cdot n_{i+1})$
*   **Effect**: Enforces $C^1$ continuity (smooth tangents/normals). Resists high-frequency noise.
*   **Usage**: Critical for preventing jagged edges, especially when Laplacian smoothing is disabled.

### D. Edge Length Consistency (`EDGE_LENGTH`)
*   **Formulation**: Penalizes the variance of edge lengths.
*   **Effect**: Encourages uniform edge lengths.
*   **Usage**: **Redundant** when Tangential Smoothing is active. The Tangential Laplacian force naturally distributes vertices uniformly along the contour (like a spring network), rendering explicit edge length penalties unnecessary.

## 2. Template Parameter Regularization
These losses act on the learnable parameters of the intensity template (e.g., $\sigma$, peak distance, amplitude).

### A. Anchoring (`ANCHOR_*`)
*   **Goal**: Prevent parameters from drifting too far from their initialization or physically plausible values.
*   **Formulation**: $L = ||\theta - \theta_{init}||^2$
*   **Usage**: Essential for implicit models (Neural Fields, Grids) to resolve ambiguity.

### B. Parameter Smoothness (`SMOOTH_*`)
*   **Goal**: Ensure template parameters vary smoothly along the contour.
*   **Formulation**: Laplacian smoothing applied to the parameter field.
*   **Usage**: Used for `PerPointTemplateModel` and `GridTemplateModel`.

## 3. Refinement Strategies

### Vertex-Based (`ContourRefiner`)
*   **Challenge**: Prone to high-frequency noise and shrinkage.
*   **Strategy**: **Tangential Smoothing**.
    *   `CONTOUR_LAPLACIAN`: 0.0 (Disable shrinking)
    *   `TANGENTIAL_LAPLACIAN`: High (e.g., 5.0) for spacing.
    *   `NORMAL_CONSISTENCY`: Moderate (e.g., 2.0) for smoothness.

### B-Spline (`BSplineContourRefiner`)
*   **Challenge**: Control points can bunch up; curve is inherently smooth but can loop.
*   **Strategy**: Regularize **Control Points**.
    *   `TANGENTIAL_LAPLACIAN`: Applied to control points to keep them uniform.
    *   `NORMAL_CONSISTENCY`: Low (curve is already $C^2$).

### RBF (`RBFContourRefiner`)
*   **Challenge**: Deformation field can become singular.
*   **Strategy**: Regularize the **Deformed Contour**.
    *   Standard geometric losses applied to the output vertices.

## 4. Adaptive Regularization
*   **Mechanism**: Dynamically adjusts regularization weights during optimization to maintain a target ratio between the Data Loss and Regularization Loss.
*   **Goal**: Prevents regularization from dominating early (preventing fitting) or vanishing late (allowing noise).
*   **Config**: `AdaptiveRegularizationProps` in `props.py`.