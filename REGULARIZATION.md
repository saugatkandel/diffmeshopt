# Regularization Strategy

This document outlines the regularization techniques used in `diffmeshopt` to ensure mesh quality and prevent degenerate solutions during optimization.

## 1. Weight Derivation (Force Balance Heuristic)
To avoid arbitrary hyperparameter tuning, we derive default regularization weights ($\lambda$) using a **Force Balance Approximation**. We treat the optimization as a physical system where the data force pulls vertices towards image features, and the regularization force resists deformation.

*   **Data Force ($F_{data}$)**: The gradient of the correlation loss is roughly proportional to the inverse of the template width ($1/\sigma_{template}$).
*   **Elastic Force ($F_{reg}$)**: The gradient of an L2 penalty ($\lambda x^2$) is $2 \lambda x$, where $x$ is displacement.
*   **Equilibrium**: We want forces to balance at a maximum reasonable displacement $D$ (e.g., 5 pixels).
    $$ F_{data} \approx F_{reg} \implies \frac{1}{\sigma_{template}} \approx 2 \lambda D $$
*   **Result**:
    $$ \lambda \approx \frac{1}{2 \cdot D \cdot \sigma_{template}} $$

**Note on Data Term**:
The Data Term ($F_{data}$) is composed of:
1.  **Correlation Loss**: Matches the sampled intensity profile to the template.
2.  **Shape Loss**: Penalizes deviation of the template shape from the mean sampled profile. This is treated as a data constraint, not a regularizer.

**Example**: For a target displacement limit $D=5$px and template width $\sigma=1.0$, the weight should be $\lambda \approx 0.1$.

## 2. Geometric Regularization
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

### E. Contour Anchor (`CONTOUR_ANCHOR`)
*   **Formulation**: $L = ||v - v_{init}||^2$
*   **Effect**: Penalizes deviation from the initial contour.
*   **Usage**:
    *   **Vertex Refiner**: Acts as a "soft tether" or trust region. Useful when initialization is reliable and we want to prevent drift in ambiguous image regions. However, high weights can prevent fitting.
    *   **B-Spline Refiner**: Critical for regularizing control points. Prevents them from drifting along the curve (tangential drift) or collapsing, ensuring the parameterization remains well-behaved.

### F. RBF Weight Decay (`RBF_WEIGHT_DECAY`)
*   **Formulation**: $L = \sum w_i^2$ (where $w_i$ are RBF weights).
*   **Effect**: Minimizes the deformation energy of the RBF field.
*   **Usage**: Primary regularizer for `RBFContourRefiner`.

## 3. Template Parameter Regularization
These losses act on the learnable parameters of the intensity template (e.g., $\sigma$, peak distance, amplitude).

### A. Anchoring (`ANCHOR_*`)
*   **Goal**: Prevent parameters from drifting too far from their initialization or physically plausible values.
*   **Formulation**: $L = ||\theta - \theta_{init}||^2$
*   **Usage**: Essential for implicit models (Neural Fields, Grids) to resolve ambiguity.

### B. Parameter Smoothness (`SMOOTH_*`)
*   **Goal**: Ensure template parameters vary smoothly along the contour.
*   **Formulation**: Laplacian smoothing applied to the parameter field.
*   **Usage**: Used for `PerPointTemplateModel` and `GridTemplateModel`.

## 4. Refinement Strategies (Recipes)
We define high-level `RegularizationStrategy` enums that map to specific weight configurations ("recipes") for each refiner type.

### Vertex-Based (`ContourRefiner`)
*   **Strategy**: `TANGENTIAL_SMOOTHING`
*   **Weights**:
    *   `TANGENTIAL_LAPLACIAN`: **5.0** (Strong spacing constraint).
    *   `NORMAL_CONSISTENCY`: **2.0** (Moderate fairing).
    *   `CONTOUR_LAPLACIAN`: **0.0** (Disable shrinking).
    *   `CONTOUR_ANCHOR`: **0.1** (Safety).

### B-Spline (`BSplineContourRefiner`)
*   **Strategy**: `TANGENTIAL_SMOOTHING`
*   **Weights**:
    *   `TANGENTIAL_LAPLACIAN`: **5.0** (Applied to control points to ensure uniform parameterization).
    *   `NORMAL_CONSISTENCY`: **0.0** (Disabled: B-splines are inherently $C^2$ smooth).
    *   `CONTOUR_ANCHOR`: **0.1** (Safety).

### RBF (`RBFContourRefiner`)
*   **Strategy**: `TANGENTIAL_SMOOTHING` (effectively Weight Decay)
*   **Weights**:
    *   `RBF_WEIGHT_DECAY`: **0.1** (Physics-based default).
    *   `TANGENTIAL_LAPLACIAN`: **0.0** (Not needed, centers are fixed).
    *   `NORMAL_CONSISTENCY`: **0.0** (Not needed, field is smooth).

## 5. Adaptive Regularization
*   **Mechanism**: Dynamically adjusts regularization weights during optimization to maintain a target ratio between the Data Loss and Regularization Loss.
*   **Goal**: Prevents regularization from dominating early (preventing fitting) or vanishing late (allowing noise).
*   **Config**: `AdaptiveRegularizationProps` in `props.py`.