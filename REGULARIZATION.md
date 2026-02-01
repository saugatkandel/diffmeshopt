# Regularization System

## Overview

The regularization system in `diffmeshopt` controls the geometric quality of the optimized contour and the plausibility of the learned template parameters. It is designed to be **type-safe**, **configurable**, and **adaptive**.

### Key Concepts

1.  **Single Source of Truth**: All regularizers are defined in the `RegularizerType` enum and configured in `RegularizerDefaults` (`props.py`).
2.  **Dynamic Architecture**: Weight buffers are automatically registered in `ContourLoss` based on the enum, eliminating manual synchronization code.
3.  **Adaptive Weights**: Weights can automatically adjust during optimization to maintain a target ratio between regularization and data loss.
4.  **Domain Separation**:
    *   **Contour Regularizers**: Operate on geometry (positions, normals).
    *   **Template Regularizers**: Operate on learned parameters (sigma, peak_dist).

---

## Configuration

### 1. Regularizer Types

Regularizers follow the naming convention `<DOMAIN>_<OPERATOR>`. Defined in `diffmeshopt/opt2d/props.py`.

| Enum | Name | Domain | Purpose |
|------|------|--------|---------|
| `TANGENTIAL_LAPLACIAN` | `tangential_laplacian` | Contour | Even point distribution (sliding) |
| `NORMAL_CONSISTENCY` | `normal_consistency` | Contour | Smooth curvature (fairing) |
| `CONTOUR_LAPLACIAN` | `contour_laplacian` | Contour | Position smoothness (can cause shrinking) |
| `EDGE_LENGTH` | `edge_length` | Contour | Uniform edge lengths |
| `TEMPLATE_PARAM_ANCHOR` | `template_param_anchor` | Template | Keep params near initialization |
| `TEMPLATE_PARAM_LAPLACIAN` | `template_param_laplacian` | Template | Spatial smoothness of params |
| `TEMPLATE_SHAPE` | `template_shape` | Template | Profile shape consistency |

### 2. Setting Weights

You can configure weights via `ContourRefinerProps`.

**Using Enum (Recommended):**
```python
from diffmeshopt.opt2d.props import ContourRefinerProps, RegularizerType

props = ContourRefinerProps(
    initial_loss_weights={
        RegularizerType.TANGENTIAL_LAPLACIAN.value: 5.0,
        RegularizerType.NORMAL_CONSISTENCY.value: 2.0,
    }
)
```

**Using Recipes:**
Pre-configured recipes are available in `diffmeshopt/opt2d/regularizer_recipes.py`.

```python
from diffmeshopt.opt2d.regularizer_recipes import TANGENTIAL_SMOOTHING_BSPLINE

props = ContourRefinerProps(
    initial_loss_weights=TANGENTIAL_SMOOTHING_BSPLINE.copy()
)
```

### 3. Fine-Grained Template Control

While `RegularizerType.TEMPLATE_PARAM_ANCHOR` controls the **global** weight of the anchor loss, you can control **which** parameters are anchored using `TemplateProps`.

```python
props = TemplateProps(
    # Relative weights (0.0 to 1.0)
    anchor_sigma=1.0,        # Anchor sigma to initialization
    anchor_peak_dist=0.0,    # Let peak_dist vary freely
    anchor_sigma_ratio=0.0,  # Let asymmetry vary freely
    
    # Smoothing weights (for PerPoint/BSpline models)
    smooth_sigma=1.0,
    smooth_peak_dist=1.0
)
```

The total anchor loss is a weighted sum of individual parameter losses, then multiplied by the global `TEMPLATE_PARAM_ANCHOR` weight.

---

## Adaptive Regularization

The system can automatically adjust regularization weights during optimization to prevent over-smoothing or noise.

### How It Works

At specific intervals, the system computes:
$$ w_{new} = \alpha \cdot w_{target} + (1 - \alpha) \cdot w_{current} $$
where:
$$ w_{target} = \frac{\text{target\_ratio} \times L_{data}}{L_{reg}} $$

This maintains the relationship $L_{reg} \approx \text{target\_ratio} \times L_{data}$.

### Usage

Enable it via `AdaptiveRegularizationProps`:

```python
from diffmeshopt.opt2d.props import AdaptiveRegularizationProps

props.adaptive_reg = AdaptiveRegularizationProps(
    enabled=True,
    update_interval=10,
    warmup_steps=5
)
```

Target ratios are defined in `RegularizerDefaults` in `props.py`.

### Target Ratio Interpretation

The `target_ratio` controls the relative influence of regularization:

- **0.01 (1%)**: Very weak regularization, mostly data-driven
- **0.1 (10%)**: Balanced, regularization provides gentle guidance
- **0.5 (50%)**: Strong regularization, conservative updates
- **1.0 (100%)**: Equal balance (often too strong)

**Rule of thumb**: Start with 0.1, decrease if over-smoothing, increase if noisy.

### When to Use Adaptive Weights

**Use heuristic adaptive** when:
- Different refiner types (vertex, B-spline, RBF)
- Different contour resolutions (50 points vs 500 points)
- You want reasonable defaults without tuning

**Use dynamic adaptive** when:
- Image quality varies significantly
- Multi-stage optimization (coarse → fine)
- You're unsure of the right weight balance
- Experimentation and research

**Use static weights** when:
- You've already tuned weights for your problem
- Maximum reproducibility needed
- Performance-critical (avoid overhead)

### Limitations & Future Work

#### Current Limitations

1. **Assumes `L ∝ ||∇L||`**: May break for non-smooth losses
2. **EMA only**: No advanced control (PID, adaptive alpha)
3. **No freeze mechanism**: Weights continue adapting indefinitely
4. **Per-loss ratios are manual**: Template models don't declare metadata

#### Potential Improvements

1. **Gradient-based option**: For pathological cases
2. **Freeze after convergence**: Stop adapting when stable
3. **Template metadata**: Models declare their preferred ratios
4. **Hierarchical adaptation**: Adapt groups, not individual losses
5. **Diagnostic tools**: Auto-detect when assumption breaks

---

## Strategies

### Tangential Smoothing (Prevent Shrinking)

Standard Laplacian smoothing pulls vertices toward the center of curvature, causing contours to shrink. **Tangential Smoothing** prevents this by projecting the smoothing force onto the tangent vector.

**Configuration:**
*   `CONTOUR_LAPLACIAN`: **0.0** (Disable shrinking force)
*   `TANGENTIAL_LAPLACIAN`: **> 0.0** (Distribute vertices evenly)
*   `NORMAL_CONSISTENCY`: **> 0.0** (Smooth orientation/curvature)

**Factory Helper:**
Use `create_tangential_smoothing_refiner` to automatically configure any refiner (Vertex, B-Spline, RBF) for this strategy.

```python
from diffmeshopt.opt2d.optimize import create_tangential_smoothing_refiner, BSplineContourRefiner

refiner = create_tangential_smoothing_refiner(
    BSplineContourRefiner,
    initial_contour,
    props,
    template_model
)
```

---

## Developer Guide

### Architecture: Dynamic Weight Registry

To avoid manual synchronization errors, `ContourLoss` dynamically creates weight buffers based on the `RegularizerType` enum.

```
RegularizerType Enum (props.py)
    ↓ (Single Source of Truth)
ContourLoss.__init__ (loss.py)
    ↓ (Iterates enum)
self.register_buffer("w_<name>", ...)
```

### Adding a New Regularizer

1.  **Add to Enum** (`diffmeshopt/opt2d/props.py`):
    ```python
    class RegularizerType(Enum):
        ...
        NEW_REGULARIZER = "new_regularizer"
    ```

2.  **Add Defaults** (`diffmeshopt/opt2d/props.py`):
    ```python
    RegularizerType.NEW_REGULARIZER: RegularizerConfig(
        static_weight=1.0, target_ratio=0.1
    )
    ```

3.  **Implement Logic** (`diffmeshopt/opt2d/loss.py`):
    In `ContourLoss.forward()`:
    ```python
    # Compute
    loss_val = self.new_loss_fn(...)
    
    # Store raw (for adaptive)
    self._raw_losses[RegularizerType.NEW_REGULARIZER.value] = loss_val
    
    # Add weighted to total (using dynamic getter)
    total_loss += self.get_weight(RegularizerType.NEW_REGULARIZER) * loss_val
    ```