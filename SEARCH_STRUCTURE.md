# Hyperparameter Search Structure

This document explains the hierarchical hyperparameter search strategy implemented in `diffmeshopt/opt2d/hyperparameter_search.py`.

## 1. Overview
The search uses **Optuna** to find the optimal combination of:
1.  **Refiner Architecture**: How the contour is represented and deformed.
2.  **Template Model**: How the intensity profile is modeled.
3.  **Regularization Strategy**: How geometric and parameter constraints are applied.

The search is **hierarchical** and uses **curriculum learning** to efficiently filter out unstable configurations.

## 2. Search Space

### A. Refiner Types (`refiner`)
*   `vertex`: Direct optimization of mesh vertices (`ContourRefiner`).
*   `bspline`: Optimization of B-Spline control points (`BSplineContourRefiner`).
*   `rbf`: Optimization of Radial Basis Function control points (`RBFContourRefiner`).

### B. Template Models (`template`)
*   `global`: Single set of parameters for the whole contour.
*   `per_point`: Independent parameters per vertex (with smoothness regularization).
*   `bspline`: Parameters vary along the contour defined by a 1D B-Spline.
*   `neural`: Parameters predicted by a coordinate-based neural network (Neural Field).

### C. Regularization Strategy (`reg_mode`)
*   `static`: Fixed weights for all regularization terms throughout optimization.
*   `adaptive`: Weights are dynamically adjusted to maintain a target ratio between the regularization loss and the data loss.

### D. Hyperparameters
The search samples the following parameters:

| Parameter | Type | Range / Options | Description |
| :--- | :--- | :--- | :--- |
| `symmetric` | Categorical | `True`, `False` | Whether the template profile is symmetric. |
| `num_cp` | Integer | 20 - 150 | Number of control points (for B-Spline/RBF refiners). |
| `rbf_sigma` | Float | 0.5 - 10.0 | Kernel width for RBF refiner. |
| `w_{reg}` | Float (Log) | 1e-4 - 10.0 | Static weight for regularizer `{reg}`. |
| `smoothness_window_size` | Categorical | `1`, `3`, `5` | Window size for smoothness regularization. |
| `ratio_{reg}` | Float (Log) | 1e-4 - 1.0 | Target ratio for adaptive regularizer `{reg}`. |

**Regularizers (`{reg}`):**
*   `laplacian`: Laplacian smoothing (shrinkage).
*   `edge`: Edge length consistency.
*   `normal`: Normal consistency (fairing).
*   `tangential`: Tangential smoothing (vertex distribution).
*   `anchor`: Template parameter anchoring.
*   `smooth_param`: Template parameter smoothness.

### E. Fixed Parameters & Constraints
Some parameters are fixed to avoid degenerate solutions or physical implausibility:
*   **`sigma`**: Initialization fixed to 0.75. While most template models optimize `sigma` during refinement, the starting value is fixed in this search to represent a standard membrane thickness prior.
*   **`min_peak_ratio`**: Fixed to 2.0. This enforces the **Rayleigh Criterion** for resolution.
    *   *Reasoning*: If the two Gaussian peaks are closer than $\approx 2\sigma$, they merge into a single peak, making the orientation and separation unresolvable (degenerate). We constrain the search to physically resolvable membrane profiles.

## 3. Curriculum Learning
To efficiently find robust hyperparameter configurations, the search evaluates each trial on a curriculum of progressively harder tasks. Unstable or poorly performing configurations are "pruned" (stopped early) to save computational resources.

### A. Curriculum Stages
The curriculum consists of three stages, each using a different set of initial contours:

*   **Stage 0 (Easy): `original`**
    *   **Task**: The initial contour is the clean, unperturbed segmentation from the data.
    *   **Purpose**: This is a basic sanity check. If a set of hyperparameters cannot refine a good initial guess, it is fundamentally flawed and will not work on more challenging inputs.

*   **Stage 1 (Medium): `shrink_5_perturb_3` & `expand_5_perturb_3`**
    *   **Task**: The initial contour is first shrunk/expanded by 5 pixels and then a moderate amount of smooth noise is added.
    *   **Purpose**: This tests the model's basin of attraction. Can the optimizer correct for small-to-medium initialization errors? Configurations that are too stiff (over-regularized) or too loose (under-regularized) might fail here.

*   **Stage 2 (Hard): `shrink_10_perturb_5` & `expand_10_perturb_5`**
    *   **Task**: The initial contour is significantly shrunk/expanded by 10 pixels and has a larger amount of noise.
    *   **Purpose**: This is the ultimate robustness test. Only configurations with a wide and smooth convergence basin will succeed. This stage selects for hyperparameters that are resilient to very poor initializations.

### B. Pruning Mechanisms
The `objective` function implements two layers of pruning that work together with the curriculum.

#### Layer 1: Hard Failure Pruning
For *any single case* within a stage, if the optimization fails catastrophically, the trial is immediately stopped.

*   **Mechanism**: If the final Chamfer distance is worse than **5.0 pixels**, or if the optimization diverges (producing `NaN` or `inf`), the trial is pruned.
*   **Benefit**: This is extremely efficient. A hyperparameter set that causes the contour to explode or collapse on the easiest `original` case is discarded in minutes, without wasting hours on the harder stages.

#### Layer 2: Optuna's Median Pruner
After each *stage* is successfully completed, the average score for that stage is reported to Optuna, which can then decide to prune the trial based on its performance relative to other trials.

*   **Mechanism**: The search is configured with a `MedianPruner`. After Stage 0, for example, the pruner looks at the scores of all other trials that have also completed Stage 0. If the current trial's score is worse than the median, it's pruned.
*   **Benefit**: This prunes configurations that are not outright failures but are clearly underperforming. For instance, a trial might pass Stage 0 with a score of 2.5. If the median score for Stage 0 across all other trials is 0.8, this trial is likely not on a path to becoming the best one and can be safely stopped.

## 4. Objective Function & Pruning
The objective function calculates the **Chamfer Distance** between the optimized contour and the ground truth.

The final score for a successful trial is the sum of the average Chamfer Distances across all evaluated curriculum stages.

## 5. Output
*   `results.parquet`: A Polars DataFrame containing all trial parameters and scores.
*   `metadata.json`: Contains the best parameters, best value, and experiment metadata.
*   `study.db`: An SQLite database storing the full Optuna study state.

## 6. Recommended Number of Trials

The hyperparameter search space is complex, with approximately 14 effective parameters in the most complex configurations (e.g., RBF refiner with adaptive regularization). These include categorical choices, integer ranges, and continuous float values.

A common rule of thumb for Bayesian optimization samplers like `TPESampler` is to budget **10-20 trials per hyperparameter**.

*   **Minimum Viable Search**: `14 params * 10 trials/param = 140 trials`
*   **Recommended Search**: `14 params * 20 trials/param = 280 trials`
*   **Thorough Search**: `14 params * 35 trials/param ≈ 500 trials`

The default setting of **`n-trials=500`** is chosen for a thorough search. This is further justified by the curriculum learning and pruning mechanisms, which terminate unpromising trials early. This allows the search to explore a large number of configurations while efficiently allocating most of the computational budget to the most promising ones.