# Experiment Log

Track each run from notebooks/scripts in a consistent format.

---

## Run ID: `<YYYYMMDD-HHMM>_<short_name>`
- **Date:** `<YYYY-MM-DD>`
- **Author:** `<name>`
- **Branch / Commit:** `<branch> @ <short_sha>`
- **Notebook / Script:** `<path>`
- **Device:** `<cpu|cuda|mps>`
- **Seed(s):** `<e.g., 1234, 145>`
- **Status:** `<planned|running|complete|failed>`

### 1) Goal
- `<what this run is testing>`

### 2) Data
- **Dataset path:** `<path>`
- **Slice / sample key:** `<e.g., shrink_5_perturb_3>`
- **Preprocessing:** `<trim/invert/smoothing details>`
- **Frame mapping metadata:**  
  - `row_start`: `<int>`  
  - `col_start`: `<int>`  
  - `untrimmed_shape`: `<H, W>`

### 3) Configuration
- **Refiner:** `<bspline|rbf>`
- **Template:** `<fixed|global|per_point|bspline|neural>`
- **Data loss type:** `<bigaussian_correlation|bigaussian_wasserstein>`
- **Key hyperparameters:**
  - `learning_rate`: `<float>`
  - `num_cp`: `<int|null>`
  - `rbf_sigma`: `<float>`
  - `profile_length`: `<int>`
  - `profile_width`: `<int>`
  - `shape_loss_weight`: `<float>`
  - `min_peak_ratio`: `<float>`
- **Regularization weights:**
  - `w_laplacian`: `<float>`
  - `w_edge`: `<float>`
  - `w_normal`: `<float>`
  - `w_tangential`: `<float>`
  - `w_anchor`: `<float>`
  - `w_contour_anchor`: `<float>`
  - `w_smooth_param`: `<float>`
- **Overrides dict name:** `initial_regularization_weights`

### 4) Runtime Notes
- **Total steps:** `<int>`
- **Log interval / save interval:** `<int>/<int>`
- **Any warnings/debug flags:** `<details>`
- **MPS fallback used:** `<yes/no>`

### 5) Quantitative Results
- **Initial metrics (GT vs init contour):**
  - `mean_dist`: `<float>`
  - `hausdorff`: `<float>`
- **Final metrics (GT vs refined contour):**
  - `mean_dist`: `<float>`
  - `hausdorff`: `<float>`
- **From-init change:**
  - `chamfer_from_init`: `<float>`
- **Loss snapshot (final):**
  - `data_loss`: `<float>`
  - `shape_loss`: `<float>`
  - `regularization_total`: `<float>`
  - `shape/data ratio`: `<float>`

### 6) Template Parameter Diagnostics
- `peak_dist mean ± std`: `<float ± float>`
- `sigma1 mean ± std`: `<float ± float>`
- `sigma2 mean ± std`: `<float ± float>`
- Drift vs original defaults:
  - `peak_dist`: `<delta>`
  - `sigma`: `<delta>`

### 7) Visual Checks
- **Image + contour overlay:** `<good|ok|bad>`
- **Normals/profile alignment:** `<good|ok|bad>`
- **RBF field sanity (if RBF):** `<good|ok|bad>`
- **Failure regions:** `<e.g., lower-left corner underfit>`

### 8) Interpretation
- `<1-3 bullets on what worked/failed and why>`

### 9) Next Action
- `<single concrete next experiment>`
- `<parameter changes planned>`

### 10) Artifacts
- **Output dir:** `<path>`
- **Key figures:** `<paths>`
- **Checkpoint(s):** `<paths>`
- **TensorBoard/log files:** `<paths>`

---

## Quick Copy Template

```text
Run ID:
Date:
Branch / Commit:
Notebook / Script:
Device:
Seed(s):
Goal:

Config:
- refiner:
- template:
- data_loss_type:
- learning_rate:
- num_cp:
- rbf_sigma:
- profile_length/profile_width:
- shape_loss_weight:
- regs: lap/edge/normal/tangential/anchor/contour_anchor/smooth_param

Results:
- initial mean_dist/hausdorff:
- final mean_dist/hausdorff:
- chamfer_from_init:
- final data_loss/shape_loss/shape:data ratio:
- template drift (peak_dist, sigma1, sigma2):

Visual outcome:
Failure regions:

Interpretation:
Next action:
Artifacts:
```