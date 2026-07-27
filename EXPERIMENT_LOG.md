# Experiment Log

Generated/updated entries should follow the same structure as `ExperimentLogEntry.to_markdown()`.

---

<!-- run_id: <run_id> -->
## Run Summary

| Field | Value |
| --- | --- |
| Run ID | `<run_id>` |
| Author | `<name>` |
| Branch / Commit | `<branch> @ <short_sha>` |
| Notebook / Script | `<path>` |
| Device | `<cpu|cuda|mps>` |
| Seed(s) | `<e.g., 145>` |
| Status | `<planned|running|complete|failed>` |

### 1) Goal
- `<what this run is testing>`

### 2) Data

| Field | Value | Field | Value |
| --- | --- | --- | --- |
| Dataset path | `<path>` | Slice / sample key | `<e.g., shrink_5_perturb_3>` |
| Preprocessing | `<trim/invert/smoothing details>` |  |  |

| Field | Value | Field | Value | Field | Value |
| --- | --- | --- | --- | --- | --- |
| row_start | `<int>` | col_start | `<int>` | untrimmed_shape | `<H, W>` |

### 3) Configuration

| Field | Value | Field | Value |
| --- | --- | --- | --- |
| Refiner | `<from RefinerType enum value>` | Template | `<from TemplateType enum value>` |
| Data loss type | `<from DataLossType enum value>` | Learning rate | `<float>` |
| Num cp | `<int|null>` | RBF sigma | `<float>` |
| Profile length | `<int>` | Profile width | `<int>` |
| Shape loss weight | `<float>` | Min peak ratio | `<float>` |

**Regularization weights**

| Field | Value | Field | Value | Field | Value |
| --- | --- | --- | --- | --- | --- |
| contour_laplacian | `<float>` | edge_length | `<float>` | normal_consistency | `<float>` |
| tangential_laplacian | `<float>` | contour_anchor | `<float>` | rbf_weight_decay | `<float>` |

### 4) Runtime Notes

| Field | Value | Field | Value |
| --- | --- | --- | --- |
| global_step | `<int>` |  |  |

### 5) Quantitative Results

| Field | Value | Field | Value | Field | Value |
| --- | --- | --- | --- | --- | --- |
| initial_mean_dist | `<float>` | initial_hausdorff | `<float>` | mean_dist | `<float>` |
| hausdorff_dist | `<float>` | p95_dist | `<float>` | total_loss | `<float>` |
| data_loss | `<float>` | chamfer_from_init | `<float>` |  |  |

### 6) Template Diagnostics

| Field | Value | Field | Value | Field | Value |
| --- | --- | --- | --- | --- | --- |
| peak_dist_mean | `<float>` | peak_dist_std | `<float>` | sigma1_mean | `<float>` |
| sigma1_std | `<float>` | sigma2_mean | `<float>` | sigma2_std | `<float>` |

### 7) Visual Checks
- `<none|notes>`

### 8) Interpretation
- `<interpretation bullet 1>`
- `<interpretation bullet 2>`

### 9) Next Action
- `<next action>`

### 10) Artifacts

| Field | Value | Field | Value |
| --- | --- | --- | --- |
| output_dir | `<path>` | final_state | `<path>/final.pkl` |