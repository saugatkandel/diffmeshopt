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

---

<!-- run_id: 20260728-1618_rbf_wasserstein -->
## Run Summary

| Field | Value |
| --- | --- |
| Run ID | 20260728-1618_rbf_wasserstein |
| Author | Saugat Kandel |
| Branch / Commit | main @ 3f70eed |
| Notebook / Script | perturbed_search_rbf_s3p5_wasserstein.ipynb |
| Device | cpu |
| Seed(s) | 145 |
| Status | complete |

### 1) Goal
- Test optimization with RBF refiner and Wasserstein loss on perturbed data.

### 2) Data


### 3) Configuration
| Field | Value | Field | Value |
| --- | --- | --- | --- |
| Data loss type | bigaussian_wasserstein | Learning rate | 0.5 |
| Num cp | 40 | Profile length | 51 |
| Profile width | 5 | Shape loss weight | 0.1 |
| Min peak ratio | 3 |  |  |

**Regularization weights**

| Field | Value | Field | Value | Field | Value |
| --- | --- | --- | --- | --- | --- |
| contour_laplacian | 0 | edge_length | 0 | normal_consistency | 0 |
| tangential_laplacian | 0 | contour_anchor | 0 | rbf_weight_decay | 0.001 |

### 4) Runtime Notes
| Field | Value | Field | Value |
| --- | --- | --- | --- |
| global_step | 10000 |  |  |

### 5) Quantitative Results
| Field | Value | Field | Value | Field | Value |
| --- | --- | --- | --- | --- | --- |
| mean_dist | 2.19003 | total_loss | 0.543941 | data_loss | 0.529604 |
| initial_mean_dist | 2.95872 | initial_hausdorff_dist | 6.30662 | initial_p95_dist | 5.18607 |

### 6) Template Diagnostics
| Field | Value | Field | Value | Field | Value |
| --- | --- | --- | --- | --- | --- |
| peak_dist_mean | 6.1667476 | sigma1_mean | 2.0551715 | sigma2_mean | 2.0551715 |
| peak_dist_std | 0.21822384 | sigma1_std | 0.07272673 | sigma2_std | 0.07272673 |

### 7) Visual Checks
| Field | Value | Field | Value |
| --- | --- | --- | --- |
| contour_bump | Still a bump at presumably the location where the contour is closed. | template | Template looks single gaussian. |
| fits | Fitting doesn't seem great. |  |  |

### 8) Interpretation
- Wasserstein loss is not helping the alignment like I would have hoped.
- Template fitting needs work. One option is to make the contour anchor harder.
- Cyclic contour needs more testing.

### 9) Next Action
- Test longer cyclic contour.

### 10) Artifacts
| Field | Value | Field | Value |
| --- | --- | --- | --- |
| output_dir | ../output/verification_test_5A/rbf/20260728-1618_rbf_wasserstein/2026-07-28_16-18-40 | final_state | ../output/verification_test_5A/rbf/20260728-1618_rbf_wasserstein/2026-07-28_16-18-40/final.pkl |

---

<!-- run_id: 20260728-1651_rbf_wasserstein -->
## Run Summary

| Field | Value |
| --- | --- |
| Run ID | 20260728-1651_rbf_wasserstein |
| Author | Saugat Kandel |
| Branch / Commit | main @ 3f70eed |
| Notebook / Script | perturbed_search_rbf_s3p5_wasserstein.ipynb |
| Device | cpu |
| Seed(s) | 145 |
| Status | complete |

### 1) Goal
- See if the bump is improved by increasing the number of cycle points. Trying 5.

### 2) Data


### 3) Configuration
| Field | Value | Field | Value |
| --- | --- | --- | --- |
| Refiner | rbf | Template | bspline |
| Data loss type | bigaussian_wasserstein | Learning rate | 0.5 |
| Num cp | 40 | Profile length | 51 |
| Profile width | 5 | Shape loss weight | 0.1 |
| Min peak ratio | 3 |  |  |

**Regularization weights**

| Field | Value | Field | Value | Field | Value |
| --- | --- | --- | --- | --- | --- |
| contour_laplacian | 0 | edge_length | 0 | normal_consistency | 0 |
| tangential_laplacian | 0 | contour_anchor | 0 | rbf_weight_decay | 0.001 |

### 4) Runtime Notes
| Field | Value | Field | Value |
| --- | --- | --- | --- |
| global_step | 10000 |  |  |

### 5) Quantitative Results
| Field | Value | Field | Value | Field | Value |
| --- | --- | --- | --- | --- | --- |
| mean_dist | 1.99161 | total_loss | 0.567042 | data_loss | 0.544858 |
| initial_mean_dist | 3.03585 | initial_hausdorff_dist | 6.59841 | initial_p95_dist | 5.32975 |

### 6) Template Diagnostics
| Field | Value | Field | Value | Field | Value |
| --- | --- | --- | --- | --- | --- |
| peak_dist_mean | 6.2117496 | sigma1_mean | 2.0701628 | sigma2_mean | 2.0701628 |
| peak_dist_std | 0.13793439 | sigma1_std | 0.04596798 | sigma2_std | 0.04596798 |

### 7) Visual Checks
| Field | Value | Field | Value |
| --- | --- | --- | --- |
| contour_bump | Still a bump at presumably the location where the contour is closed. | template | Template looks single gaussian. |
| fits | Fitting doesn't seem great. |  |  |

### 8) Interpretation
- Wasserstein loss is not helping the alignment like I would have hoped.
- Template fitting needs work. One option is to make the contour anchor harder.
- Cyclic contour needs more testing.

### 9) Next Action
- Test longer cyclic contour. Tried 5, go for 20 next. 

### 10) Artifacts
| Field | Value | Field | Value |
| --- | --- | --- | --- |
| output_dir | ../output/verification_test_5A/rbf/20260728-1651_rbf_wasserstein/2026-07-28_16-51-11 | final_state | ../output/verification_test_5A/rbf/20260728-1651_rbf_wasserstein/2026-07-28_16-51-11/final.pkl |

---

<!-- run_id: 20260728-1733_rbf_wasserstein -->
## Run Summary

| Field | Value |
| --- | --- |
| Run ID | 20260728-1733_rbf_wasserstein |
| Author | Saugat Kandel |
| Branch / Commit | main @ 3f70eed |
| Notebook / Script | perturbed_search_rbf_s3p5_wasserstein.ipynb |
| Device | cpu |
| Seed(s) | 145 |
| Status | complete |

### 1) Goal
- See if the bump is improved by increasing the number of cycle points. Trying 20.

### 2) Data


### 3) Configuration
| Field | Value | Field | Value |
| --- | --- | --- | --- |
| Refiner | rbf | Template | bspline |
| Data loss type | bigaussian_wasserstein | Learning rate | 0.5 |
| Num cp | 40 | Profile length | 51 |
| Profile width | 5 | Shape loss weight | 0.1 |
| Min peak ratio | 3 |  |  |

**Regularization weights**

| Field | Value | Field | Value | Field | Value |
| --- | --- | --- | --- | --- | --- |
| contour_laplacian | 0 | edge_length | 0 | normal_consistency | 0 |
| tangential_laplacian | 0 | contour_anchor | 0 | rbf_weight_decay | 0.001 |

### 4) Runtime Notes
| Field | Value | Field | Value |
| --- | --- | --- | --- |
| global_step | 10000 |  |  |

### 5) Quantitative Results
| Field | Value | Field | Value | Field | Value |
| --- | --- | --- | --- | --- | --- |
| mean_dist | 2.09109 | total_loss | 0.564437 | data_loss | 0.545401 |
| initial_mean_dist | 3.01746 | initial_hausdorff_dist | 5.93717 | initial_p95_dist | 5.0327 |

### 6) Template Diagnostics
| Field | Value | Field | Value | Field | Value |
| --- | --- | --- | --- | --- | --- |
| peak_dist_mean | 6.1109157 | sigma1_mean | 2.0365493 | sigma2_mean | 2.0365493 |
| peak_dist_std | 0.18276176 | sigma1_std | 0.060907975 | sigma2_std | 0.060907975 |

### 7) Visual Checks
| Field | Value | Field | Value |
| --- | --- | --- | --- |
| contour_bump | Still a bump at presumably the location where the contour is closed. | template | Template looks single gaussian. |
| fits | Fitting doesn't seem great. |  |  |

### 8) Interpretation
- Wasserstein loss is not helping the alignment like I would have hoped.
- Template fitting needs work. One option is to make the contour anchor harder.
- Cyclic contour needs more testing.

### 9) Next Action
- Set aside the contour bump question for longer term investigation.
- Revert to question about alignment and template fit.

### 10) Artifacts
| Field | Value | Field | Value |
| --- | --- | --- | --- |
| output_dir | ../output/verification_test_5A/rbf/20260728-1733_rbf_wasserstein/2026-07-28_17-33-08 | final_state | ../output/verification_test_5A/rbf/20260728-1733_rbf_wasserstein/2026-07-28_17-33-08/final.pkl |

---

<!-- run_id: 20260728-1810_rbf_wasserstein -->
## Run Summary

| Field | Value |
| --- | --- |
| Run ID | 20260728-1810_rbf_wasserstein |
| Author | Saugat Kandel |
| Branch / Commit | main @ 3f70eed |
| Notebook / Script | perturbed_search_rbf_s3p5_wasserstein.ipynb |
| Device | cpu |
| Seed(s) | 145 |
| Status | complete |

### 1) Goal
- See if I can improve the alignment and template by adding template shape anchor.

### 2) Data


### 3) Configuration
| Field | Value | Field | Value |
| --- | --- | --- | --- |
| Refiner | rbf | Template | bspline |
| Data loss type | bigaussian_wasserstein | Learning rate | 0.5 |
| Num cp | 40 | Profile length | 51 |
| Profile width | 5 | Shape loss weight | 0.1 |
| Min peak ratio | 3 |  |  |

**Regularization weights**

| Field | Value | Field | Value | Field | Value |
| --- | --- | --- | --- | --- | --- |
| contour_laplacian | 0 | edge_length | 0 | normal_consistency | 0 |
| tangential_laplacian | 0 | contour_anchor | 0.1 | rbf_weight_decay | 0.001 |

### 4) Runtime Notes
| Field | Value | Field | Value |
| --- | --- | --- | --- |
| global_step | 10000 |  |  |

### 5) Quantitative Results
| Field | Value | Field | Value | Field | Value |
| --- | --- | --- | --- | --- | --- |
| mean_dist | 2.184 | total_loss | 0.557167 | data_loss | 0.541028 |
| initial_mean_dist | 2.9809 | initial_hausdorff_dist | 6.102 | initial_p95_dist | 5.09565 |

### 6) Template Diagnostics
| Field | Value | Field | Value | Field | Value |
| --- | --- | --- | --- | --- | --- |
| peak_dist_mean | 6.055817 | sigma1_mean | 2.018197 | sigma2_mean | 2.018197 |
| peak_dist_std | 0.27205303 | sigma1_std | 0.09066599 | sigma2_std | 0.09066599 |

### 7) Visual Checks
| Field | Value | Field | Value |
| --- | --- | --- | --- |
| template | Template still looks single gaussian. | fits | Fitting doesn't seem great. |

### 8) Interpretation
- I changed the CONTOUR_ANCHOR instead of SHAPE_ANCHOR. This is obviously WRONG.
- Wasserstein loss is not helping the alignment like I would have hoped.

### 9) Next Action
- Apply shape anchor instead of CONTOUR ANCHOR.

### 10) Artifacts
| Field | Value | Field | Value |
| --- | --- | --- | --- |
| output_dir | ../output/verification_test_5A/rbf/20260728-1810_rbf_wasserstein/2026-07-28_18-10-35 | final_state | ../output/verification_test_5A/rbf/20260728-1810_rbf_wasserstein/2026-07-28_18-10-35/final.pkl |

---

---

<!-- run_id: 20260728-1830_rbf_wasserstein -->
## Run Summary

| Field | Value |
| --- | --- |
| Run ID | 20260728-1830_rbf_wasserstein |
| Author | Saugat Kandel |
| Branch / Commit | main @ 3f70eed |
| Notebook / Script | perturbed_search_rbf_s3p5_wasserstein.ipynb |
| Device | cpu |
| Seed(s) | 145 |
| Status | complete |

### 1) Goal
- See if I can improve the alignment and template by ACTUALLY APPLYING template shape anchor.

### 2) Data


### 3) Configuration
| Field | Value | Field | Value |
| --- | --- | --- | --- |
| Refiner | rbf | Template | bspline |
| Data loss type | bigaussian_wasserstein | Learning rate | 0.5 |
| Num cp | 40 | Profile length | 51 |
| Profile width | 5 | Shape loss weight | 0.1 |
| Min peak ratio | 3 |  |  |

**Regularization weights**

| Field | Value | Field | Value | Field | Value |
| --- | --- | --- | --- | --- | --- |
| contour_laplacian | 0 | edge_length | 0 | normal_consistency | 0 |
| tangential_laplacian | 0 | contour_anchor | 0 | rbf_weight_decay | 0.001 |
| anchor_sigma | 0.1 | anchor_peak_dist | 0.1 |  |  |

### 4) Runtime Notes
| Field | Value | Field | Value |
| --- | --- | --- | --- |
| global_step | 10000 |  |  |

### 5) Quantitative Results
| Field | Value | Field | Value | Field | Value |
| --- | --- | --- | --- | --- | --- |
| mean_dist | 1.58618 | total_loss | 0.672829 | data_loss | 0.635982 |
| initial_mean_dist | 3.47151 | initial_hausdorff_dist | 7.44773 | initial_p95_dist | 6.74815 |

### 6) Template Diagnostics
| Field | Value | Field | Value | Field | Value |
| --- | --- | --- | --- | --- | --- |
| peak_dist_mean | 4.3153224 | sigma1_mean | 0.83704686 | sigma2_mean | 0.83704686 |
| peak_dist_std | 0.24156901 | sigma1_std | 0.042178255 | sigma2_std | 0.042178255 |

### 7) Visual Checks
| Field | Value | Field | Value |
| --- | --- | --- | --- |
| template | Template looks bette. | fits | Fitting actually looks better, but the top and bottom left contours are wonky. Fit is probably too flexible. |

### 8) Interpretation
- Using shape anchor definitely improves the fit. But obviously the prior vs optimization issue is there.
- If the fit is too flexible, then changing the number of control points is a worthwhile experiment.
- Not clear if Wasserstein loss is having any effect here. Obviously it is the more principled approach ( I think).
- Should probably re-verify if the KL divergence comes out to be the normalized correlation loss.

### 9) Next Action
- Decrease number of control points to 30.

### 10) Artifacts
| Field | Value | Field | Value |
| --- | --- | --- | --- |
| output_dir | ../output/verification_test_5A/rbf/20260728-1830_rbf_wasserstein/2026-07-28_18-30-41 | final_state | ../output/verification_test_5A/rbf/20260728-1830_rbf_wasserstein/2026-07-28_18-30-41/final.pkl |
