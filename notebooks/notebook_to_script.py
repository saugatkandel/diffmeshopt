# %% [markdown]
# # Full Comparison: Refiners x Template Models x Symmetry
#
# This notebook performs a comprehensive comparison of different geometric refiners combined with different template parameterization models and symmetry settings.
#
# **Refiners:**
# *   `ContourRefiner`: Direct vertex optimization.
# *   `BSplineContourRefiner`: Optimization of B-Spline control points.
#
# **Template Models:**
# *   `Fixed`: Constant parameters.
# *   `Global`: Single learnable set of parameters.
# *   `PerPoint`: Learnable parameters for each vertex.
# *   `BSpline`: Spatially varying parameters (1D B-spline along contour).
# *   `Neural`: Spatially varying parameters (2D Neural Field).
# *   `Grid`: Spatially varying parameters (2D learnable grid).
# *   `Splat`: Spatially varying parameters (Gaussian splatting).
#
# **Symmetry:**
# *   `Symmetric`: `sigma1` = `sigma2`, `amp1` = `amp2`.
# *   `Asymmetric`: All parameters are independent.

# %%

import shutil
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from tqdm.notebook import tqdm

import diffmeshopt.opt2d.vis as vis2d
from diffmeshopt.opt2d.evaluation import compute_contour_metrics, compute_reference_free_metrics
from diffmeshopt.opt2d.optimize import BSplineContourRefiner, ContourRefiner
from diffmeshopt.opt2d.props import (
    BSplineContourRefinerProps,
    BSplineTemplateProps,
    ContourRefinerProps,
    GaussianSplatTemplateProps,
    GridTemplateProps,
    NeuralFieldTemplateProps,
    TemplateProps,
)
from diffmeshopt.opt2d.template import TemplateModelFactory
from diffmeshopt.opt2d.trainer import AtomicParquetLogger, OptimizationTrainer, TrainerConfig

# %%
# 1. Load Data
data_path = Path("../data/2d_training_data.pkl")
if not data_path.exists():
    data_path = Path("/workspace/diffmeshopt/data/2d_training_data.pkl")

print(f"Loading data from {data_path}...")
data = joblib.load(data_path)
image_np = -data["image"]  # Invert intensity so membrane is dark
contour_np = data["contour"]
gt_contour_np = data["gt"]

plt.figure(figsize=(6, 6))
plt.imshow(image_np, cmap="gray")
plt.plot(contour_np[:, 1], contour_np[:, 0], "r-", label="Init")
plt.plot(gt_contour_np[:, 1], gt_contour_np[:, 0], "g--", label="GT")
plt.legend()
plt.title("Input Data")
plt.axis("off")
plt.show()

# %%
# 2. Define Experiment Configurations
refiners = ["Vertex", "BSpline"]
templates = ["fixed", "global", "per_point", "bspline", "neural", "grid", "splat"]
symmetries = [True, False]

experiments = []
for r in refiners:
    for t in templates:
        for s in symmetries:
            if t in ["fixed", "global"] and s is False:
                continue
            experiments.append((r, t, s))

print(f"Defined {len(experiments)} experiments.")


def get_config(refiner_type, template_mode, symmetric, steps):
    # 1. Select Optimization Props based on Refiner
    if refiner_type == "BSpline":
        refiner_props = BSplineContourRefinerProps(
            num_steps=steps,
            learning_rate=0.05,
            laplacian_loss_weight=5.0,
            edge_length_loss_weight=1.0,
            contour_num_control_points=64,
            num_sampled_profiles=128,  # smaller batch size for performance
        )
        refiner_cls = BSplineContourRefiner
    else:  # Vertex
        refiner_props = ContourRefinerProps(
            num_steps=steps,
            learning_rate=0.05,
            laplacian_loss_weight=50.0,
            edge_length_loss_weight=10.0,
            num_sampled_profiles=128,
        )
        refiner_cls = ContourRefiner

    # 2. Select Template Props based on Mode
    template_props = TemplateProps(
        symmetric=symmetric,
    )

    if template_mode == "bspline":
        template_props = BSplineTemplateProps(
            **template_props.__dict__, bspline_num_control_points=8
        )
    elif template_mode == "neural":
        template_props = NeuralFieldTemplateProps(**template_props.__dict__, neural_hidden_dim=16)
        # Neural fields can be sensitive to high learning rates
        refiner_props.learning_rate = 0.01
    elif template_mode == "grid":
        template_props = GridTemplateProps(**template_props.__dict__, grid_size=8)
    elif template_mode == "splat":
        template_props = GaussianSplatTemplateProps(**template_props.__dict__, splat_num_splats=4)

    return refiner_cls, refiner_props, template_props


# %%
# 3. Run Experiments
results = []
steps = 500
device = "cuda" if torch.cuda.is_available() else "cpu"

# Move data to device once
image_torch = torch.from_numpy(image_np).float().to(device)
contour_torch = torch.from_numpy(contour_np).float().to(device)
gt_contour_torch = torch.from_numpy(gt_contour_np).float().to(device)
img_shape = image_torch.shape

# Clean experiment directory
exp_dir = Path("../output/experiments/comparison/")
if exp_dir.exists():
    shutil.rmtree(exp_dir)

for refiner_name, template_mode, symmetric in tqdm(experiments, desc="All Experiments"):
    label = f"{refiner_name}_{template_mode}_{'sym' if symmetric else 'asym'}"
    print(f"Running: {label}")

    refiner_cls, refiner_props, template_props = get_config(
        refiner_name, template_mode, symmetric, steps
    )

    # Instantiate Template Model
    factory_kwargs = {
        "num_vertices": len(contour_torch),
        "image_shape": img_shape,
    }
    template_model = TemplateModelFactory.create(
        template_mode, props=template_props, **factory_kwargs
    ).to(device)

    # Instantiate Refiner
    refiner = refiner_cls(
        initial_contour=contour_torch.clone(),
        props=refiner_props,
        template_model=template_model,
    )

    # Train using OptimizationTrainer
    run_dir = exp_dir / label

    config = TrainerConfig(
        output_dir=run_dir,
        image=image_np,
        gt_contour=gt_contour_np,
        save_images=True,
        use_tensorboard=True,
        log_interval=10,
        save_interval=200,
        max_steps=steps,
    )

    trainer = OptimizationTrainer(refiner, config, run_dir=run_dir)
    trainer.fit(steps_to_run=steps)

    # Get metrics from logger history
    logger = next((l for l in trainer.trainer.loggers if isinstance(l, AtomicParquetLogger)), None)
    history = logger.history if logger else []
    loss_history = [h["total_loss"] for h in history if "total_loss" in h]
    final_loss = loss_history[-1] if loss_history else np.nan

    # Evaluate
    final_contour = refiner.contour.detach().cpu().numpy()
    with torch.no_grad():
        params_torch = refiner.template_model.get_params(coordinates=refiner.contour)
        params_np = {
            k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
            for k, v in params_torch.items()
        }
    geo_metrics = compute_contour_metrics(final_contour, gt_contour_np)
    ref_free_metrics = compute_reference_free_metrics(final_contour)

    # Store
    res_entry = {
        "Refiner": refiner_name,
        "Template": template_mode,
        "Symmetric": symmetric,
        "Label": label,
        "Final Loss": final_loss,
        "Mean Dist": geo_metrics.get("mean_dist", np.nan),
        "Hausdorff": geo_metrics.get("hausdorff_dist", np.nan),
        "Perimeter": ref_free_metrics.get("perimeter", np.nan),
        "Roughness": ref_free_metrics.get("roughness", np.nan),
        "Contour": final_contour,
        "Template Params": params_np,
        "Loss History": loss_history,
        "RefinerObj": refiner.cpu(),  # Move to CPU to free up GPU memory
    }
    results.append(res_entry)

# %%
# 4. Quantitative Comparison (Table & Bar Plot)
if not results:
    print("No results to display.")
else:
    # Filter out complex objects that Polars doesn't handle well
    simple_results = [
        {
            k: v
            for k, v in r.items()
            if k not in ["Contour", "Template Params", "Loss History", "RefinerObj"]
        }
        for r in results
    ]
    df = pl.DataFrame(simple_results)

    # display(
    #    df.select(
    #        ["Refiner", "Template", "Symmetric", "Perimeter", "Roughness", "Final Loss"]
    #    ).sort(["Refiner", "Template", "Symmetric"])
    # )

    # Bar plot using matplotlib
    templates = sorted(df["Template"].unique().to_list())
    refiners = sorted(df["Refiner"].unique().to_list())
    print(f"Plotting for refiners: {refiners}")
    metrics = ["Perimeter", "Roughness"]
    symmetries = [True, False]

    # Create subplots: rows for metrics, columns for symmetries
    fig, axes = plt.subplots(
        len(metrics), len(symmetries), figsize=(14, 10), sharex=True, sharey="row"
    )

    # Ensure axes is 2D array
    if len(metrics) == 1 and len(symmetries) == 1:
        axes = np.array([[axes]])
    elif len(metrics) == 1:
        axes = axes[np.newaxis, :]
    elif len(symmetries) == 1:
        axes = axes[:, np.newaxis]

    # Dynamic width based on number of refiners
    width = 0.8 / len(refiners) if len(refiners) > 0 else 0.4

    for row, metric in enumerate(metrics):
        for col, is_symmetric in enumerate(symmetries):
            ax = axes[row, col]

            # Filter for symmetry
            subset_sym = df.filter(pl.col("Symmetric") == is_symmetric)

            x = np.arange(len(templates))

            # Plot bars for each refiner
            for i, refiner in enumerate(refiners):
                subset_ref = subset_sym.filter(pl.col("Refiner") == refiner)

                # Map template to value to ensure alignment
                val_map = {
                    row["Template"]: row[metric] for row in subset_ref.iter_rows(named=True)
                }
                vals = [val_map.get(t, 0) for t in templates]

                # Offset bars
                offset = width * (i - (len(refiners) - 1) / 2)

                ax.bar(x + offset, vals, width, label=refiner)

            sym_label = "Symmetric" if is_symmetric else "Asymmetric"
            ax.set_title(f"{metric} ({sym_label})")

            if col == 0:
                ax.set_ylabel(metric)

            if row == len(metrics) - 1:
                ax.set_xticks(x)
                ax.set_xticklabels(templates, rotation=45, ha="right")

            ax.grid(axis="y", linestyle="--", alpha=0.7)

            # Legend
            if row == 0 and col == 0:
                ax.legend(title="Refiner")

    plt.suptitle("Quantitative Metrics: Refiner Comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# %%
import random

import torch

from diffmeshopt.opt2d.geometry import compute_normals

# %%
if not results:
    print("No results to visualize.")
else:
    # Setup for consistent cropping
    random.seed(42)
    num_crops = 5
    crop_size = 60
    H, W = image_np.shape[:2]
    half_size = crop_size // 2

    # Select crop indices based on initial contour (consistent across all experiments)
    if len(contour_np) > 0:
        indices = list(range(len(contour_np)))
        if len(indices) >= num_crops:
            crop_indices = sorted(random.sample(indices, num_crops))
        else:
            crop_indices = list(range(len(contour_np)))
    else:
        crop_indices = []

    fig, axes = plt.subplots(
        len(results), num_crops, figsize=(3 * num_crops, 3 * len(results)), constrained_layout=True
    )

    if len(results) == 1:
        axes = axes[None, :]
    if num_crops == 1:
        axes = axes[:, None]

    for i, res in enumerate(results):
        label = res["Label"]
        final_contour = res["Contour"]
        params = res["Template Params"]

        # Compute normals for the final contour
        normals_torch = compute_normals(torch.from_numpy(final_contour).float())
        normals = normals_torch.numpy()

        # Helper to broadcast param
        def get_p(key):
            val = params.get(key, 1.0)
            if np.isscalar(val) or val.ndim == 0:
                return np.full(len(final_contour), val)
            return val.flatten()

        peak_dist = get_p("peak_dist")

        if "sigma1" in params and "sigma2" in params:
            sigma1 = get_p("sigma1")
            sigma2 = get_p("sigma2")
        else:
            sigma = get_p("sigma")
            sigma1 = sigma
            sigma2 = sigma

        # Calculate lines
        # Center is final_contour
        # Peaks
        peak1 = final_contour - normals * (peak_dist[:, None] / 2)
        peak2 = final_contour + normals * (peak_dist[:, None] / 2)

        # Widths (1 sigma away from peaks)
        p1_outer = peak1 - normals * sigma1[:, None]
        p2_outer = peak2 + normals * sigma2[:, None]

        for j, idx in enumerate(crop_indices):
            ax = axes[i, j]

            # Crop center from INITIAL contour
            cy, cx = contour_np[idx]

            y_min = max(0, int(cy - half_size))
            y_max = min(H, int(cy + half_size))
            x_min = max(0, int(cx - half_size))
            x_max = min(W, int(cx + half_size))

            ax.imshow(image_np, cmap="gray")

            # Plot GT
            if gt_contour_np is not None:
                ax.plot(gt_contour_np[:, 1], gt_contour_np[:, 0], "g-.", alpha=0.8, lw=1)

            ax.plot(contour_np[:, 1], contour_np[:, 0], "k-.", alpha=0.8, lw=1)
            # Plot Center
            ax.plot(final_contour[:, 1], final_contour[:, 0], "r-", alpha=0.8, lw=1.5)

            # Plot Peaks
            ax.plot(peak1[:, 1], peak1[:, 0], "c-", alpha=0.8, lw=1)
            ax.plot(peak2[:, 1], peak2[:, 0], "c-", alpha=0.8, lw=1)

            # Plot Widths
            ax.plot(p1_outer[:, 1], p1_outer[:, 0], "y:", alpha=0.8, lw=1)
            ax.plot(p2_outer[:, 1], p2_outer[:, 0], "y:", alpha=0.8, lw=1)

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_max, y_min)
            ax.axis("off")

            if j == 0:
                ax.text(
                    -0.1,
                    0.5,
                    label,
                    transform=ax.transAxes,
                    va="center",
                    ha="right",
                    fontsize=9,
                    rotation=0,
                )

    plt.suptitle("Qualitative Comparison: Zoomed Sections", fontsize=16)
    plt.show()

# %%
# 6. Learned Parameters Visualization
if results:
    # Group results by parameter type for plotting"
    param_groups = ["peak_dist", "sigma", "amp"]
    fig, axes = plt.subplots(
        len(param_groups), 1, figsize=(12, 4 * len(param_groups)), sharex=True
    )
    if len(param_groups) == 1:
        axes = [axes]

    for ax, p_name in zip(axes, param_groups):
        ax.set_title(f"Learned {p_name}")
        ax.set_ylabel("Value")

        for res in results:
            params = res["Template Params"]
            label = res["Label"]
            contour_len = len(res["Contour"])

            # Handle symmetric/asymmetric params
            keys_to_plot = []
            if p_name in params:
                keys_to_plot.append((p_name, ""))
            else:
                if f"{p_name}1" in params:
                    keys_to_plot.append((f"{p_name}1", "1"))
                if f"{p_name}2" in params:
                    keys_to_plot.append((f"{p_name}2", "2"))

            for key, suffix in keys_to_plot:
                val = params[key]
                # Broadcast scalar to array for plotting
                if np.isscalar(val) or val.ndim == 0:
                    y = np.full(contour_len, val)
                    linestyle = "--"
                elif val.ndim == 1 and len(val) == 1:
                    y = np.full(contour_len, val[0])
                    linestyle = "--"
                else:
                    y = val.flatten()
                    linestyle = "-"
                x = np.arange(len(y))
                lbl = f"{label} {suffix}".strip()
                ax.plot(x, y, label=lbl, linestyle=linestyle, alpha=0.7)

        # Legend outside
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="x-small", ncol=1)
        ax.grid(True, alpha=0.3)

    plt.xlabel("Contour Vertex Index")
    plt.tight_layout()
    plt.show()

# %%
# 6. Visualize Learned Template Parameters
for res in results:
    # Only plot for optimizable, non-fixed models
    if res["Template"] not in ["fixed"]:
        refiner = res["RefinerObj"]
        # get_params needs coordinates, use the final contour
        final_contour_torch = torch.from_numpy(res["Contour"]).to(device)
        params = refiner.template_model.get_params(coordinates=final_contour_torch)

        plt.figure(figsize=(10, 3))
        vis2d.plot_parameter_curves(params, ax=plt.gca())
        plt.title(f"Learned Parameters: {res['Label']}")
        plt.show()

# %%
