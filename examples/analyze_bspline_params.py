"""
Script to analyze the effect of B-Spline control point count on refinement quality.

Experiment Design:
------------------
This script evaluates how the number of B-Spline control points affects the ability
of the contour to fit a target shape.

1. Setup:
   - Target: A Gaussian ring at radius 40.0 px.
   - Initialization: A circular contour at radius 30.0 px.
   - Task: The contour must expand to fit the target.

2. Variable:
   - Number of Control Points (CPs): [8, 16, 32, 64].
   - Low CP count acts as a strong low-pass filter (stiff curve).
   - High CP count allows fitting higher frequency details but may overfit noise.

Expected Results:
-----------------
1. Fit Error: Should decrease as CP count increases, plateauing once the spline
   is flexible enough to represent the target circle.
2. Smoothness: Lower CP counts produce inherently smoother curves.
3. Convergence: All configurations should converge, but low CP counts might
   underfit (bias), while high CP counts might capture pixel grid noise (variance).

Visualizations:
---------------
1. Fit Error vs Complexity: Chamfer distance vs Number of Control Points.
2. Contour Comparison: Visual overlay of final contours for different CP counts.
3. Zoomed View: Detailed look at the fit accuracy.
4. Loss Components: Analysis of final data loss vs regularization costs.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from diffmeshopt.opt2d.config import (
    BSplineContourRefinerProps,
    RegularizationStrategy,
    RegularizerType,
    TemplateProps,
)
from diffmeshopt.opt2d.evaluation import compute_contour_metrics
from diffmeshopt.opt2d.refiner import BSplineContourRefiner
from diffmeshopt.opt2d.template import TemplateModelFactory


def analyze_bspline_control_points():
    # Setup data (Circle expansion task: Radius 30 -> 40)
    H, W = 100, 100
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    center = torch.tensor([50.0, 50.0])
    dist = torch.sqrt((y - center[0]) ** 2 + (x - center[1]) ** 2)
    target_radius = 40.0
    image = torch.exp(-((dist - target_radius) ** 2) / (2 * 2.0**2))
    image = image.unsqueeze(0).unsqueeze(0)

    theta = torch.linspace(0, 2 * np.pi, 100)[:-1]
    init_radius = 36.0  # Closer to target (40.0) to ensure gradient overlap
    initial_contour = torch.stack(
        [
            init_radius * torch.sin(theta) + center[0],
            init_radius * torch.cos(theta) + center[1],
        ],
        dim=1,
    )
    # GT Contour for metrics
    gt_contour = torch.stack(
        [
            target_radius * torch.sin(theta) + center[0],
            target_radius * torch.cos(theta) + center[1],
        ],
        dim=1,
    )

    template = TemplateModelFactory.create("fixed", TemplateProps(sigma=2.0, peak_dist=0.0))

    cp_counts = [8, 16, 32, 64]
    results = []

    print(f"Running analysis for CP counts: {cp_counts}")
    for num_cp in cp_counts:
        props = BSplineContourRefinerProps(
            num_steps=100,
            learning_rate=0.5,
            profile_length=21,
            contour_num_control_points=num_cp,
            regularization_strategy=RegularizationStrategy.TANGENTIAL_SMOOTHING,
        )
        refiner = BSplineContourRefiner(initial_contour.clone(), props, template)

        final_losses = {}
        for _ in tqdm(range(100), desc=f"CPs={num_cp}", leave=False):
            final_losses = refiner.step(image)

        final = refiner.contour
        metrics = compute_contour_metrics(final, gt_contour)
        results.append(
            {
                "num_cp": num_cp,
                "contour": final.detach().cpu().numpy(),
                "mean_dist": metrics["mean_dist"],
                "losses": final_losses,
            }
        )

    # --- Visualization ---
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("B-Spline Complexity Analysis: Effect of Control Point Count", fontsize=16)
    gs = fig.add_gridspec(2, 2)

    # 1. Fit Error vs Complexity
    ax_err = fig.add_subplot(gs[0, 0])
    cp_vals = [r["num_cp"] for r in results]
    err_vals = [r["mean_dist"] for r in results]

    ax_err.plot(cp_vals, err_vals, "o-", linewidth=2)
    ax_err.set_xlabel("Number of Control Points")
    ax_err.set_ylabel("Chamfer Distance (px)")
    ax_err.set_title("Fit Error vs Complexity")
    ax_err.grid(True)
    ax_err.set_xticks(cp_counts)

    # 2. Contour Comparison
    ax_vis = fig.add_subplot(gs[0, 1])
    ax_vis.imshow(image[0, 0], cmap="gray", origin="upper")

    # Plot Initial & Target
    init_c = initial_contour.numpy()
    init_c_closed = np.vstack([init_c, init_c[0]])
    ax_vis.plot(init_c_closed[:, 1], init_c_closed[:, 0], "r--", label="Initial", linewidth=1.5)

    gt_c = gt_contour.numpy()
    gt_c_closed = np.vstack([gt_c, gt_c[0]])
    ax_vis.plot(gt_c_closed[:, 1], gt_c_closed[:, 0], "g:", label="Target", linewidth=2)

    # Plot contours
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=0, vmax=len(cp_counts) - 1)

    for i, res in enumerate(results):
        c = res["contour"]
        c_closed = np.vstack([c, c[0]])
        color = cmap(norm(i))
        ax_vis.plot(
            c_closed[:, 1], c_closed[:, 0], color=color, alpha=0.8, label=f"{res['num_cp']} CPs"
        )

    ax_vis.set_title("Final Contours")
    ax_vis.legend(fontsize="small")
    ax_vis.set_xlim(0, 100)
    ax_vis.set_ylim(0, 100)

    # 3. Zoomed View
    ax_zoom = fig.add_subplot(gs[1, 0])
    ax_zoom.imshow(image[0, 0], cmap="gray", origin="upper")
    ax_zoom.plot(gt_c_closed[:, 1], gt_c_closed[:, 0], "g:", linewidth=2)

    for i, res in enumerate(results):
        c = res["contour"]
        c_closed = np.vstack([c, c[0]])
        color = cmap(norm(i))
        ax_zoom.plot(c_closed[:, 1], c_closed[:, 0], color=color, alpha=0.8)

    # Zoom on top arc
    ax_zoom.set_xlim(30, 70)
    ax_zoom.set_ylim(0, 40)
    ax_zoom.set_title("Zoomed View (Top Arc)")

    # 4. Loss Components
    ax_loss = fig.add_subplot(gs[1, 1])
    data_losses = [r["losses"]["data_loss"] for r in results]
    # Tangential Laplacian is the main regularizer here
    reg_losses = [
        r["losses"].get(RegularizerType.TANGENTIAL_LAPLACIAN.value, 0.0) for r in results
    ]

    x = np.arange(len(cp_counts))
    width = 0.35

    ax_loss.bar(x - width / 2, data_losses, width, label="Data Loss")
    ax_loss.bar(x + width / 2, reg_losses, width, label="Tangential Reg Loss")

    ax_loss.set_ylabel("Loss Value")
    ax_loss.set_title("Final Loss Components")
    ax_loss.set_xticks(x)
    ax_loss.set_xticklabels([str(cp) for cp in cp_counts])
    ax_loss.set_xlabel("Number of Control Points")
    ax_loss.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    analyze_bspline_control_points()
