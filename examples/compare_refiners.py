"""
Script to compare different refiners on synthetic data.

Experiment Design:
------------------
Compare three refinement strategies on a simple circle expansion task.
1. Vertex Refiner: Direct optimization of points.
2. B-Spline Refiner: Optimization of control points.
3. RBF Refiner: Optimization of deformation field weights.

Visualizations:
---------------
1. Convergence: Loss history over time.
2. Final Contours: Overlay of all methods.
3. Zoomed View: Detail of fit quality.
4. Metrics: Bar chart of final Chamfer distance.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from diffmeshopt.opt2d.config import (
    BSplineContourRefinerProps,
    ContourRefinerProps,
    RBFContourRefinerProps,
    RegularizationStrategy,
    TemplateProps,
)
from diffmeshopt.opt2d.evaluation import compute_contour_metrics
from diffmeshopt.opt2d.generate_2d_data import generate_synthetic_data
from diffmeshopt.opt2d.refiner import (
    BSplineContourRefiner,
    ContourRefiner,
    RBFContourRefiner,
)
from diffmeshopt.opt2d.template import TemplateModelFactory


def compare_refiners_on_synthetic():
    # Generate data
    img_np, contour_np, gt_np = generate_synthetic_data(
        shape=(128, 128), radius=40, center=(64, 64)
    )

    # Convert to torch
    image = torch.from_numpy(img_np).float().unsqueeze(0).unsqueeze(0)

    # Re-initialize contour closer to target (radius 40)
    # generate_synthetic_data initializes at 30, which is 10px away.
    # With sigma=1.0, this is too far. We need ~3-4px.
    theta = torch.linspace(0, 2 * 3.14159, 100)[:-1]
    center = torch.tensor([64.0, 64.0])
    init_radius = 37.0  # Closer to target (40.0)
    initial_contour = torch.stack(
        [
            init_radius * torch.sin(theta) + center[0],
            init_radius * torch.cos(theta) + center[1],
        ],
        dim=1,
    )

    # GT Contour
    gt_contour = torch.stack(
        [
            40.0 * torch.sin(theta) + center[0],
            40.0 * torch.cos(theta) + center[1],
        ],
        dim=1,
    )

    # Template
    template_props = TemplateProps(sigma=1.0, peak_dist=4.0)
    template = TemplateModelFactory.create("fixed", template_props)

    # 1. Vertex Refiner
    props_vertex = ContourRefinerProps(
        num_steps=100,
        learning_rate=0.5,
        profile_length=21,
        regularization_strategy=RegularizationStrategy.TANGENTIAL_SMOOTHING,
    )
    refiner_vertex = ContourRefiner(initial_contour.clone(), props_vertex, template)

    # 2. B-Spline Refiner
    props_bspline = BSplineContourRefinerProps(
        num_steps=100,
        learning_rate=0.5,
        profile_length=21,
        regularization_strategy=RegularizationStrategy.TANGENTIAL_SMOOTHING,
        contour_num_control_points=32,
    )
    refiner_bspline = BSplineContourRefiner(initial_contour.clone(), props_bspline, template)

    # 3. RBF Refiner
    props_rbf = RBFContourRefinerProps(
        num_steps=100,
        learning_rate=0.5,
        profile_length=21,
        regularization_strategy=RegularizationStrategy.TANGENTIAL_SMOOTHING,
        rbf_num_control_points=20,
        rbf_kernel_sigma=0.0,  # Auto
    )
    refiner_rbf = RBFContourRefiner(initial_contour.clone(), props_rbf, template)

    # Run
    refiners = [("Vertex", refiner_vertex), ("B-Spline", refiner_bspline), ("RBF", refiner_rbf)]
    results = []

    for name, refiner in refiners:
        loss_history = []
        for _ in tqdm(range(100), desc=f"Running {name}"):
            losses = refiner.step(image)
            loss_history.append(losses["total_loss"])

        final = refiner.contour
        metrics = compute_contour_metrics(final, gt_contour)

        results.append(
            {
                "name": name,
                "contour": final.detach().cpu().numpy(),
                "loss_history": loss_history,
                "chamfer": metrics["mean_dist"],
            }
        )

    # --- Visualization ---
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Comparison of Refinement Strategies", fontsize=16)
    gs = fig.add_gridspec(2, 2)

    # 1. Convergence
    ax_conv = fig.add_subplot(gs[0, 0])
    for res in results:
        ax_conv.plot(res["loss_history"], label=res["name"])
    ax_conv.set_xlabel("Step")
    ax_conv.set_ylabel("Total Loss")
    ax_conv.set_title("Convergence")
    ax_conv.legend()
    ax_conv.grid(True)

    # 2. Final Contours
    ax_vis = fig.add_subplot(gs[0, 1])
    ax_vis.imshow(img_np, cmap="gray", origin="upper")

    gt_c = gt_contour.numpy()
    gt_c_closed = np.vstack([gt_c, gt_c[0]])
    ax_vis.plot(gt_c_closed[:, 1], gt_c_closed[:, 0], "g:", label="GT", linewidth=2, alpha=0.7)

    for res in results:
        c = res["contour"]
        c_closed = np.vstack([c, c[0]])
        ax_vis.plot(c_closed[:, 1], c_closed[:, 0], label=res["name"])

    ax_vis.legend()
    ax_vis.set_title("Final Contours")

    # 3. Zoomed View
    ax_zoom = fig.add_subplot(gs[1, 0])
    ax_zoom.imshow(img_np, cmap="gray", origin="upper")
    ax_zoom.plot(gt_c_closed[:, 1], gt_c_closed[:, 0], "g:", linewidth=2, alpha=0.7)
    for res in results:
        c = res["contour"]
        c_closed = np.vstack([c, c[0]])
        ax_zoom.plot(c_closed[:, 1], c_closed[:, 0])
    ax_zoom.set_xlim(40, 90)
    ax_zoom.set_ylim(10, 60)
    ax_zoom.set_title("Zoomed View")

    # 4. Metrics
    ax_met = fig.add_subplot(gs[1, 1])
    names = [r["name"] for r in results]
    vals = [r["chamfer"] for r in results]
    ax_met.bar(names, vals)
    ax_met.set_ylabel("Chamfer Distance (px)")
    ax_met.set_title("Final Fit Error")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_refiners_on_synthetic()
