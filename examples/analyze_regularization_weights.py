"""
Script to analyze the effect of regularization weights on refinement metrics.

Experiment Design:
------------------
1. Setup:
   - Target: Gaussian ring at radius 40.0 px.
   - Initialization: Circle at radius 36.0 px.
   - Task: Expand to fit target while maintaining regularity.

2. Variable:
   - Tangential Laplacian Weight: Controls vertex spacing uniformity.
     Range: [0.0, 0.1, 1.0, 5.0, 10.0, 50.0]

Expected Results:
-----------------
1. Error: Very high weights might prevent fitting (bias). Very low weights might allow bunching.
2. Perimeter: Should remain close to the circle's perimeter.
3. Regularity: Higher weights ensure uniform edge lengths.

Visualizations:
---------------
1. Metrics: Chamfer Distance and Perimeter vs Weight.
2. Contour Evolution: Visual overlay of final contours.
3. Zoomed View: Detail of vertex distribution.
4. Edge Length Variance: Metric for vertex spacing uniformity.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from diffmeshopt.opt2d.config import (
    ContourRefinerProps,
    RegularizerType,
    TemplateProps,
)
from diffmeshopt.opt2d.evaluation import compute_contour_metrics
from diffmeshopt.opt2d.refiner import VertexContourRefiner
from diffmeshopt.opt2d.regularizer_recipes import TANGENTIAL_SMOOTHING_VERTEX
from diffmeshopt.opt2d.template import TemplateModelFactory


def analyze_weights():
    # 1. Setup Synthetic Problem
    # Target: Circle at radius 40
    # Initial: Circle at radius 30 (needs to expand)
    # Image: Distance transform to target
    H, W = 100, 100
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    center = torch.tensor([50.0, 50.0])
    dist = torch.sqrt((y - center[0]) ** 2 + (x - center[1]) ** 2)

    # Gaussian ridge at radius 40
    target_radius = 40.0
    image = torch.exp(-((dist - target_radius) ** 2) / (2 * 2.0**2))
    image = image.unsqueeze(0).unsqueeze(0)

    # Initial contour at radius 30
    theta = torch.linspace(0, 2 * np.pi, 100)[:-1]
    init_radius = 36.0  # Closer to target (40.0) to ensure gradient overlap
    initial_contour = torch.stack(
        [init_radius * torch.sin(theta) + center[0], init_radius * torch.cos(theta) + center[1]],
        dim=1,
    )

    # GT Contour
    gt_contour = torch.stack(
        [
            target_radius * torch.sin(theta) + center[0],
            target_radius * torch.cos(theta) + center[1],
        ],
        dim=1,
    )

    template = TemplateModelFactory.create("fixed", TemplateProps(sigma=2.0, peak_dist=0.0))

    # 2. Sweep Weights
    # We will sweep Tangential Laplacian (Spacing) and Contour Anchor (Safety)
    weights = [0.0, 0.1, 1.0, 5.0, 10.0, 50.0]

    results = []

    print("Sweeping Tangential Laplacian Weights...")
    for w in weights:
        # Use standard recipe and override specific weights
        loss_weights = TANGENTIAL_SMOOTHING_VERTEX.copy()
        loss_weights[RegularizerType.TANGENTIAL_LAPLACIAN.value] = w
        loss_weights[RegularizerType.CONTOUR_ANCHOR.value] = 0.0
        loss_weights[RegularizerType.NORMAL_CONSISTENCY.value] = 1.0

        props = ContourRefinerProps(
            num_steps=50,
            learning_rate=0.5,
            profile_length=21,
            initial_loss_weights=loss_weights,
        )
        refiner = VertexContourRefiner(initial_contour.clone(), props, template)

        for _ in range(50):
            refiner.step(image)

        # Metrics
        final = refiner.contour
        metrics = compute_contour_metrics(final, gt_contour)

        # Perimeter (to check for bunching/expansion)
        diffs = final - torch.roll(final, -1, 0)
        edge_lengths = torch.norm(diffs, dim=1)
        perimeter = edge_lengths.sum().item()
        edge_var = torch.var(edge_lengths).item()

        results.append({
            "weight": w,
            "contour": final.detach().cpu().numpy(),
            "chamfer": metrics["mean_dist"],
            "perimeter": perimeter,
            "edge_var": edge_var,
        })

    # 3. Plot
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Effect of Tangential Laplacian Weight on Refinement", fontsize=16)
    gs = fig.add_gridspec(2, 2)

    # 1. Metrics
    ax_met = fig.add_subplot(gs[0, 0])
    color = "tab:red"
    ax_met.set_xlabel("Tangential Laplacian Weight")
    ax_met.set_ylabel("Chamfer Distance (Error)", color=color)
    ax_met.plot(weights, [r["chamfer"] for r in results], marker="o", color=color)
    ax_met.tick_params(axis="y", labelcolor=color)
    ax_met.set_xscale("symlog", linthresh=0.1)

    ax2 = ax_met.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Perimeter Length", color=color)
    ax2.plot(weights, [r["perimeter"] for r in results], marker="s", linestyle="--", color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    ax_met.set_title("Metrics vs Weight")
    ax_met.grid(True)

    # 2. Contour Evolution
    ax_vis = fig.add_subplot(gs[0, 1])
    ax_vis.imshow(image[0, 0], cmap="gray", origin="upper")

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=0, vmax=len(weights) - 1)

    for i, res in enumerate(results):
        c = res["contour"]
        c_closed = np.vstack([c, c[0]])
        color = cmap(norm(i))
        ax_vis.plot(
            c_closed[:, 1], c_closed[:, 0], color=color, alpha=0.8, label=f"w={res['weight']}"
        )

    ax_vis.set_title("Contour Evolution")
    ax_vis.legend(fontsize="small")
    ax_vis.set_xlim(0, 100)
    ax_vis.set_ylim(0, 100)

    # 3. Zoomed View
    ax_zoom = fig.add_subplot(gs[1, 0])
    ax_zoom.imshow(image[0, 0], cmap="gray", origin="upper")
    for i, res in enumerate(results):
        c = res["contour"]
        c_closed = np.vstack([c, c[0]])
        color = cmap(norm(i))
        ax_zoom.plot(c_closed[:, 1], c_closed[:, 0], "o-", color=color, alpha=0.8, markersize=3)

    ax_zoom.set_xlim(30, 70)
    ax_zoom.set_ylim(0, 40)
    ax_zoom.set_title("Zoomed View (Vertex Distribution)")

    # 4. Edge Length Variance
    ax_var = fig.add_subplot(gs[1, 1])
    ax_var.plot(weights, [r["edge_var"] for r in results], "o-", color="purple")
    ax_var.set_xscale("symlog", linthresh=0.1)
    ax_var.set_xlabel("Weight")
    ax_var.set_ylabel("Edge Length Variance")
    ax_var.set_title("Vertex Spacing Uniformity (Lower is Better)")
    ax_var.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    analyze_weights()
