"""
Script to analyze the effect of vertex resolution (count) on refinement quality.

Experiment Design:
------------------
1. Setup: Target radius 40.0, Init radius 36.0.
2. Variable: Number of vertices [32, 64, 128, 256].

Expected Results:
-----------------
1. Error: Discretization error decreases as N increases.
2. Convergence: Higher N might require more steps or different learning rates (though Adam handles this well).

Visualizations:
---------------
1. Fit Error vs Resolution.
2. Contour Comparison.
3. Zoomed View (showing polygonal facets).
4. Loss Components.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from diffmeshopt.opt2d.config import (
    ContourRefinerProps,
    RegularizationStrategy,
    TemplateProps,
)
from diffmeshopt.opt2d.evaluation import compute_contour_metrics
from diffmeshopt.opt2d.geometry import smooth_contour
from diffmeshopt.opt2d.refiner import ContourRefiner
from diffmeshopt.opt2d.template import TemplateModelFactory


def analyze_vertex_resolution():
    # Setup data (Circle expansion task: Radius 30 -> 40)
    H, W = 100, 100
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    center = torch.tensor([50.0, 50.0])
    dist = torch.sqrt((y - center[0]) ** 2 + (x - center[1]) ** 2)
    target_radius = 40.0
    image = torch.exp(-((dist - target_radius) ** 2) / (2 * 2.0**2))
    image = image.unsqueeze(0).unsqueeze(0)

    # Base initial contour (Radius 30)
    theta = torch.linspace(0, 2 * np.pi, 200)[:-1]
    init_radius = 36.0  # Closer to target (40.0) to ensure gradient overlap
    base_contour_np = np.stack(
        [
            init_radius * np.sin(theta) + center[0].numpy(),
            init_radius * np.cos(theta) + center[1].numpy(),
        ],
        axis=1,
    ).astype(np.float32)

    # GT Contour for metrics
    gt_contour = torch.stack(
        [
            target_radius * torch.sin(theta) + center[0],
            target_radius * torch.cos(theta) + center[1],
        ],
        dim=1,
    )

    template = TemplateModelFactory.create("fixed", TemplateProps(sigma=2.0, peak_dist=0.0))

    resolutions = [32, 64, 128, 256]
    results = []

    for num_verts in resolutions:
        # Resample initial contour to specific resolution
        resampled_contour_np = smooth_contour(base_contour_np, num_points=num_verts)
        initial_contour = torch.from_numpy(resampled_contour_np)

        props = ContourRefinerProps(
            num_steps=100,
            learning_rate=0.5,
            profile_length=21,
            regularization_strategy=RegularizationStrategy.TANGENTIAL_SMOOTHING,
        )
        refiner = ContourRefiner(initial_contour.clone(), props, template)

        final_losses = {}
        for _ in range(100):
            final_losses = refiner.step(image)

        final = refiner.contour
        metrics = compute_contour_metrics(final, gt_contour)
        results.append(
            {
                "num_verts": num_verts,
                "contour": final.detach().cpu().numpy(),
                "mean_dist": metrics["mean_dist"],
                "losses": final_losses,
            }
        )

    # --- Visualization ---
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Effect of Vertex Resolution on Refinement", fontsize=16)
    gs = fig.add_gridspec(2, 2)

    # 1. Fit Error vs Resolution
    ax_err = fig.add_subplot(gs[0, 0])
    res_vals = [r["num_verts"] for r in results]
    err_vals = [r["mean_dist"] for r in results]
    ax_err.plot(res_vals, err_vals, "o-")
    ax_err.set_xlabel("Number of Vertices")
    ax_err.set_ylabel("Chamfer Distance (px)")
    ax_err.set_title("Fit Error vs Resolution")
    ax_err.grid(True)
    ax_err.set_xticks(resolutions)

    # 2. Contour Comparison
    ax_vis = fig.add_subplot(gs[0, 1])
    ax_vis.imshow(image[0, 0], cmap="gray", origin="upper")

    gt_c = gt_contour.numpy()
    gt_c_closed = np.vstack([gt_c, gt_c[0]])
    ax_vis.plot(gt_c_closed[:, 1], gt_c_closed[:, 0], "g:", label="GT", alpha=0.5, linewidth=2)

    for res in results:
        c = res["contour"]
        c_closed = np.vstack([c, c[0]])
        ax_vis.plot(c_closed[:, 1], c_closed[:, 0], label=f"{res['num_verts']} Verts")

    ax_vis.legend()
    ax_vis.set_title("Final Contours")

    # 3. Zoomed View
    ax_zoom = fig.add_subplot(gs[1, 0])
    ax_zoom.imshow(image[0, 0], cmap="gray", origin="upper")
    ax_zoom.plot(gt_c_closed[:, 1], gt_c_closed[:, 0], "g:", linewidth=2)

    for res in results:
        c = res["contour"]
        c_closed = np.vstack([c, c[0]])
        ax_zoom.plot(c_closed[:, 1], c_closed[:, 0], "o-", markersize=3)

    ax_zoom.set_xlim(30, 70)
    ax_zoom.set_ylim(0, 40)
    ax_zoom.set_title("Zoomed View (Discretization)")

    # 4. Loss Components
    ax_loss = fig.add_subplot(gs[1, 1])
    data_losses = [r["losses"]["data_loss"] for r in results]

    x = np.arange(len(resolutions))
    ax_loss.bar(x, data_losses, width=0.5, label="Data Loss")
    ax_loss.set_xticks(x)
    ax_loss.set_xticklabels([str(r) for r in resolutions])
    ax_loss.set_xlabel("Resolution")
    ax_loss.set_title("Final Data Loss")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    analyze_vertex_resolution()
