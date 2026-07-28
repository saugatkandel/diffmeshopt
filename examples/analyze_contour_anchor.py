"""
Script to analyze the effect of the Contour Anchor weight on maximum displacement.

Experiment Design:
------------------
This script sets up a "tug-of-war" between two opposing forces:
1. Data Force: Tries to pull the contour towards a bright Gaussian ring in the image.
   - Target Radius: 34.0 px
   - Template Sigma: 2.0 px
   - Force magnitude is proportional to the gradient of the cross-correlation.

2. Anchor Force: Tries to keep the contour at its initialization.
   - Initial Radius: 30.0 px
   - Force magnitude is proportional to 2 * weight * displacement (derivative of L2 loss).

We also compare static weights against an **Adaptive Regularization** strategy,
which dynamically adjusts the anchor weight to maintain a target ratio between
regularization loss and data loss.

Theoretical Expectation:
------------------------
At equilibrium, F_data = F_anchor.
- F_data ~ 1 / sigma (approximate max gradient of Gaussian correlation). With sigma=2.0, F_data ~ 0.5.
- F_anchor = 2 * w * d
- Therefore, displacement d ~ 0.5 / (2 * w) = 0.25 / w.

Visualizations:
---------------
Part 1: Vertex Refiner
1. Displacement vs Weight: Log-log plot comparing measured displacement to theory.
   Includes markers for the "Standard" static weight (0.1) and the final Adaptive weight.
2. Contour Evolution: Visual overlay of how the contour "stretches" towards the target
   but gets held back by the anchor as weight increases.
   Highlights the Standard (w=0.1) and Adaptive contours.
3. Zoomed View: Detailed look at the gap between target and final contour.
4. Loss Components: Bar chart showing the energy balance between fitting the data
   and satisfying the anchor constraint. Includes the Adaptive case.

Part 2: B-Spline Refiner
1. Displacement Comparison: Compares the displacement of the evaluated contour vs.
   the underlying control points. This reveals how the B-spline basis transmits
   the anchor force (applied to control points) to the curve.
2. Contour Evolution: Similar to Vertex Refiner.
3. Control Point Evolution: Visualizes the "hull" of control points to check for
   tangential drift or bunching.
4. Loss Components: Energy balance.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from diffmeshopt.opt2d.config import (
    AdaptiveRegularizationProps,
    BSplineContourRefinerProps,
    ContourRefinerProps,
    RegularizerType,
    TemplateProps,
)
from diffmeshopt.opt2d.refiner import BSplineContourRefiner, VertexContourRefiner
from diffmeshopt.opt2d.regularizer_recipes import (
    TANGENTIAL_SMOOTHING_BSPLINE,
    TANGENTIAL_SMOOTHING_VERTEX,
)
from diffmeshopt.opt2d.template import TemplateModelFactory


def analyze_vertex_anchor():
    # Setup: Strong pull away from initialization
    # Init at 30, Target at 50.
    H, W = 100, 100
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    center = torch.tensor([50.0, 50.0])
    dist = torch.sqrt((y - center[0]) ** 2 + (x - center[1]) ** 2)
    target_radius = 34.0  # Closer to init (30.0) to ensure gradient overlap (2 sigma)
    image = torch.exp(-((dist - target_radius) ** 2) / (2 * 2.0**2))
    image = image.unsqueeze(0).unsqueeze(0)

    theta = torch.linspace(0, 2 * np.pi, 100)[:-1]
    init_radius = 30.0
    initial_contour = torch.stack(
        [
            init_radius * torch.sin(theta) + center[0],
            init_radius * torch.cos(theta) + center[1],
        ],
        dim=1,
    )

    template = TemplateModelFactory.create("fixed", TemplateProps(sigma=2.0, peak_dist=0.0))

    weights = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    results = []

    # Ensure dense sampling to avoid "spiky" movement if resolution is increased
    num_vertices = len(initial_contour)

    print(f"Running analysis for weights: {weights}")
    for w in weights:
        # Use standard recipe and override anchor weight
        loss_weights = TANGENTIAL_SMOOTHING_VERTEX.copy()
        loss_weights[RegularizerType.CONTOUR_ANCHOR.value] = w
        loss_weights[RegularizerType.NORMAL_CONSISTENCY.value] = (
            1.0  # Override to match experiment
        )

        props = ContourRefinerProps(
            num_steps=100,
            learning_rate=0.5,
            profile_length=21,  # Reduced from 51 to avoid hitting image boundaries (50+30+25 > 100)
            num_sampled_profiles=num_vertices,  # Sample all vertices for accurate analysis
            initial_loss_weights=loss_weights,
        )
        refiner = VertexContourRefiner(initial_contour.clone(), props, template)

        final_losses = {}
        for _ in tqdm(range(100), desc=f"w={w}", leave=False):
            final_losses = refiner.step(image)

        # Calculate mean radial displacement from initialization
        final_contour = refiner.contour.detach().cpu()
        final_radius = torch.norm(final_contour - center, dim=1).mean().item()
        displacement = final_radius - init_radius

        results.append({
            "weight": w,
            "displacement": displacement,
            "contour": final_contour.numpy(),
            "losses": final_losses,
        })

    # --- Run Adaptive Case for Comparison ---
    print("Running Adaptive analysis...")
    loss_weights_adaptive = TANGENTIAL_SMOOTHING_VERTEX.copy()
    loss_weights_adaptive[RegularizerType.CONTOUR_ANCHOR.value] = 0.1
    loss_weights_adaptive[RegularizerType.NORMAL_CONSISTENCY.value] = 1.0

    props_adaptive = ContourRefinerProps(
        num_steps=100,
        learning_rate=0.5,
        profile_length=21,
        num_sampled_profiles=num_vertices,
        initial_loss_weights=loss_weights_adaptive,
        adaptive_reg=AdaptiveRegularizationProps(enabled=True, update_interval=5, warmup_steps=10),
    )
    refiner_adaptive = VertexContourRefiner(initial_contour.clone(), props_adaptive, template)
    final_losses_adaptive = {}
    for _ in range(100):
        final_losses_adaptive = refiner_adaptive.step(image)

    final_contour_adaptive = refiner_adaptive.contour.detach().cpu()
    final_radius_adaptive = torch.norm(final_contour_adaptive - center, dim=1).mean().item()
    disp_adaptive = final_radius_adaptive - init_radius
    final_weight_adaptive = refiner_adaptive.loss_fn.get_weight(
        RegularizerType.CONTOUR_ANCHOR
    ).item()

    # --- Visualization ---
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Vertex Refiner: Tug-of-War between Data Force and Anchor Force", fontsize=16)
    gs = fig.add_gridspec(2, 2)

    # 1. Displacement Analysis
    ax_disp = fig.add_subplot(gs[0, 0])
    measured_disp = [r["displacement"] for r in results]
    ax_disp.plot(weights, measured_disp, "o-", label="Measured")

    # Theoretical curve: d ~ 0.25/w
    w_dense = np.linspace(0.01, 5.0, 100)
    d_theory_plot = 0.25 / w_dense
    d_theory_plot = np.minimum(d_theory_plot, 4.0)
    ax_disp.plot(w_dense, d_theory_plot, "r--", label="Theoretical (Approx)")

    # Standard weight marker
    ax_disp.axvline(0.1, color="gray", linestyle="--", alpha=0.7, label="Static Standard (w=0.1)")

    # Adaptive result
    ax_disp.plot(
        final_weight_adaptive,
        disp_adaptive,
        "r*",
        markersize=15,
        label=f"Adaptive (Final w={final_weight_adaptive:.3f})",
    )

    ax_disp.set_xscale("symlog", linthresh=0.01)
    ax_disp.set_xlabel("Anchor Weight")
    ax_disp.set_ylabel("Displacement (px)")
    ax_disp.set_title("Displacement vs Anchor Weight")
    ax_disp.grid(True)
    ax_disp.legend()

    # 2. Visual Inspection
    ax_vis = fig.add_subplot(gs[0, 1])
    ax_vis.imshow(image[0, 0], cmap="gray", origin="upper")

    # Plot Initial & Target
    init_c = initial_contour.numpy()
    init_c_closed = np.vstack([init_c, init_c[0]])
    t = np.linspace(0, 2 * np.pi, 100)
    target_x = target_radius * np.cos(t) + 50
    target_y = target_radius * np.sin(t) + 50

    ax_vis.plot(init_c_closed[:, 1], init_c_closed[:, 0], "r--", label="Initial", linewidth=1.5)
    ax_vis.plot(target_x, target_y, "g:", label="Target", linewidth=1.5)

    # Plot contours with colormap
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=0, vmax=len(weights) - 1)

    for i, res in enumerate(results):
        c = res["contour"]
        c_closed = np.vstack([c, c[0]])
        color = cmap(norm(i))
        # Only label a few to avoid clutter
        if res["weight"] == 0.1:
            label = f"w={res['weight']} (Std)"
        elif i in [0, 6]:
            label = f"w={res['weight']}"
        else:
            label = None
        ax_vis.plot(c_closed[:, 1], c_closed[:, 0], color=color, alpha=0.8, label=label)

    c_adapt = final_contour_adaptive.numpy()
    c_adapt_closed = np.vstack([c_adapt, c_adapt[0]])
    ax_vis.plot(c_adapt_closed[:, 1], c_adapt_closed[:, 0], "m-.", linewidth=2, label="Adaptive")

    ax_vis.set_title("Contour Evolution (Color=Weight)")
    ax_vis.legend(fontsize="small")
    ax_vis.set_xlim(0, 100)
    ax_vis.set_ylim(0, 100)

    # 3. Zoomed View
    ax_zoom = fig.add_subplot(gs[1, 0])
    ax_zoom.imshow(image[0, 0], cmap="gray", origin="upper")
    ax_zoom.plot(init_c_closed[:, 1], init_c_closed[:, 0], "r--", linewidth=2)
    ax_zoom.plot(target_x, target_y, "g:", linewidth=2)

    for i, res in enumerate(results):
        c = res["contour"]
        c_closed = np.vstack([c, c[0]])
        color = cmap(norm(i))
        ax_zoom.plot(c_closed[:, 1], c_closed[:, 0], color=color, alpha=0.8)

    # Zoom on top arc to show gap clearly
    # Center (50, 50). Top is at (50, 16) for target, (50, 20) for init.
    ax_zoom.set_xlim(35, 65)
    ax_zoom.set_ylim(5, 35)
    ax_zoom.set_title("Zoomed View (Top Arc)")

    # 4. Loss Magnitudes
    ax_loss = fig.add_subplot(gs[1, 1])

    data_losses = [r["losses"]["data_loss"] for r in results]
    anchor_losses = [r["losses"].get("contour_anchor_loss", 0.0) for r in results]

    # Add adaptive losses
    data_losses.append(final_losses_adaptive["data_loss"])
    anchor_losses.append(final_losses_adaptive.get("contour_anchor_loss", 0.0))

    x = np.arange(len(weights) + 1)
    width = 0.35

    rects1 = ax_loss.bar(x - width / 2, data_losses, width, label="Data Loss")
    rects2 = ax_loss.bar(x + width / 2, anchor_losses, width, label="Weighted Anchor Loss")

    ax_loss.set_ylabel("Loss Value")
    ax_loss.set_title("Final Loss Components")
    ax_loss.set_xticks(x)

    xticklabels = []
    for w in weights:
        xticklabels.append(f"{w} (Std)" if w == 0.1 else str(w))
    xticklabels.append("Adaptive")

    ax_loss.set_xticklabels(xticklabels)
    ax_loss.set_xlabel("Anchor Weight")
    ax_loss.legend()

    # Calculate theoretical values at measured points for comparison
    d_theory_points = []
    for w in weights:
        if w <= 1e-6:
            val = 4.0
        else:
            val = min(0.25 / w, 4.0)
        d_theory_points.append(val)

    plt.tight_layout()
    plt.show()


def analyze_bspline_anchor():
    """
    Analyzes the effect of anchor weight on B-Spline control points.

    Unlike the vertex refiner, the anchor force here is applied to the *control points*,
    while the data force is applied to the *evaluated contour*. The B-spline basis
    functions act as a transmission mechanism.

    We compare:
    - Contour Displacement: How far the curve moves.
    - Control Point Displacement: How far the parameters move.
    - Adaptive Regularization: How the dynamic weight compares to static baselines.
    """
    # Setup: Same as vertex analysis for comparison
    H, W = 100, 100
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    center = torch.tensor([50.0, 50.0])
    dist = torch.sqrt((y - center[0]) ** 2 + (x - center[1]) ** 2)
    target_radius = 34.0
    image = torch.exp(-((dist - target_radius) ** 2) / (2 * 2.0**2))
    image = image.unsqueeze(0).unsqueeze(0)

    theta = torch.linspace(0, 2 * np.pi, 100)[:-1]
    init_radius = 30.0
    initial_contour = torch.stack(
        [
            init_radius * torch.sin(theta) + center[0],
            init_radius * torch.cos(theta) + center[1],
        ],
        dim=1,
    )

    template = TemplateModelFactory.create("fixed", TemplateProps(sigma=2.0, peak_dist=0.0))

    weights = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    results = []

    print(f"Running B-Spline analysis for weights: {weights}")
    for w in weights:
        loss_weights = TANGENTIAL_SMOOTHING_BSPLINE.copy()
        loss_weights[RegularizerType.CONTOUR_ANCHOR.value] = w

        props = BSplineContourRefinerProps(
            num_steps=100,
            learning_rate=0.5,
            profile_length=21,
            num_sampled_profiles=len(initial_contour),
            num_control_points=16,  # Sparse control points
            initial_loss_weights=loss_weights,
        )
        refiner = BSplineContourRefiner(initial_contour.clone(), props, template)

        final_losses = {}
        for _ in tqdm(range(100), desc=f"BSpline w={w}", leave=False):
            final_losses = refiner.step(image)

        # Calculate displacements
        final_contour = refiner.contour.detach().cpu()
        final_radius = torch.norm(final_contour - center, dim=1).mean().item()
        contour_disp = final_radius - init_radius

        # Control point displacement
        final_cp = refiner.control_points.detach().cpu()
        init_cp = refiner.initial_control_points.detach().cpu()
        # Mean Euclidean distance of CPs from their start
        cp_disp = torch.norm(final_cp - init_cp, dim=1).mean().item()

        results.append({
            "weight": w,
            "contour_disp": contour_disp,
            "cp_disp": cp_disp,
            "contour": final_contour.numpy(),
            "control_points": final_cp.numpy(),
            "losses": final_losses,
        })

    # --- Run Adaptive Case for Comparison ---
    print("Running Adaptive B-Spline analysis...")
    loss_weights_adaptive = TANGENTIAL_SMOOTHING_BSPLINE.copy()
    loss_weights_adaptive[RegularizerType.CONTOUR_ANCHOR.value] = 0.1

    props_adaptive = BSplineContourRefinerProps(
        num_steps=100,
        learning_rate=0.5,
        profile_length=21,
        num_sampled_profiles=len(initial_contour),
        num_control_points=16,
        initial_loss_weights=loss_weights_adaptive,
        adaptive_reg=AdaptiveRegularizationProps(enabled=True, update_interval=5, warmup_steps=10),
    )
    refiner_adaptive = BSplineContourRefiner(initial_contour.clone(), props_adaptive, template)
    final_losses_adaptive = {}
    for _ in range(100):
        final_losses_adaptive = refiner_adaptive.step(image)

    final_contour_adaptive = refiner_adaptive.contour.detach().cpu()
    final_radius_adaptive = torch.norm(final_contour_adaptive - center, dim=1).mean().item()
    disp_adaptive = final_radius_adaptive - init_radius
    final_weight_adaptive = refiner_adaptive.loss_fn.get_weight(
        RegularizerType.CONTOUR_ANCHOR
    ).item()

    # --- Visualization ---
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "B-Spline Refiner: Anchor Force on Control Points vs Contour Displacement", fontsize=16
    )
    gs = fig.add_gridspec(2, 2)

    # 1. Displacement Analysis (Contour vs CP)
    ax_disp = fig.add_subplot(gs[0, 0])
    meas_c_disp = [r["contour_disp"] for r in results]
    meas_cp_disp = [r["cp_disp"] for r in results]

    ax_disp.plot(weights, meas_c_disp, "o-", label="Contour Disp")
    ax_disp.plot(weights, meas_cp_disp, "s--", label="Control Point Disp")

    # Theoretical curve (same as vertex for reference)
    w_dense = np.linspace(0.01, 5.0, 100)
    d_theory_plot = 0.25 / w_dense
    d_theory_plot = np.minimum(d_theory_plot, 4.0)
    ax_disp.plot(w_dense, d_theory_plot, "r:", label="Theoretical (Vertex)")

    # Standard weight marker
    ax_disp.axvline(0.1, color="gray", linestyle="--", alpha=0.7, label="Static Standard (w=0.1)")

    # Adaptive result
    ax_disp.plot(
        final_weight_adaptive,
        disp_adaptive,
        "r*",
        markersize=15,
        label=f"Adaptive (Final w={final_weight_adaptive:.3f})",
    )

    ax_disp.set_xscale("symlog", linthresh=0.01)
    ax_disp.set_xlabel("Anchor Weight")
    ax_disp.set_ylabel("Displacement (px)")
    ax_disp.set_title("Displacement: Contour vs Control Points")
    ax_disp.grid(True)
    ax_disp.legend()

    # 2. Contour Evolution
    ax_vis = fig.add_subplot(gs[0, 1])
    ax_vis.imshow(image[0, 0], cmap="gray", origin="upper")

    # Target
    t = np.linspace(0, 2 * np.pi, 100)
    target_x = target_radius * np.cos(t) + 50
    target_y = target_radius * np.sin(t) + 50
    ax_vis.plot(target_x, target_y, "g:", label="Target", linewidth=1.5)

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=0, vmax=len(weights) - 1)

    for i, res in enumerate(results):
        c = res["contour"]
        c_closed = np.vstack([c, c[0]])
        color = cmap(norm(i))
        if res["weight"] == 0.1:
            label = f"w={res['weight']} (Std)"
        elif i in [0, 6]:
            label = f"w={res['weight']}"
        else:
            label = None
        ax_vis.plot(c_closed[:, 1], c_closed[:, 0], color=color, alpha=0.8, label=label)

    c_adapt = final_contour_adaptive.numpy()
    c_adapt_closed = np.vstack([c_adapt, c_adapt[0]])
    ax_vis.plot(c_adapt_closed[:, 1], c_adapt_closed[:, 0], "m-.", linewidth=2, label="Adaptive")

    ax_vis.set_title("Contour Evolution")
    ax_vis.legend(fontsize="small")
    ax_vis.set_xlim(0, 100)
    ax_vis.set_ylim(0, 100)

    # 3. Control Point Evolution (Zoomed)
    ax_cp = fig.add_subplot(gs[1, 0])
    ax_cp.imshow(image[0, 0], cmap="gray", origin="upper")

    for i, res in enumerate(results):
        cp = res["control_points"]
        # Close the loop for plotting
        cp_closed = np.vstack([cp, cp[0]])
        color = cmap(norm(i))
        ax_cp.plot(cp_closed[:, 1], cp_closed[:, 0], "o-", color=color, alpha=0.6, markersize=4)

    # Zoom on top arc
    ax_cp.set_xlim(30, 70)
    ax_cp.set_ylim(0, 40)
    ax_cp.set_title("Control Point Polygon Evolution (Zoomed)")

    # 4. Loss Components
    ax_loss = fig.add_subplot(gs[1, 1])
    data_losses = [r["losses"]["data_loss"] for r in results]
    anchor_losses = [r["losses"].get("contour_anchor_loss", 0.0) for r in results]

    # Add adaptive losses
    data_losses.append(final_losses_adaptive["data_loss"])
    anchor_losses.append(final_losses_adaptive.get("contour_anchor_loss", 0.0))

    x = np.arange(len(weights) + 1)
    width = 0.35
    ax_loss.bar(x - width / 2, data_losses, width, label="Data Loss")
    ax_loss.bar(x + width / 2, anchor_losses, width, label="Weighted Anchor Loss")
    ax_loss.set_xticks(x)

    xticklabels = []
    for w in weights:
        xticklabels.append(f"{w} (Std)" if w == 0.1 else str(w))
    xticklabels.append("Adaptive")

    ax_loss.set_xticklabels(xticklabels)
    ax_loss.set_title("Final Loss Components")
    ax_loss.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    analyze_vertex_anchor()
    analyze_bspline_anchor()
