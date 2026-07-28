"""
Script to compare different template models on data with spatially varying parameters.

Experiment Design:
------------------
1. Data: Synthetic image with a bi-Gaussian profile where sigma varies sinusoidally.
2. Task: Recover the varying sigma profile while keeping the contour fixed.
   Here we lock the contour to isolate template learning capabilities.
3. Models: Fixed, Global, Per-Point, B-Spline.

Visualizations:
---------------
1. Sigma Profile: Learned sigma vs Ground Truth sigma along the contour.
2. Visual Map: Color-coded contour showing learned sigma values.
3. Loss Evolution: Convergence of total loss over optimization steps.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from diffmeshopt.opt2d.config import (
    AdaptiveRegularizationProps,
    BSplineTemplateProps,
    ContourRefinerProps,
    NeuralFieldTemplateProps,
    RegularizerConfig,
    RegularizerDefaults,
    RegularizerType,
    TemplateProps,
)
from diffmeshopt.opt2d.refiner import VertexContourRefiner
from diffmeshopt.opt2d.regularizer_recipes import TANGENTIAL_SMOOTHING_VERTEX
from diffmeshopt.opt2d.template import TemplateModelFactory


def get_sigma_profile(angle: torch.Tensor) -> torch.Tensor:
    """Defines the spatially varying sigma profile: 1.5 + 0.5 * sin(2*theta)."""
    return 1.5 + 0.5 * torch.sin(2 * angle)


def generate_varying_sigma_data(
    shape: tuple[int, int] = (100, 100),
    center: tuple[float, float] = (50.0, 50.0),
    radius: float = 40.0,
    peak_dist: float = 6.0,
):
    H, W = shape
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    center_t = torch.tensor(center)
    dist = torch.sqrt((y - center_t[0]) ** 2 + (x - center_t[1]) ** 2)

    # Angle for varying sigma
    angle = torch.atan2(y - center_t[0], x - center_t[1])

    # Sigma varies from 1.0 to 2.0
    sigma_map = get_sigma_profile(angle)

    # Image: Bi-Gaussian
    d = dist - radius
    image = torch.exp(-((d - peak_dist / 2) ** 2) / (2 * sigma_map**2)) + torch.exp(
        -((d + peak_dist / 2) ** 2) / (2 * sigma_map**2)
    )
    image = image.unsqueeze(0).unsqueeze(0)

    return image


def analyze_templates():
    # Configuration
    H, W = 100, 100
    center_coords = (50.0, 50.0)
    radius = 40.0
    peak_dist = 6.0

    image = generate_varying_sigma_data(
        shape=(H, W), center=center_coords, radius=radius, peak_dist=peak_dist
    )

    # Initial contour at correct location
    theta = torch.linspace(0, 2 * np.pi, 100)[:-1]
    center = torch.tensor(center_coords)
    contour = torch.stack(
        [
            radius * torch.sin(theta) + center[0],
            radius * torch.cos(theta) + center[1],
        ],
        dim=1,
    )

    # GT Sigma along contour
    # angle corresponds to theta (but check coordinate system: y,x vs sin,cos)
    # y = r sin, x = r cos -> atan2(y, x) = atan2(sin, cos) = theta
    gt_sigma = get_sigma_profile(theta)

    # We use a bi-Gaussian template to match the data.
    models = [
        ("fixed", "fixed", TemplateProps(sigma=1.5, peak_dist=peak_dist)),
        ("global", "global", TemplateProps(sigma=1.5, peak_dist=peak_dist)),
        ("per_point", "per_point", TemplateProps(sigma=1.5, peak_dist=peak_dist)),
        (
            "bspline",
            "bspline",
            BSplineTemplateProps(sigma=1.5, peak_dist=peak_dist, num_control_points=16),
        ),
        (
            "neural",
            "neural",
            NeuralFieldTemplateProps(sigma=1.5, peak_dist=peak_dist, hidden_dim=32),
        ),
    ]

    results = {}
    loss_histories = {}

    # Setup Adaptive Regularization Defaults
    # We want to allow the template parameters to vary, but not drift wildly.
    reg_defaults = RegularizerDefaults.get_defaults()

    # For B-Spline/PerPoint: Allow variation (smoothness cost should be ~1% of data cost)
    reg_defaults.regularizers[RegularizerType.SMOOTH_SIGMA] = RegularizerConfig(
        static_weight=0.1, target_ratio=0.01
    )
    reg_defaults.regularizers[RegularizerType.SMOOTH_PEAK_DIST] = RegularizerConfig(
        static_weight=0.1, target_ratio=0.01
    )

    # For Neural/Grid: Anchor to initialization (anchor cost ~1% of data cost)
    # This prevents the unconstrained MLP from drifting to degenerate solutions
    reg_defaults.regularizers[RegularizerType.ANCHOR_SIGMA] = RegularizerConfig(
        static_weight=0.1, target_ratio=0.01
    )

    for name, mode, t_props in models:
        print(f"Running {name} model...")

        # Use standard recipe but lock contour
        loss_weights = TANGENTIAL_SMOOTHING_VERTEX.copy()
        loss_weights[RegularizerType.CONTOUR_ANCHOR.value] = 100.0  # Lock contour

        # Start with weak regularization to allow learning
        loss_weights[RegularizerType.SMOOTH_SIGMA.value] = 0.01
        loss_weights[RegularizerType.ANCHOR_SIGMA.value] = 0.01

        # Lock contour to focus on template optimization
        props = ContourRefinerProps(
            num_steps=1000,
            learning_rate=0.1,
            profile_length=21,
            initial_loss_weights=loss_weights,
            adaptive_reg=AdaptiveRegularizationProps(
                enabled=True, update_interval=10, warmup_steps=50
            ),
            _reg_defaults=reg_defaults,
        )

        template_model = TemplateModelFactory.create(
            mode, t_props, num_vertices=len(contour), image_shape=image.shape[-2:]
        )
        refiner = VertexContourRefiner(contour.clone(), props, template_model)

        history = []
        for _ in tqdm(range(1000), desc=name, leave=False):
            losses = refiner.step(image)
            history.append(losses["total_loss"])
        loss_histories[name] = history

        # Extract learned sigma
        with torch.no_grad():
            params = refiner.template_model.get_params(coordinates=refiner.contour)
            # params["sigma1"] is (N,) or scalar
            s = params["sigma1"]
            if s.ndim == 0:
                s = s.expand(len(contour))
            results[name] = s.cpu().numpy()

    # Plot
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Template Model Comparison: Spatially Varying Sigma", fontsize=16)
    gs = fig.add_gridspec(2, 2)

    # 1. Sigma Profile
    ax_prof = fig.add_subplot(gs[0, 0])
    ax_prof.plot(gt_sigma.numpy(), "k--", label="Ground Truth", linewidth=2)

    for name, res in results.items():
        ax_prof.plot(res, label=name)

    ax_prof.set_xlabel("Contour Index")
    ax_prof.set_ylabel("Sigma")
    ax_prof.set_title("Learned Sigma Profile")
    ax_prof.legend()
    ax_prof.grid(True)

    # 2. Loss Evolution
    ax_loss = fig.add_subplot(gs[1, 0])
    for name, history in loss_histories.items():
        ax_loss.plot(history, label=name)
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Total Loss")
    ax_loss.set_title("Convergence")
    ax_loss.legend()
    ax_loss.grid(True)

    # 3. Visual Map (Best Model - B-Spline or Per-Point)
    ax_vis = fig.add_subplot(gs[:, 1])
    ax_vis.imshow(image[0, 0], cmap="gray", origin="upper")

    # Visualize the B-Spline result as it's usually the best trade-off
    sigma_vis = results["bspline"]
    sc = ax_vis.scatter(contour[:, 1], contour[:, 0], c=sigma_vis, cmap="viridis", s=20)
    plt.colorbar(sc, ax=ax_vis, label="Learned Sigma")
    ax_vis.set_title("Learned Sigma Map (B-Spline Model)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    analyze_templates()
