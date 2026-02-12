"""
Script to analyze the Data Force (gradient of correlation) vs Distance.

Experiment Design:
------------------
1. Create a synthetic image with a Gaussian ring at a fixed target radius.
2. Move a probe point across the image, passing through the target.
3. At each position, sample an intensity profile and compute the BiGaussianLoss.
4. Compute the gradient of the loss with respect to position (-dL/dr).
   This gradient represents the "Data Force" pulling the contour.

Expected Results:
-----------------
1. Zero Force at Peak: At the exact target radius, the correlation is maximized (loss minimized),
   so the gradient is zero.
2. Restoring Force: Slightly away from the target, the force should point towards the target
   (positive if r < target, negative if r > target).
3. Peak Force: The force magnitude peaks roughly at +/- sqrt(2) * sigma from the target.
4. Basin of Attraction: Beyond +/- 3 sigma, the Gaussian tail is negligible.
   The correlation becomes constant (near zero) and the gradient (force) vanishes.
   This defines the "capture range" of the optimization.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from diffmeshopt.opt2d.loss import BiGaussianLoss
from diffmeshopt.opt2d.sampling import sample_at_points
from diffmeshopt.opt2d.template import TemplateProps


def analyze_force_dropoff():
    # 1. Setup Image: Gaussian Ring at radius 50
    H, W = 100, 100
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    center = torch.tensor([50.0, 50.0])
    dist_map = torch.sqrt((y - center[0]) ** 2 + (x - center[1]) ** 2)

    target_radius = 30.0
    sigma_image = 2.0
    image = torch.exp(-((dist_map - target_radius) ** 2) / (2 * sigma_image**2))
    image = image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # 2. Setup Probe Points at various distances
    # We move a point from radius 10 to 50 (crossing the target at 30)
    radii = np.linspace(10.0, 50.0, 200)
    forces = []
    correlations = []

    # Template matches image
    template_props = TemplateProps(sigma=sigma_image, peak_dist=0.0)
    # Use small profile length to avoid boundary issues during this test
    loss_fn = BiGaussianLoss(template_props, num_samples=21, sample_step=1.0)

    print("Analyzing force dropoff...")
    for r in radii:
        # Place a single point at radius r
        # Position: (50, 50+r) -> purely horizontal offset for simplicity
        pos = torch.tensor([[50.0, 50.0 + r]], dtype=torch.float32, requires_grad=True)

        # Normal: (0, 1) (pointing outwards)
        normal = torch.tensor([[0.0, 1.0]], dtype=torch.float32)

        # Generate sample points for profile
        # shape: (1, num_samples, 1, 2)
        # offsets: -10 to +10
        offsets = (torch.arange(21) - 10).float()

        # sample_points = pos + normal * offset
        # Reshaping is critical here:
        # Must be (1, 21, 1, 2) so sample_at_points returns (1, 21) after averaging last dim
        sample_points = pos.view(1, 1, 1, 2) + normal.view(1, 1, 1, 2) * offsets.view(1, -1, 1, 1)

        # Sample image
        # sample_at_points expects (..., 2)
        profiles = sample_at_points(image, sample_points)  # (1, 21)

        # Check for valid signal to avoid "ringing" at the tail
        # When the profile is flat (far from target), std is ~0.
        # Normalizing by ~0 causes numerical instability in gradients.
        if profiles.std() < 1e-4:
            forces.append(0.0)
            correlations.append(0.0)
        else:
            # Compute Loss
            loss = loss_fn(profiles)

            # Compute Gradient (Force)
            loss.backward()

            # Force = -Gradient (Gradient points uphill on loss surface, we want downhill)
            # grad is dL/dr.
            # If r < 50, moving +r decreases loss. So dL/dr < 0. Force > 0.
            grad = pos.grad[0, 1].item()
            force = -grad

            forces.append(force)
            correlations.append(1.0 - loss.item())

    # Theoretical Calculations (Numerical)
    # We compute the exact gradient of the loss function on ideal synthetic profiles
    # to account for normalization and finite window effects.
    theory_forces = []
    d_vals = radii - target_radius
    x_coords = loss_fn.x  # (21,)

    for d in d_vals:
        # Shift parameter: d = r - target
        # Profile samples at (r + x). Image peak is at target.
        # Intensity ~ exp(-((r + x - target)**2) / 2sigma^2)
        #             exp(-((x + d)**2) / 2sigma^2)
        shift = torch.tensor(float(d), requires_grad=True)

        # Generate ideal profile
        profile = torch.exp(-((x_coords + shift) ** 2) / (2 * sigma_image**2))
        profile = profile.unsqueeze(0)  # (1, L)

        # Compute loss
        loss = loss_fn(profile)
        loss.backward()

        # Force = -dL/dr = -dL/d(shift)
        theory_forces.append(-shift.grad.item())

    theory_forces = np.array(theory_forces)

    # Extract peaks from theoretical curve
    peak_idx = np.argmax(theory_forces)
    peak_offset = abs(d_vals[peak_idx])
    peak_magnitude = theory_forces[peak_idx]

    # Force at 3 sigma (Basin edge)
    dropoff_dist = 3.0 * sigma_image
    idx_dropoff = np.argmin(np.abs(np.abs(d_vals) - dropoff_dist))
    dropoff_magnitude = abs(theory_forces[idx_dropoff])

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel("Radius (px)")
    ax1.set_ylabel("Data Force (-dL/dr)", color="tab:blue")
    ax1.plot(
        radii,
        forces,
        color="tab:blue",
        label="Force (Measured)",
        linewidth=2,
        marker="o",
        markevery=10,
        markersize=4,
    )
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.plot(
        radii,
        theory_forces,
        color="tab:blue",
        linestyle=":",
        alpha=0.8,
        label="Force (Theory)",
        linewidth=2,
    )
    ax1.axhline(0, color="grey", linestyle="--", linewidth=0.5)
    ax1.axvline(target_radius, color="black", linestyle=":", label="Target")

    # Expected peak force locations
    ax1.axvline(
        target_radius - peak_offset,
        color="tab:red",
        linestyle="--",
        alpha=0.5,
        label=f"Peak Loc ({peak_offset:.1f}px)",
    )
    ax1.axvline(target_radius + peak_offset, color="tab:red", linestyle="--", alpha=0.5)

    # Expected peak force magnitude
    ax1.axhline(
        peak_magnitude,
        color="tab:purple",
        linestyle=":",
        alpha=0.5,
        label=f"Peak Mag ({peak_magnitude:.2f})",
    )
    ax1.axhline(-peak_magnitude, color="tab:purple", linestyle=":", alpha=0.5)

    # Expected dropoff magnitude (at 3 sigma)
    ax1.axhline(
        dropoff_magnitude,
        color="tab:brown",
        linestyle=":",
        alpha=0.5,
        label=f"Dropoff Mag ({dropoff_magnitude:.2f})",
    )
    ax1.axhline(-dropoff_magnitude, color="tab:brown", linestyle=":", alpha=0.5)

    # Basin of attraction visualization
    # Force is non-zero roughly within +/- 3 sigma
    ax1.axvspan(
        target_radius - 3 * sigma_image,
        target_radius + 3 * sigma_image,
        color="green",
        alpha=0.1,
        label="Basin (3σ)",
    )

    # Secondary axis for Correlation
    ax2 = ax1.twinx()
    ax2.set_ylabel("Correlation", color="tab:orange")
    ax2.plot(
        radii,
        correlations,
        color="tab:orange",
        linestyle="--",
        label="Correlation (Measured)",
        marker="x",
        markevery=10,
        markersize=4,
    )
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize="small")

    plt.title(f"Data Force vs Distance (Sigma={sigma_image})")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    analyze_force_dropoff()
