"""
Script to visualize the RBF deformation field on the surrounding space.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from diffmeshopt.opt2d.config import RBFContourRefinerProps, TemplateProps
from diffmeshopt.opt2d.refiner import RBFContourRefiner
from diffmeshopt.opt2d.template import TemplateModelFactory


def visualize_field():
    # 1. Setup RBF Refiner
    # Circle contour
    theta = torch.linspace(0, 2 * np.pi, 50)[:-1]
    radius = 30.0
    center = torch.tensor([50.0, 50.0])
    initial_contour = torch.stack(
        [radius * torch.sin(theta) + center[0], radius * torch.cos(theta) + center[1]], dim=1
    )

    props = RBFContourRefinerProps(
        rbf_num_control_points=8,
        rbf_kernel_sigma=20.0,  # Large sigma for global influence
    )
    template = TemplateModelFactory.create("fixed", TemplateProps())
    refiner = RBFContourRefiner(initial_contour, props, template)

    # 2. Apply manual deformation
    # Push the first control point (top of circle) upwards
    with torch.no_grad():
        # Find control point nearest to top
        cp = refiner.control_points
        # Top is roughly (80, 50) in (row, col) if center is (50,50) and radius 30?
        # sin(0)=0, cos(0)=1 -> (50, 80). Wait, stack is (sin, cos).
        # row = 30*sin + 50, col = 30*cos + 50.
        # We want to move a point. Let's just move index 0.
        refiner.rbf_weights[0] = torch.tensor([20.0, 0.0])  # Move +20 in row (y)
        refiner.rbf_weights[1] = torch.tensor([0.0, 20.0])  # Move +20 in col (x)

    # 3. Create a dense grid to visualize the field
    H, W = 100, 100
    y = torch.linspace(0, H, 20)
    x = torch.linspace(0, W, 20)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    grid_points = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1)

    # 4. Compute deformation
    with torch.no_grad():
        displacement = refiner.compute_deformation(grid_points)
        deformed_points = grid_points + displacement

    # 5. Plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot Grid
    gp = grid_points.numpy()
    dp = deformed_points.numpy()

    # Plot displacement vectors on grid
    ax.quiver(
        gp[:, 1],
        gp[:, 0],
        displacement[:, 1].numpy(),
        displacement[:, 0].numpy(),
        color="gray",
        alpha=0.5,
        scale=1,
        scale_units="xy",
        angles="xy",
    )

    # Plot deformed grid points
    ax.scatter(dp[:, 1], dp[:, 0], c="blue", s=5, alpha=0.5, label="Deformed Grid")

    # Plot Contour
    ic = initial_contour.numpy()
    fc = refiner.contour.detach().numpy()
    ax.plot(ic[:, 1], ic[:, 0], "k--", label="Initial Contour")
    ax.plot(fc[:, 1], fc[:, 0], "r-", linewidth=2, label="Deformed Contour")

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.invert_yaxis()
    ax.legend()
    ax.set_title("RBF Deformation Field Influence")
    plt.show()


if __name__ == "__main__":
    visualize_field()
