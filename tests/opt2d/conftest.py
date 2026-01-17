import numpy as np
import pytest
import torch

from diffmeshopt.opt2d.props import OptimizationProps, SamplingProps, TemplateProps


@pytest.fixture
def synthetic_data():
    """Creates a synthetic image and a circular contour."""
    # Create a simple image (gradient)
    H, W = 100, 100
    image = torch.zeros((1, 1, H, W), dtype=torch.float32)
    for i in range(H):
        image[0, 0, i, :] = i / H

    # Create a simple circular contour
    num_points = 50
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    r = 30
    cx, cy = 50, 50
    # Shape (N, 2) -> (row, col) i.e., (y, x)
    contour = np.stack([cy + r * np.sin(theta), cx + r * np.cos(theta)], axis=1)

    return image, contour


@pytest.fixture
def props():
    """Creates default property objects for testing."""
    opt_props = OptimizationProps(
        lr=0.1,
        w_data=1.0,
        w_laplacian=0.1,
        w_edge=0.1,
        w_sigma_reg=1.0,
        w_template_shape=0.1,
    )
    samp_props = SamplingProps(num_samples=20, sample_step=1.0, width=1, batch_size=32)
    temp_props = TemplateProps(peak_dist=10.0, sigma=2.0, num_samples=20)
    return opt_props, samp_props, temp_props
