import numpy as np
import pytest
import torch

from diffmeshopt.opt2d.optimize import ContourRefiner
from diffmeshopt.opt2d.props import OptimizationProps, SamplingProps, TemplateProps


def test_optimization_convergence():
    """
    Test that the optimizer actually moves a contour towards a target.
    Target: A bright vertical double-ridge at x=50.
    Initial contour: A vertical line at x=40.
    """
    H, W = 100, 100
    x = np.arange(W)
    y = np.arange(H)
    xv, yv = np.meshgrid(x, y)

    # Create image with a BiGaussian ridge centered at x=50
    # Peaks at 45 and 55 (dist=10).
    image = np.exp(-((xv - 45) ** 2) / (2 * 2**2)) + np.exp(-((xv - 55) ** 2) / (2 * 2**2))
    image = image.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min())

    # Initial contour: vertical line at x=40
    # (y, x)
    num_points = 10
    ys = np.linspace(10, 90, num_points)
    xs = np.full_like(ys, 40.0)
    initial_contour = np.stack([ys, xs], axis=1)

    opt_props = OptimizationProps(lr=1.0, w_data=1.0, w_laplacian=0.0, w_edge=0.0)
    samp_props = SamplingProps(num_samples=21, sample_step=1.0, width=1, batch_size=num_points)
    temp_props = TemplateProps(peak_dist=10.0, sigma=2.0, num_samples=21)

    refiner = ContourRefiner(
        image=image,
        initial_contour=initial_contour,
        optimization_props=opt_props,
        sampling_props=samp_props,
        template_props=temp_props,
        optimize_template=False,
    )

    # Run optimization
    for _ in range(20):
        refiner.step()

    final_x = refiner.contour.detach().cpu().numpy()[:, 1].mean()

    # Should have moved towards 50 (from 40)
    assert final_x > 40.0
    # Should be close to 50
    assert np.abs(final_x - 50.0) < 2.0
