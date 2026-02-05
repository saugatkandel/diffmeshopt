import numpy as np
import pytest
import torch

from diffmeshopt.opt2d.config import (
    BSplineContourRefinerProps,
    ContourRefinerProps,
    RBFContourRefinerProps,
    RegularizerType,
    TemplateProps,
)
from diffmeshopt.opt2d.refiner import (
    BSplineContourRefiner,
    ContourRefiner,
    RBFContourRefiner,
    TangentialSmoothingContourRefiner,
)
from diffmeshopt.opt2d.template import TemplateModelFactory


def test_optimization_convergence():
    """
    A numerical integration test to verify the optimizer moves a contour towards a known target.
    Target: A bright vertical double-ridge centered at x=50.
    Initial contour: A vertical line at x=40.
    """
    H, W = 100, 100
    x = torch.arange(W, dtype=torch.float32)
    y = torch.arange(H, dtype=torch.float32)
    xv, yv = torch.meshgrid(x, y, indexing="xy")

    # Use a wider sigma to ensure a good basin of attraction for the optimizer
    sigma = 2.0
    # Create image with a BiGaussian ridge centered at x=50
    # Peaks at 45 and 55 (dist=10).
    image = torch.exp(-((xv - 45) ** 2) / (2 * sigma**2)) + torch.exp(
        -((xv - 55) ** 2) / (2 * sigma**2)
    )
    image = (image - image.min()) / (image.max() - image.min())
    image = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims

    # Initial contour: vertical line at x=45 (closer to 50 to avoid local minima)
    num_points = 10
    ys = torch.linspace(10, 90, num_points)
    xs = torch.full_like(ys, 48.0)
    initial_contour = torch.stack([ys, xs], axis=1)  # (y, x) format

    # Use the new properties system
    props = ContourRefinerProps(
        num_steps=200,
        learning_rate=0.5,
        initial_loss_weights={
            RegularizerType.CONTOUR_LAPLACIAN.value: 0.0,
            RegularizerType.EDGE_LENGTH.value: 0.0,
            RegularizerType.TANGENTIAL_LAPLACIAN.value: 0.0,
            RegularizerType.NORMAL_CONSISTENCY.value: 0.0,
            RegularizerType.ANCHOR_SIGMA.value: 0.0,
            RegularizerType.ANCHOR_PEAK_DIST.value: 0.0,
            RegularizerType.SMOOTH_SIGMA.value: 0.0,
            RegularizerType.SMOOTH_PEAK_DIST.value: 0.0,
            RegularizerType.TEMPLATE_SHAPE.value: 0.0,
        },
        # Sampling props
        profile_length=21,
        profile_width=1,
        num_sampled_profiles=num_points,  # Sample all points
    )
    template_props = TemplateProps(peak_dist=10.0, sigma=sigma)
    template_model = TemplateModelFactory.create("fixed", props=template_props)

    refiner = ContourRefiner(
        initial_contour=initial_contour,
        props=props,
        template_model=template_model,
    )

    # Run optimization
    history = list(refiner.refine(image))
    final_contour = history[-1]["contour"]
    final_x = final_contour.detach().cpu().numpy()[:, 1].mean()  # Contour is (y, x)

    # Should have moved towards 50 (from 45)
    assert final_x > 45.0
    assert np.abs(final_x - 50.0) < 1.0, f"Final x {final_x} not close to target 50.0"


def test_tangential_smoothing_convergence():
    """
    Verify TangentialSmoothingContourRefiner converges to a target.
    """
    H, W = 100, 100
    x = torch.arange(W, dtype=torch.float32)
    y = torch.arange(H, dtype=torch.float32)
    xv, yv = torch.meshgrid(x, y, indexing="xy")

    sigma = 2.0
    # Create image with a BiGaussian ridge centered at x=50
    image = torch.exp(-((xv - 45) ** 2) / (2 * sigma**2)) + torch.exp(
        -((xv - 55) ** 2) / (2 * sigma**2)
    )
    image = (image - image.min()) / (image.max() - image.min())
    image = image.unsqueeze(0).unsqueeze(0)

    # Initial contour: vertical line at x=45
    num_points = 10
    ys = torch.linspace(10, 90, num_points)
    xs = torch.full_like(ys, 48.0)
    initial_contour = torch.stack([ys, xs], axis=1)

    props = ContourRefinerProps(
        num_steps=200,
        learning_rate=0.5,
        initial_loss_weights={
            RegularizerType.TANGENTIAL_LAPLACIAN.value: 1.0,
            RegularizerType.NORMAL_CONSISTENCY.value: 0.1,
            RegularizerType.CONTOUR_LAPLACIAN.value: 0.0,
            RegularizerType.EDGE_LENGTH.value: 0.0,
            RegularizerType.ANCHOR_SIGMA.value: 0.0,
            RegularizerType.ANCHOR_PEAK_DIST.value: 0.0,
            RegularizerType.SMOOTH_SIGMA.value: 0.0,
            RegularizerType.SMOOTH_PEAK_DIST.value: 0.0,
            RegularizerType.TEMPLATE_SHAPE.value: 0.0,
        },
        profile_length=21,
        num_sampled_profiles=num_points,
    )
    template_model = TemplateModelFactory.create(
        "fixed", props=TemplateProps(peak_dist=10.0, sigma=sigma)
    )

    refiner = TangentialSmoothingContourRefiner(
        initial_contour=initial_contour, props=props, template_model=template_model
    )

    history = list(refiner.refine(image))
    final_x = history[-1]["contour"].detach().cpu().numpy()[:, 1].mean()
    assert np.abs(final_x - 50.0) < 1.0, f"Final x {final_x} not close to target 50.0"


def test_bspline_convergence():
    """
    Verify BSplineContourRefiner converges to a target.
    Target: A bright vertical double-ridge centered at x=50.
    Initial contour: A vertical line at x=48.
    """
    H, W = 100, 100
    x = torch.arange(W, dtype=torch.float32)
    y = torch.arange(H, dtype=torch.float32)
    xv, yv = torch.meshgrid(x, y, indexing="xy")

    sigma = 2.0
    # Create image with a BiGaussian ridge centered at x=50
    image = torch.exp(-((xv - 45) ** 2) / (2 * sigma**2)) + torch.exp(
        -((xv - 55) ** 2) / (2 * sigma**2)
    )
    image = (image - image.min()) / (image.max() - image.min())
    image = image.unsqueeze(0).unsqueeze(0)

    # Initial contour: vertical line at x=48
    num_points = 20
    ys = torch.linspace(10, 90, num_points)
    xs = torch.full_like(ys, 48.0)
    initial_contour = torch.stack([ys, xs], axis=1)

    props = BSplineContourRefinerProps(
        num_steps=200,
        learning_rate=0.5,
        initial_loss_weights={
            RegularizerType.CONTOUR_LAPLACIAN.value: 0.0,
            RegularizerType.EDGE_LENGTH.value: 0.0,
            RegularizerType.TANGENTIAL_LAPLACIAN.value: 0.0,
            RegularizerType.NORMAL_CONSISTENCY.value: 0.0,
            RegularizerType.ANCHOR_SIGMA.value: 0.0,
            RegularizerType.ANCHOR_PEAK_DIST.value: 0.0,
            RegularizerType.SMOOTH_SIGMA.value: 0.0,
            RegularizerType.SMOOTH_PEAK_DIST.value: 0.0,
            RegularizerType.TEMPLATE_SHAPE.value: 0.0,
        },
        profile_length=21,
        num_sampled_profiles=num_points,
        contour_num_control_points=10,
    )

    template_props = TemplateProps(peak_dist=10.0, sigma=sigma)
    template_model = TemplateModelFactory.create("fixed", props=template_props)

    refiner = BSplineContourRefiner(initial_contour, props, template_model)
    history = list(refiner.refine(image))
    final_x = history[-1]["contour"].detach().cpu().numpy()[:, 1].mean()
    assert np.abs(final_x - 50.0) < 1.0, f"Final x {final_x} not close to target 50.0"


def test_rbf_convergence():
    """
    Verify RBFContourRefiner converges to a target.
    Target: A bright vertical double-ridge centered at x=50.
    Initial contour: A vertical line at x=48.
    """
    H, W = 100, 100
    x = torch.arange(W, dtype=torch.float32)
    y = torch.arange(H, dtype=torch.float32)
    xv, yv = torch.meshgrid(x, y, indexing="xy")

    sigma = 2.0
    # Create image with a BiGaussian ridge centered at x=50
    image = torch.exp(-((xv - 45) ** 2) / (2 * sigma**2)) + torch.exp(
        -((xv - 55) ** 2) / (2 * sigma**2)
    )
    image = (image - image.min()) / (image.max() - image.min())
    image = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims

    # Initial contour: vertical line at x=48
    num_points = 20
    ys = torch.linspace(10, 90, num_points)
    xs = torch.full_like(ys, 48.0)
    initial_contour = torch.stack([ys, xs], axis=1)  # (y, x) format

    # RBF Props
    props = RBFContourRefinerProps(
        num_steps=200,
        learning_rate=0.5,
        initial_loss_weights={
            # Disable other regularizers to isolate convergence capability
            RegularizerType.CONTOUR_LAPLACIAN.value: 0.0,
            RegularizerType.EDGE_LENGTH.value: 0.0,
            RegularizerType.TANGENTIAL_LAPLACIAN.value: 0.0,
            RegularizerType.NORMAL_CONSISTENCY.value: 0.0,
            RegularizerType.ANCHOR_SIGMA.value: 0.0,
            RegularizerType.ANCHOR_PEAK_DIST.value: 0.0,
            RegularizerType.SMOOTH_SIGMA.value: 0.0,
            RegularizerType.SMOOTH_PEAK_DIST.value: 0.0,
            RegularizerType.TEMPLATE_SHAPE.value: 0.0,
        },
        profile_length=21,
        num_sampled_profiles=num_points,
        rbf_num_control_points=10,
        rbf_kernel_sigma=20.0,
    )

    template_props = TemplateProps(peak_dist=10.0, sigma=sigma)
    template_model = TemplateModelFactory.create("fixed", props=template_props)

    refiner = RBFContourRefiner(initial_contour, props, template_model)
    history = list(refiner.refine(image))
    final_x = history[-1]["contour"].detach().cpu().numpy()[:, 1].mean()
    assert np.abs(final_x - 50.0) < 1.0, f"Final x {final_x} not close to target 50.0"
