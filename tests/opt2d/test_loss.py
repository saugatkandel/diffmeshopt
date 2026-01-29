import pytest
import torch

from diffmeshopt.opt2d.loss import (
    BiGaussianLoss,
    EdgeLengthConsistencyLoss,
    LaplacianSmoothingLoss,
    TemplateProps,
)


@pytest.fixture
def simple_contour():
    """A simple contour for testing geometric losses."""
    """A simple 16-point circular contour for testing geometric losses."""
    theta = torch.linspace(0, 2 * torch.pi, 17, dtype=torch.float32)[:-1]  # 16 points
    radius = 10.0
    contour = torch.stack([radius * torch.cos(theta), radius * torch.sin(theta)], dim=1)
    contour.requires_grad = True
    return contour


def test_laplacian_smoothing_loss(simple_contour):
    """Test that Laplacian loss penalizes non-smoothness and has correct gradient."""
    # 1. A smooth circular contour has non-zero curvature, so Laplacian is non-zero but small/stable
    loss_fn = LaplacianSmoothingLoss(window_size=3)
    loss_smooth = loss_fn(simple_contour)
    assert loss_smooth > 0

    # 2. Create a perturbed version
    perturbed_contour = simple_contour.clone()
    with torch.no_grad():
        # Move one point outwards along its radius vector (which is the x-axis for theta=0)
        perturbed_contour.data[0] *= 1.2

    # 3. Assert loss on perturbed is > loss on smooth
    loss_perturbed = loss_fn(perturbed_contour)
    assert loss_perturbed > loss_smooth

    # 4. Check gradient direction on the perturbed point
    # We need a fresh tensor for gradient checking
    perturbed_contour_grad = perturbed_contour.detach().clone()
    perturbed_contour_grad.requires_grad = True
    loss_for_grad = loss_fn(perturbed_contour_grad)
    loss_for_grad.backward()

    assert perturbed_contour_grad.grad is not None
    # The gradient points in the direction of steepest ascent (increasing loss).
    # p0 was at (10, 0) and was moved to (12, 0).
    # Moving the point further out increases the Laplacian loss, so the gradient is positive (outwards).
    # The gradient vector should be approximately in the (+1, 0) direction.
    grad_p0 = perturbed_contour_grad.grad[0]
    assert grad_p0[0] > 0  # x-component should be positive
    assert torch.isclose(grad_p0[1], torch.tensor(0.0), atol=1e-4)  # y-component should be ~0


def test_edge_length_consistency_loss(simple_contour):
    """Test that edge length loss penalizes variance in edge lengths."""
    loss_fn = EdgeLengthConsistencyLoss()

    # For a perfect circle, all edges have similar lengths. Variance should be near 0.
    loss = loss_fn(simple_contour)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)

    # Test with non-uniform edges by perturbing a point
    non_uniform_contour = simple_contour.clone()
    with torch.no_grad():
        non_uniform_contour.data[1] *= 1.5

    loss_non_uniform = loss_fn(non_uniform_contour)
    assert loss_non_uniform > loss


def test_bigaussian_loss():
    """Test the BiGaussian cross-correlation loss."""
    peak_dist, sigma, amplitude = 5.0, 1.0, 1.0
    props = TemplateProps(peak_dist=peak_dist, sigma=sigma, amp=amplitude)
    loss_fn = BiGaussianLoss(template_props=props, num_samples=21)

    # Create a perfect bi-gaussian profile
    profile_coords = torch.linspace(-10, 10, 21)

    gauss1 = torch.exp(-0.5 * ((profile_coords - peak_dist / 2) / sigma) ** 2)
    gauss2 = torch.exp(-0.5 * ((profile_coords + peak_dist / 2) / sigma) ** 2)
    template_profile = amplitude * (gauss1 + gauss2)
    template_profile = template_profile.unsqueeze(0)  # Batch of 1

    # Case 1: Perfect match -> loss should be 0
    loss_match = loss_fn(template_profile.clone())
    assert torch.isclose(loss_match, torch.tensor(0.0), atol=1e-6)

    # Case 2: No correlation (flat signal) -> loss should be 1
    loss_no_corr = loss_fn(torch.ones_like(template_profile))
    assert torch.isclose(loss_no_corr, torch.tensor(1.0))

    # Case 3: Anti-correlated -> loss should be 2
    loss_anti_corr = loss_fn(-template_profile.clone())
    assert torch.isclose(loss_anti_corr, torch.tensor(2.0), atol=1e-6)
