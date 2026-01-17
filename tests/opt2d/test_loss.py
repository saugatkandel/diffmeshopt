import torch

from diffmeshopt.opt2d.loss import (
    BiGaussianLoss,
    EdgeLengthConsistencyLoss,
    LaplacianSmoothingLoss,
)


def test_bi_gaussian_loss(props):
    """Test BiGaussianLoss computation and gradients."""
    _, _, temp_props = props
    loss_fn = BiGaussianLoss(temp_props)

    # Create dummy profiles (batch_size=10, len=20)
    profiles = torch.randn(10, temp_props.num_samples, requires_grad=True)

    loss = loss_fn(profiles)

    # Check output is scalar
    assert loss.ndim == 0
    assert not torch.isnan(loss)

    # Check gradients
    loss.backward()
    assert profiles.grad is not None


def test_bi_gaussian_loss_with_params(props):
    """Test BiGaussianLoss with per-point parameters."""
    _, _, temp_props = props
    loss_fn = BiGaussianLoss(temp_props)

    batch_size = 10
    profiles = torch.randn(batch_size, temp_props.num_samples, requires_grad=True)
    peak_dist = torch.full((batch_size,), temp_props.peak_dist, requires_grad=True)
    sigma = torch.full((batch_size,), temp_props.sigma, requires_grad=True)

    loss = loss_fn(profiles, peak_dist, sigma)

    assert loss.ndim == 0
    assert not torch.isnan(loss)

    loss.backward()
    assert profiles.grad is not None
    assert peak_dist.grad is not None
    assert sigma.grad is not None


def test_geometric_losses():
    """Test Laplacian and Edge Length consistency losses."""
    laplacian_loss = LaplacianSmoothingLoss(window_size=1)
    edge_loss = EdgeLengthConsistencyLoss()

    # Dummy contour (N=50, 2)
    contour = torch.randn(50, 2, requires_grad=True)

    l_loss = laplacian_loss(contour)
    e_loss = edge_loss(contour)

    assert l_loss.ndim == 0
    assert e_loss.ndim == 0

    # Check gradients
    (l_loss + e_loss).backward()
    assert contour.grad is not None
