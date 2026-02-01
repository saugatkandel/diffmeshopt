import pytest
import torch
import torch.nn.functional as F

from diffmeshopt.opt2d.loss import (
    BiGaussianLoss,
    ContourLoss,
    EdgeLengthConsistencyLoss,
    LaplacianSmoothingLoss,
    NormalConsistencyLoss,
    TemplateProps,
)
from diffmeshopt.opt2d.props import RegularizerType


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


def test_tangential_laplacian_loss(simple_contour):
    """Test that tangential Laplacian loss penalizes bunching but not scaling."""
    loss_fn = LaplacianSmoothingLoss(window_size=1, mode="tangential")

    # Compute normals for the circle (pointing outwards)
    # For a circle centered at 0, normal is just normalized position
    normals = F.normalize(simple_contour, dim=-1)

    # 1. Perfect circle with uniform spacing -> Tangential Laplacian should be 0
    # (The Laplacian vector is purely radial, so projection onto tangent is 0)
    loss_uniform = loss_fn(simple_contour, normals=normals)
    assert torch.isclose(loss_uniform, torch.tensor(0.0), atol=1e-5)

    # 2. Radial expansion (Scaling) -> Should NOT increase tangential loss
    # Standard Laplacian loss would increase here, but tangential should not.
    scaled_contour = simple_contour * 1.5
    loss_scaled = loss_fn(scaled_contour, normals=normals)
    assert torch.isclose(loss_scaled, torch.tensor(0.0), atol=1e-5)

    # 3. Tangential perturbation (Bunching) -> Should increase loss
    bunched_contour = simple_contour.clone()
    # Move point 1 towards point 0 along the circle
    with torch.no_grad():
        # Interpolate between p0 and p1
        new_p1 = simple_contour[0] * 0.2 + simple_contour[1] * 0.8
        # Project back to radius 10 to ensure we only changed spacing, not shape
        bunched_contour[1] = F.normalize(new_p1, dim=0) * 10.0

    # We need to recompute normals for the bunched contour if we want to be exact,
    # but using the original radial normals is fine for testing the projection logic
    # on the vertices.
    loss_bunched = loss_fn(bunched_contour, normals=normals)
    assert loss_bunched > loss_uniform


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


def test_normal_consistency_loss():
    """Test that normal consistency loss penalizes angle variations."""
    loss_fn = NormalConsistencyLoss()

    # 1. Flat line (normals all [0, 1]) -> Loss should be 0
    normals_flat = torch.tensor([[0.0, 1.0]] * 10)
    loss_flat = loss_fn(normals_flat)
    assert torch.isclose(loss_flat, torch.tensor(0.0))

    # 2. Circle (normals rotate slowly) -> Loss small but > 0
    theta = torch.linspace(0, 2 * torch.pi, 11)[:-1]
    normals_circle = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    loss_circle = loss_fn(normals_circle)
    assert loss_circle > 0

    # 3. ZigZag (normals flip 180 deg) -> Loss high
    normals_zigzag = torch.tensor([[0.0, 1.0], [0.0, -1.0]] * 5)
    loss_zigzag = loss_fn(normals_zigzag)
    # Dot product is -1, so 1 - (-1) = 2
    assert torch.isclose(loss_zigzag, torch.tensor(2.0))


def test_bigaussian_loss():
    """Test the BiGaussian cross-correlation loss."""
    peak_dist, sigma, amplitude = 5.0, 1.0, 1.0
    props = TemplateProps(peak_dist=peak_dist, sigma=sigma, amp=amplitude)
    loss_fn = BiGaussianLoss(template_props=props, num_samples=21, sample_step=1.0)

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


def test_laplacian_smoothing_loss_with_edges():
    """Test Graph Laplacian calculation with explicit edges."""
    # 3 points in a line: p0=(-1,0), p1=(0,0), p2=(1,0)
    points = torch.tensor([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])

    # Bidirectional edges: 0<->1, 1<->2
    # Note: LaplacianSmoothingLoss uses index_add_, so we define directed edges for neighbors
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)

    loss_fn = LaplacianSmoothingLoss()

    # L[0] = p0 - p1 = (-1, 0) -> norm sq = 1
    # L[1] = p1 - (p0+p2)/2 = 0 - 0 = 0 -> norm sq = 0
    # L[2] = p2 - p1 = (1, 0) -> norm sq = 1
    # Mean loss = (1 + 0 + 1) / 3 = 2/3

    loss = loss_fn(points, edges=edges)
    assert torch.isclose(loss, torch.tensor(2.0 / 3.0))


def test_edge_length_consistency_loss_with_edges():
    """Test edge length loss with explicit edges (Graph mode)."""
    # Triangle: (0,0), (3,0), (0,4). Edges lengths: 3, 5, 4.
    points = torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    # Edges: 0->1, 1->2, 2->0
    edges = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

    loss_fn = EdgeLengthConsistencyLoss()
    loss = loss_fn(points, edges=edges)

    lengths = torch.tensor([3.0, 5.0, 4.0])
    mean_l = lengths.mean()
    expected_loss = ((lengths - mean_l) ** 2).mean() / (mean_l**2 + 1e-8)

    assert torch.isclose(loss, expected_loss)


def test_normal_consistency_loss_with_edges():
    """Test normal consistency loss with explicit edges."""
    # Two normals pointing in opposite directions
    normals = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    edges = torch.tensor([[0], [1]], dtype=torch.long)  # Edge 0->1

    loss_fn = NormalConsistencyLoss()
    loss = loss_fn(normals, edges=edges)
    # Dot product is -1. Loss = 1 - (-1) = 2.
    assert torch.isclose(loss, torch.tensor(2.0))


def test_contour_loss_dynamic_architecture():
    """
    Test that ContourLoss correctly implements the dynamic regularizer architecture.
    Verifies:
    1. Buffers are created for all RegularizerType entries.
    2. initial_weights correctly override defaults.
    3. get_weight/set_weight API works.
    4. Forward pass returns all expected loss keys.
    """
    # 1. Test Initialization & Buffer Registration
    initial_weights = {
        RegularizerType.TANGENTIAL_LAPLACIAN.value: 5.0,
        "normal_consistency": 2.0,
    }
    # Use small num_samples for simpler dummy data
    loss_fn = ContourLoss(initial_weights=initial_weights, num_samples=10)

    for reg_type in RegularizerType:
        # Check buffer existence
        buffer_name = f"w_{reg_type.value}"
        assert hasattr(loss_fn, buffer_name), f"Missing buffer for {reg_type.name}"

        # Check value initialization
        weight = loss_fn.get_weight(reg_type).item()
        if reg_type == RegularizerType.TANGENTIAL_LAPLACIAN:
            assert weight == 5.0
        elif reg_type == RegularizerType.NORMAL_CONSISTENCY:
            assert weight == 2.0
        else:
            assert weight == 0.0

    # 2. Test API
    loss_fn.set_weight(RegularizerType.EDGE_LENGTH, 1.5)
    assert loss_fn.get_weight("edge_length").item() == 1.5

    # 3. Test Forward Pass Structure
    # Create dummy inputs matching num_samples=10
    profiles = torch.randn(1, 10)
    points = torch.randn(10, 2)

    # Run forward
    results = loss_fn(profiles, points)

    # Verify output structure
    assert "total_loss" in results
    assert "data_loss" in results

    for reg_type in RegularizerType:
        key = f"{reg_type.value}_loss"
        assert key in results, f"Missing return key: {key}"
        # Since we set EDGE_LENGTH to 1.5 and points are random, loss should be > 0
        if reg_type == RegularizerType.EDGE_LENGTH:
            assert results[key].item() > 0
