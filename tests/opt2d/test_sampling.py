import torch

from diffmeshopt.opt2d.geometry import compute_normals
from diffmeshopt.opt2d.props import SamplingProps
from diffmeshopt.opt2d.sampling import sample_profiles, sample_profiles_stochastic


def test_sampling_shapes(synthetic_data, props):
    """Test profile sampling returns correct shapes."""
    image, contour_np = synthetic_data
    _, samp_props, _ = props

    contour = torch.from_numpy(contour_np).float()
    normals = compute_normals(contour)

    profiles = sample_profiles(image, contour, normals, samp_props)

    # Expected shape: (N, num_samples)
    assert profiles.shape == (len(contour), samp_props.num_samples)


def test_stochastic_sampling(synthetic_data, props):
    """Test stochastic sampling returns correct batch size."""
    image, contour_np = synthetic_data
    _, samp_props, _ = props

    contour = torch.from_numpy(contour_np).float()
    batch_size = 10

    profiles, indices = sample_profiles_stochastic(
        image, contour, samp_props, batch_size=batch_size
    )

    assert profiles.shape == (batch_size, samp_props.num_samples)
    assert indices.shape == (batch_size,)
    # Check indices are within range
    assert (indices >= 0).all() and (indices < len(contour)).all()


def test_sampling_values():
    """Test that sampling returns correct values from a known image."""
    # Create a 10x10 image where pixel value equals x coordinate
    H, W = 10, 10
    # Shape (1, 1, H, W)
    x_grid = torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W)
    image = x_grid.view(1, 1, H, W)

    # Define a vertical line contour at x=5
    # Points: (2, 5), (3, 5), ... (7, 5)
    # (y, x) format
    num_points = 5
    y_coords = torch.arange(2, 7, dtype=torch.float32)
    x_coords = torch.full_like(y_coords, 5.0)
    contour = torch.stack([y_coords, x_coords], dim=1)

    # Normals pointing right: (0, 1)
    normals = torch.zeros_like(contour)
    normals[:, 1] = 1.0

    # Sampling props
    # sample_step = 1.0, num_samples = 3, width = 1
    # Samples should be at x = 5-1, 5, 5+1 => 4, 5, 6
    samp_props = SamplingProps(num_samples=3, sample_step=1.0, width=1)

    profiles = sample_profiles(image, contour, normals, samp_props)

    # Expected profiles: [4, 5, 6] for all points
    expected = torch.tensor([4.0, 5.0, 6.0]).view(1, 3).expand(num_points, 3)

    assert torch.allclose(profiles, expected, atol=1e-5)
