import pytest
import torch

from diffmeshopt.opt2d.sampling import _get_stratified_indices, sample_profiles


@pytest.fixture
def simple_image_and_contour():
    """Create a simple image and a contour for testing."""
    # A 20x20 image with a vertical bright line at x=10
    image = torch.zeros(1, 1, 20, 20, dtype=torch.float32)
    image[:, :, :, 10] = 1.0

    # A small square contour straddling the line
    contour = torch.tensor(
        [
            [9.0, 9.0],
            [11.0, 9.0],
            [11.0, 11.0],
            [9.0, 11.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    return image, contour


def test_get_stratified_indices():
    """Unit test for the stratified sampling logic."""
    num_vertices = 100
    num_samples = 10
    indices = _get_stratified_indices(num_vertices, num_samples)

    assert len(indices) == num_samples
    assert len(set(indices.tolist())) == num_samples  # unique indices

    # Check if indices are from different strata
    stratum_size = num_vertices / num_samples
    for i in range(num_samples):
        assert i * stratum_size <= indices[i] < (i + 1) * stratum_size


def test_sample_profiles_shape_and_mask(simple_image_and_contour):
    """Unit test for the output shape and validity mask of sample_profiles."""
    image, contour = simple_image_and_contour

    profiles, valid_mask = sample_profiles(
        image, contour, profile_length=5, profile_width=1, sample_step=1.0
    )

    assert profiles.shape == (contour.shape[0], 5)
    assert valid_mask.shape == (contour.shape[0],)
    assert torch.all(valid_mask)


def test_sample_profiles_out_of_bounds(simple_image_and_contour):
    """Unit test that out-of-bounds sampling is correctly masked."""
    image, _ = simple_image_and_contour
    contour_edge = torch.tensor(
        [
            [1.0, 10.0],  # Normal points left, will go off-image
            [19.0, 10.0],  # Normal points right, will go off-image
            [10.0, 1.0],  # Normal points up, will go off-image
            [10.0, 19.0],  # Normal points down, will go off-image
        ],
        dtype=torch.float32,
    )

    _, valid_mask = sample_profiles(
        image, contour_edge, profile_length=5, profile_width=1, sample_step=1.0
    )

    # All vertices should lead to invalid profiles
    assert not torch.any(valid_mask)
