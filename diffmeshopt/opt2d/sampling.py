import numpy as np
import torch
import torch.nn.functional as F

from diffmeshopt.opt2d.geometry import compute_normals


def _convert_to_tensor(array: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(array, np.ndarray):
        return torch.from_numpy(array).float()
    return array


def get_sampling_points(
    contour: torch.Tensor,
    normals: torch.Tensor,
    num_samples: int,
    sample_step: float,
    width: int,
) -> torch.Tensor:
    """
    Generates the coordinates for sampling profiles.
    Returns tensor of shape (N, num_samples, width, 2) in (y, x) pixel coordinates.
    """

    # Compute tangents from normals: (ny, nx) -> (nx, -ny)
    # normals is (y, x)
    tangents = torch.stack([normals[:, 1], -normals[:, 0]], dim=-1)

    # Offsets centered at 0. Using arange ensures step size is exactly sample_step.
    offsets = (
        torch.arange(num_samples, device=contour.device, dtype=torch.float32)
        - (num_samples - 1) / 2
    ) * sample_step

    # Tangent offsets for width averaging
    # width=3 -> [-1, 0, 1]
    tangent_offsets = torch.linspace(
        -(width - 1) / 2, (width - 1) / 2, width, device=contour.device
    )

    # Calculate sample points: p = v + offset_n * n + offset_t * t
    # Shape: (N, K, W, 2)
    # Use None indexing for cleaner broadcasting
    sample_points = (
        contour[:, None, None, :]
        + normals[:, None, None, :] * offsets[None, :, None, None]
        + tangents[:, None, None, :] * tangent_offsets[None, None, :, None]
    )

    return sample_points


def sample_at_points(image: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """
    Samples the image at the given points using bilinear interpolation.
    points: (..., 2) tensor of coordinates in (y, x) format.
    image: (B, C, H, W) tensor.
    Returns: (...) tensor of sampled intensities.
    """
    if image.ndim == 2:
        image = image[None, None, ...]
    elif image.ndim == 3:
        image = image[None, ...]

    # Normalize coordinates to [-1, 1] for grid_sample
    H, W = image.shape[-2:]

    # points is (..., 2) -> (y, x)
    # grid_sample expects (x, y)

    # Flatten to (1, 1, num_points, 2) for grid_sample
    original_shape = points.shape[:-1]
    num_points = points.numel() // 2
    flat_points = points.view(1, 1, num_points, 2)

    grid_x = (flat_points[..., 1] / (W - 1)) * 2 - 1
    grid_y = (flat_points[..., 0] / (H - 1)) * 2 - 1
    grid = torch.stack([grid_x, grid_y], dim=-1)

    # Sample
    samples = F.grid_sample(image, grid, align_corners=True, padding_mode="border")

    return samples.view(*original_shape).mean(dim=-1)


def sample_profiles(
    image: torch.Tensor,
    contour: torch.Tensor,
    profile_length: int,
    profile_width: int,
    sample_step: float = 1.0,
    normals: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample intensity profiles from the image along the normals.
    """
    if normals is None:
        normals = compute_normals(contour)
    points = get_sampling_points(contour, normals, profile_length, sample_step, profile_width)
    samples = sample_at_points(image, points)

    # Calculate validity mask
    # points: (N, K, width, 2) in (y, x) pixel coordinates
    H, W = image.shape[-2:]
    y = points[..., 0]
    x = points[..., 1]

    # Profile is valid only if all sample points are strictly within image bounds
    valid_mask = (y >= 0) & (y <= H - 1) & (x >= 0) & (x <= W - 1)
    valid_mask = valid_mask.all(dim=-1).all(dim=-1)  # (N,)

    return samples, valid_mask


def _get_stratified_indices(
    num_total: int, batch_size: int, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """
    Selects a batch of indices that are roughly evenly spaced.
    """
    if batch_size >= num_total:
        return torch.arange(num_total, device=device)

    step = num_total / batch_size

    # Stratified sampling: one random point per bin of size 'step'
    bin_starts = torch.arange(batch_size, device=device) * step
    jitter = torch.rand(batch_size, device=device) * step

    indices = bin_starts + jitter
    return (indices % num_total).long()


def sample_profiles_stochastic(
    image: torch.Tensor | np.ndarray,
    contour: torch.Tensor | np.ndarray,
    profile_length: int,
    profile_width: int,
    sample_step: float,
    num_samples: int,
    normals: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Samples profiles for a pseudo-uniformly distributed random subset of contour vertices.
    """

    # 1. Select a subset of indices that are spaced out along the contour
    sub_indices = _get_stratified_indices(contour.shape[0], num_samples, device=contour.device)

    # 2. Create the coarse contour from the selected vertices
    coarse_contour = contour[sub_indices]

    # 3. Compute normals on this new, smaller, coarse contour.
    if normals is not None:
        coarse_normals = normals[sub_indices]
    else:
        coarse_normals = compute_normals(coarse_contour)

    # 4. Sample profiles at the locations of the coarse contour vertices using their normals
    profiles, valid_mask = sample_profiles(
        image, coarse_contour, profile_length, profile_width, sample_step, coarse_normals
    )

    return profiles, sub_indices, valid_mask
