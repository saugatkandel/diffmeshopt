import torch
import torch.nn.functional as F

from diffmeshopt.opt2d.geometry import compute_normals
from diffmeshopt.opt2d.props import SamplingProps


def get_sampling_points(
    contour: torch.Tensor,
    normals: torch.Tensor,
    sampling_props: SamplingProps | None = None,
) -> torch.Tensor:
    """
    Generates the coordinates for sampling profiles.
    Returns tensor of shape (N, num_samples, width, 2) in (y, x) pixel coordinates.
    """
    if sampling_props is None:
        sampling_props = SamplingProps()

    # N = contour.shape[0]
    K = sampling_props.num_samples
    sample_step = sampling_props.sample_step
    width = sampling_props.width

    # Compute tangents from normals: (ny, nx) -> (nx, -ny)
    # normals is (y, x)
    tangents = torch.stack([normals[:, 1], -normals[:, 0]], dim=-1)

    # Offsets centered at 0. Using arange ensures step size is exactly sample_step.
    offsets = (
        torch.arange(K, device=contour.device, dtype=torch.float32) - (K - 1) / 2
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
    normals: torch.Tensor,
    sampling_props: SamplingProps | None = None,
) -> torch.Tensor:
    """
    Sample intensity profiles from the image along the normals.
    """
    if sampling_props is None:
        sampling_props = SamplingProps()

    points = get_sampling_points(contour, normals, sampling_props)
    samples = sample_at_points(image, points)
    return samples


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
    image: torch.Tensor,
    contour: torch.Tensor,
    sampling_props: SamplingProps | None = None,
    batch_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Samples profiles for a pseudo-uniformly distributed random subset of contour vertices.
    """
    if sampling_props is None:
        sampling_props = SamplingProps()

    bs = batch_size if batch_size is not None else sampling_props.batch_size

    # 1. Select a subset of indices that are spaced out along the contour
    sub_indices = _get_stratified_indices(contour.shape[0], bs, device=contour.device)

    # 2. Create the coarse contour from the selected vertices
    coarse_contour = contour[sub_indices]

    # 3. Compute normals on this new, smaller, coarse contour.
    coarse_normals = compute_normals(coarse_contour)

    # 4. Sample profiles at the locations of the coarse contour vertices using their normals
    profiles = sample_profiles(image, coarse_contour, coarse_normals, sampling_props)

    return profiles, sub_indices
