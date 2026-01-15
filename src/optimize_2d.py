from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import make_splprep

from src.loss_2d import (BiGaussianLoss, EdgeLengthConsistencyLoss,
                         LaplacianSmoothingLoss)


def compute_normals(contour):
    """
    Compute normals for a 2D closed contour.
    contour: (N, 2)
    """
    # Central differences
    v_next = torch.roll(contour, shifts=-1, dims=0)
    v_prev = torch.roll(contour, shifts=1, dims=0)
    tangents = v_next - v_prev
    tangents = F.normalize(tangents, dim=-1)

    # Rotate 90 degrees: (x, y) -> (-y, x)
    # contour is (y, x), so tangent is (dy, dx).
    # normal should be (-dx, dy)
    normals = torch.stack([-tangents[:, 1], tangents[:, 0]], dim=-1)
    return normals


def get_sampling_grid(contour, normals, num_samples=21, sample_step=1.0, width=3):
    """
    Generates the coordinates for sampling profiles.
    Returns tensor of shape (N, num_samples, width, 2) in (y, x) pixel coordinates.
    """
    N = contour.shape[0]
    K = num_samples

    # Compute tangents from normals: (ny, nx) -> (nx, -ny)
    # normals is (y, x)
    tangents = torch.stack([normals[:, 1], -normals[:, 0]], dim=-1)

    # Offsets centered at 0. Using arange ensures step size is exactly sample_step.
    offsets = (torch.arange(K, device=contour.device, dtype=torch.float32) - (K - 1) / 2) * sample_step

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


def sample_at_points(image, points):
    """
    Samples the image at the given points using bilinear interpolation.
    points: (..., 2) tensor of coordinates in (y, x) format.
    image: (B, C, H, W) tensor.
    Returns: (...) tensor of sampled intensities.
    """
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

    return samples.view(*original_shape)


def sample_profiles(image, contour, normals, num_samples=21, sample_step=1.0, width=3):
    """
    Sample intensity profiles from the image along the normals.

    This function performs the core differentiable sampling. For each vertex, it
    generates a grid of sampling points along its normal and tangent, then uses
    bilinear interpolation (`grid_sample`) to get image intensities.

    The `sample_step` is in pixel units. A value of 1.0 means we create a profile
    where each point is 1 pixel apart in the image. Sub-pixel accuracy during
    optimization is achieved because the vertex coordinates are continuous, and
    `grid_sample` provides gradients with respect to these continuous coordinates.

    The `width` parameter averages samples along the tangent to reduce noise and
    make the profile more robust.
    """
    grid = get_sampling_grid(contour, normals, num_samples, sample_step, width)
    samples = sample_at_points(image, grid)
    return samples.mean(dim=-1)


def _get_stratified_indices(num_total, batch_size, device):
    """
    Selects a batch of indices that are roughly evenly spaced.
    This is a form of stratified sampling on a circular contour, similar in
    spirit to 1D Poisson disk sampling for ensuring spread.
    """
    if batch_size >= num_total:
        return torch.arange(num_total, device=device)

    step = num_total / batch_size
    
    # Stratified sampling: one random point per bin of size 'step'
    bin_starts = torch.arange(batch_size, device=device) * step
    jitter = torch.rand(batch_size, device=device) * step
    
    indices = bin_starts + jitter
    return (indices % num_total).long()


def sample_profiles_stochastic(image, contour, batch_size, num_samples=21, sample_step=1.0, width=3):
    """
    Samples profiles for a pseudo-uniformly distributed random subset of contour vertices.
    The subset of vertices is treated as a new, coarser contour, and normals are
    calculated on this coarse contour for stability.
    """
    # 1. Select a subset of indices that are spaced out along the contour
    sub_indices = _get_stratified_indices(contour.shape[0], batch_size, device=contour.device)

    # 2. Create the coarse contour from the selected vertices
    coarse_contour = contour[sub_indices]

    # 3. Compute normals on this new, smaller, coarse contour.
    # This provides stable normals because the baseline for the tangent is wider.
    coarse_normals = compute_normals(coarse_contour)

    # 4. Sample profiles at the locations of the coarse contour vertices using their normals
    profiles = sample_profiles(
        image, coarse_contour, coarse_normals, num_samples, sample_step, width
    )

    return profiles, sub_indices


def smooth_contour(contour_np, num_points=256):
    """
    Smooths and resamples a contour using B-splines.
    """
    try:
        if not np.allclose(contour_np[0], contour_np[-1]):
            contour_closed = np.vstack([contour_np, contour_np[0]])
        else:
            contour_closed = contour_np

        diffs = np.diff(contour_closed, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        u = np.concatenate([[0], np.cumsum(dists)])
        u = u / u[-1]

        spl = make_splprep(contour_closed.T, u=u, s=len(contour_closed))
        u_new = np.linspace(0, 1, num_points, endpoint=False)
        return spl(u_new).T.astype(np.float32)
    except Exception as e:
        print(f"Warning: Spline smoothing failed ({e}). Using raw contour.")
        return contour_np


class ContourRefiner(nn.Module):
    def __init__(
        self,
        initial_contour,
        lr=1.0,
        w_data=1.0,
        w_laplacian=0.1,
        w_edge=0.1,
    ):
        super().__init__()
        self.contour = nn.Parameter(torch.from_numpy(initial_contour).float())

        self.w_data = w_data
        self.w_laplacian = w_laplacian
        self.w_edge = w_edge

        # Loss Functions
        self.data_loss_fn = BiGaussianLoss()
        self.laplacian_loss_fn = LaplacianSmoothingLoss()
        self.edge_loss_fn = EdgeLengthConsistencyLoss()

        # Optimizer
        self.optimizer = torch.optim.Adam([self.contour], lr=lr)

    def step(self, image, batch_size):
        self.optimizer.zero_grad()

        # --- Data Loss (Stochastic) ---
        profiles, _ = sample_profiles_stochastic(
            image, self.contour, batch_size=batch_size
        )
        data_loss = self.data_loss_fn(profiles)

        # --- Regularization Losses (Global) ---
        laplacian_loss = self.laplacian_loss_fn(self.contour)
        edge_loss = self.edge_loss_fn(self.contour)

        # --- Total Loss ---
        total_loss = (
            self.w_data * data_loss
            + self.w_laplacian * laplacian_loss
            + self.w_edge * edge_loss
        )

        total_loss.backward()
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "data_loss": data_loss.item(),
            "laplacian_loss": laplacian_loss.item(),
            "edge_loss": edge_loss.item(),
        }
