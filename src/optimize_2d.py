from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import make_splprep


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


def sample_profiles(image, contour, normals, num_samples=21, sample_step=1.0, width=3):
    """
    Sample intensity profiles from the image along the normals.
    Averages intensity across a rectangle of specified width along the tangent.
    image: (1, 1, H, W) tensor
    contour: (N, 2) tensor (y, x)
    normals: (N, 2) tensor
    """
    N = contour.shape[0]
    K = num_samples

    # Compute tangents from normals: (ny, nx) -> (nx, -ny)
    # normals is (y, x)
    tangents = torch.stack([normals[:, 1], -normals[:, 0]], dim=-1)

    # Offsets: [-K//2, ..., K//2]
    offsets = torch.linspace(-(K // 2), K // 2, K, device=contour.device) * sample_step

    # Tangent offsets for width averaging
    # width=3 -> [-1, 0, 1]
    tangent_offsets = torch.linspace(
        -(width - 1) / 2, (width - 1) / 2, width, device=contour.device
    )

    # Calculate sample points: p = v + offset * n
    # Shape: (N, K, 2)
    sample_points = contour.unsqueeze(1) + normals.unsqueeze(1) * offsets.view(1, K, 1)
    # Calculate sample points: p = v + offset_n * n + offset_t * t
    # Shape: (N, K, W, 2)
    sample_points = (
        contour.view(N, 1, 1, 2)
        + normals.view(N, 1, 1, 2) * offsets.view(1, K, 1, 1)
        + tangents.view(N, 1, 1, 2) * tangent_offsets.view(1, 1, width, 1)
    )

    # Normalize coordinates to [-1, 1] for grid_sample
    # Image shape is (H, W). grid_sample expects (x, y).
    # Our contour is (row, col) -> (y, x).
    H, W = image.shape[-2:]

    sample_points_norm = sample_points.clone()
    # x = col = index 1, y = row = index 0
    sample_points_norm[..., 0] = (sample_points[..., 1] / (W - 1)) * 2 - 1  # x
    sample_points_norm[..., 1] = (sample_points[..., 0] / (H - 1)) * 2 - 1  # y

    # grid_sample expects (B, C, H_in, W_in) input and (B, H_out, W_out, 2) grid
    # We reshape grid to (1, 1, N*K, 2) to sample all points at once
    grid = sample_points_norm.view(1, 1, -1, 2)

    # Sample
    samples = F.grid_sample(image, grid, align_corners=True, padding_mode="border")

    # Output is (1, 1, 1, N*K*W) -> reshape to (N, K, W) and mean over W
    return samples.view(N, K, width).mean(dim=2)


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
