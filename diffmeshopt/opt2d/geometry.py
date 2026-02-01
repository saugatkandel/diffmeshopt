import logging

import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import make_splprep


def compute_normals(contour: torch.Tensor | np.ndarray, neighbor_shift: int = 1) -> torch.Tensor:
    """
    Compute normals for a 2D closed contour using central differences.

    Assumes the contour is a closed loop (cyclic), so the last point connects to the first.
    contour: (N, 2)
    """
    if isinstance(contour, np.ndarray):
        contour = torch.from_numpy(contour).float()

    # Central differences
    v_next = torch.roll(contour, shifts=-neighbor_shift, dims=0)
    v_prev = torch.roll(contour, shifts=neighbor_shift, dims=0)
    tangents = v_next - v_prev
    tangents = F.normalize(tangents, dim=-1)

    # Rotate 90 degrees: (x, y) -> (-y, x)
    # contour is (y, x), so tangent is (dy, dx).
    # normal should be (-dx, dy)
    normals = torch.stack([-tangents[:, 1], tangents[:, 0]], dim=-1)
    return normals


def smooth_contour(
    contour_np: np.ndarray, num_points: int = 256, return_spline: bool = False
) -> np.ndarray:
    """
    Smooths and resamples a contour using B-splines.

    Returns a closed loop with unique vertices (start point not repeated).
    """
    try:
        if not np.allclose(contour_np[0], contour_np[-1]):
            contour_closed = np.vstack([contour_np, contour_np[0]])
        else:
            contour_closed = contour_np

        # make_splprep returns (spl, u)
        spl, _ = make_splprep(contour_closed.T, s=len(contour_closed))

        u_new = np.linspace(0, 1, num_points, endpoint=False)
        if return_spline:
            return spl(u_new).T.astype(np.float32), spl

        return spl(u_new).T.astype(np.float32)
    except Exception as e:
        logging.warning(f"Spline smoothing failed ({e}). Using raw contour.")
        return contour_np


def compute_cubic_bspline_weights(
    u: torch.Tensor, num_cp: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes weights and indices for cubic B-spline interpolation.
    u: Tensor of parameter values in [0, num_cp].
    Returns:
        indices: (..., 4) LongTensor of control point indices.
        weights: (..., 4) FloatTensor of basis function values.
    """
    i = torch.floor(u).long()
    t = u - i

    # Cubic B-spline basis functions
    b0 = (1 - t) ** 3 / 6
    b1 = (3 * t**3 - 6 * t**2 + 4) / 6
    b2 = (-3 * t**3 + 3 * t**2 + 3 * t + 1) / 6
    b3 = t**3 / 6

    # Indices of the 4 control points (wrapped)
    idx_0 = (i - 1) % num_cp
    idx_1 = i % num_cp
    idx_2 = (i + 1) % num_cp
    idx_3 = (i + 2) % num_cp

    indices = torch.stack([idx_0, idx_1, idx_2, idx_3], dim=-1)
    weights = torch.stack([b0, b1, b2, b3], dim=-1)

    return indices, weights


def get_bspline_matrix(
    num_cp: int, num_samples: int, device: torch.device = "cpu"
) -> torch.Tensor:
    """
    Creates a matrix to evaluate a closed cubic B-spline at uniform intervals.
    Resulting matrix M is (num_samples, num_cp).
    Contour = M @ ControlPoints
    """
    # u runs from 0 to num_cp (periodic)
    u = torch.linspace(0, num_cp, num_samples + 1, device=device)[:-1]

    indices, weights = compute_cubic_bspline_weights(u, num_cp)

    # Construct dense matrix
    M = torch.zeros(num_samples, num_cp, device=device)
    rows = torch.arange(num_samples, device=device)

    for k in range(4):
        M[rows, indices[:, k]] += weights[:, k]

    return M


def get_bspline_derivative_matrix(
    num_control_points: int, num_eval_points: int, device: torch.device = "cpu"
) -> torch.Tensor:
    """
    Constructs the matrix M such that M @ control_points = tangents.
    Assumes uniform cubic B-splines and closed loop.
    """
    # u runs from 0 to num_control_points
    u = torch.linspace(0, num_control_points, num_eval_points + 1, device=device)[:-1]

    i = torch.floor(u).long()
    t = u - i

    # Derivatives of cubic B-spline basis functions
    db0 = -0.5 * (1 - t) ** 2
    db1 = 0.5 * (3 * t**2 - 4 * t)
    db2 = 0.5 * (-3 * t**2 + 2 * t + 1)
    db3 = 0.5 * t**2

    M_deriv = torch.zeros((num_eval_points, num_control_points), device=device)

    # Indices of the 4 control points (wrapped)
    idx_0 = (i - 1) % num_control_points
    idx_1 = i % num_control_points
    idx_2 = (i + 1) % num_control_points
    idx_3 = (i + 2) % num_control_points

    rows = torch.arange(num_eval_points, device=device)

    M_deriv[rows, idx_0] += db0
    M_deriv[rows, idx_1] += db1
    M_deriv[rows, idx_2] += db2
    M_deriv[rows, idx_3] += db3

    return M_deriv
