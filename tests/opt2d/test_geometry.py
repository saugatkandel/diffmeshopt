import numpy as np
import pytest
import torch
from scipy.interpolate import BSpline

from diffmeshopt.opt2d.geometry import compute_normals, get_bspline_matrix


def test_compute_normals_circle():
    """Test normal computation on a perfect circle."""
    num_points = 100
    # Create a circle (y, x)
    # We use endpoint=False to have evenly spaced points on the closed loop
    theta = torch.linspace(0, 2 * np.pi, num_points, endpoint=False)
    r = 10.0

    # (y, x) convention
    y = r * torch.sin(theta)
    x = r * torch.cos(theta)
    contour = torch.stack([y, x], dim=1)

    normals = compute_normals(contour)

    # Check shape
    assert normals.shape == contour.shape

    # Check unit length
    norms = torch.norm(normals, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    # Check direction
    # For a circle defined as (r*sin(t), r*cos(t)), the tangent is (r*cos(t), -r*sin(t)).
    # The outward normal is (sin(t), cos(t)).
    # This matches the position vector normalized.
    pos_norm = contour / r

    # Dot product should be 1 (parallel) or -1 (anti-parallel)
    dot = (normals * pos_norm).sum(dim=1)

    # We expect them to be either all 1 or all -1.
    assert torch.allclose(dot.abs(), torch.ones_like(dot), atol=1e-5)

    # Check consistency (all pointing same way relative to surface)
    sign = dot.sign().mean()
    assert sign.abs() > 0.9  # Should be consistently 1 or -1


def test_bspline_matrix_partition_of_unity():
    """Test that B-spline basis functions sum to 1 at any point."""
    num_cp = 10
    num_eval = 50
    # Assuming default degree is 3
    try:
        M = get_bspline_matrix(num_cp, num_eval, degree=3)
    except TypeError:
        M = get_bspline_matrix(num_cp, num_eval)

    # Sum across control points for each evaluation point
    row_sums = M.sum(dim=1)

    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_bspline_matrix_against_scipy():
    """
    Test that matrix multiplication yields same results as scipy.interpolate.BSpline.
    This ensures mathematical correctness of the B-spline basis functions.
    """
    num_cp = 10
    num_eval = 100
    degree = 3

    # 1. Generate random control points
    np.random.seed(42)
    cp_np = np.random.randn(num_cp, 2)
    cp_torch = torch.from_numpy(cp_np).float()

    # 2. Evaluate using our matrix
    try:
        M = get_bspline_matrix(num_cp, num_eval, degree=degree)
    except TypeError:
        M = get_bspline_matrix(num_cp, num_eval)

    eval_torch = M @ cp_torch
    eval_np = eval_torch.numpy()

    # 3. Evaluate using scipy
    # Setup knots for periodic cubic spline on domain [0, num_cp]
    # Knots are [-3, -2, -1, 0, 1, ..., N, N+1, N+2, N+3]
    t = np.arange(-degree, num_cp + degree + 1)

    # Coefficients: wrap around for periodicity
    # Scipy BSpline requires coefficients to match knots length - degree - 1
    # len(t) = N + 2k + 1.
    # Required coeffs = N + k.
    # We append the first k control points to the end.
    c = np.vstack([cp_np, cp_np[:degree]])

    # Evaluation points
    # get_bspline_matrix typically evaluates on [0, num_cp] with endpoint=False
    u = np.linspace(0, num_cp, num_eval, endpoint=False)

    spl = BSpline(t, c, k=degree, extrapolate="periodic")
    scipy_eval = spl(u)

    # Check closeness
    assert np.allclose(eval_np, scipy_eval, atol=1e-4, rtol=1e-4)
