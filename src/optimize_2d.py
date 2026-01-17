import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import make_splprep

from src.loss_2d import ContourLoss
from src.props_2d import OptimizationProps, SamplingProps, TemplateProps


def compute_normals(contour: torch.Tensor | np.ndarray, neighbor_shift: int = 1) -> torch.Tensor:
    """
    Compute normals for a 2D closed contour.
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

    This function performs the core differentiable sampling. For each vertex, it
    generates a grid of sampling points along its normal and tangent, then uses
    bilinear interpolation (`grid_sample`) to get image intensities.

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


def sample_profiles_stochastic(
    image: torch.Tensor,
    contour: torch.Tensor,
    sampling_props: SamplingProps | None = None,
    batch_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Samples profiles for a pseudo-uniformly distributed random subset of contour vertices.
    The subset of vertices is treated as a new, coarser contour, and normals are
    calculated on this coarse contour for stability.
    """
    if sampling_props is None:
        sampling_props = SamplingProps()

    bs = batch_size if batch_size is not None else sampling_props.batch_size

    # 1. Select a subset of indices that are spaced out along the contour
    sub_indices = _get_stratified_indices(contour.shape[0], bs, device=contour.device)

    # 2. Create the coarse contour from the selected vertices
    coarse_contour = contour[sub_indices]

    # 3. Compute normals on this new, smaller, coarse contour.
    # This provides stable normals because the baseline for the tangent is wider.
    coarse_normals = compute_normals(coarse_contour)

    # 4. Sample profiles at the locations of the coarse contour vertices using their normals
    profiles = sample_profiles(image, coarse_contour, coarse_normals, sampling_props)

    return profiles, sub_indices


def smooth_contour(
    contour_np: np.ndarray, num_points: int = 256, return_spline: bool = False
) -> np.ndarray:
    """
    Smooths and resamples a contour using B-splines.
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
        print(f"Warning: Spline smoothing failed ({e}). Using raw contour.")
        return contour_np


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

    # Indices of the control points
    i = torch.floor(u).long()
    t = u - i

    # Cubic B-spline basis functions
    b0 = (1 - t) ** 3 / 6
    b1 = (3 * t**3 - 6 * t**2 + 4) / 6
    b2 = (-3 * t**3 + 3 * t**2 + 3 * t + 1) / 6
    b3 = t**3 / 6

    # Construct dense matrix
    M = torch.zeros(num_samples, num_cp, device=device)

    # Handle wrapping indices for closed loop
    idx_0 = (i - 1) % num_cp
    idx_1 = i % num_cp
    idx_2 = (i + 1) % num_cp
    idx_3 = (i + 2) % num_cp

    rows = torch.arange(num_samples, device=device)
    M[rows, idx_0] += b0
    M[rows, idx_1] += b1
    M[rows, idx_2] += b2
    M[rows, idx_3] += b3

    return M


class ContourRefiner(nn.Module):
    def __init__(
        self,
        image: np.ndarray,
        initial_contour: np.ndarray,
        optimization_props: OptimizationProps = None,
        template_props: TemplateProps = None,
        sampling_props: SamplingProps | None = None,
        laplacian_window_size: int = 1,
        optimize_template: bool = False,
    ):
        super().__init__()
        self.register_buffer("image", torch.from_numpy(image).float())
        self.contour = nn.Parameter(torch.from_numpy(initial_contour).float())

        if optimization_props is None:
            optimization_props = OptimizationProps()

        if sampling_props is None:
            sampling_props = SamplingProps()

        if template_props is None:
            template_props = TemplateProps()

        self.optimization_props = optimization_props
        self.sampling_props = sampling_props
        self.template_props = template_props

        self.loss_fn = ContourLoss(
            optimization_props=self.optimization_props,
            template_props=self.template_props,
            laplacian_window_size=laplacian_window_size,
        )

        # Optimizer
        lr = self.optimization_props.lr
        self.optimizer = torch.optim.Adam([self.contour], lr=lr)
        self.optimize_template = optimize_template
        if optimize_template:
            # Use log trick to ensure positive parameters
            self.log_peak_dist = nn.Parameter(
                torch.log(torch.tensor(float(self.template_props.peak_dist)))
            )
            self.log_sigma = nn.Parameter(
                torch.log(torch.tensor(float(self.template_props.sigma)))
            )
            # Store initial log sigma for regularization (Gaussian prior)
            self.register_buffer("initial_log_sigma", self.log_sigma.detach().clone())
            self.optimizer = torch.optim.Adam(
                [self.contour, self.log_peak_dist, self.log_sigma], lr=lr
            )
        else:
            self.optimizer = torch.optim.Adam([self.contour], lr=lr)

    def _forward_propagate(self):
        # --- Data Loss (Stochastic) ---
        profiles, _ = sample_profiles_stochastic(
            self.image,
            self.contour,
            sampling_props=self.sampling_props,
        )

        peak_dist = None
        sigma = None
        log_sigma = None
        initial_log_sigma = None

        if self.optimize_template:
            peak_dist = torch.exp(self.log_peak_dist)
            sigma = torch.exp(self.log_sigma)
            log_sigma = self.log_sigma
            initial_log_sigma = self.initial_log_sigma

        losses = self.loss_fn(
            profiles,
            self.contour,
            optimize_template=self.optimize_template,
            peak_dist=peak_dist,
            sigma=sigma,
            log_sigma=log_sigma,
            initial_log_sigma=initial_log_sigma,
        )

        losses["total_loss"].backward()
        return losses

    def step(self) -> dict[str, float]:
        self.optimizer.zero_grad()

        losses = self._forward_propagate()
        self.optimizer.step()
        return {k: v.item() for k, v in losses.items()}

    def update_contour(self, contour_np: np.ndarray) -> None:
        """
        Updates the contour state in-place and clears gradients.
        Assumes the number of vertices remains constant.
        """
        with torch.no_grad():
            self.contour.copy_(torch.from_numpy(contour_np).float().to(self.contour.device))
            self.contour.grad = None


class BSplineContourRefiner(nn.Module):
    def __init__(
        self,
        image: np.ndarray,
        initial_contour: np.ndarray,
        optimization_props: OptimizationProps = None,
        template_props: TemplateProps = None,
        sampling_props: SamplingProps | None = None,
        num_control_points: int = 40,
        num_eval_points: int = 200,
        laplacian_window_size: int = 1,
        optimize_template: bool = False,
    ):
        super().__init__()
        self.register_buffer("image", torch.from_numpy(image).float())

        # 1. Fit initial control points to the initial contour
        # We create a temporary matrix for the initial contour length
        M_init = get_bspline_matrix(num_control_points, len(initial_contour))
        target = torch.from_numpy(initial_contour).float()
        # Solve linear system M_init @ P = V for P (control points)
        # P = (M^T M)^-1 M^T V
        # Using least squares
        initial_cp = torch.linalg.lstsq(M_init, target).solution

        self.control_points = nn.Parameter(initial_cp)

        # 2. Precompute evaluation matrix for the desired resolution
        self.register_buffer("M_eval", get_bspline_matrix(num_control_points, num_eval_points))

        if optimization_props is None:
            optimization_props = OptimizationProps()

        if sampling_props is None:
            sampling_props = SamplingProps()

        if template_props is None:
            template_props = TemplateProps()

        self.optimization_props = optimization_props
        self.sampling_props = sampling_props
        self.template_props = template_props

        self.loss_fn = ContourLoss(
            optimization_props=self.optimization_props,
            template_props=self.template_props,
            laplacian_window_size=laplacian_window_size,
        )

        # Optimizer
        lr = self.optimization_props.lr
        self.optimizer = torch.optim.Adam([self.control_points], lr=lr)
        self.optimize_template = optimize_template
        if optimize_template:
            # Use log trick to ensure positive parameters
            self.log_peak_dist = nn.Parameter(
                torch.log(torch.tensor(float(self.template_props.peak_dist)))
            )
            self.log_sigma = nn.Parameter(
                torch.log(torch.tensor(float(self.template_props.sigma)))
            )
            # Store initial log sigma for regularization (Gaussian prior)
            self.register_buffer("initial_log_sigma", self.log_sigma.detach().clone())
            self.optimizer = torch.optim.Adam(
                [self.control_points, self.log_peak_dist, self.log_sigma], lr=lr
            )
        else:
            self.optimizer = torch.optim.Adam([self.control_points], lr=lr)

    @property
    def contour(self):
        # Generate dense contour from control points
        return self.M_eval @ self.control_points

    def _forward_propagate(self):
        # --- Data Loss (Sampled from dense spline) ---
        # We sample from the generated smooth contour
        profiles, _ = sample_profiles_stochastic(
            self.image, self.contour, sampling_props=self.sampling_props
        )

        peak_dist = None
        sigma = None
        log_sigma = None
        initial_log_sigma = None

        if self.optimize_template:
            peak_dist = torch.exp(self.log_peak_dist)
            sigma = torch.exp(self.log_sigma)
            log_sigma = self.log_sigma
            initial_log_sigma = self.initial_log_sigma

        losses = self.loss_fn(
            profiles,
            self.control_points,
            optimize_template=self.optimize_template,
            peak_dist=peak_dist,
            sigma=sigma,
            log_sigma=log_sigma,
            initial_log_sigma=initial_log_sigma,
        )

        losses["total_loss"].backward()
        return losses

    def step(self) -> dict[str, float]:
        self.optimizer.zero_grad()
        losses = self._forward_propagate()
        self.optimizer.step()

        return {k: v.item() for k, v in losses.items()}
