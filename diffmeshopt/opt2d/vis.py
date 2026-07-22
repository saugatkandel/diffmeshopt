from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes

import diffmeshopt.opt2d.debug as debug_module
import diffmeshopt.opt2d.geometry as geometry
import diffmeshopt.opt2d.sampling as sampling
from diffmeshopt.opt2d.config import ContourRefinerProps, TemplateProps
from diffmeshopt.opt2d.geometry import get_bspline_matrix
from diffmeshopt.opt2d.loss import BiGaussianBaseLoss


def plot_prior_and_landscape_from_contour(
    image: np.ndarray,
    contour: np.ndarray,
    refiner_props: ContourRefinerProps | None = None,
    template_props: TemplateProps | None = None,
):
    if template_props is None:
        template_props = TemplateProps()

    if refiner_props is None:
        refiner_props = ContourRefinerProps()

    sample_profiles, _, _ = sampling.sample_profiles_stochastic(
        torch.from_numpy(image).float(),
        torch.from_numpy(contour).float(),
        profile_length=refiner_props.profile_length,
        profile_width=refiner_props.profile_width,
        sample_step=refiner_props.sample_step,
        num_samples=refiner_props.num_sampled_profiles,
    )

    sample_profiles = sample_profiles.detach().cpu().numpy()

    # Compute x coordinates for plotting
    num_samples = refiner_props.profile_length
    step = refiner_props.sample_step
    x = (np.arange(num_samples) - (num_samples - 1) / 2.0) * step

    plot_prior_and_landscape_from_profiles(
        sample_profiles,  # (N, L)
        x=x,
        template_props=template_props,
    )


def plot_prior_and_landscape_from_profiles(
    profiles: np.ndarray | torch.Tensor,
    x: np.ndarray | None = None,
    template_props: TemplateProps | None = None,
) -> None:
    """
    Visualizes the BiGaussian prior and the resulting loss landscape.
    This addresses 'Objective 1: Validate Loss Landscape' from plan_2d.md.

    Args:
        x (np.array): 1D coordinates along the normal vector.
        profiles (np.array or torch.Tensor): Batch of 1D intensity profiles.
            template_props (TemplateProps): Properties for the template (peak_dist, sigma).
    """
    if template_props is None:
        template_props = TemplateProps()

    if x is None:
        x = np.arange(profiles.shape[1]) - (profiles.shape[1] - 1) / 2.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    template, mean_profile, std_profile = plot_profile_statistics(
        profiles,
        x=x,
        ax=ax1,
        title="BiGaussian Prior Profile",
        template_props=template_props,
    )

    # Visualize the Loss Landscape using Cross-Correlation for various shifts.
    _plot_landscape_from_signal(x, mean_profile, template, ax2)

    plt.tight_layout()
    plt.show()


def plot_profile_statistics(
    profiles: np.ndarray | torch.Tensor,
    x: np.ndarray | None = None,
    title: str = "Profile Statistics",
    ax: Axes | None = None,
    template: np.ndarray | torch.Tensor | None = None,
    template_props: TemplateProps | None = None,
    norm: int = 1,
):
    """
    Visualizes the mean and spread of a batch of profiles.

    Args:
        profiles (np.array or torch.Tensor): Shape (N, L) where N is batch size, L is profile length.
        x (np.array): Optional x-axis coordinates.
        title (str): Plot title.
        ax (matplotlib.axes.Axes): Optional axes to plot on. If None, creates a new figure.
        template (np.array or torch.Tensor): Optional template profile to overlay.
    """
    if template_props is None:
        template_props = TemplateProps()

    if isinstance(profiles, torch.Tensor):
        profiles = profiles.detach().cpu().numpy()

    with debug_module.debug_warning(
        "Temporary setting for profile norm calculations. Not sure how it works with L1 vs L2."
    ):
        if norm not in [1, 2]:
            raise ValueError("norm must be 1 or 2 for L1 or L2 norm.")

        if norm == 2:
            mean_profile = np.mean(profiles, axis=0)
            std_profile = np.std(profiles, axis=0)
        elif norm == 1:
            mean_profile = np.median(profiles, axis=0)
            std_profile = np.median(np.abs(profiles - mean_profile), axis=0)

        if x is None:
            x = np.arange(profiles.shape[1]) - (profiles.shape[1] - 1) / 2.0

    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        show_plot = True

    ax.plot(x, mean_profile, label=f"L{norm} Mean Profile", color="blue", linewidth=2)

    ax.fill_between(
        x,
        mean_profile - std_profile,
        mean_profile + std_profile,
        color="blue",
        alpha=0.2,
        label=f"L{norm} Standard Deviation",
    )

    if template is not None:
        if isinstance(template, torch.Tensor):
            template = template.detach().cpu().numpy()
    else:
        template = BiGaussianBaseLoss.get_bigaussian_profile(
            x=torch.tensor(x, dtype=torch.float32),
            peak_dist=template_props.peak_dist,
            sigma=template_props.sigma,
        ).numpy()

    ax.plot(x, template, label="Template", color="red", linestyle="--", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Sample Index / Distance")
    ax.set_ylabel("Intensity")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    if show_plot:
        plt.show()
    return template, mean_profile, std_profile


def _plot_landscape_from_signal(x, y_signal, template, ax):
    """Visualize the Loss Landscape using Cross-Correlation for various shifts."""

    shifts = np.linspace(-8, 8, 100)
    correlations = []

    for s in shifts:
        # Shift the signal by -s (equivalent to sampling at x+s)
        # We use interpolation to simulate sampling at non-integer coordinates
        y_sampled = np.interp(x + s, x, y_signal)
        S_norm = (y_sampled - np.mean(y_sampled)) / (np.std(y_sampled) + 1e-6)

        # Cross-correlation
        corr = np.mean(S_norm * template)
        correlations.append(corr)

    ax.plot(shifts, correlations, color="red", label="Cross-Correlation")
    ax.axvline(0, color="green", linestyle="--", label="Ideal Shift (0)")
    ax.set_title("Loss Landscape (Objective 1)")
    ax.set_xlabel("Shift Parameter $s$")
    ax.set_ylabel("Correlation Score")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)


def _compute_peak_boundary_lines(
    contour: np.ndarray,
    normals: np.ndarray,
    template_params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Helper to compute peak and boundary lines for visualization."""
    N = len(contour)

    def _get_param(name, default_val=1.0):
        val = template_params.get(name, default_val)
        if isinstance(val, torch.Tensor):
            val = val.detach().cpu().numpy()
        if np.ndim(val) == 0:
            return np.full(N, val)
        return val

    peak_dist = _get_param("peak_dist", 4.0)
    sigma1 = _get_param("sigma1", 1.0)

    if "sigma2" in template_params:
        sigma2 = _get_param("sigma2")
    else:
        sigma2 = sigma1

    # Calculate offsets
    p1 = contour - normals * (peak_dist[:, None] / 2.0)
    p2 = contour + normals * (peak_dist[:, None] / 2.0)
    b1 = contour - normals * (peak_dist[:, None] / 2.0 + 2 * sigma1[:, None])
    b2 = contour + normals * (peak_dist[:, None] / 2.0 + 2 * sigma2[:, None])

    return p1, p2, b1, b2


def plot_contour_normals(
    image: np.ndarray,
    contour: np.ndarray,
    ax: Axes | None = None,
    stochastic: bool = True,
    refiner_props: ContourRefinerProps | None = None,
    template_params: dict[str, Any] | None = None,
    plot_normals: bool = True,
) -> None:
    """
    Visualizes the contour and its normals on top of the image.
    Useful for verifying normal calculation and sampling direction.

    Args:
        image (np.array): 2D image array.
        contour (np.array): (N, 2) array of (row, col) coordinates.
        ax (matplotlib.axes.Axes): Optional axes.
        stochastic (bool): If True, use stochastic sampling for normals (simulating optimization step).
        refiner_props (ContourRefinerProps): Props containing sampling configuration.
        template_params (dict): Optional dictionary of template parameters (peak_dist, sigma1, sigma2) to visualize peaks/boundaries.
        plot_normals (bool): If True, plots the yellow normal lines.
    """
    if refiner_props is None:
        refiner_props = ContourRefinerProps()

    profile_len = refiner_props.profile_length
    num_lines = refiner_props.num_sampled_profiles

    # Initialize figure
    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
        show_plot = True

    ax.imshow(image, cmap="gray")
    ax.plot(contour[:, 1], contour[:, 0], "r-", linewidth=1, label="Contour")

    # Calculate normals
    contour_tensor = torch.from_numpy(contour).float()
    full_normals = geometry.compute_normals(contour_tensor).numpy()

    # Plot peaks and boundaries if template params are provided
    if template_params is not None:
        p1, p2, b1, b2 = _compute_peak_boundary_lines(contour, full_normals, template_params)
        ax.plot(p1[:, 1], p1[:, 0], "c--", linewidth=1.5, alpha=0.8, label="Peaks")
        ax.plot(p2[:, 1], p2[:, 0], "c--", linewidth=1.5, alpha=0.8)
        ax.plot(b1[:, 1], b1[:, 0], "m:", linewidth=1.5, alpha=0.8, label="Boundaries")
        ax.plot(b2[:, 1], b2[:, 0], "m:", linewidth=1.5, alpha=0.8)

    title_suffix = ""
    if plot_normals:
        if stochastic:
            # Simulate the stochastic sampling used in optimization
            indices = (
                sampling._get_stratified_indices(
                    len(contour), num_lines, device=contour_tensor.device
                )
                .cpu()
                .numpy()
            )

            # Extract coarse contour and compute normals on it
            coarse_contour = contour_tensor[indices]
            normals = geometry.compute_normals(coarse_contour).numpy()

            active_points = contour[indices]
            active_normals = normals

            title_suffix = f"(stochastic batch: {len(indices)})"
        else:
            # Compute normals on full contour
            normals = full_normals

            N_points = len(contour)
            # Ensure we don't try to plot more lines than points
            num_lines_to_plot = min(num_lines, N_points)

            if num_lines_to_plot > 0:
                indices = np.linspace(0, N_points - 1, num_lines_to_plot, dtype=int)
                active_points = contour[indices]
                active_normals = normals[indices]
            else:
                active_points = []
                active_normals = []

            title_suffix = f"(showing {len(active_points)} of {N_points})"

        for i in range(len(active_points)):
            nr, nc = active_normals[i]
            r0, c0 = active_points[i]
            half_len = (profile_len - 1) / 2.0
            r_start, c_start = r0 - nr * half_len, c0 - nc * half_len
            r_end, c_end = r0 + nr * half_len, c0 + nc * half_len
            ax.plot([c_start, c_end], [r_start, r_end], "y-", alpha=0.8, linewidth=1)

    ax.set_title(f"Contour Details {title_suffix}")
    ax.legend()

    if show_plot:
        plt.show()


def plot_contour_crops(
    image: np.ndarray,
    contour: np.ndarray,
    crop_indices: list[int],
    init_contour: np.ndarray | None = None,
    gt_contour: np.ndarray | None = None,
    crop_size: int = 60,
    template_params: dict[str, Any] | None = None,
    num_cols: int = 4,
) -> plt.Figure:
    """
    Creates a grid of cropped visualizations centered on specific contour vertices.
    """
    num_crops = len(crop_indices)
    num_rows = int(np.ceil(num_crops / num_cols))

    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2), constrained_layout=True
    )
    axes = axes.flatten()

    H, W = image.shape[:2]
    half_size = crop_size // 2

    # Precompute lines if needed
    p1, p2, b1, b2 = None, None, None, None
    if template_params is not None:
        contour_tensor = torch.from_numpy(contour).float()
        normals = geometry.compute_normals(contour_tensor).numpy()
        p1, p2, b1, b2 = _compute_peak_boundary_lines(contour, normals, template_params)

    for i, ax in enumerate(axes):
        if i >= num_crops:
            ax.axis("off")
            continue

        idx = crop_indices[i]
        # Center on initial contour if available (to track drift), else current
        if init_contour is not None:
            cy, cx = init_contour[idx]
        else:
            cy, cx = contour[idx]

        y_min = max(0, int(cy - half_size))
        y_max = min(H, int(cy + half_size))
        x_min = max(0, int(cx - half_size))
        x_max = min(W, int(cx + half_size))

        ax.imshow(image, cmap="gray")

        if p1 is not None:
            ax.plot(p1[:, 1], p1[:, 0], "c--", linewidth=1.5, alpha=0.8, label="Peaks")
            ax.plot(p2[:, 1], p2[:, 0], "c--", linewidth=1.5, alpha=0.8)
            ax.plot(b1[:, 1], b1[:, 0], "m:", linewidth=1.5, alpha=0.8, label="Boundaries")
            ax.plot(b2[:, 1], b2[:, 0], "m:", linewidth=1.5, alpha=0.8)
            # ax.scatter(p1[:, 1], p1[:, 0], c="c", s=20, label="Peaks")
            # ax.scatter(p2[:, 1], p2[:, 0], c="c", s=20)
            # ax.scatter(b1[:, 1], b1[:, 0], c="m", s=20, label="Boundaries")
            # ax.scatter(
            #    b2[:, 1],
            #    b2[:, 0],
            #    c="m",
            #    s=20,
            # )

        if init_contour is not None:
            ax.plot(
                init_contour[:, 1],
                init_contour[:, 0],
                "r--",
                alpha=0.6,
                linewidth=1.5,
                label="Initial",
            )

        ax.plot(contour[:, 1], contour[:, 0], "b-", linewidth=2, label="Refined")
        # ax.scatter(contour[idx, 1], contour[idx, 0], c="b", s=30, label="Refined")

        if gt_contour is not None:
            ax.plot(gt_contour[:, 1], gt_contour[:, 0], "k:", alpha=0.8, linewidth=2, label="GT")

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)
        ax.axis("off")

    # Legend
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        ncol=len(by_label),
        bbox_to_anchor=(0.5, 1.02),
    )
    return fig


def plot_rbf_deformation(
    initial_contour: np.ndarray | torch.Tensor,
    final_contour: np.ndarray | torch.Tensor,
    control_points: np.ndarray | torch.Tensor,
    weights: np.ndarray | torch.Tensor,
    ax: Axes | None = None,
    title: str = "RBF Deformation Field",
) -> None:
    """
    Visualizes the RBF deformation field.

    Args:
        initial_contour: (N, 2) array of initial vertices (row, col).
        final_contour: (N, 2) array of deformed vertices (row, col).
        control_points: (K, 2) array of RBF centers (row, col).
        weights: (K, 2) array of RBF weights/vectors (row, col).
        ax: Optional matplotlib axes.
        title: Plot title.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
        show_plot = True
    else:
        show_plot = False

    # Convert to numpy
    def _to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    init_c = _to_np(initial_contour)
    final_c = _to_np(final_contour)
    cp = _to_np(control_points)
    w = _to_np(weights)

    # Plot contours
    # (row, col) -> plot(x=col, y=row)
    ax.plot(init_c[:, 1], init_c[:, 0], "k--", alpha=0.5, label="Initial")
    ax.plot(final_c[:, 1], final_c[:, 0], "b-", linewidth=2, label="Deformed")

    # Plot control points
    ax.scatter(cp[:, 1], cp[:, 0], c="red", s=30, zorder=5, label="Control Points")

    # Plot weight vectors
    # Quiver expects (x, y, u, v). Our data is (row, col).
    # x = col, y = row. u = w_col, v = w_row.
    ax.quiver(
        cp[:, 1],
        cp[:, 0],
        w[:, 1],
        w[:, 0],
        color="red",
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.005,
        label="Weights",
    )

    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal")

    if show_plot:
        ax.invert_yaxis()
        plt.show()


def plot_bspline_basis(
    num_cp: int = 8,
    num_eval: int = 200,
    ax: Axes | None = None,
) -> None:
    """
    Visualizes the B-Spline basis functions (columns of M) to show overlap.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
        show_plot = True
    else:
        show_plot = False

    # Get the matrix
    M = get_bspline_matrix(num_cp, num_eval).cpu().numpy()
    x = np.arange(num_eval)

    # Plot each basis function
    for i in range(num_cp):
        ax.plot(x, M[:, i], linewidth=2, label=f"CP {i}")
        ax.fill_between(x, 0, M[:, i], alpha=0.1)

    ax.set_title(f"Cubic B-Spline Basis Functions (Cyclic, N_CP={num_cp})")
    ax.set_xlabel("Evaluation Point Index")
    ax.set_ylabel("Influence Weight")
    ax.grid(True, linestyle="--", alpha=0.5)

    if show_plot:
        plt.tight_layout()
        plt.show()


def compare_bspline_basis_functions(
    configs: list[int] | None = None,
    num_eval: int = 200,
) -> None:
    """
    Plots B-Spline basis functions for different numbers of control points.
    """
    if configs is None:
        configs = [5, 20]

    fig, axes = plt.subplots(len(configs), 1, figsize=(10, 4 * len(configs)), sharex=True)
    if len(configs) == 1:
        axes = [axes]

    for ax, num_cp in zip(axes, configs):
        plot_bspline_basis(num_cp, num_eval, ax=ax)

    plt.tight_layout()
    plt.show()


def plot_parameter_curves(
    params: dict[str, torch.Tensor],
    ax: Axes | None = None,
) -> None:
    """
    Plots the unrolled parameter values along the contour.
    params: Dictionary of tensors (e.g. from template_model.get_params())
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
        show_plot = True
    else:
        show_plot = False

    x = None
    for name, tensor in params.items():
        # Ensure 1D
        if tensor.ndim == 0:
            continue

        y = tensor.detach().cpu().numpy()
        if x is None:
            x = np.arange(len(y))

        ax.plot(x, y, label=name, linewidth=2)

    if x is not None:
        ax.set_title("Spatially Varying Template Parameters")
        ax.set_xlabel("Contour Vertex Index")
        ax.set_ylabel("Parameter Value")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

    if show_plot:
        plt.tight_layout()
        plt.show()
