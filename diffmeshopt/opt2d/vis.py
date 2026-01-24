import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes

import diffmeshopt.opt2d.geometry as geometry
import diffmeshopt.opt2d.sampling as sampling
from diffmeshopt.opt2d.geometry import get_bspline_matrix
from diffmeshopt.opt2d.loss import BiGaussianLoss
from diffmeshopt.opt2d.props import SamplingProps, TemplateProps


def plot_prior_and_landscape_from_contour(
    image: np.ndarray,
    contour: np.ndarray,
    sampling_props: SamplingProps | None = None,
    template_props: TemplateProps | None = None,
):
    if template_props is None:
        template_props = TemplateProps()

    if sampling_props is None:
        sampling_props = SamplingProps()

    sample_profiles, _, _ = sampling.sample_profiles_stochastic(
        torch.from_numpy(image).float(),
        torch.from_numpy(contour).float(),
        sampling_props=sampling_props,
    )

    sample_profiles = sample_profiles.detach().cpu().numpy()

    plot_prior_and_landscape_from_profiles(
        sample_profiles,  # (N, L)
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
        peak_dist (float): Distance between peaks.
        sigma (float): Width of peaks.
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

    mean_profile = np.mean(profiles, axis=0)
    std_profile = np.std(profiles, axis=0)

    if x is None:
        x = np.arange(profiles.shape[1]) - (profiles.shape[1] - 1) / 2.0

    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        show_plot = True

    ax.plot(x, mean_profile, label="Mean Profile", color="blue", linewidth=2)

    ax.fill_between(
        x,
        mean_profile - std_profile,
        mean_profile + std_profile,
        color="blue",
        alpha=0.2,
        label="Standard Deviation",
    )

    if template is not None:
        if isinstance(template, torch.Tensor):
            template = template.detach().cpu().numpy()
    else:
        template = BiGaussianLoss.get_bigaussian_profile(
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
    """Visualize the Loss Landscape using ross-Correlation for various shifts."""

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


def plot_contour_normals(
    image: np.ndarray,
    contour: np.ndarray,
    ax: Axes | None = None,
    stochastic: bool = True,
    sampling_props: SamplingProps | None = None,
) -> None:
    """
    Visualizes the contour and its normals on top of the image.
    Useful for verifying normal calculation and sampling direction.

    Args:
        image (np.array): 2D image array.
        contour (np.array): (N, 2) array of (row, col) coordinates.
        profile_len (int): Length of the profile line to visualize centered at vertex.
        num_lines (int): Number of normal lines to plot. If stochastic is True, this is the batch size.
        ax (matplotlib.axes.Axes): Optional axes.
        stochastic (bool): If True, use stochastic sampling for normals (simulating optimization step).
    """
    if sampling_props is None:
        sampling_props = SamplingProps()

    profile_len = sampling_props.num_samples
    num_lines = sampling_props.batch_size

    # Initialize figure
    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
        show_plot = True

    ax.imshow(image, cmap="gray")
    ax.plot(contour[:, 1], contour[:, 0], "r-", linewidth=1, label="Contour")

    # Calculate normals
    contour_tensor = torch.from_numpy(contour).float()

    if stochastic:
        # Simulate the stochastic sampling used in optimization
        indices = (
            sampling._get_stratified_indices(len(contour), num_lines, device=contour_tensor.device)
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
        normals = geometry.compute_normals(contour_tensor).numpy()

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

        # Center point
        r0, c0 = active_points[i]

        # Define line segment for profile
        # From -profile_len/2 to +profile_len/2 along normal
        half_len = (profile_len - 1) / 2.0

        r_start = r0 - nr * half_len
        c_start = c0 - nc * half_len
        r_end = r0 + nr * half_len
        c_end = c0 + nc * half_len

        # Plot line (x=col, y=row)
        ax.plot([c_start, c_end], [r_start, r_end], "y-", alpha=0.8, linewidth=1)

    ax.set_title(f"Contour Normals {title_suffix}")
    ax.legend()

    if show_plot:
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
