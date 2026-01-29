import abc
import logging
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffmeshopt.opt2d.geometry import compute_cubic_bspline_weights, get_bspline_matrix
from diffmeshopt.opt2d.props import TemplateProps


class TemplateMode(Enum):
    PER_POINT = "per_point"
    GLOBAL = "global"
    FIXED = "fixed"
    BSPLINE = "bspline"
    NEURAL = "neural"
    GRID = "grid"
    SPLAT = "splat"


class TemplateModelFactory:
    @staticmethod
    def create(
        mode: str | TemplateMode, props, num_vertices: int = None, image_shape: tuple = None
    ):
        logging.info(f"Creating template model: {mode}")
        # Convert string to Enum if necessary
        if isinstance(mode, str):
            try:
                mode = TemplateMode(mode.lower())
            except ValueError:
                raise ValueError(
                    "Unknown template_mode: "
                    + f"{mode}. Available modes: {[m.value for m in TemplateMode]}"
                ) from TemplateMode

        model = None
        # Validation and Instantiation
        if mode == TemplateMode.PER_POINT:
            if num_vertices is None:
                raise ValueError("num_vertices is required for PER_POINT mode")
            model = PerPointTemplateModel(num_vertices, props)

        elif mode == TemplateMode.GLOBAL:
            model = GlobalOptimizableTemplateModel(props)

        elif mode == TemplateMode.FIXED:
            model = FixedTemplateModel(props)

        elif mode == TemplateMode.BSPLINE:
            model = BSplineTemplateModel(props)

        elif mode == TemplateMode.NEURAL:
            if image_shape is None:
                raise ValueError("image_shape (H, W) is required for NEURAL mode")
            model = NeuralFieldTemplateModel(props, image_shape)

        elif mode == TemplateMode.GRID:
            if image_shape is None:
                raise ValueError("image_shape (H, W) is required for GRID mode")
            model = GridTemplateModel(props, image_shape)

        elif mode == TemplateMode.SPLAT:
            if image_shape is None:
                raise ValueError("image_shape (H, W) is required for SPLAT mode")
            model = GaussianSplatTemplateModel(props, image_shape)

        if model is not None:
            model.mode = mode
            return model

        raise ValueError(f"Factory implementation missing for mode: {mode}")


class BaseTemplateModel(nn.Module, abc.ABC):
    def __init__(self, props: TemplateProps):
        super().__init__()
        self.props = props
        self.image_shape = None  # Set by refiner if needed
        self.mode = None
        # Register initial values to keep them on the correct device
        self.register_buffer("peak_dist_init", torch.tensor(float(props.peak_dist)))
        self.register_buffer("sigma_init", torch.tensor(float(props.sigma)))
        self.register_buffer("amp_init", torch.tensor(float(props.amp)))
        self.register_buffer("sigma_ratio_init", torch.tensor(float(props.sigma_ratio)))
        self.register_buffer("amp_ratio_init", torch.tensor(float(props.amp_ratio)))

    @abc.abstractmethod
    def get_params(
        self,
        batch_indices: torch.Tensor | None = None,
        coordinates: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Returns a dictionary of parameters (peak_dist, sigma) for the given indices.
        coordinates: (N, 2) tensor of spatial positions (used for Neural Fields).
        If batch_indices is None, returns parameters for all points.
        """
        pass

    def get_regularization_loss(self) -> dict[str, torch.Tensor]:
        return {}


class FixedTemplateModel(BaseTemplateModel):
    def get_params(
        self,
        batch_indices: torch.Tensor | None = None,
        coordinates: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # Return scalar tensors which will be broadcasted by the loss function
        if self.props.symmetric:
            sigma_ratio = torch.tensor(1.0, device=self.sigma_init.device)
            amp_ratio = torch.tensor(1.0, device=self.amp_init.device)
        else:
            sigma_ratio = self.sigma_ratio_init
            amp_ratio = self.amp_ratio_init

        return {
            "peak_dist": self.peak_dist_init,
            "sigma1": self.sigma_init,
            "sigma2": self.sigma_init * sigma_ratio,
            "amp1": self.amp_init,
            "amp2": self.amp_init * amp_ratio,
        }


class GlobalOptimizableTemplateModel(BaseTemplateModel):
    def __init__(self, props: TemplateProps):
        super().__init__(props)
        # Single scalar parameters for the whole contour
        self.log_sigma = nn.Parameter(torch.tensor(float(props.sigma)).log())
        if not props.symmetric:
            self.log_sigma_ratio = nn.Parameter(torch.tensor(float(props.sigma_ratio)).log())
            self.log_amp_ratio = nn.Parameter(torch.tensor(float(props.amp_ratio)).log())

        # Reparameterization: peak_dist = sigma * (2.0 + excess)
        # This enforces peak_dist > min_peak_ratio * sigma structurally.
        # New reparam: peak_dist = (sigma1+sigma2) * (min_peak_ratio/2 + excess)
        sigma2_init = props.sigma * (1.0 if props.symmetric else props.sigma_ratio)
        init_ratio = props.peak_dist / (props.sigma + sigma2_init)
        init_excess = max(init_ratio - props.min_peak_ratio / 2.0, 1e-6)
        self.log_excess = nn.Parameter(torch.tensor(init_excess).log())

    def get_params(
        self,
        batch_indices: torch.Tensor | None = None,
        coordinates: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        sigma1 = self.log_sigma.exp()
        excess = self.log_excess.exp()

        if self.props.symmetric:
            sigma2 = sigma1
            amp_ratio = torch.tensor(1.0, device=sigma1.device)
        else:
            sigma2 = sigma1 * self.log_sigma_ratio.exp()
            amp_ratio = self.log_amp_ratio.exp()

        peak_dist = (sigma1 + sigma2) * (self.props.min_peak_ratio / 2.0 + excess)

        return {
            "peak_dist": peak_dist,
            "sigma1": sigma1,
            "sigma2": sigma2,
            "amp1": self.amp_init,
            "amp2": self.amp_init * amp_ratio,
        }

    def get_regularization_loss(self) -> dict[str, torch.Tensor]:
        # Weak prior to stay near initialization
        # Reconstruct log_peak_dist for the prior
        sigma1 = self.log_sigma.exp()
        if self.props.symmetric:
            sigma2 = sigma1
        else:
            sigma2 = sigma1 * self.log_sigma_ratio.exp()
        peak_dist = (sigma1 + sigma2) * (self.props.min_peak_ratio / 2.0 + self.log_excess.exp())
        log_peak_dist = peak_dist.log()

        prior = (self.log_sigma - self.sigma_init.log()).pow(2) + (
            log_peak_dist - self.peak_dist_init.log()
        ).pow(2)

        if not self.props.symmetric:
            prior = (
                prior
                + (self.log_sigma_ratio - self.sigma_ratio_init.log()).pow(2)
                + (self.log_amp_ratio - self.amp_ratio_init.log()).pow(2)
            )

        return {"sigma_reg": prior * 0.1}  # Scaled down slightly


class PerPointTemplateModel(BaseTemplateModel):
    def __init__(self, num_points: int, props: TemplateProps):
        super().__init__(props)
        # Parameters for each point
        self.log_sigma = nn.Parameter(torch.full((num_points,), float(props.sigma)).log())
        if not props.symmetric:
            self.log_sigma_ratio = nn.Parameter(
                torch.full((num_points,), float(props.sigma_ratio)).log()
            )
            self.log_amp_ratio = nn.Parameter(
                torch.full((num_points,), float(props.amp_ratio)).log()
            )

        sigma2_init = props.sigma * (1.0 if props.symmetric else props.sigma_ratio)
        init_ratio = props.peak_dist / (props.sigma + sigma2_init)
        init_excess = max(init_ratio - props.min_peak_ratio / 2.0, 1e-6)
        self.log_excess = nn.Parameter(torch.full((num_points,), init_excess).log())

    def get_params(
        self,
        batch_indices: torch.Tensor | None = None,
        coordinates: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        sigma1 = self.log_sigma.exp()
        excess = self.log_excess.exp()

        if self.props.symmetric:
            sigma2 = sigma1
            amp2 = self.amp_init.expand_as(sigma1)
        else:
            sigma_ratio = self.log_sigma_ratio.exp()
            amp_ratio = self.log_amp_ratio.exp()
            sigma2 = sigma1 * sigma_ratio
            amp2 = self.amp_init * amp_ratio

        peak_dist = (sigma1 + sigma2) * (self.props.min_peak_ratio / 2.0 + excess)

        if batch_indices is not None:
            return {
                "peak_dist": peak_dist[batch_indices],
                "sigma1": sigma1[batch_indices],
                "sigma2": sigma2[batch_indices],
                "amp1": self.amp_init.expand_as(sigma1)[batch_indices],
                "amp2": amp2[batch_indices],
            }
        return {
            "peak_dist": peak_dist,
            "sigma1": sigma1,
            "sigma2": sigma2,
            "amp1": self.amp_init.expand_as(sigma1),
            "amp2": amp2,
        }

    def get_regularization_loss(self) -> dict[str, torch.Tensor]:
        # Regularization: Gaussian prior on log(sigma) centered at initialization
        prior = (self.log_sigma - self.sigma_init.log()).pow(2).mean()
        if not self.props.symmetric:
            prior = prior + (self.log_sigma_ratio - self.sigma_ratio_init.log()).pow(2).mean()
        # Reconstruct log_peak_dist for smoothness
        sigma1 = self.log_sigma.exp()
        if self.props.symmetric:
            sigma2 = sigma1
        else:
            sigma2 = sigma1 * self.log_sigma_ratio.exp()
        peak_dist = (sigma1 + sigma2) * (self.props.min_peak_ratio / 2.0 + self.log_excess.exp())
        log_peak_dist = peak_dist.log()

        # Smoothness: Penalize changes along the contour
        diff_sigma = self.log_sigma - torch.roll(self.log_sigma, shifts=1, dims=0)
        diff_peak = log_peak_dist - torch.roll(log_peak_dist, shifts=1, dims=0)
        smoothness = diff_sigma.pow(2).mean() + diff_peak.pow(2).mean()

        if not self.props.symmetric:
            diff_sigma_ratio = self.log_sigma_ratio - torch.roll(
                self.log_sigma_ratio, shifts=1, dims=0
            )
            diff_amp_ratio = self.log_amp_ratio - torch.roll(self.log_amp_ratio, shifts=1, dims=0)
            smoothness = smoothness + diff_sigma_ratio.pow(2).mean() + diff_amp_ratio.pow(2).mean()

        return {"sigma_reg": prior, "template_smooth": smoothness}


class BSplineTemplateModel(BaseTemplateModel):
    def __init__(self, props: TemplateProps, **kwargs):  # Accept extra kwargs
        super().__init__(props)
        self.num_cp = getattr(props, "bspline_num_control_points", 10)

        # Initialize control points in log space for positivity
        # Parameters: peak_dist, sigma1, sigma2, amp1, amp2
        # Reparameterize: peak_dist = (sigma1 + sigma2) * (1.0 + excess)
        # Factor is min_peak_ratio / 2.0 because we sum two sigmas
        init_ratio = props.peak_dist / (2 * props.sigma)  # Assuming sigma1=sigma2=sigma
        init_excess = max(init_ratio - props.min_peak_ratio / 2.0, 1e-6)
        # Vectorized control points
        # Channels: excess, sigma1, amp1, [sigma2, amp2]
        self.channel_names = ["excess", "sigma1", "amp1"]
        init_vals = [init_excess, float(props.sigma), float(props.amp)]

        if not props.symmetric:
            self.channel_names.extend(["sigma2", "amp2"])
            init_vals.extend(
                [float(props.sigma * props.sigma_ratio), float(props.amp * props.amp_ratio)]
            )

        init_tensor = (
            torch.tensor(init_vals, dtype=torch.float32).unsqueeze(1).repeat(1, self.num_cp)
        )
        self.log_control_points = nn.Parameter(init_tensor.log())

    def get_params(
        self,
        batch_indices: torch.Tensor | None = None,
        coordinates: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if coordinates is None:
            raise ValueError("BSplineTemplateModel requires coordinates to compute arc length")

        # 1. Compute arc lengths along the contour
        # We detach coordinates here to prevent gradients from flowing through the arc-length calculation.
        # This ensures vertices move to match the image, not to slide along the spline to find a better parameter.
        coords_detached = coordinates.detach()

        # Calculate distances between consecutive points (assuming closed loop)
        # Re-calculate diffs forward: 0->1, 1->2, ...
        diffs_fwd = coords_detached - torch.roll(coords_detached, shifts=-1, dims=0)
        dists_fwd = torch.norm(diffs_fwd, dim=1)

        # cum_dists[i] = length of segment 0->...->i+1
        cum_dists = torch.cumsum(dists_fwd, dim=0)
        total_length = cum_dists[-1]

        # t[0] = 0, t[1] = dist(0,1), etc.
        t = torch.cat([torch.zeros(1, device=coordinates.device), cum_dists[:-1]])
        t = t / (total_length + 1e-8)  # Normalize to [0, 1]

        if batch_indices is not None:
            t = t[batch_indices]

        # 2. Evaluate B-spline (Linear interpolation for robustness)
        params = self._evaluate_spline(t)

        if self.props.symmetric:
            sigma2 = params["sigma1"]
            amp2 = params["amp1"]
        else:
            sigma2 = params["sigma2"]
            amp2 = params["amp2"]

        # Enforce separation: peak_dist > sigma1 + sigma2
        peak_dist = (params["sigma1"] + sigma2) * (
            self.props.min_peak_ratio / 2.0 + params["excess"]
        )

        return {
            "peak_dist": peak_dist,
            "sigma1": params["sigma1"],
            "sigma2": sigma2,
            "amp1": params["amp1"],
            "amp2": amp2,
        }

    def _evaluate_spline(self, t: torch.Tensor) -> dict[str, torch.Tensor]:
        # t in [0, 1]
        # Map to B-spline parameter u in [0, num_cp]
        u = t * self.num_cp
        indices, weights = compute_cubic_bspline_weights(u, self.num_cp)

        # Vectorized interpolation
        # self.log_control_points: (C, num_cp)
        p = self.log_control_points.exp()

        # Gather values: (C, N, 4) using advanced indexing
        p_gathered = p[:, indices]

        # Weighted sum along the 4 control points
        val = (p_gathered * weights.unsqueeze(0)).sum(dim=-1)  # (C, N)

        res = {name: val[i] for i, name in enumerate(self.channel_names)}
        return res

    def get_regularization_loss(self) -> dict[str, torch.Tensor]:
        # Proximal-type regularization:
        # 1. Keep sigma close to initialization (prevents collapse to 0 or explosion)
        # 2. Enforce smoothness along the contour (spatial coupling)

        sigma_ref = self.sigma_init.log()

        # sigma1 is always at index 1
        log_sigma1_cp = self.log_control_points[1]

        # Deviation from prior (Proximal term towards initialization)
        reg_sigma = (log_sigma1_cp - sigma_ref).pow(2).mean()

        # Smoothness (first differences of control points)
        # This acts as a spatial regularizer
        smooth_sigma = (log_sigma1_cp[1:] - log_sigma1_cp[:-1]).pow(2).mean()

        if not self.props.symmetric:
            sigma2_ref = (self.sigma_init * self.sigma_ratio_init).log()
            log_sigma2_cp = self.log_control_points[3]
            reg_sigma = reg_sigma + (log_sigma2_cp - sigma2_ref).pow(2).mean()
            smooth_sigma = smooth_sigma + (log_sigma2_cp[1:] - log_sigma2_cp[:-1]).pow(2).mean()

        return {"sigma_reg": reg_sigma, "template_smooth": smooth_sigma}


class NeuralFieldTemplateModel(BaseTemplateModel):
    def __init__(self, props: TemplateProps, image_shape: tuple[int, int]):
        super().__init__(props)
        # Coordinate-based MLP: (x, y) -> (peak_dist, sigma1, sigma2, amp1, amp2)
        self.image_shape = image_shape
        layers = []
        hidden_dim = getattr(props, "neural_hidden_dim", 32)
        num_layers = getattr(props, "neural_num_layers", 2)

        in_dim = 2
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        self.net = nn.Sequential(*layers)
        # Output 5 parameters (or 3 if symmetric)
        out_dim = 3 if props.symmetric else 5
        self.head = nn.Linear(in_dim, out_dim)

        # Initialize head to zero so we start exactly at the initial props values
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def get_params(
        self,
        batch_indices: torch.Tensor | None = None,
        coordinates: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if coordinates is None:
            raise ValueError("NeuralFieldTemplateModel requires coordinates")

        # Detach coordinates to prevent gradients flowing into vertex positions
        coords_to_eval = coordinates.detach()

        if batch_indices is not None:
            coords_to_eval = coords_to_eval[batch_indices]

        # Normalize coordinates to [-1, 1] for MLP stability
        H, W = self.image_shape
        # coordinates are (y, x)
        norm_x = (coords_to_eval[:, 1] / (W - 1)) * 2 - 1
        norm_y = (coords_to_eval[:, 0] / (H - 1)) * 2 - 1
        norm_coords = torch.stack([norm_x, norm_y], dim=-1)

        out = self.head(self.net(norm_coords))

        # Base values in log space
        base_peak = self.peak_dist_init.log()
        base_sigma = self.sigma_init.log()
        base_amp = self.amp_init.log()

        # Apply learned residuals in log space (ensures positivity)
        if self.props.symmetric:
            # out: [d_peak, d_s1, d_a1]
            return {
                "peak_dist": (base_peak + out[:, 0]).exp(),
                "sigma1": (base_sigma + out[:, 1]).exp(),
                "sigma2": (base_sigma + out[:, 1]).exp(),
                "amp1": (base_amp + out[:, 2]).exp(),
                "amp2": (base_amp + out[:, 2]).exp(),
            }
        else:
            # out: [d_peak, d_s1, d_s2, d_a1, d_a2]
            base_sigma2 = (self.sigma_init * self.sigma_ratio_init).log()
            base_amp2 = (self.amp_init * self.amp_ratio_init).log()
            return {
                "peak_dist": (base_peak + out[:, 0]).exp(),
                "sigma1": (base_sigma + out[:, 1]).exp(),
                "sigma2": (base_sigma2 + out[:, 2]).exp(),
                "amp1": (base_amp + out[:, 3]).exp(),
                "amp2": (base_amp2 + out[:, 4]).exp(),
            }


class GridTemplateModel(BaseTemplateModel):
    def __init__(self, props: TemplateProps, image_shape: tuple[int, int]):
        super().__init__(props)
        self.image_shape = image_shape
        # Learnable grid: (1, 5, H, W)
        grid_size = getattr(props, "grid_size", 32)
        # Channels: peak_dist, sigma1, [sigma2], amp1, [amp2]
        num_channels = 3 if props.symmetric else 5
        self.grid = nn.Parameter(torch.zeros(1, num_channels, grid_size, grid_size))

    def get_params(
        self,
        batch_indices: torch.Tensor | None = None,
        coordinates: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if coordinates is None:
            raise ValueError("GridTemplateModel requires coordinates")

        coords_detached = coordinates.detach()
        if batch_indices is not None:
            coords_detached = coords_detached[batch_indices]

        # Normalize coordinates to [-1, 1] for grid_sample
        # coordinates are (y, x) in pixels
        H, W = self.image_shape
        norm_x = (coords_detached[:, 1] / (W - 1)) * 2 - 1
        norm_y = (coords_detached[:, 0] / (H - 1)) * 2 - 1
        grid_coords = torch.stack([norm_x, norm_y], dim=-1).view(1, 1, -1, 2)

        # Sample from grid
        num_channels = 3 if self.props.symmetric else 5
        out = F.grid_sample(self.grid, grid_coords, align_corners=True).view(num_channels, -1).T

        # Apply residuals to base values
        base_peak = self.peak_dist_init.log()
        base_sigma = self.sigma_init.log()
        base_amp = self.amp_init.log()

        if self.props.symmetric:
            return {
                "peak_dist": (base_peak + out[:, 0]).exp(),
                "sigma1": (base_sigma + out[:, 1]).exp(),
                "sigma2": (base_sigma + out[:, 1]).exp(),
                "amp1": (base_amp + out[:, 2]).exp(),
                "amp2": (base_amp + out[:, 2]).exp(),
            }
        else:
            base_sigma2 = (self.sigma_init * self.sigma_ratio_init).log()
            base_amp2 = (self.amp_init * self.amp_ratio_init).log()
            return {
                "peak_dist": (base_peak + out[:, 0]).exp(),
                "sigma1": (base_sigma + out[:, 1]).exp(),
                "sigma2": (base_sigma2 + out[:, 2]).exp(),
                "amp1": (base_amp + out[:, 3]).exp(),
                "amp2": (base_amp2 + out[:, 4]).exp(),
            }


class GaussianSplatTemplateModel(BaseTemplateModel):
    def __init__(self, props: TemplateProps, image_shape: tuple[int, int]):
        super().__init__(props)
        self.image_shape = image_shape
        num_splats = getattr(props, "splat_num_splats", 32)
        H, W = image_shape

        # Initialize splats randomly in the image domain
        self.centers = nn.Parameter(torch.rand(num_splats, 2) * torch.tensor([H, W]))
        # Splat influence radius (inverse scale)
        self.log_radius = nn.Parameter(torch.ones(num_splats) * 3.0)
        # Parameter payloads (residuals)
        num_payloads = 3 if props.symmetric else 5
        self.payloads = nn.Parameter(torch.zeros(num_splats, num_payloads))

    def get_params(
        self,
        batch_indices: torch.Tensor | None = None,
        coordinates: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if coordinates is None:
            raise ValueError("GaussianSplatTemplateModel requires coordinates")

        coords_detached = coordinates.detach()
        if batch_indices is not None:
            coords_detached = coords_detached[batch_indices]

        # Compute RBF weights: exp(-dist^2 / (2 * radius^2))
        # coordinates: (B, 2), centers: (K, 2)
        dists_sq = torch.cdist(coords_detached, self.centers, p=2) ** 2  # (B, K)
        radii_sq = self.log_radius.exp() ** 2
        weights = torch.exp(-dists_sq / (2 * radii_sq.unsqueeze(0)))  # (B, K)

        # Normalize weights (Shepard's method)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        # Interpolate payloads
        out = weights @ self.payloads  # (B, 5)

        base_peak = self.peak_dist_init.log()
        base_sigma = self.sigma_init.log()
        base_amp = self.amp_init.log()

        if self.props.symmetric:
            return {
                "peak_dist": (base_peak + out[:, 0]).exp(),
                "sigma1": (base_sigma + out[:, 1]).exp(),
                "sigma2": (base_sigma + out[:, 1]).exp(),
                "amp1": (base_amp + out[:, 2]).exp(),
                "amp2": (base_amp + out[:, 2]).exp(),
            }
        else:
            base_sigma2 = (self.sigma_init * self.sigma_ratio_init).log()
            base_amp2 = (self.amp_init * self.amp_ratio_init).log()
            return {
                "peak_dist": (base_peak + out[:, 0]).exp(),
                "sigma1": (base_sigma + out[:, 1]).exp(),
                "sigma2": (base_sigma2 + out[:, 2]).exp(),
                "amp1": (base_amp + out[:, 3]).exp(),
                "amp2": (base_amp2 + out[:, 4]).exp(),
            }
