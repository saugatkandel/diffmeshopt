import abc
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffmeshopt.opt2d.geometry import get_bspline_matrix
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
            if num_vertices is None:
                raise ValueError("num_vertices is required for BSPLINE mode")
            model = BSplineTemplateModel(num_vertices, props)

        elif mode == TemplateMode.NEURAL:
            model = NeuralFieldTemplateModel(props)

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


class TemplateModel(nn.Module, abc.ABC):
    def __init__(self, props: TemplateProps):
        super().__init__()
        self.props = props
        self.image_shape = None  # Set by refiner if needed
        self.mode = None
        # Register initial values to keep them on the correct device
        self.register_buffer("peak_dist_init", torch.tensor(float(props.peak_dist)))
        self.register_buffer("sigma_init", torch.tensor(float(props.sigma)))

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


class FixedTemplateModel(TemplateModel):
    def get_params(
        self,
        batch_indices: torch.Tensor | None = None,
        coordinates: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # Return scalar tensors which will be broadcasted by the loss function
        return {
            "peak_dist": self.peak_dist_init,
            "sigma": self.sigma_init,
        }


class GlobalOptimizableTemplateModel(TemplateModel):
    def __init__(self, props: TemplateProps):
        super().__init__(props)
        # Single scalar parameters for the whole contour
        self.log_peak_dist = nn.Parameter(torch.tensor(float(props.peak_dist)).log())
        self.log_sigma = nn.Parameter(torch.tensor(float(props.sigma)).log())

    def get_params(
        self,
        batch_indices: torch.Tensor | None = None,
        coordinates: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return {
            "peak_dist": self.log_peak_dist.exp(),
            "sigma": self.log_sigma.exp(),
        }

    def get_regularization_loss(self) -> dict[str, torch.Tensor]:
        # Weak prior to stay near initialization
        prior = (self.log_sigma - self.sigma_init.log()).pow(2) + (
            self.log_peak_dist - self.peak_dist_init.log()
        ).pow(2)
        return {"sigma_reg": prior}


class PerPointTemplateModel(TemplateModel):
    def __init__(self, num_points: int, props: TemplateProps):
        super().__init__(props)
        # Parameters for each point
        self.log_peak_dist = nn.Parameter(torch.full((num_points,), float(props.peak_dist)).log())
        self.log_sigma = nn.Parameter(torch.full((num_points,), float(props.sigma)).log())

    def get_params(
        self,
        batch_indices: torch.Tensor | None = None,
        coordinates: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if batch_indices is not None:
            return {
                "peak_dist": self.log_peak_dist[batch_indices].exp(),
                "sigma": self.log_sigma[batch_indices].exp(),
            }
        return {
            "peak_dist": self.log_peak_dist.exp(),
            "sigma": self.log_sigma.exp(),
        }

    def get_regularization_loss(self) -> dict[str, torch.Tensor]:
        # Regularization: Gaussian prior on log(sigma) centered at initialization
        prior = (self.log_sigma - self.sigma_init.log()).pow(2).mean()

        # Smoothness: Penalize changes along the contour
        diff_sigma = self.log_sigma - torch.roll(self.log_sigma, shifts=1, dims=0)
        diff_peak = self.log_peak_dist - torch.roll(self.log_peak_dist, shifts=1, dims=0)
        smoothness = diff_sigma.pow(2).mean() + diff_peak.pow(2).mean()

        return {"sigma_reg": prior, "template_smooth": smoothness}


class BSplineTemplateModel(TemplateModel):
    def __init__(self, num_eval_points: int, props: TemplateProps):
        super().__init__(props)
        self.num_cp = props.num_control_points
        self.num_eval = num_eval_points

        # Precompute B-spline evaluation matrix
        # Shape: (num_eval, num_cp)
        self.register_buffer("M", get_bspline_matrix(self.num_cp, self.num_eval))

        # Initialize control points in log space for positivity
        # Parameters: peak_dist, sigma1, sigma2, amp1, amp2
        self.log_peak_dist_cp = nn.Parameter(
            torch.full((self.num_cp,), float(props.peak_dist)).log()
        )
        self.log_sigma1_cp = nn.Parameter(torch.full((self.num_cp,), float(props.sigma)).log())
        self.log_sigma2_cp = nn.Parameter(torch.full((self.num_cp,), float(props.sigma)).log())
        self.log_amp1_cp = nn.Parameter(torch.full((self.num_cp,), float(props.amp)).log())
        self.log_amp2_cp = nn.Parameter(torch.full((self.num_cp,), float(props.amp)).log())

    def get_params(
        self,
        batch_indices: torch.Tensor | None = None,
        coordinates: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # Evaluate B-splines to get dense parameters
        # (num_eval, num_cp) @ (num_cp,) -> (num_eval,)
        peak_dist = (self.M @ self.log_peak_dist_cp).exp()
        sigma1 = (self.M @ self.log_sigma1_cp).exp()
        sigma2 = (self.M @ self.log_sigma2_cp).exp()
        amp1 = (self.M @ self.log_amp1_cp).exp()
        amp2 = (self.M @ self.log_amp2_cp).exp()

        if batch_indices is not None:
            return {
                "peak_dist": peak_dist[batch_indices],
                "sigma1": sigma1[batch_indices],
                "sigma2": sigma2[batch_indices],
                "amp1": amp1[batch_indices],
                "amp2": amp2[batch_indices],
            }
        return {
            "peak_dist": peak_dist,
            "sigma1": sigma1,
            "sigma2": sigma2,
            "amp1": amp1,
            "amp2": amp2,
        }


class NeuralFieldTemplateModel(TemplateModel):
    def __init__(self, props: TemplateProps):
        super().__init__(props)
        # Coordinate-based MLP: (x, y) -> (peak_dist, sigma1, sigma2, amp1, amp2)
        layers = []
        in_dim = 2
        for _ in range(props.neural_num_layers):
            layers.append(nn.Linear(in_dim, props.neural_hidden_dim))
            layers.append(nn.ReLU())
            in_dim = props.neural_hidden_dim

        self.net = nn.Sequential(*layers)
        # Output 5 parameters
        self.head = nn.Linear(in_dim, 5)

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

        # Normalize coordinates roughly to help optimization?
        # For now, we assume the MLP can handle the scale or we rely on normalization elsewhere.
        # Using raw coordinates.
        out = self.head(self.net(coordinates))

        # Base values in log space
        base_peak = self.peak_dist_init.log()
        base_sigma = self.sigma_init.log()
        base_amp = torch.tensor(self.props.amp, device=out.device).log()

        # Apply learned residuals in log space (ensures positivity)
        # out: [d_peak, d_s1, d_s2, d_a1, d_a2]
        return {
            "peak_dist": (base_peak + out[:, 0]).exp(),
            "sigma1": (base_sigma + out[:, 1]).exp(),
            "sigma2": (base_sigma + out[:, 2]).exp(),
            "amp1": (base_amp + out[:, 3]).exp(),
            "amp2": (base_amp + out[:, 4]).exp(),
        }


class GridTemplateModel(TemplateModel):
    def __init__(self, props: TemplateProps, image_shape: tuple[int, int]):
        super().__init__(props)
        self.image_shape = image_shape
        # Learnable grid: (1, 5, H, W)
        # 5 channels: peak_dist, sigma1, sigma2, amp1, amp2
        self.grid = nn.Parameter(torch.zeros(1, 5, props.grid_size, props.grid_size))

    def get_params(
        self,
        batch_indices: torch.Tensor | None = None,
        coordinates: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if coordinates is None:
            raise ValueError("GridTemplateModel requires coordinates")

        # Normalize coordinates to [-1, 1] for grid_sample
        # coordinates are (y, x) in pixels
        H, W = self.image_shape
        norm_x = (coordinates[:, 1] / (W - 1)) * 2 - 1
        norm_y = (coordinates[:, 0] / (H - 1)) * 2 - 1
        grid_coords = torch.stack([norm_x, norm_y], dim=-1).view(1, 1, -1, 2)

        # Sample from grid
        out = F.grid_sample(self.grid, grid_coords, align_corners=True).view(5, -1).T

        # Apply residuals to base values
        base_peak = self.peak_dist_init.log()
        base_sigma = self.sigma_init.log()
        base_amp = torch.tensor(self.props.amp, device=out.device).log()

        return {
            "peak_dist": (base_peak + out[:, 0]).exp(),
            "sigma1": (base_sigma + out[:, 1]).exp(),
            "sigma2": (base_sigma + out[:, 2]).exp(),
            "amp1": (base_amp + out[:, 3]).exp(),
            "amp2": (base_amp + out[:, 4]).exp(),
        }


class GaussianSplatTemplateModel(TemplateModel):
    def __init__(self, props: TemplateProps, image_shape: tuple[int, int]):
        super().__init__(props)
        self.image_shape = image_shape
        num_splats = props.num_splats
        H, W = image_shape

        # Initialize splats randomly in the image domain
        self.centers = nn.Parameter(torch.rand(num_splats, 2) * torch.tensor([H, W]))
        # Splat influence radius (inverse scale)
        self.log_radius = nn.Parameter(torch.ones(num_splats) * 3.0)
        # Parameter payloads (residuals)
        self.payloads = nn.Parameter(torch.zeros(num_splats, 5))

    def get_params(
        self,
        batch_indices: torch.Tensor | None = None,
        coordinates: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if coordinates is None:
            raise ValueError("GaussianSplatTemplateModel requires coordinates")

        # Compute RBF weights: exp(-dist^2 / (2 * radius^2))
        # coordinates: (B, 2), centers: (K, 2)
        dists_sq = torch.cdist(coordinates, self.centers, p=2) ** 2  # (B, K)
        radii_sq = self.log_radius.exp() ** 2
        weights = torch.exp(-dists_sq / (2 * radii_sq.unsqueeze(0)))  # (B, K)

        # Normalize weights (Shepard's method)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        # Interpolate payloads
        out = weights @ self.payloads  # (B, 5)

        base_peak = self.peak_dist_init.log()
        base_sigma = self.sigma_init.log()
        base_amp = torch.tensor(self.props.amp, device=out.device).log()

        return {
            "peak_dist": (base_peak + out[:, 0]).exp(),
            "sigma1": (base_sigma + out[:, 1]).exp(),
            "sigma2": (base_sigma + out[:, 2]).exp(),
            "amp1": (base_amp + out[:, 3]).exp(),
            "amp2": (base_amp + out[:, 4]).exp(),
        }
