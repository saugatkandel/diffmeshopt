import abc
import logging
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffmeshopt.opt2d.geometry import compute_cubic_bspline_weights, get_bspline_matrix
from diffmeshopt.opt2d.loss import LaplacianSmoothingLoss
from diffmeshopt.opt2d.props import RegularizerType, TemplateProps


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

        # Detect spatial dimension from image_shape
        spatial_dim = 2
        if image_shape is not None and len(image_shape) == 3:
            spatial_dim = 3

        elif mode == TemplateMode.GLOBAL:
            model = GlobalOptimizableTemplateModel(props)

        elif mode == TemplateMode.FIXED:
            model = FixedTemplateModel(props)

        elif mode == TemplateMode.BSPLINE:
            model = BSplineTemplateModel(props)

        elif mode == TemplateMode.NEURAL:
            if image_shape is None:
                raise ValueError("image_shape (H, W) is required for NEURAL mode")
            model = NeuralFieldTemplateModel(props, image_shape, spatial_dim=spatial_dim)

        elif mode == TemplateMode.GRID:
            if image_shape is None:
                raise ValueError("image_shape (H, W) is required for GRID mode")
            model = GridTemplateModel(props, image_shape, spatial_dim=spatial_dim)

        elif mode == TemplateMode.SPLAT:
            if image_shape is None:
                raise ValueError("image_shape (H, W) is required for SPLAT mode")
            model = GaussianSplatTemplateModel(props, image_shape, spatial_dim=spatial_dim)

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
        Returns a dictionary of parameters (peak_dist, sigma1, sigma2, amp1, amp2) for the given indices.
        coordinates: (N, 2) tensor of spatial positions (used for Neural Fields).
        If batch_indices is None, returns parameters for all points.
        """
        pass

    def get_regularization_loss(self) -> dict[str, torch.Tensor]:
        return {}

    def _get_channel_weights(self, prefix: str = "anchor") -> list[float]:
        """Map props weights to parameter indices for residual-based models.

        Args:
            prefix: 'anchor' or 'smooth' to select the type of weight.

        Returns:
            List of floats indicating relative weight for each parameter channel:
            - Symmetric: 3 params [peak_dist, sigma, amp]
            - Asymmetric: 5 params [peak_dist, sigma1, sigma2, amp1, amp2]
        """

        # Helper to get value safely
        def get_w(name):
            return getattr(self.props, f"{prefix}_{name}", 0.0)

        # Note: amp/amp1/amp2 usually don't have explicit smooth/anchor props in the
        # simplified list, defaulting to 0.0 or 1.0 as appropriate.
        # For now, we assume amp is not heavily regularized or uses defaults.

        if self.props.symmetric:
            return [
                get_w("peak_dist"),  # Channel 0: peak_dist
                get_w("sigma"),  # Channel 1: sigma1
                0.0,  # Channel 2: amp1
            ]
        else:
            return [
                get_w("peak_dist"),  # Channel 0: peak_dist
                get_w("sigma"),  # Channel 1: sigma1
                get_w("sigma_ratio"),  # Channel 2: sigma2
                0.0,  # Channel 3: amp1
                get_w("amp_ratio"),  # Channel 4: amp2
            ]

    def _compute_channel_anchor_loss(
        self, learned_corrections: torch.Tensor, param_dim: int
    ) -> torch.Tensor:
        """Anchor learned parameter corrections toward zero (residual-based models).

        Residual models learn additive corrections to initialization values.
        Multiple parameter types (peak_dist, sigma, amp, etc.) are stored along one dimension.
        Anchoring penalizes large corrections for selected parameter types.

        Args:
            learned_corrections: Tensor storing corrections for multiple parameter types.
                                Examples:
                                - NeuralField: head.weight shaped (num_params, hidden_dim)
                                  → num_params rows, one per parameter type
                                - Grid: shaped (1, num_params, H, W)
                                  → num_params feature maps, one per parameter type
                                - Splat: payloads shaped (num_splats, num_params)
                                  → num_params values per splat, one per parameter type
                                where num_params = 3 (symmetric) or 5 (asymmetric)
            param_dim: Which dimension indexes parameter types (peak_dist, sigma, etc.):
                      - param_dim=0: parameter types are rows (weight matrices)
                      - param_dim=1: parameter types are feature dimension (grids/payloads)

        Returns:
            Scalar anchor loss (L2 penalty on selected parameter type corrections)
        """
        anchor_loss = torch.tensor(0.0, device=learned_corrections.device)
        weights = self._get_channel_weights("anchor")

        for param_idx, weight in enumerate(weights):
            if weight > 0:
                if param_dim == 0:  # Weight matrix: params indexed by rows
                    anchor_loss = anchor_loss + learned_corrections[param_idx].pow(2).mean()
                elif param_dim == 1:  # Grid/payloads: params indexed by feature dimension
                    anchor_loss = anchor_loss + learned_corrections[:, param_idx].pow(2).mean()

        return anchor_loss

    def _compute_explicit_param_anchor(self) -> torch.Tensor:
        """Anchor explicit parameters toward their initialization values.

        Explicit models (Global, PerPoint) store parameters directly as learnable tensors.
        This method penalizes deviations from initial values based on anchor flags.

        Used by:
            - GlobalOptimizableTemplateModel (single global parameters)
            - PerPointTemplateModel (per-vertex parameters)

        Returns:
            Scalar anchor loss (L2 penalty on log-space parameter deviations)
        """
        prior = torch.tensor(0.0, device=self.sigma_init.device)

        if self.props.anchor_sigma > 0:
            loss = (self.log_sigma - self.sigma_init.log()).pow(2).mean()
            prior = prior + loss * self.props.anchor_sigma

        if self.props.anchor_peak_dist > 0:
            # Reconstruct current peak_dist
            sigma1 = self.log_sigma.exp()
            if self.props.symmetric:
                sigma2 = sigma1
            else:
                sigma2 = sigma1 * self.log_sigma_ratio.exp()
            peak_dist = (sigma1 + sigma2) * (
                self.props.min_peak_ratio / 2.0 + self.log_excess.exp()
            )
            loss = (peak_dist.log() - self.peak_dist_init.log()).pow(2).mean()
            prior = prior + loss * self.props.anchor_peak_dist

        if not self.props.symmetric:
            if self.props.anchor_sigma_ratio > 0:
                loss = (self.log_sigma_ratio - self.sigma_ratio_init.log()).pow(2).mean()
                prior = prior + loss * self.props.anchor_sigma_ratio
            if self.props.anchor_amp_ratio > 0:
                loss = (self.log_amp_ratio - self.amp_ratio_init.log()).pow(2).mean()
                prior = prior + loss * self.props.anchor_amp_ratio

        return prior

    def set_topology(self, edges: torch.Tensor | None) -> None:
        """
        Sets the connectivity for regularization (optional).
        edges: (2, E) LongTensor of vertex indices.
        """
        pass

    def _decode_log_residuals(self, learned_corrections: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert learned additive corrections to final parameter values.

        Residual-based models (Neural, Grid, Splat) learn corrections in log-space:
            final_param = exp(log(init_value) + learned_correction)
        This ensures positive values and makes the model start at initialization.

        Args:
            learned_corrections: (N, num_params) tensor of additive corrections in log-space.
                                num_params = number of parameter types:
                                  - 3 for symmetric: [peak_dist, sigma, amp]
                                  - 5 for asymmetric: [peak_dist, sigma1, sigma2, amp1, amp2]

        Returns:
            Dictionary with decoded parameters: peak_dist, sigma1, sigma2, amp1, amp2
        """
        log_peak_init = self.peak_dist_init.log()
        log_sigma_init = self.sigma_init.log()
        log_amp_init = self.amp_init.log()

        # Helper to enforce min_peak_ratio constraint
        def enforce_min_separation(peak_dist, sigma1, sigma2):
            """Ensure peaks are separated by at least min_peak_ratio * sigma."""
            min_dist = (sigma1 + sigma2) * (self.props.min_peak_ratio / 2.0)
            return torch.max(peak_dist, min_dist)

        if self.props.symmetric:
            # Symmetric: learn 3 corrections [peak_dist, sigma, amp]
            peak_dist = (log_peak_init + learned_corrections[:, 0]).exp()
            sigma1 = (log_sigma_init + learned_corrections[:, 1]).exp()
            sigma2 = sigma1  # Symmetric: sigma2 = sigma1

            peak_dist = enforce_min_separation(peak_dist, sigma1, sigma2)

            amp = (log_amp_init + learned_corrections[:, 2]).exp()
            return {
                "peak_dist": peak_dist,
                "sigma1": sigma1,
                "sigma2": sigma2,
                "amp1": amp,
                "amp2": amp,  # Symmetric: amp2 = amp1
            }
        else:
            # Asymmetric: learn 5 corrections [peak_dist, sigma1, sigma2, amp1, amp2]
            log_sigma2_init = (self.sigma_init * self.sigma_ratio_init).log()
            log_amp2_init = (self.amp_init * self.amp_ratio_init).log()

            peak_dist = (log_peak_init + learned_corrections[:, 0]).exp()
            sigma1 = (log_sigma_init + learned_corrections[:, 1]).exp()
            sigma2 = (log_sigma2_init + learned_corrections[:, 2]).exp()

            peak_dist = enforce_min_separation(peak_dist, sigma1, sigma2)

            return {
                "peak_dist": peak_dist,
                "sigma1": sigma1,
                "sigma2": sigma2,
                "amp1": (log_amp_init + learned_corrections[:, 3]).exp(),
                "amp2": (log_amp2_init + learned_corrections[:, 4]).exp(),
            }


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
    """Single set of template parameters shared across all vertices.

    This is an explicit parameterization: parameters are learned directly.
    Appropriate when all vertices should have identical template properties.
    """

    def __init__(self, props: TemplateProps):
        super().__init__(props)
        # Single scalar parameters shared across entire contour
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
        """Weak prior to stay near initialization (respects anchor flags)."""
        prior = self._compute_explicit_param_anchor()
        return {
            RegularizerType.TEMPLATE_PARAM_ANCHOR.value: prior * 0.1
        }  # Proximal reg: keep params near initialization


class PerPointTemplateModel(BaseTemplateModel):
    """Independent template parameters at each vertex.

    This is an explicit parameterization with spatial regularization.
    Each vertex has its own parameters, regularized for spatial coherence
    along the contour (Laplacian smoothness).
    """

    def __init__(self, num_points: int, props: TemplateProps):
        super().__init__(props)
        # Independent learnable parameters at each vertex
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
        self.edges = None
        # Use Laplacian loss for 2nd order smoothness (penalize curvature/kinks, not slope)
        window_size = getattr(props, "smoothness_window_size", 1)
        self.laplacian_loss_fn = LaplacianSmoothingLoss(window_size=window_size)

    def set_topology(self, edges: torch.Tensor | None) -> None:
        if edges is not None:
            self.register_buffer("edges", edges, persistent=False)
        else:
            self.edges = None

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
        """Anchor and smoothness for per-point parameters.

        Anchoring: Keep parameters close to initialization
        Smoothness: Spatial coherence along contour (Laplacian)
        """
        # Anchoring using shared helper
        prior = self._compute_explicit_param_anchor()

        # Smoothness: Penalize changes along the contour
        # Need peak_dist for smoothness computation
        sigma1 = self.log_sigma.exp()
        if self.props.symmetric:
            sigma2 = sigma1
        else:
            sigma2 = sigma1 * self.log_sigma_ratio.exp()
        peak_dist = (sigma1 + sigma2) * (self.props.min_peak_ratio / 2.0 + self.log_excess.exp())
        log_peak_dist = peak_dist.log()

        # Smoothness: Penalize changes along the contour
        if self.edges is not None:
            # General graph smoothness (Dirichlet energy) for meshes
            # edges: (2, E)
            idx_u, idx_v = self.edges[0], self.edges[1]

            diff_sigma = self.log_sigma[idx_u] - self.log_sigma[idx_v]
            diff_peak = log_peak_dist[idx_u] - log_peak_dist[idx_v]
            smoothness = diff_sigma.pow(2).mean() + diff_peak.pow(2).mean()

            if not self.props.symmetric:
                diff_sigma_ratio = self.log_sigma_ratio[idx_u] - self.log_sigma_ratio[idx_v]
                diff_amp_ratio = self.log_amp_ratio[idx_u] - self.log_amp_ratio[idx_v]
                smoothness = (
                    smoothness + diff_sigma_ratio.pow(2).mean() + diff_amp_ratio.pow(2).mean()
                )
        else:
            # 1D Contour smoothness using Laplacian (2nd order)
            # Stack parameters into (N, C)
            params_list = [self.log_sigma, log_peak_dist]
            if not self.props.symmetric:
                params_list.extend([self.log_sigma_ratio, self.log_amp_ratio])

            # (N, C)
            params_stacked = torch.stack(params_list, dim=1)
            # We can't easily apply different weights inside the vectorized laplacian loss
            # without changing how params are stacked or the loss fn.
            # For simplicity in PerPoint, we assume uniform smoothness weight for now,
            # or we could compute it per channel.
            smoothness = self.laplacian_loss_fn(params_stacked)

        return {
            RegularizerType.TEMPLATE_PARAM_ANCHOR.value: prior,
            RegularizerType.TEMPLATE_PARAM_LAPLACIAN.value: smoothness,
        }


class BSplineTemplateModel(BaseTemplateModel):
    """Template parameters vary smoothly along contour via B-spline interpolation.

    This is an explicit parameterization with built-in smoothness.
    Parameters are defined at control points and interpolated along the curve.
    B-spline basis provides C² continuity automatically.
    """

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
        """Proximal regularization and smoothness for B-spline control points.

        Design: Anchor at CONTROL POINTS, not sampled curve points.

        Rationale:
        - Control points are the actual learned representation
        - More efficient than evaluating at many sample points
        - B-spline's smoothness property means anchoring control points
          naturally encourages smooth behavior in the evaluated curve

        Components:
        1. Anchoring: Keep control points near initialization (prevents collapse)
        2. Smoothness: Penalize differences between adjacent control points
        """
        # Anchoring (proximal regularization)
        reg_anchor = torch.tensor(0.0, device=self.log_control_points.device)

        # sigma1 is always at index 1
        log_sigma1_cp = self.log_control_points[1]

        if self.props.anchor_sigma > 0:
            sigma_ref = self.sigma_init.log()
            reg_anchor = (
                reg_anchor + (log_sigma1_cp - sigma_ref).pow(2).mean() * self.props.anchor_sigma
            )

        if self.props.anchor_peak_dist > 0:
            # peak_dist is at index 0
            log_peak_dist_cp = self.log_control_points[0]
            peak_dist_ref = self.peak_dist_init.log()
            reg_anchor = (
                reg_anchor
                + (log_peak_dist_cp - peak_dist_ref).pow(2).mean() * self.props.anchor_peak_dist
            )

        # Smoothness (first differences of control points) - always applied
        smooth = (log_sigma1_cp[1:] - log_sigma1_cp[:-1]).pow(2).mean() * self.props.smooth_sigma

        # Asymmetric case
        if not self.props.symmetric:
            log_sigma2_cp = self.log_control_points[3]  # sigma2 at index 3
            log_amp2_cp = self.log_control_points[4]  # amp2 at index 4

            if self.props.anchor_sigma_ratio > 0:
                sigma2_ref = (self.sigma_init * self.sigma_ratio_init).log()
                reg_anchor = (
                    reg_anchor
                    + (log_sigma2_cp - sigma2_ref).pow(2).mean() * self.props.anchor_sigma_ratio
                )

            if self.props.anchor_amp_ratio > 0:
                amp2_ref = (self.amp_init * self.amp_ratio_init).log()
                reg_anchor = (
                    reg_anchor
                    + (log_amp2_cp - amp2_ref).pow(2).mean() * self.props.anchor_amp_ratio
                )

            # Smoothness on additional channels
            smooth = (
                smooth
                + (log_sigma2_cp[1:] - log_sigma2_cp[:-1]).pow(2).mean()
                * self.props.smooth_sigma_ratio
            )
            smooth = (
                smooth
                + (log_amp2_cp[1:] - log_amp2_cp[:-1]).pow(2).mean() * self.props.smooth_amp_ratio
            )

        return {
            RegularizerType.TEMPLATE_PARAM_ANCHOR.value: reg_anchor,
            RegularizerType.TEMPLATE_PARAM_LAPLACIAN.value: smooth,
        }


class NeuralFieldTemplateModel(BaseTemplateModel):
    """Template parameters as a learned continuous function of spatial position.

    This is an implicit (residual) parameterization.
    A neural network maps (x, y) coordinates to parameter corrections.
    The MLP architecture provides implicit smoothness.

    Output: additive corrections in log-space to initialization values.
    """

    def __init__(self, props: TemplateProps, image_shape: tuple, spatial_dim: int = 2):
        super().__init__(props)
        # MLP: (x, y) -> parameter corrections [d_peak, d_sigma1, ...]
        self.image_shape = image_shape
        self.spatial_dim = spatial_dim
        layers = []
        hidden_dim = getattr(props, "neural_hidden_dim", 32)
        num_layers = getattr(props, "neural_num_layers", 2)

        in_dim = spatial_dim
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

        # Normalize coordinates to [-1, 1]
        # coordinates are (y, x) or (z, y, x)
        norm_coords = coords_to_eval.clone()
        for d in range(self.spatial_dim):
            # image_shape is (H, W) or (D, H, W)
            # coords are (y, x) or (z, y, x)
            # This matches if we assume standard indexing order
            size = self.image_shape[d]
            norm_coords[:, d] = (coords_to_eval[:, d] / (size - 1)) * 2 - 1

        # MLP expects features in last dim, which is already the case

        out = self.head(self.net(norm_coords))

        return self._decode_log_residuals(out)

    def get_regularization_loss(self) -> dict[str, torch.Tensor]:
        """Anchor network outputs for implicit smoothness.

        Neural field provides smoothness through the network architecture.
        Only anchoring is needed to prevent drift from initialization.
        """
        # Anchor output layer weights (one row per parameter type)
        anchor_loss = self._compute_channel_anchor_loss(self.head.weight, param_dim=0)

        # Also anchor bias terms for selected parameter types
        weights = self._get_channel_weights("anchor")
        for param_idx, weight in enumerate(weights):
            if weight > 0:
                anchor_loss = anchor_loss + self.head.bias[param_idx].pow(2) * weight

        return {RegularizerType.TEMPLATE_PARAM_ANCHOR.value: anchor_loss}


class GridTemplateModel(BaseTemplateModel):
    """Template parameters interpolated from a learned spatial grid.

    This is an implicit (residual) parameterization.
    A learnable grid stores parameter corrections at regular spatial locations.
    Values at arbitrary positions are obtained via bilinear/trilinear interpolation.
    Grid interpolation provides implicit smoothness.

    Grid values: additive corrections in log-space to initialization values.
    """

    def __init__(self, props: TemplateProps, image_shape: tuple, spatial_dim: int = 2):
        super().__init__(props)
        self.image_shape = image_shape
        self.spatial_dim = spatial_dim
        # Learnable grid: (1, num_params, H, W) or (1, num_params, D, H, W)
        # where num_params is the number of parameter types we're learning
        grid_size = getattr(props, "grid_size", 32)
        # Parameter types: peak_dist, sigma1, [sigma2], amp1, [amp2]
        num_params = 3 if props.symmetric else 5

        if spatial_dim == 3:
            self.grid = nn.Parameter(torch.zeros(1, num_params, grid_size, grid_size, grid_size))
        else:
            self.grid = nn.Parameter(torch.zeros(1, num_params, grid_size, grid_size))

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

        # Normalize coordinates to [-1, 1]
        # grid_sample expects (x, y, z) order (last dim is x)
        # coordinates are (y, x) or (z, y, x)

        norm_coords_list = []
        # Iterate backwards to map (z, y, x) -> (x, y, z)
        for d in reversed(range(self.spatial_dim)):
            size = self.image_shape[d]
            norm_val = (coords_detached[:, d] / (size - 1)) * 2 - 1
            norm_coords_list.append(norm_val)

        grid_coords = torch.stack(norm_coords_list, dim=-1)

        # Reshape for grid_sample:
        # 2D: (1, 1, N, 2)
        # 3D: (1, 1, 1, N, 3)
        view_shape = [1] * (self.spatial_dim) + [-1, self.spatial_dim]
        grid_coords = grid_coords.view(*view_shape)

        # Sample from grid (interpolate parameter values at query positions)
        num_params = 3 if self.props.symmetric else 5
        out = F.grid_sample(self.grid, grid_coords, align_corners=True).view(num_params, -1).T

        return self._decode_log_residuals(out)

    def get_regularization_loss(self) -> dict[str, torch.Tensor]:
        """Anchor grid values for implicit smoothness.

        Grid interpolation provides smoothness naturally.
        Only anchoring is needed to prevent drift from initialization.
        """
        # Grid shape: (1, num_params, H, W) - anchor across H,W for each param type
        anchor_loss = self._compute_channel_anchor_loss(self.grid[0], param_dim=0)
        return {RegularizerType.TEMPLATE_PARAM_ANCHOR.value: anchor_loss}


class GaussianSplatTemplateModel(BaseTemplateModel):
    """Template parameters interpolated from scattered Gaussian splats.

    This is an implicit (residual) parameterization.
    Parameters are weighted combinations of learnable 'splats' (RBF centers).
    Each splat has a position, radius, and parameter payload.
    RBF interpolation provides implicit smoothness.

    Splat payloads: additive corrections in log-space to initialization values.
    """

    def __init__(self, props: TemplateProps, image_shape: tuple, spatial_dim: int = 2):
        super().__init__(props)
        self.image_shape = image_shape
        self.spatial_dim = spatial_dim
        num_splats = getattr(props, "splat_num_splats", 32)

        shape_tensor = torch.tensor(image_shape, dtype=torch.float32)

        # Initialize splats randomly in the image domain
        self.centers = nn.Parameter(torch.rand(num_splats, spatial_dim) * shape_tensor)
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

        return self._decode_log_residuals(out)

    def get_regularization_loss(self) -> dict[str, torch.Tensor]:
        """Anchor splat payloads for implicit smoothness.

        RBF interpolation provides smoothness naturally.
        Only anchoring is needed to prevent drift from initialization.
        """
        # Payloads shape: (num_splats, num_params) - anchor across splats for each param type
        anchor_loss = self._compute_channel_anchor_loss(self.payloads, param_dim=1)
        return {RegularizerType.TEMPLATE_PARAM_ANCHOR.value: anchor_loss}
