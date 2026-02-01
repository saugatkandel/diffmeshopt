import abc
import copy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffmeshopt.opt2d.geometry import (
    compute_normals,
    get_bspline_derivative_matrix,
    get_bspline_matrix,
)
from diffmeshopt.opt2d.loss import ContourLoss
from diffmeshopt.opt2d.props import (
    BSplineContourRefinerProps,
    ContourRefinerProps,
    RBFContourRefinerProps,
    RegularizerType,
)
from diffmeshopt.opt2d.sampling import sample_profiles_stochastic
from diffmeshopt.opt2d.template import BaseTemplateModel


class ContourRefinerBase(nn.Module, abc.ABC):
    """Abstract base class for contour refinement.

    Architecture:
    1. Contour representation (subclass-specific):
       - ContourRefiner: Direct vertex optimization
       - BSplineContourRefiner: B-spline control points
       - RBFContourRefiner: RBF displacement field

    2. Template model (shared):
       - Defines intensity profile parameters (peak_dist, sigma, etc.)
       - Can be fixed, global, per-point, or implicit (neural/grid/splat)

    3. Loss function (shared):
       - Data loss: Match sampled profiles to template
       - Regularizers: Smooth geometry and template parameters

    Workflow:
    - step(image) → compute losses → backward → optimizer.step()
    - Template parameters passed from template_model.get_params()
    - Template regularization from template_model.get_regularization_loss()
    - Weights can adapt during optimization (AdaptiveRegularizationProps)
    """

    def __init__(
        self,
        props: ContourRefinerProps,
        template_model: BaseTemplateModel,
    ):
        super().__init__()
        self.props = props
        self.template_model = template_model

        # Validate configuration consistency
        # This ensures that all defined regularizers have default configs in RegularizerDefaults
        self.props._reg_defaults.validate()

        # Get initial weights (computed from ratios if not explicitly set)
        # Dynamically extract all weights from props based on RegularizerType
        initial_weights = {reg.value: props.get_initial_weight(reg) for reg in RegularizerType}

        # Note: data loss always has weight=1.0
        self.loss_fn = ContourLoss(
            initial_weights=initial_weights,
            template_props=template_model.props,
            num_samples=props.profile_length,
            sample_step=props.sample_step,
            laplacian_window_size=props.laplacian_window_size,
        )

        self.optimizer = None
        self._initial_state = None

        # Adaptive regularization
        self.step_counter = 0
        self.adaptive_config = props.adaptive_reg

    @property
    @abc.abstractmethod
    def contour(self) -> torch.Tensor:
        """The evaluated contour points. Must be implemented by subclasses."""
        pass

    @property
    @abc.abstractmethod
    def points_for_regularization(self) -> torch.Tensor:
        """The points to apply geometric regularization to (e.g., vertices or control points)."""
        pass

    @property
    def normals(self) -> torch.Tensor:
        """Returns unit normals at each contour point. Default is finite differences."""
        return compute_normals(self.contour)

    def create_optimizer(self) -> torch.optim.Optimizer:
        """Creates the optimizer configured for this refiner."""
        return torch.optim.Adam(self.parameters(), lr=self.props.learning_rate)

    def configure_optimizer(self):
        self.optimizer = self.create_optimizer()

    def sample_image_features(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample intensity profiles from the image.

        Returns:
            Tuple of (profiles, sub_indices, valid_mask)
            - profiles: (N, L) sampled intensities
            - sub_indices: (N,) indices of sampled vertices
            - valid_mask: (N,) boolean mask of valid samples
        """
        # We pass self.normals (computed on full contour) to ensure accurate sampling directions.
        # If we let sample_profiles_stochastic compute normals on the subsampled vertices,
        # it would calculate chord vectors rather than true tangents, leading to errors
        # at high curvature regions or sparse sampling rates.
        profiles, sub_indices, valid_mask = sample_profiles_stochastic(
            image,
            self.contour,
            profile_length=self.props.profile_length,
            profile_width=self.props.profile_width,
            sample_step=self.props.sample_step,
            num_samples=self.props.num_sampled_profiles,
            normals=self.normals,
        )
        return profiles, sub_indices, valid_mask

    def compute_losses(
        self,
        image: torch.Tensor,
        sampling_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute all losses for the current contour state.

        Args:
            image: Input image
            sampling_data: Optional pre-computed (profiles, sub_indices, valid_mask) to avoid re-sampling.
        """
        if sampling_data is None:
            sampling_data = self.sample_image_features(image)

        profiles, sub_indices, valid_mask = sampling_data

        template_params = self.template_model.get_params(
            batch_indices=sub_indices, coordinates=self.contour
        )

        # Template regularization from template model (anchor, smoothness)
        # Returns dict like {"template_param_anchor": loss, "template_param_laplacian": loss}
        reg_losses = self.template_model.get_regularization_loss()

        # Check if explicit edges are defined (e.g. for 3D meshes)
        edges = getattr(self, "edges", None)

        losses = self.loss_fn(
            profiles,
            self.points_for_regularization,
            vertices=self.contour,
            normals=self.normals,
            edges=edges,
            **template_params,
            mask=valid_mask,
            reg_losses=reg_losses,
        )

        return losses

    def forward_pass(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Performs the full forward pass: sampling, adaptive update, and loss computation.

        This unifies the logic for both manual stepping and Lightning training steps,
        ensuring consistent behavior for adaptive regularization and sampling.
        """
        # 1. Sample image features once for this step
        sampling_data = self.sample_image_features(image)

        # 2. Adaptive weight adjustment (if enabled)
        if (
            self.adaptive_config is not None
            and self.adaptive_config.enabled
            and self.step_counter >= self.adaptive_config.warmup_steps
            and self.step_counter % self.adaptive_config.update_interval == 0
        ):
            self._update_adaptive_weights(image, sampling_data=sampling_data)

        # 3. Compute Losses
        return self.compute_losses(image, sampling_data=sampling_data)

    def step(self, image: torch.Tensor) -> dict[str, float]:
        if self.optimizer is None:
            self.configure_optimizer()

        self.optimizer.zero_grad()
        losses = self.forward_pass(image)
        losses["total_loss"].backward()
        self.optimizer.step()
        self.step_counter += 1
        return {k: v.item() for k, v in losses.items()}

    def refine(self, image: torch.Tensor):
        """Generator that yields optimization history for a number of steps."""
        if self.optimizer is None:
            self.configure_optimizer()

        for step in range(self.props.num_steps):
            losses = self.step(image)
            history_item = {
                "step": step,
                "contour": self.contour.detach().clone(),
                **losses,
            }
            yield history_item

    def _update_adaptive_weights(
        self,
        image: torch.Tensor,
        sampling_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ):
        """Adjust regularization weights based on loss magnitudes."""
        with torch.no_grad():
            # Compute losses to access raw values
            _ = self.compute_losses(image, sampling_data=sampling_data)
            raw_losses = self.loss_fn._raw_losses

            # Data loss magnitude (reference for scaling)
            L_data = raw_losses.get("data", torch.tensor(0.0)).item()
            if L_data < 1e-8:
                return  # Can't adapt if data loss is negligible

            # EMA parameters
            alpha = self.adaptive_config.ema_alpha

            # Update each regularization weight
            # Note: get_weight/set_weight work for all regularizers automatically
            # because ContourLoss dynamically registers weight buffers from RegularizerType
            for loss_name, L_reg_tensor in raw_losses.items():
                if loss_name == "data":
                    continue  # Don't adapt data loss weight

                L_reg = L_reg_tensor.item()
                if L_reg < 1e-8:
                    continue  # Skip if regularization is negligible

                # Get target ratio from props (single source of truth)
                target_ratio = self.props.get_target_ratio(loss_name)

                # Compute target weight: w = (target_ratio * L_data) / L_reg
                w_target = target_ratio * L_data / L_reg

                # Clamp to reasonable range
                w_target = np.clip(
                    w_target, self.adaptive_config.min_weight, self.adaptive_config.max_weight
                )

                # Get current weight buffer
                weight_attr = f"w_{loss_name}"
                try:
                    current_w = self.loss_fn.get_weight(loss_name).item()
                    # EMA update: w_new = alpha * w_target + (1 - alpha) * w_old
                    new_w = alpha * w_target + (1 - alpha) * current_w
                    # Update weight in-place
                    self.loss_fn.set_weight(loss_name, new_w)
                except KeyError:
                    # Weight not found (shouldn't happen if RegularizerType is consistent)
                    continue

    def export_state(self) -> dict:
        """
        Exports the current state of the contour and template parameters as numpy arrays.
        """
        with torch.no_grad():
            contour_np = self.contour.detach().cpu().numpy()
            params_torch = self.template_model.get_params(coordinates=self.contour)
            params_np = {
                k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
                for k, v in params_torch.items()
            }
            return {
                "contour": contour_np,
                "template_params": params_np,
                "mode": self.template_model.mode.value if self.template_model.mode else "unknown",
            }

    def capture_initial_state(self):
        """Captures the current state dict as the initial state."""
        self._initial_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}

    def reset(self):
        """Resets the parameters to the captured initial state."""
        if self._initial_state is not None:
            self.load_state_dict(self._initial_state)
            self.optimizer = None
        else:
            logging.warning("reset() called but no initial state was captured.")

    def _set_template_topology(self, num_points: int, device: torch.device):
        """
        Sets the default 1D cyclic topology for the template model.
        """
        indices = torch.arange(num_points, device=device)
        next_indices = torch.roll(indices, shifts=-1, dims=0)
        edges = torch.stack([indices, next_indices], dim=0)
        self.template_model.set_topology(edges)


class ContourRefiner(ContourRefinerBase):
    """Refines a 2D contour by optimizing its vertex positions directly."""

    def __init__(
        self,
        initial_contour: torch.Tensor,
        props: ContourRefinerProps,
        template_model: BaseTemplateModel,
    ):
        super().__init__(props, template_model)
        self.contour_param = nn.Parameter(initial_contour.clone())
        self.capture_initial_state()

    @property
    def contour(self) -> torch.Tensor:
        return self.contour_param

    @property
    def points_for_regularization(self) -> torch.Tensor:
        return self.contour_param


class BSplineContourRefiner(ContourRefinerBase):
    """Refines a 2D contour by optimizing B-spline control points."""

    def __init__(
        self,
        initial_contour: torch.Tensor,
        props: BSplineContourRefinerProps,
        template_model: BaseTemplateModel,
    ):
        super().__init__(props, template_model)
        num_control_points = props.contour_num_control_points
        num_eval_points = len(initial_contour)

        logging.info(
            f"Initializing BSplineContourRefiner with {num_control_points} control points, {num_eval_points} eval points."
        )

        # Fit initial control points to the initial contour
        M_init = get_bspline_matrix(
            num_control_points, len(initial_contour), device=initial_contour.device
        )
        target = initial_contour.float()
        initial_cp = torch.linalg.lstsq(M_init, target).solution
        self.control_points = nn.Parameter(initial_cp)

        # Precompute evaluation and derivative matrices
        self.register_buffer(
            "M_eval",
            get_bspline_matrix(num_control_points, num_eval_points, device=initial_contour.device),
            persistent=False,
        )
        self.register_buffer(
            "M_deriv",
            get_bspline_derivative_matrix(
                num_control_points, num_eval_points, device=initial_contour.device
            ),
            persistent=False,
        )
        self.capture_initial_state()

    @property
    def contour(self):
        return self.M_eval @ self.control_points

    @property
    def points_for_regularization(self) -> torch.Tensor:
        return self.control_points

    @property
    def normals(self) -> torch.Tensor:
        """Analytical normals from B-spline derivatives."""
        tangents = self.M_deriv @ self.control_points
        normals = torch.stack([-tangents[:, 1], tangents[:, 0]], dim=1)
        return F.normalize(normals, dim=-1)


def create_tangential_smoothing_refiner(
    refiner_class: type[ContourRefinerBase],
    initial_contour: torch.Tensor,
    props: ContourRefinerProps,
    template_model: BaseTemplateModel,
    adaptive: bool = True,
    enable_adaptive_weights: bool = False,
) -> ContourRefinerBase:
    """
    Factory function that creates a refiner configured for Tangential Smoothing.

    Tangential Smoothing prevents shrinking by:
    - Disabling shrinking forces (Laplacian/Edge weights = 0)
    - Enabling Tangential Spacing regularization (vertices slide along surface)
    - Enabling Normal Consistency (Fairing) regularization (surface smoothness)

    Args:
        refiner_class: The refiner class to instantiate (ContourRefiner, BSplineContourRefiner, etc.)
        initial_contour: Initial contour vertices
        props: Refiner properties (will be copied and modified)
        template_model: Template model for intensity prior
        adaptive: If True, uses heuristic adaptive weights based on contour properties
                  and refiner type. If False, uses fixed defaults.
        enable_adaptive_weights: If True, enables dynamic weight adaptation during optimization
                                 based on loss magnitudes (experimental).

    Returns:
        Configured refiner instance

    Example:
        >>> # Static heuristic weights
        >>> refiner = create_tangential_smoothing_refiner(BSplineContourRefiner, contour, props, template)
        >>> # Enable dynamic adaptation
        >>> refiner = create_tangential_smoothing_refiner(
        ...     BSplineContourRefiner, contour, props, template, enable_adaptive_weights=True
        ... )
    """
    from diffmeshopt.opt2d.props import AdaptiveRegularizationProps

    # Clone props to avoid side effects
    props = copy.copy(props)

    # Enforce the Tangential Smoothing configuration (disable shrinking)
    props.initial_loss_weights[RegularizerType.CONTOUR_LAPLACIAN.value] = 0.0
    props.initial_loss_weights[RegularizerType.EDGE_LENGTH.value] = 0.0

    # Set tangential regularization weights
    if enable_adaptive_weights:
        # Enable dynamic weight adaptation during optimization
        from diffmeshopt.opt2d.props import RegularizerConfig, RegularizerDefaults

        adaptive_cfg = AdaptiveRegularizationProps(
            enabled=True,
            update_interval=10,
            ema_alpha=0.1,
            warmup_steps=5,
        )

        # Customize regularizer defaults for tangential smoothing
        reg_defaults = RegularizerDefaults.get_defaults()

        # Adjust fairing ratio based on refiner type for tangential smoothing
        if refiner_class.__name__ == "BSplineContourRefiner":
            # Weaker for smooth parameterizations
            reg_defaults.regularizers[RegularizerType.NORMAL_CONSISTENCY] = RegularizerConfig(
                static_weight=0.5, target_ratio=0.05
            )
        elif refiner_class.__name__ == "RBFContourRefiner":
            # Even weaker for RBF
            reg_defaults.regularizers[RegularizerType.NORMAL_CONSISTENCY] = RegularizerConfig(
                static_weight=0.2, target_ratio=0.02
            )
        else:
            # Standard for vertex-based (keep defaults)
            pass

        props.adaptive_reg = adaptive_cfg
        props._reg_defaults = reg_defaults

    elif adaptive:
        # Use heuristic adaptive weights based on structure
        if props.initial_loss_weights.get(RegularizerType.TANGENTIAL_LAPLACIAN.value, 0.0) == 0.0:
            # Adaptive spacing weight based on contour resolution
            num_vertices = len(initial_contour)

            # For B-splines, use control points instead
            if hasattr(props, "contour_num_control_points"):
                num_effective = props.contour_num_control_points
            elif hasattr(props, "rbf_num_control_points"):
                num_effective = props.rbf_num_control_points
            else:
                num_effective = num_vertices

            # Base weight scaled by log of resolution
            base_spacing = 2.0
            props.initial_loss_weights[RegularizerType.TANGENTIAL_LAPLACIAN.value] = (
                base_spacing * np.log(1 + num_effective / 50)
            )

        if props.initial_loss_weights.get(RegularizerType.NORMAL_CONSISTENCY.value, 0.0) == 0.0:
            # Adaptive fairing weight based on refiner type
            if refiner_class.__name__ == "BSplineContourRefiner":
                props.initial_loss_weights[RegularizerType.NORMAL_CONSISTENCY.value] = 0.5
            elif refiner_class.__name__ == "RBFContourRefiner":
                props.initial_loss_weights[RegularizerType.NORMAL_CONSISTENCY.value] = 0.1
            else:
                props.initial_loss_weights[RegularizerType.NORMAL_CONSISTENCY.value] = 2.0
    else:
        # Fixed defaults
        if props.initial_loss_weights.get(RegularizerType.TANGENTIAL_LAPLACIAN.value, 0.0) == 0.0:
            props.initial_loss_weights[RegularizerType.TANGENTIAL_LAPLACIAN.value] = 5.0
        if props.initial_loss_weights.get(RegularizerType.NORMAL_CONSISTENCY.value, 0.0) == 0.0:
            props.initial_loss_weights[RegularizerType.NORMAL_CONSISTENCY.value] = 1.0

    spacing_w = props.get_initial_weight(RegularizerType.TANGENTIAL_LAPLACIAN)
    fairing_w = props.get_initial_weight(RegularizerType.NORMAL_CONSISTENCY)

    logging.info(
        f"Tangential smoothing configured for {refiner_class.__name__}: "
        f"spacing={spacing_w:.2f}, fairing={fairing_w:.2f}, "
        f"adaptive_weights={'enabled' if enable_adaptive_weights else 'disabled'}"
    )

    # Instantiate the refiner with modified props
    return refiner_class(initial_contour, props, template_model)


class TangentialSmoothingContourRefiner(ContourRefiner):
    """
    A specialized refiner that enforces Tangential Smoothing to prevent shrinking.

    This is a convenience wrapper around create_tangential_smoothing_refiner() for
    backward compatibility. For new code, consider using the factory function directly.

    Note: This only works with vertex-based refinement. For B-splines or RBF,
    use create_tangential_smoothing_refiner() with the appropriate class.
    """

    def __init__(
        self,
        initial_contour: torch.Tensor,
        props: ContourRefinerProps,
        template_model: BaseTemplateModel,
    ):
        # Clone props to avoid side effects on the passed object
        props = copy.copy(props)
        props.initial_loss_weights = props.initial_loss_weights.copy()

        # Enforce the specific configuration for this strategy
        props.initial_loss_weights[RegularizerType.CONTOUR_LAPLACIAN.value] = 0.0
        props.initial_loss_weights[RegularizerType.EDGE_LENGTH.value] = 0.0

        # Set default tangential regularization weights if not specified
        if props.initial_loss_weights.get(RegularizerType.TANGENTIAL_LAPLACIAN.value, 0.0) == 0.0:
            props.initial_loss_weights[RegularizerType.TANGENTIAL_LAPLACIAN.value] = 5.0
        if props.initial_loss_weights.get(RegularizerType.NORMAL_CONSISTENCY.value, 0.0) == 0.0:
            props.initial_loss_weights[RegularizerType.NORMAL_CONSISTENCY.value] = (
                2.0  # Higher for vertex-based
            )

        super().__init__(initial_contour, props, template_model)


class RBFContourRefiner(ContourRefinerBase):
    """
    Refines a contour using Radial Basis Function (RBF) deformation.
    This approach is mesh-free and generalizes trivially to 3D.
    """

    def __init__(
        self,
        initial_contour: torch.Tensor,
        props: RBFContourRefinerProps,
        template_model: BaseTemplateModel,
    ):
        super().__init__(props, template_model)

        self.register_buffer("initial_contour", initial_contour.clone())

        # 1. Select Control Points (Centers)
        # Simple strided subsampling of the initial contour
        num_cp = props.rbf_num_control_points
        num_points = len(initial_contour)
        if num_cp >= num_points:
            indices = torch.arange(num_points, device=initial_contour.device)
        else:
            indices = torch.linspace(
                0, num_points - 1, num_cp, device=initial_contour.device
            ).long()

        self.register_buffer("control_points", initial_contour[indices].clone())

        # 2. Initialize Weights (Parameters)
        # Weights represent the displacement vectors at the control points
        # Initialize to zero (no deformation)
        self.rbf_weights = nn.Parameter(torch.zeros_like(self.control_points))

        # 3. Precompute Kernel Matrix (Gaussian)
        # Phi_ij = exp(-||x_i - c_j||^2 / 2sigma^2)
        dists = torch.cdist(self.initial_contour, self.control_points)  # (N, K)
        kernel_matrix = torch.exp(-(dists.pow(2)) / (2 * props.rbf_kernel_sigma**2))

        # Normalize rows (Partition of Unity) to ensure translation reproduction
        # This prevents vertices "sticking" in gaps between control points
        kernel_sum = kernel_matrix.sum(dim=1, keepdim=True)
        self.register_buffer("kernel_matrix", kernel_matrix / (kernel_sum + 1e-8))

        self.capture_initial_state()

    @property
    def contour(self) -> torch.Tensor:
        # V' = V + Phi @ W
        displacement = self.kernel_matrix @ self.rbf_weights
        return self.initial_contour + displacement

    @property
    def points_for_regularization(self) -> torch.Tensor:
        # We regularize the output contour to ensure valid geometry
        return self.contour
