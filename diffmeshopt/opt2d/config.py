from dataclasses import dataclass, field
from enum import Enum


class RegularizerType(Enum):
    """Central registry of all regularizer loss types in the system.

    Single source of truth for loss names - prevents typos and enables
    IDE autocomplete, type safety, and refactoring support.

    Naming convention: <domain>_<operator>
    - Contour regularizers: operate on contour geometry (positions, normals)
    - Template regularizers: operate on template parameters (sigma, peak_dist, etc.)

    Note: DATA loss is not included here as it always has weight=1.0
    and is never adaptive. It serves as the reference for all regularizers.
    """

    # Contour geometry regularizers
    CONTOUR_LAPLACIAN = "contour_laplacian"  # Laplacian smoothing on positions
    EDGE_LENGTH = "edge_length"  # Edge length uniformity
    TANGENTIAL_LAPLACIAN = "tangential_laplacian"  # Tangential Laplacian for point spacing
    NORMAL_CONSISTENCY = "normal_consistency"  # Normal/curvature smoothness (fairing)
    CONTOUR_ANCHOR = "contour_anchor"  # Anchor to initialization (L2 distance)

    # Template parameter regularizers
    # Anchors (L2 proximity to initialization)
    ANCHOR_SIGMA = "anchor_sigma"
    ANCHOR_PEAK_DIST = "anchor_peak_dist"
    ANCHOR_SIGMA_RATIO = "anchor_sigma_ratio"
    ANCHOR_AMP_RATIO = "anchor_amp_ratio"

    # Smoothness (Laplacian/Spatial)
    SMOOTH_SIGMA = "smooth_sigma"
    SMOOTH_PEAK_DIST = "smooth_peak_dist"
    SMOOTH_SIGMA_RATIO = "smooth_sigma_ratio"
    SMOOTH_AMP_RATIO = "smooth_amp_ratio"

    # Refiner parameter regularizers
    RBF_WEIGHT_DECAY = "rbf_weight_decay"  # Penalize RBF weight magnitude


class RegularizationStrategy(Enum):
    """High-level strategies for configuring regularization weights."""

    MANUAL = "manual"  # Use provided weights directly
    TANGENTIAL_SMOOTHING = "tangential_smoothing"  # Prevent shrinking, enforce spacing
    STRONG_SMOOTHING = "strong_smoothing"  # High regularization for noisy data
    MINIMAL = "minimal"  # Data-driven, minimal constraints


@dataclass
class RegularizerConfig:
    """Configuration for a single regularizer - single source of truth."""

    static_weight: float  # Default weight for static (non-adaptive) mode
    target_ratio: float  # Target ratio for adaptive mode (relative to data loss)


@dataclass
class RegularizerDefaults:
    """Single source of truth for all regularizers in the codebase.

    To add/remove a regularizer:
    1. Add/remove entry in RegularizerType enum
    2. Add/remove entry in this regularizers dict
    3. Add/remove loss computation in ContourLoss.forward()

    - static_weight: Used when adaptive_reg is disabled
    - target_ratio: Target for adaptive weight adjustment (L_reg/L_data ratio)
    """

    regularizers: dict[RegularizerType, RegularizerConfig] = field(
        default_factory=lambda: {
            # Weight Derivation (Force Balance Heuristic):
            # We want the regularization force to balance the data force at a specific
            # displacement limit D (e.g., 5 pixels).
            # F_data approx 1 / sigma_template
            # F_reg approx 2 * weight * D
            # weight approx 1 / (2 * D * sigma_template)
            # For D=5.0, sigma=1.0 -> weight = 0.1
            # Contour geometry regularizers
            RegularizerType.TANGENTIAL_LAPLACIAN: RegularizerConfig(
                static_weight=5.0,  # Strong force to ensure uniform spacing
                target_ratio=0.0,  # Static constraint
            ),
            RegularizerType.NORMAL_CONSISTENCY: RegularizerConfig(
                static_weight=2.0,
                target_ratio=0.0,  # Static constraint
            ),
            RegularizerType.CONTOUR_LAPLACIAN: RegularizerConfig(
                static_weight=0.0, target_ratio=0.0
            ),
            RegularizerType.EDGE_LENGTH: RegularizerConfig(static_weight=0.0, target_ratio=0.0),
            RegularizerType.CONTOUR_ANCHOR: RegularizerConfig(
                static_weight=0.1,
                target_ratio=0.01,  # Allows ~5px movement
            ),
            # Template parameter regularizers
            RegularizerType.ANCHOR_SIGMA: RegularizerConfig(static_weight=0.0, target_ratio=0.0),
            RegularizerType.ANCHOR_PEAK_DIST: RegularizerConfig(
                static_weight=0.0, target_ratio=0.0
            ),
            RegularizerType.ANCHOR_SIGMA_RATIO: RegularizerConfig(
                static_weight=0.0, target_ratio=0.01
            ),
            RegularizerType.ANCHOR_AMP_RATIO: RegularizerConfig(
                static_weight=0.0, target_ratio=0.01
            ),
            RegularizerType.SMOOTH_SIGMA: RegularizerConfig(static_weight=1.0, target_ratio=0.05),
            RegularizerType.SMOOTH_PEAK_DIST: RegularizerConfig(
                static_weight=1.0, target_ratio=0.05
            ),
            RegularizerType.SMOOTH_SIGMA_RATIO: RegularizerConfig(
                static_weight=1.0, target_ratio=0.05
            ),
            RegularizerType.SMOOTH_AMP_RATIO: RegularizerConfig(
                static_weight=0.0, target_ratio=0.01
            ),
            RegularizerType.RBF_WEIGHT_DECAY: RegularizerConfig(
                static_weight=0.1,
                target_ratio=0.0,  # Static (physics-based, allows ~5px movement)
            ),
        }
    )

    @classmethod
    def get_defaults(cls) -> "RegularizerDefaults":
        """Get the default regularizer configuration."""
        return cls()

    @classmethod
    def get_all_regularizer_names(cls) -> set[str]:
        """Get all regularizer names (string values from RegularizerType enum).

        Useful for validation and consistency checks.

        Returns:
            Set of regularizer name strings (e.g., {"contour_laplacian", ...})
        """
        return {reg.value for reg in RegularizerType}

    def validate(self) -> None:
        """Validate that all RegularizerType entries have configs.

        Call this explicitly when you want to check consistency.
        Not called automatically to avoid issues with PyTorch Lightning serialization.
        """
        defined_regs = set(self.regularizers.keys())
        all_regs = set(RegularizerType)

        missing = all_regs - defined_regs
        if missing:
            raise ValueError(
                f"RegularizerDefaults missing configs for: {[r.value for r in missing]}. "
                f"Add entries to the regularizers dict."
            )

        extra = defined_regs - all_regs
        if extra:
            raise ValueError(
                f"RegularizerDefaults has configs for unknown regularizers: {[r.value for r in extra]}. "
                f"Either add to RegularizerType enum or remove from dict."
            )


@dataclass
class AdaptiveRegularizationProps:
    """Configuration for adaptive regularization weight adjustment.

    Note: Target ratios are defined in RegularizerDefaults (single source of truth).

    Parameters:
        enabled (bool): Whether to enable adaptive weight adjustment.
        update_interval (int): How often (in steps) to update weights.
        ema_alpha (float): Exponential moving average factor for weight updates (0.0 to 1.0).
                           Higher values mean faster adaptation, lower means smoother.
        warmup_steps (int): Number of steps to wait before starting adaptation.
        min_weight (float): Minimum allowed weight value to prevent collapse.
        max_weight (float): Maximum allowed weight value to prevent explosion.
    """

    enabled: bool = False
    update_interval: int = 10
    ema_alpha: float = 0.1
    warmup_steps: int = 5
    min_weight: float = 1e-6
    max_weight: float = 100.0


@dataclass
class ContourRefinerProps:
    """Properties for the ContourRefiner.

    Loss weights can be overridden via initial_loss_weights dict.
    Keys must match those in RegularizerDefaults (single source of truth).

    When adaptive_reg is enabled:
    - Weights start from static defaults in RegularizerDefaults
    - Can be overridden with initial_loss_weights
    - Adapt during optimization according to target_ratios

    When adaptive_reg is None/disabled:
    - Uses static defaults from RegularizerDefaults
    - Can be overridden with initial_loss_weights

    Parameters:
        num_steps (int): Number of optimization steps.
        learning_rate (float): Learning rate for the optimizer (Adam).
        initial_loss_weights (dict): Overrides for default regularizer weights.
                                     Keys should match RegularizerType values.
        profile_length (int): Length of the sampled intensity profile in pixels.
        profile_width (int): Width of the sampling strip (averaging across tangent).
        sample_step (float): Step size between samples in the profile (usually 1.0 pixel).
        num_sampled_profiles (int): Number of profiles to sample stochastically per step.
        laplacian_window_size (int): Window size for Laplacian smoothing of the contour.
        shape_loss_weight (float): Weight for the template shape consistency loss (part of data term).
        adaptive_reg (AdaptiveRegularizationProps | None): Configuration for adaptive weights.
        _reg_defaults (RegularizerDefaults): Internal registry of default weights.
    """

    num_steps: int = 100
    learning_rate: float = 0.1
    # Optional overrides for initial weights (dict keys match RegularizerDefaults)
    # Note: data loss always has weight=1.0 and cannot be overridden
    initial_loss_weights: dict[str, float] = field(
        default_factory=dict  # Regularizers: if not specified, uses RegularizerDefaults
    )
    # Sampling
    profile_length: int = 51
    profile_width: int = 1
    sample_step: float = 1.0
    num_sampled_profiles: int = 256
    # Geometry
    laplacian_window_size: int = 3
    shape_loss_weight: float = 1.0
    # Adaptive regularization (optional)
    adaptive_reg: AdaptiveRegularizationProps | None = None
    regularization_strategy: RegularizationStrategy = RegularizationStrategy.TANGENTIAL_SMOOTHING
    # Global regularizer defaults (single source of truth)
    _reg_defaults: RegularizerDefaults = field(default_factory=RegularizerDefaults.get_defaults)

    def get_initial_weight(self, loss_name: RegularizerType | str) -> float:
        """Get initial weight for a regularizer at optimization start.

        Priority order (first match wins):
        1. Explicit value in initial_loss_weights dict (user override)
        2. Static default from RegularizerDefaults (global defaults)
        3. Zero if regularizer not in RegularizerDefaults (unknown loss)

        This method is called once at ContourLoss initialization.
        If adaptive_reg is enabled, weights will be adjusted during optimization.

        Args:
            loss_name: Regularizer name (RegularizerType enum or string)

        Returns:
            Initial weight value (float >= 0)
        """
        # Convert string to RegularizerType if needed
        if isinstance(loss_name, str):
            try:
                loss_name = RegularizerType(loss_name)
            except ValueError:
                # Unknown loss name, return 0.0
                return 0.0

        # Check if explicitly overridden (support both string and enum keys)
        if loss_name in self.initial_loss_weights:
            return self.initial_loss_weights[loss_name]
        if loss_name.value in self.initial_loss_weights:
            return self.initial_loss_weights[loss_name.value]

        # Use static default from single source of truth
        if loss_name in self._reg_defaults.regularizers:
            return self._reg_defaults.regularizers[loss_name].static_weight

        return 0.0

    def get_target_ratio(self, loss_name: RegularizerType | str) -> float:
        """Get target ratio for adaptive regularizer weight adjustment.

        Target ratio defines the desired balance: L_reg / L_data
        Used only when adaptive_reg is enabled.

        During optimization, adaptive weight adjustment computes:
            w_new = (target_ratio * L_data) / L_reg
        to maintain the target balance as losses evolve.

        Args:
            loss_name: Regularizer name (RegularizerType enum or string)

        Returns:
            Target ratio from RegularizerDefaults, or 0.1 if not found (fallback)
        """
        # Convert string to RegularizerType if needed
        if isinstance(loss_name, str):
            try:
                loss_name = RegularizerType(loss_name)
            except ValueError:
                return 0.1  # Fallback for unknown losses

        if loss_name in self._reg_defaults.regularizers:
            return self._reg_defaults.regularizers[loss_name].target_ratio
        return 0.1  # Fallback default


@dataclass
class BSplineContourRefinerProps(ContourRefinerProps):
    """Properties for the BSplineContourRefiner."""

    contour_num_control_points: int = 64


@dataclass
class RBFContourRefinerProps(ContourRefinerProps):
    """Properties for the RBFContourRefiner."""

    rbf_num_control_points: int = 32
    rbf_kernel_sigma: float = 0.0  # If <= 0.0, auto-calculated from control point spacing


@dataclass
class TemplateProps:
    """Configuration for template (bi-Gaussian intensity profile) models.

    Template parameters define the characteristic double-peak intensity profile.

    Regularization weights are now handled via ContourRefinerProps.initial_loss_weights
    using RegularizerType keys (e.g. "anchor_sigma", "smooth_peak_dist").

    Parameters:
        sigma (float): Standard deviation (width) of the main Gaussian peak.
                       Controls the blur/thickness of the membrane wall.
        peak_dist (float): Distance between the centers of the two Gaussian peaks in pixels.
                           Represents the membrane thickness.
        min_peak_ratio (float): Constraint to prevent peaks from merging.
                                Enforces peak_dist >= (sigma1 + sigma2) * (min_peak_ratio / 2.0).
                                If symmetric, peak_dist >= sigma * min_peak_ratio.
                                A value of 4.0 ensures peaks are separated by at least 4 sigmas.
        sigma_ratio (float): Ratio of the second Gaussian's width to the first (sigma2 / sigma1).
                             Only used if symmetric=False.
        amp_ratio (float): Ratio of the second Gaussian's amplitude to the first (amp2 / amp1).
                           Only used if symmetric=False.
        symmetric (bool): If True, enforces sigma1=sigma2 and amp1=amp2.
                          Symmetric templates often provide more stable gradients for positioning.
        smoothness_window_size (int): Window size for spatial smoothness regularization of
                                      template parameters (used in PerPoint models).
    """

    # Common parameters
    sigma: float = 0.75
    peak_dist: float = 4.5
    min_peak_ratio: float = 2.0
    sigma_ratio: float = 1.0
    amp_ratio: float = 1.0
    symmetric: bool = True
    smoothness_window_size: int = 3

    @property
    def amp(self) -> float:
        return 1.0

    def model_copy(self, update: dict = None):
        if update is None:
            update = {}
        return self.__class__(**{**self.__dict__, **update})


@dataclass
class BSplineTemplateProps(TemplateProps):
    # BSpline Template specific
    bspline_num_control_points: int = 10


@dataclass
class NeuralFieldTemplateProps(TemplateProps):
    # Neural Field Template specific
    neural_hidden_dim: int = 32
    neural_num_layers: int = 2


@dataclass
class GridTemplateProps(TemplateProps):
    # Grid Template specific
    grid_size: int = 32


@dataclass
class GaussianSplatTemplateProps(TemplateProps):
    # Gaussian Splat Template specific
    splat_num_splats: int = 32


@dataclass
class TrainerProps:
    """Properties for the OptimizationTrainer."""

    output_dir: str = "output"
    checkpoint_interval: int = 100
