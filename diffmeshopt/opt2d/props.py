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

    # Template parameter regularizers
    TEMPLATE_PARAM_ANCHOR = "template_param_anchor"  # L2 proximity to initial parameter values
    TEMPLATE_PARAM_LAPLACIAN = "template_param_laplacian"  # Laplacian spatial smoothness on params
    TEMPLATE_SHAPE = "template_shape"  # Shape/profile consistency


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
            # Contour geometry regularizers
            RegularizerType.TANGENTIAL_LAPLACIAN: RegularizerConfig(
                static_weight=1.0, target_ratio=0.1
            ),
            RegularizerType.NORMAL_CONSISTENCY: RegularizerConfig(
                static_weight=1.0, target_ratio=0.1
            ),
            RegularizerType.CONTOUR_LAPLACIAN: RegularizerConfig(
                static_weight=0.0, target_ratio=0.1
            ),
            RegularizerType.EDGE_LENGTH: RegularizerConfig(static_weight=0.0, target_ratio=0.1),
            # Template parameter regularizers
            RegularizerType.TEMPLATE_PARAM_ANCHOR: RegularizerConfig(
                static_weight=0.1, target_ratio=0.01
            ),
            RegularizerType.TEMPLATE_PARAM_LAPLACIAN: RegularizerConfig(
                static_weight=1.0, target_ratio=0.05
            ),
            RegularizerType.TEMPLATE_SHAPE: RegularizerConfig(static_weight=1.0, target_ratio=0.1),
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
    # Adaptive regularization (optional)
    adaptive_reg: AdaptiveRegularizationProps | None = None
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
    rbf_kernel_sigma: float = 20.0


@dataclass
class TemplateProps:
    """Configuration for template (bi-Gaussian intensity profile) models.

    Template parameters define the characteristic double-peak intensity profile:
    - peak_dist: Distance between the two Gaussian peaks
    - sigma: Width of the Gaussians (symmetric) or sigma1 (asymmetric)
    - amp: Amplitude of the Gaussians (symmetric) or amp1 (asymmetric)
    - sigma_ratio: sigma2 = sigma1 * sigma_ratio (asymmetric only)
    - amp_ratio: amp2 = amp1 * amp_ratio (asymmetric only)

    Anchor flags control which parameters are regularized toward initialization:
    - anchor_sigma: Keep sigma close to initial value (prevents collapse)
    - anchor_peak_dist: Keep peak_dist close to initial value (rare - usually let vary)
    - anchor_sigma_ratio: Keep sigma ratio close to initial value (asymmetric only)
    - anchor_amp_ratio: Keep amp ratio close to initial value (asymmetric - rarely used)

    The anchor flags are used by template_model.get_regularization_loss() to compute
    the template_param_anchor loss term, which is then weighted in ContourLoss.
    """

    # Common parameters
    sigma: float = 0.75
    peak_dist: float = 4.5
    amp: float = 1.0
    min_peak_ratio: float = 4.0
    sigma_ratio: float = 1.0
    amp_ratio: float = 1.0
    symmetric: bool = False
    smoothness_window_size: int = 1

    # Relative regularization weights per parameter (0.0 to disable)
    # Anchoring: Keep close to initialization
    anchor_sigma: float = 1.0
    anchor_peak_dist: float = 0.0
    anchor_sigma_ratio: float = 1.0
    anchor_amp_ratio: float = 0.0

    # Smoothing: Penalize spatial variation (Laplacian)
    smooth_sigma: float = 1.0
    smooth_peak_dist: float = 1.0
    smooth_sigma_ratio: float = 1.0
    smooth_amp_ratio: float = 1.0

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
