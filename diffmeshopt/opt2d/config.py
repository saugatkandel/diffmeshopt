from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum

from diffmeshopt.opt2d.enums import DataLossType, RefinerType, RegularizerType, TemplateType


class ReplaceableMixin:
    def replace(self, **changes):
        """Return a new instance with updated fields.

        This method creates a new instance of the dataclass with the specified
        changes applied. It does not modify the original instance.

        Args:
            **changes: Field names and their new values to update.

        Returns:
            A new instance of the dataclass with the specified changes.

        Caveats:
            - It is a bit less general than dataclasses.replace() because it assumes
             the class is a dataclass and uses __dict__.
            - It assumes dataclass fields are all accessible via __dict__,
             which is true for standard dataclasses but may not hold for all
             custom implementations.
            - If I want to use slots=True or init=False, or other advanced dataclass features,
             this method may need to be updated to handle those cases.
        """
        # Create a copy of the current instance's __dict__
        current_fields = deepcopy(self.__dict__)
        # Update the fields with the provided changes
        current_fields.update(changes)
        # Create a new instance of the same class with updated fields
        return self.__class__(**current_fields)


class RegularizationStrategy(Enum):
    """High-level strategies for configuring regularization weights."""

    MANUAL = "manual"  # Use provided weights directly
    TANGENTIAL_SMOOTHING = "tangential_smoothing"  # Prevent shrinking, enforce spacing
    STRONG_SMOOTHING = "strong_smoothing"  # High regularization for noisy data
    MINIMAL = "minimal"  # Data-driven, minimal constraints


@dataclass(frozen=True)
class RegularizerConfig(ReplaceableMixin):
    """Configuration for a single regularizer - single source of truth."""

    static_weight: float  # Default weight for static (non-adaptive) mode
    target_ratio: float  # Target ratio for adaptive mode (relative to data loss)


@dataclass(frozen=True)
class RegularizerDefaults(ReplaceableMixin):
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


@dataclass(frozen=True)
class AdaptiveRegularizationProps(ReplaceableMixin):
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


@dataclass(frozen=True)
class ContourRefinerProps(ReplaceableMixin):
    """Properties for contour refiners.

    This configuration is shared by vertex, B-spline, and RBF refiners.

    Regularization weights can be overridden via `initial_regularization_weights`.
    Keys must match `RegularizerType` values (either enum members or their string
    values). These overrides are validated at construction time.

    When `adaptive_reg` is enabled:
    - Weights start from static defaults in `RegularizerDefaults`
    - Can be overridden with `initial_regularization_weights`
    - Adapt during optimization according to `target_ratio`

    When `adaptive_reg` is disabled or None:
    - Uses static defaults from `RegularizerDefaults`
    - Can still be overridden with `initial_regularization_weights`

    Parameters:
        refiner_type (RefinerType): Which contour refiner this config applies to.
        num_steps (int): Number of optimization steps.
        learning_rate (float): Learning rate for the optimizer (Adam).
        data_loss_type (DataLossType): Data term used by the refiner.
        initial_regularization_weights (dict[str, float]): Overrides for default
            regularizer weights. Keys must match `RegularizerType` values.
        profile_length (int): Length of the sampled intensity profile in pixels.
        profile_width (int): Width of the sampling strip (averaging across tangent).
        sample_step (float): Step size between samples in the profile.
        num_sampled_profiles (int): Number of profiles to sample stochastically per step.
        laplacian_window_size (int): Window size for Laplacian smoothing of the contour.
        shape_loss_weight (float): Weight for the template shape consistency loss.
        closed_contour (bool): Whether the contour should be treated as cyclic/closed.
        cyclic_pad_width (int): Number of wrapped points used for cyclic padding
            in seam-sensitive local operations.
        center_symmetry_weight (float): Penalty for asymmetric profiles.
        adaptive_reg (AdaptiveRegularizationProps | None): Configuration for adaptive
            regularization weight adjustment.
        regularization_strategy (RegularizationStrategy): High-level regularization mode.
        _reg_defaults (RegularizerDefaults): Internal registry of default weights.
    """

    refiner_type: RefinerType = RefinerType.VERTEX
    num_steps: int = 100
    learning_rate: float = 0.1
    data_loss_type: DataLossType = DataLossType.BIGAUSSIAN_CORRELATION
    initial_regularization_weights: dict[str, float] = field(default_factory=dict)
    profile_length: int = 51
    profile_width: int = 5
    sample_step: float = 1.0
    num_sampled_profiles: int = 256
    laplacian_window_size: int = 3
    shape_loss_weight: float = 1.0
    closed_contour: bool = True
    cyclic_pad_width: int = 2
    center_symmetry_weight: float = 0.0
    adaptive_reg: AdaptiveRegularizationProps | None = None
    regularization_strategy: RegularizationStrategy = RegularizationStrategy.TANGENTIAL_SMOOTHING
    _reg_defaults: RegularizerDefaults = field(default_factory=RegularizerDefaults.get_defaults)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate ContourRefinerProps configuration.

        Ensures that every key in `initial_regularization_weights` matches a valid
        regularizer defined by `RegularizerType`.
        """
        valid_regularizers = RegularizerDefaults.get_all_regularizer_names()

        for key in self.initial_regularization_weights.keys():
            if isinstance(key, RegularizerType):
                key_name = key.value
            elif isinstance(key, str):
                key_name = key
            else:
                raise TypeError(
                    "initial_regularization_weights keys must be RegularizerType or str, "
                    f"got {type(key).__name__}"
                )

            if key_name not in valid_regularizers:
                raise ValueError(
                    f"Unknown regularizer '{key_name}'. "
                    f"Valid regularizers are: {sorted(valid_regularizers)}"
                )

    def get_initial_weight(self, loss_name: RegularizerType | str) -> float:
        """Get the initial weight for a regularizer at optimization start.

        Priority order:
        1. Explicit override in `initial_regularization_weights`
        2. Static default from `RegularizerDefaults`
        3. 0.0 if the regularizer is unknown

        Args:
            loss_name: Regularizer name as an enum member or string.

        Returns:
            Initial weight value.
        """
        if isinstance(loss_name, str):
            try:
                loss_name = RegularizerType(loss_name)
            except ValueError:
                return 0.0

        if loss_name in self.initial_regularization_weights:
            return self.initial_regularization_weights[loss_name]
        if loss_name.value in self.initial_regularization_weights:
            return self.initial_regularization_weights[loss_name.value]

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


@dataclass(frozen=True)
class BSplineContourRefinerProps(ContourRefinerProps):
    """Properties for the BSplineContourRefiner."""

    refiner_type: RefinerType = RefinerType.BSPLINE
    num_control_points: int = 64


@dataclass(frozen=True)
class RBFContourRefinerProps(ContourRefinerProps):
    """Properties for the RBFContourRefiner."""

    refiner_type: RefinerType = RefinerType.RBF
    num_control_points: int = 32
    kernel_sigma: float = 0.0  # If <= 0.0, auto-calculated from control point spacing
    # contour closing with ghost points
    use_ghost_control_points: bool = True
    ghost_width: int = 2


@dataclass(frozen=True)
class TemplateProps(ReplaceableMixin):
    """Configuration for template (bi-Gaussian intensity profile) models.

    Template parameters define the characteristic double-peak intensity profile.

    Regularization weights are now handled via ContourRefinerProps.initial_regularization_weights
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


@dataclass(frozen=True)
class BSplineTemplateProps(TemplateProps):
    # BSpline Template specific
    num_control_points: int = 10


@dataclass(frozen=True)
class NeuralFieldTemplateProps(TemplateProps):
    # Neural Field Template specific
    hidden_dim: int = 32
    num_layers: int = 2


@dataclass(frozen=True)
class GridTemplateProps(TemplateProps):
    # Grid Template specific
    grid_size: int = 32


@dataclass(frozen=True)
class GaussianSplatTemplateProps(TemplateProps):
    # Gaussian Splat Template specific
    splat_num_splats: int = 32


@dataclass(frozen=True)
class TrainerProps(ReplaceableMixin):
    """Properties for the OptimizationTrainer."""

    output_dir: str = "output"
    checkpoint_interval: int = 100
