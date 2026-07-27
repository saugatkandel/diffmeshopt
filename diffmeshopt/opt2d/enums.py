from enum import Enum


class RefinerType(Enum):
    """Enumeration of supported contour refiner types."""

    VERTEX = "vertex"
    BSPLINE = "bspline"
    RBF = "rbf"


class TemplateType(Enum):
    PER_POINT = "per_point"
    GLOBAL = "global"
    FIXED = "fixed"
    BSPLINE = "bspline"
    NEURAL = "neural"
    GRID = "grid"
    SPLAT = "splat"


class DataLossType(Enum):
    BIGAUSSIAN_CORRELATION = "bigaussian_correlation"
    BIGAUSSIAN_WASSERSTEIN = "bigaussian_wasserstein"


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
