"""
Common regularizer weight configurations ("recipes").

These are reference examples you can copy/modify. Simply copy the dict and
adjust values as needed for your use case.

Usage:
    from diffmeshopt.opt2d.regularizer_recipes import TANGENTIAL_SMOOTHING

    props = ContourRefinerProps(
        initial_loss_weights=TANGENTIAL_SMOOTHING.copy()  # Always copy!
    )

    # Optionally customize specific weights
    props.initial_loss_weights["normal_consistency"] = 3.0
"""

from diffmeshopt.opt2d.config import RegularizationStrategy, RegularizerType

# ============================================================================
# Standard Configurations
# ============================================================================

STANDARD = {
    # Balanced weights for general use
    RegularizerType.CONTOUR_LAPLACIAN.value: 1.0,
    RegularizerType.TANGENTIAL_LAPLACIAN.value: 1.0,
    RegularizerType.NORMAL_CONSISTENCY.value: 1.0,
    RegularizerType.EDGE_LENGTH.value: 0.0,
    RegularizerType.CONTOUR_ANCHOR.value: 0.1,  # Safety anchor (allows ~5px drift)
    # Template params
    RegularizerType.ANCHOR_SIGMA.value: 0.0,
    RegularizerType.ANCHOR_PEAK_DIST.value: 0.0,
    RegularizerType.SMOOTH_SIGMA.value: 1.0,
    RegularizerType.SMOOTH_PEAK_DIST.value: 1.0,
    RegularizerType.RBF_WEIGHT_DECAY.value: 0.1,  # Safety for RBF
}

MINIMAL = {
    # Minimal regularization - mostly data-driven
    RegularizerType.CONTOUR_LAPLACIAN.value: 0.0,
    RegularizerType.TANGENTIAL_LAPLACIAN.value: 0.0,
    RegularizerType.NORMAL_CONSISTENCY.value: 0.0,
    RegularizerType.EDGE_LENGTH.value: 0.0,
    RegularizerType.CONTOUR_ANCHOR.value: 0.0,
    # Template params
    RegularizerType.ANCHOR_SIGMA.value: 0.0,
    RegularizerType.SMOOTH_SIGMA.value: 0.1,
    RegularizerType.SMOOTH_PEAK_DIST.value: 0.1,
    RegularizerType.RBF_WEIGHT_DECAY.value: 0.1,
}


# ============================================================================
# Tangential Smoothing Strategy (Prevents Shrinking)
# ============================================================================

TANGENTIAL_SMOOTHING_VERTEX = {
    # For vertex-based refinement (ContourRefiner)
    # Disables shrinking, uses tangential regularization
    RegularizerType.CONTOUR_LAPLACIAN.value: 0.1,  # Weak shrinking to constrain expansion
    RegularizerType.TANGENTIAL_LAPLACIAN.value: 5.0,  # Primary: even point spacing
    RegularizerType.NORMAL_CONSISTENCY.value: 2.0,  # Secondary: smooth curvature
    RegularizerType.EDGE_LENGTH.value: 0.0,  # Disabled
    RegularizerType.CONTOUR_ANCHOR.value: 0.1,  # Safety anchor
    # Template params
    RegularizerType.ANCHOR_SIGMA.value: 0.0,
    RegularizerType.SMOOTH_SIGMA.value: 1.0,
    RegularizerType.SMOOTH_PEAK_DIST.value: 1.0,
    RegularizerType.RBF_WEIGHT_DECAY.value: 0.1,  # Prevent large deformations
}

TANGENTIAL_SMOOTHING_BSPLINE = {
    # For B-spline refinement (BSplineContourRefiner)
    # Shrinking is DISABLED: B-splines suffer from corner-cutting if control points are shrunk.
    RegularizerType.CONTOUR_LAPLACIAN.value: 0.0,  # Disabled
    RegularizerType.TANGENTIAL_LAPLACIAN.value: 5.0,  # Essential for control point spacing
    RegularizerType.NORMAL_CONSISTENCY.value: 0.0,  # Disabled: Splines are inherently C2 smooth
    RegularizerType.EDGE_LENGTH.value: 0.0,
    RegularizerType.CONTOUR_ANCHOR.value: 0.1,  # Safety anchor
    # Template params
    RegularizerType.ANCHOR_SIGMA.value: 0.0,
    RegularizerType.SMOOTH_SIGMA.value: 1.0,
    RegularizerType.SMOOTH_PEAK_DIST.value: 1.0,
    RegularizerType.RBF_WEIGHT_DECAY.value: 0.1,  # Prevent large deformations
}

TANGENTIAL_SMOOTHING_RBF = {
    # For RBF refinement (RBFContourRefiner)
    # Shrinking is DISABLED: RBF shape is controlled by weight decay.
    RegularizerType.CONTOUR_LAPLACIAN.value: 0.0,  # Disabled
    RegularizerType.TANGENTIAL_LAPLACIAN.value: 0.0,  # Disabled: Centers are fixed
    RegularizerType.NORMAL_CONSISTENCY.value: 0.0,  # Disabled: Field is inherently smooth
    RegularizerType.EDGE_LENGTH.value: 0.0,
    RegularizerType.CONTOUR_ANCHOR.value: 0.0,  # RBF uses WEIGHT_DECAY instead
    # Template params
    RegularizerType.ANCHOR_SIGMA.value: 0.0,
    RegularizerType.SMOOTH_SIGMA.value: 1.0,
    RegularizerType.SMOOTH_PEAK_DIST.value: 1.0,
    RegularizerType.RBF_WEIGHT_DECAY.value: 0.1,  # Prevent large deformations
}


# ============================================================================
# Specialized Configurations
# ============================================================================

STRONG_SMOOTHING = {
    # Aggressive smoothing for noisy data
    RegularizerType.CONTOUR_LAPLACIAN.value: 2.0,
    RegularizerType.TANGENTIAL_LAPLACIAN.value: 2.0,
    RegularizerType.NORMAL_CONSISTENCY.value: 3.0,
    RegularizerType.EDGE_LENGTH.value: 1.0,
    RegularizerType.CONTOUR_ANCHOR.value: 0.1,  # Anchor to help with noise
    # Template params
    RegularizerType.ANCHOR_SIGMA.value: 0.0,
    RegularizerType.SMOOTH_SIGMA.value: 2.0,
    RegularizerType.SMOOTH_PEAK_DIST.value: 2.0,
}

DATA_DRIVEN = {
    # Let data dominate, minimal geometric constraints
    RegularizerType.CONTOUR_LAPLACIAN.value: 0.1,
    RegularizerType.TANGENTIAL_LAPLACIAN.value: 0.1,
    RegularizerType.NORMAL_CONSISTENCY.value: 0.1,
    RegularizerType.EDGE_LENGTH.value: 0.0,
    RegularizerType.CONTOUR_ANCHOR.value: 0.0,
    # Template params
    RegularizerType.ANCHOR_SIGMA.value: 0.0,
    RegularizerType.SMOOTH_SIGMA.value: 0.1,
    RegularizerType.SMOOTH_PEAK_DIST.value: 0.1,
}


ANCHORED_SIGMA = STANDARD.copy()
ANCHORED_SIGMA[RegularizerType.ANCHOR_SIGMA.value] = 0.1

ANCHORED_ALL = STANDARD.copy()
ANCHORED_ALL[RegularizerType.ANCHOR_SIGMA.value] = 0.1
ANCHORED_ALL[RegularizerType.ANCHOR_PEAK_DIST.value] = 0.1
ANCHORED_ALL[RegularizerType.ANCHOR_SIGMA_RATIO.value] = 0.1
ANCHORED_ALL[RegularizerType.ANCHOR_AMP_RATIO.value] = 0.1
ANCHORED_ALL[RegularizerType.CONTOUR_ANCHOR.value] = 0.1

# ============================================================================
# Quick Reference Guide
# ============================================================================

RECIPES_BY_USE_CASE = {
    "general": STANDARD,
    "vertex_based": TANGENTIAL_SMOOTHING_VERTEX,
    "bspline": TANGENTIAL_SMOOTHING_BSPLINE,
    "rbf": TANGENTIAL_SMOOTHING_RBF,
    "noisy_data": STRONG_SMOOTHING,
    "clean_data": DATA_DRIVEN,
    "minimal": MINIMAL,
    "anchored_sigma": ANCHORED_SIGMA,
    "anchored_all": ANCHORED_ALL,
}


def resolve_strategy(
    strategy: RegularizationStrategy,
    refiner_type: str,
    base_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Resolves a high-level strategy into specific weight overrides.

    Args:
        strategy: The high-level strategy to apply.
        refiner_type: The class name of the refiner (e.g. "BSplineContourRefiner").
        base_weights: Optional existing weights to merge/override.

    Returns:
        Dictionary of {regularizer_name: weight}.
    """
    weights = base_weights.copy() if base_weights else {}

    if strategy == RegularizationStrategy.MANUAL:
        return weights

    recipe = {}

    if strategy == RegularizationStrategy.TANGENTIAL_SMOOTHING:
        # Select appropriate recipe based on refiner type
        if "BSpline" in refiner_type:
            recipe = TANGENTIAL_SMOOTHING_BSPLINE
        elif "RBF" in refiner_type:
            recipe = TANGENTIAL_SMOOTHING_RBF
        else:
            # Default to vertex-based
            recipe = TANGENTIAL_SMOOTHING_VERTEX

    elif strategy == RegularizationStrategy.STRONG_SMOOTHING:
        recipe = STRONG_SMOOTHING

    elif strategy == RegularizationStrategy.MINIMAL:
        recipe = MINIMAL

    # Merge recipe into weights
    # Strategy defines the baseline, but user-provided base_weights take precedence.
    for k, v in recipe.items():
        if k not in weights:
            weights[k] = v

    return weights


def print_recipe(name: str):
    """Print a recipe in readable format."""
    if name not in RECIPES_BY_USE_CASE:
        print(f"Unknown recipe: {name}")
        print(f"Available: {list(RECIPES_BY_USE_CASE.keys())}")
        return

    print(f"\n{name.upper()} Configuration:")
    print("-" * 60)
    recipe = RECIPES_BY_USE_CASE[name]
    for key, value in recipe.items():
        print(f"  {key:40s}: {value}")


if __name__ == "__main__":
    print("=" * 60)
    print("REGULARIZER WEIGHT RECIPES")
    print("=" * 60)

    for name in RECIPES_BY_USE_CASE.keys():
        print_recipe(name)

    print("\n" + "=" * 60)
    print("USAGE EXAMPLES:")
    print("=" * 60)
    print("""
# Example 1: Use recipe as-is with BSplineContourRefiner
from diffmeshopt.opt2d.regularizer_recipes import TANGENTIAL_SMOOTHING_BSPLINE
from diffmeshopt.opt2d.props import (
    ContourRefinerProps, BSplineContourRefinerProps,
    TemplateProps, RegularizerType
)
from diffmeshopt.opt2d.optimize import BSplineContourRefiner
from diffmeshopt.opt2d.template import TemplateModelFactory

# Setup template with per-point parameters
template_props = TemplateProps(
    sigma=2.0,
    peak_dist=10.0,
    symmetric=True
)
template = TemplateModelFactory.create(
    mode="per_point",
    props=template_props,
    num_vertices=100
)

# Use recipe for refiner
props = BSplineContourRefinerProps(
    initial_loss_weights=TANGENTIAL_SMOOTHING_BSPLINE.copy()
)

refiner = BSplineContourRefiner(
    initial_contour=my_contour,
    props=props,
    template_model=template
)

# Example 2: Customize template anchor to only apply to sigma (not other params)
# This requires creating a custom template class - see below
    """)

    print("\n" + "=" * 60)
    print("ADVANCED: Anchor only specific params (e.g., sigma only)")
    print("=" * 60)
    print("""
# To anchor only sigma but not sigma_ratio or amp_ratio, use RegularizerType weights:
# Anchor weights are now controlled via initial_loss_weights in ContourRefinerProps.
#   - ANCHOR_SIGMA: anchors sigma to initialization
#   - ANCHOR_PEAK_DIST: anchors peak_dist to initialization
#   - ANCHOR_SIGMA_RATIO: anchors sigma_ratio (asymmetry)
#   - ANCHOR_AMP_RATIO: anchors amp_ratio (asymmetry)
#
# For implicit models (NeuralField, Grid, GaussianSplat), anchoring penalizes
# the learned residuals to stay small, keeping decoded parameters near initialization.

from diffmeshopt.opt2d.props import TemplateProps
from diffmeshopt.opt2d.template import TemplateModelFactory

# Configure template
template_props = TemplateProps(
    sigma=2.0,
    peak_dist=10.0,
    symmetric=False
)

# Create template
template = TemplateModelFactory.create(
    mode="per_point",
    props=template_props,
    num_vertices=100
)

# Configure weights
from diffmeshopt.opt2d.props import BSplineContourRefinerProps
from diffmeshopt.opt2d.optimize import BSplineContourRefiner

props = BSplineContourRefinerProps(
    initial_loss_weights={
        RegularizerType.TANGENTIAL_LAPLACIAN.value: 5.0,
        RegularizerType.NORMAL_CONSISTENCY.value: 0.5,
        RegularizerType.CONTOUR_LAPLACIAN.value: 0.0,
        # Control which parameters get anchored:
        RegularizerType.ANCHOR_SIGMA.value: 1.0,      # Anchor sigma
        RegularizerType.ANCHOR_PEAK_DIST.value: 0.0,  # Free peak_dist
        RegularizerType.ANCHOR_SIGMA_RATIO.value: 0.0,# Free asymmetry
        RegularizerType.SMOOTH_SIGMA.value: 0.1,      # Weak smoothing on sigma
    }
)

refiner = BSplineContourRefiner(
    initial_contour=my_contour,
    props=props,
    template_model=template
)

# Result: Only sigma stays close to initialization (2.0)
#         sigma_ratio and amp_ratio vary freely (only smoothness constraint)
    """)
