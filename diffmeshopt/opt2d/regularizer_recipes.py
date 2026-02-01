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

from diffmeshopt.opt2d.props import RegularizerType

# ============================================================================
# Standard Configurations
# ============================================================================

STANDARD = {
    # Balanced weights for general use
    RegularizerType.CONTOUR_LAPLACIAN.value: 1.0,
    RegularizerType.TANGENTIAL_LAPLACIAN.value: 1.0,
    RegularizerType.NORMAL_CONSISTENCY.value: 1.0,
    RegularizerType.EDGE_LENGTH.value: 0.0,
    RegularizerType.TEMPLATE_PARAM_ANCHOR.value: 0.1,
    RegularizerType.TEMPLATE_PARAM_LAPLACIAN.value: 1.0,
    RegularizerType.TEMPLATE_SHAPE.value: 1.0,
}

MINIMAL = {
    # Minimal regularization - mostly data-driven
    RegularizerType.CONTOUR_LAPLACIAN.value: 0.0,
    RegularizerType.TANGENTIAL_LAPLACIAN.value: 0.0,
    RegularizerType.NORMAL_CONSISTENCY.value: 0.0,
    RegularizerType.EDGE_LENGTH.value: 0.0,
    RegularizerType.TEMPLATE_PARAM_ANCHOR.value: 0.01,
    RegularizerType.TEMPLATE_PARAM_LAPLACIAN.value: 0.1,
    RegularizerType.TEMPLATE_SHAPE.value: 0.1,
}


# ============================================================================
# Tangential Smoothing Strategy (Prevents Shrinking)
# ============================================================================

TANGENTIAL_SMOOTHING_VERTEX = {
    # For vertex-based refinement (ContourRefiner)
    # Disables shrinking, uses tangential regularization
    RegularizerType.CONTOUR_LAPLACIAN.value: 0.0,  # Disabled (causes shrinking)
    RegularizerType.TANGENTIAL_LAPLACIAN.value: 5.0,  # Primary: even point spacing
    RegularizerType.NORMAL_CONSISTENCY.value: 2.0,  # Secondary: smooth curvature
    RegularizerType.EDGE_LENGTH.value: 0.0,  # Disabled
    RegularizerType.TEMPLATE_PARAM_ANCHOR.value: 0.1,  # Standard template weights
    RegularizerType.TEMPLATE_PARAM_LAPLACIAN.value: 1.0,
    RegularizerType.TEMPLATE_SHAPE.value: 1.0,
}

TANGENTIAL_SMOOTHING_BSPLINE = {
    # For B-spline refinement (BSplineContourRefiner)
    # Weaker fairing due to smooth parameterization
    RegularizerType.CONTOUR_LAPLACIAN.value: 0.0,
    RegularizerType.TANGENTIAL_LAPLACIAN.value: 5.0,
    RegularizerType.NORMAL_CONSISTENCY.value: 0.5,  # Weaker for B-splines
    RegularizerType.EDGE_LENGTH.value: 0.0,
    RegularizerType.TEMPLATE_PARAM_ANCHOR.value: 0.1,
    RegularizerType.TEMPLATE_PARAM_LAPLACIAN.value: 1.0,
    RegularizerType.TEMPLATE_SHAPE.value: 1.0,
}

TANGENTIAL_SMOOTHING_RBF = {
    # For RBF refinement (RBFContourRefiner)
    # Very weak fairing for RBF
    RegularizerType.CONTOUR_LAPLACIAN.value: 0.0,
    RegularizerType.TANGENTIAL_LAPLACIAN.value: 5.0,
    RegularizerType.NORMAL_CONSISTENCY.value: 0.1,  # Much weaker for RBF
    RegularizerType.EDGE_LENGTH.value: 0.0,
    RegularizerType.TEMPLATE_PARAM_ANCHOR.value: 0.1,
    RegularizerType.TEMPLATE_PARAM_LAPLACIAN.value: 1.0,
    RegularizerType.TEMPLATE_SHAPE.value: 1.0,
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
    RegularizerType.TEMPLATE_PARAM_ANCHOR.value: 0.5,
    RegularizerType.TEMPLATE_PARAM_LAPLACIAN.value: 2.0,
    RegularizerType.TEMPLATE_SHAPE.value: 2.0,
}

DATA_DRIVEN = {
    # Let data dominate, minimal geometric constraints
    RegularizerType.CONTOUR_LAPLACIAN.value: 0.1,
    RegularizerType.TANGENTIAL_LAPLACIAN.value: 0.1,
    RegularizerType.NORMAL_CONSISTENCY.value: 0.1,
    RegularizerType.EDGE_LENGTH.value: 0.0,
    RegularizerType.TEMPLATE_PARAM_ANCHOR.value: 0.01,
    RegularizerType.TEMPLATE_PARAM_LAPLACIAN.value: 0.1,
    RegularizerType.TEMPLATE_SHAPE.value: 0.1,
}


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
}


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
    mode="per_point",
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
# To anchor only sigma but not sigma_ratio or amp_ratio, use TemplateProps flags:
# Anchor flags are supported by ALL template models:
#   - GlobalOptimizableTemplateModel: anchors global parameters
#   - PerPointTemplateModel: anchors per-point parameters
#   - BSplineTemplateModel: anchors B-spline control points
#   - NeuralFieldTemplateModel: anchors network output weights
#   - GridTemplateModel: anchors grid values (residuals)
#   - GaussianSplatTemplateModel: anchors splat payloads (residuals)
# 
# For implicit models (NeuralField, Grid, GaussianSplat), anchoring penalizes
# the learned residuals to stay small, keeping decoded parameters near initialization.

from diffmeshopt.opt2d.props import TemplateProps
from diffmeshopt.opt2d.template import TemplateModelFactory

# Configure template to anchor only sigma
template_props = TemplateProps(
    sigma=2.0,
    peak_dist=10.0,
    symmetric=False,
    # Control which parameters get anchored:
    anchor_sigma=1.0,         # Anchor sigma to initial value (2.0)
    anchor_peak_dist=0.0,     # Let peak_dist vary freely
    anchor_sigma_ratio=0.0,   # Let sigma_ratio vary freely
    anchor_amp_ratio=0.0      # Let amp_ratio vary freely
)

# Create template - it automatically respects anchor flags
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
        RegularizerType.TEMPLATE_PARAM_ANCHOR.value: 1.0,    # Strong anchor
        RegularizerType.TEMPLATE_PARAM_LAPLACIAN.value: 0.1,  # Weak smoothing
    }
)

refiner = BSplineContourRefiner(
    initial_contour=my_contour,
    props=props,
    template_model=template
)

# Result: Only sigma stays close to initialization (2.0)
#         sigma_ratio and amp_ratio vary freely (only smoothness constraint)
#         No custom template class needed!
    """)
