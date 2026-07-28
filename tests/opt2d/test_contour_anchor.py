import pytest
import torch

from diffmeshopt.opt2d.config import (
    BSplineContourRefinerProps,
    ContourRefinerProps,
    RegularizerType,
    TemplateProps,
)
from diffmeshopt.opt2d.refiner import BSplineContourRefiner, VertexContourRefiner
from diffmeshopt.opt2d.template import FixedTemplateModel


def test_contour_anchor_loss_vertex():
    """Verify anchor loss for direct vertex optimization."""
    # Setup: Simple triangle
    initial_contour = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    props = ContourRefinerProps()
    # Set specific weight to verify propagation
    props.initial_loss_weights = {RegularizerType.CONTOUR_ANCHOR.value: 0.5}

    template = FixedTemplateModel(TemplateProps())
    refiner = VertexContourRefiner(initial_contour, props, template)

    # 1. Verify initial loss is 0 (starts at anchor)
    losses = refiner.get_regularization_loss()
    assert losses[RegularizerType.CONTOUR_ANCHOR.value].item() == 0.0

    # 2. Perturb contour
    with torch.no_grad():
        refiner.contour_param.add_(1.0)  # Shift all coordinates by +1.0

    # Expected MSE: mean((x+1 - x)^2) = mean(1^2) = 1.0
    # Note: Since we add 1.0 to (x, y), distance^2 is 1^2 + 1^2?
    # No, the tensor operation adds 1.0 to every element.
    # (val_new - val_old)^2 = 1.0^2 = 1.0.
    # Mean over all elements is 1.0.
    losses = refiner.get_regularization_loss()
    assert torch.isclose(losses[RegularizerType.CONTOUR_ANCHOR.value], torch.tensor(1.0))

    # 3. Verify integration with compute_losses (weighting)
    dummy_image = torch.zeros((10, 10))
    all_losses = refiner.compute_losses(dummy_image)

    anchor_loss_key = f"{RegularizerType.CONTOUR_ANCHOR.value}_loss"
    # Weight 0.5 * Raw 1.0 = 0.5
    assert anchor_loss_key in all_losses
    assert torch.isclose(all_losses[anchor_loss_key], torch.tensor(0.5))


def test_contour_anchor_loss_bspline():
    """Verify anchor loss for B-spline control points."""
    # Setup: Simple square
    initial_contour = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    # 4 control points for 4 vertices allows exact fit
    props = BSplineContourRefinerProps(num_control_points=4)
    props.initial_loss_weights = {RegularizerType.CONTOUR_ANCHOR.value: 2.0}

    template = FixedTemplateModel(TemplateProps())
    refiner = BSplineContourRefiner(initial_contour, props, template)

    # 1. Verify initial loss is negligible (least squares fit might have tiny error)
    losses = refiner.get_regularization_loss()
    assert losses[RegularizerType.CONTOUR_ANCHOR.value].item() < 1e-6

    # 2. Perturb control points
    with torch.no_grad():
        # Shift first control point by +2.0 in both x and y
        refiner.control_points[0] += 2.0

    # Expected MSE:
    # Only 1 of 4 control points moved.
    # Squared diff for that point: (x+2-x)^2 + (y+2-y)^2 = 4 + 4 = 8.
    # Mean over 4 points * 2 coords = 8 elements.
    # Wait, .pow(2).mean() is over all elements in the tensor.
    # 2 elements changed by 2.0 -> diff^2 is 4.0.
    # Sum of diffs = 4.0 + 4.0 = 8.0.
    # Total elements = 4 points * 2 coords = 8.
    # Mean = 1.0.
    losses = refiner.get_regularization_loss()
    assert torch.isclose(losses[RegularizerType.CONTOUR_ANCHOR.value], torch.tensor(1.0))

    # 3. Verify weighted loss
    dummy_image = torch.zeros((10, 10))
    all_losses = refiner.compute_losses(dummy_image)

    anchor_loss_key = f"{RegularizerType.CONTOUR_ANCHOR.value}_loss"
    # Weight 2.0 * Raw 1.0 = 2.0
    assert torch.isclose(all_losses[anchor_loss_key], torch.tensor(2.0))
