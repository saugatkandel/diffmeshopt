import pytest
import torch

from diffmeshopt.opt2d.config import (
    RBFContourRefinerProps,
    RegularizerType,
    TemplateProps,
)
from diffmeshopt.opt2d.refiner import RBFContourRefiner
from diffmeshopt.opt2d.template import FixedTemplateModel


def test_rbf_weight_decay_loss():
    """Verify that RBF weight decay loss is computed correctly."""
    # Setup: 10 points line
    initial_contour = torch.stack([torch.arange(10, dtype=torch.float32), torch.zeros(10)], dim=1)
    # Props with specific weight decay
    props = RBFContourRefinerProps(
        rbf_num_control_points=5,
        initial_loss_weights={RegularizerType.RBF_WEIGHT_DECAY.value: 0.5},
    )
    template = FixedTemplateModel(TemplateProps())
    refiner = RBFContourRefiner(initial_contour, props, template)

    # 1. Verify initial loss is 0 (weights init to 0)
    losses = refiner.get_regularization_loss()
    assert losses[RegularizerType.RBF_WEIGHT_DECAY.value].item() == 0.0

    # 2. Manually set weights
    with torch.no_grad():
        refiner.rbf_weights.fill_(1.0)

    # Loss is mean(weights^2). Since all are 1.0, mean is 1.0.
    losses = refiner.get_regularization_loss()
    assert torch.isclose(losses[RegularizerType.RBF_WEIGHT_DECAY.value], torch.tensor(1.0))

    # 3. Verify integration with compute_losses (weighting)
    dummy_image = torch.zeros((10, 10))
    all_losses = refiner.compute_losses(dummy_image)

    loss_key = f"{RegularizerType.RBF_WEIGHT_DECAY.value}_loss"
    # Weight 0.5 * Raw 1.0 = 0.5
    assert loss_key in all_losses
    assert torch.isclose(all_losses[loss_key], torch.tensor(0.5))
