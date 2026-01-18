import pytest
import torch
import torch.nn as nn

from diffmeshopt.opt2d.props import TemplateProps
from diffmeshopt.opt2d.template import (
    BSplineTemplateModel,
    FixedTemplateModel,
    NeuralFieldTemplateModel,
    PerPointTemplateModel,
)


def test_fixed_template_model():
    props = TemplateProps(peak_dist=10.0, sigma=2.0)
    model = FixedTemplateModel(props)

    params = model.get_params()
    assert "peak_dist" in params
    assert "sigma" in params
    assert params["peak_dist"] == 10.0
    assert params["sigma"] == 2.0

    # Check gradients (should be none or not require grad)
    # FixedTemplateModel registers buffers, so they don't require grad by default unless wrapped?
    # Actually they are tensors.
    assert not params["peak_dist"].requires_grad


def test_per_point_template_model():
    num_points = 5
    props = TemplateProps(peak_dist=10.0, sigma=2.0)
    model = PerPointTemplateModel(num_points, props)

    # Check parameters exist
    assert isinstance(model.log_peak_dist, nn.Parameter)
    assert isinstance(model.log_sigma, nn.Parameter)

    # Check initial values
    params = model.get_params()
    assert torch.allclose(params["peak_dist"], torch.full((num_points,), 10.0))
    assert torch.allclose(params["sigma"], torch.full((num_points,), 2.0))

    # Check subsetting
    indices = torch.tensor([0, 2])
    params_sub = model.get_params(indices)
    assert params_sub["peak_dist"].shape == (2,)

    # Check regularization loss
    # Initially should be 0 since we init at props values
    reg_loss = model.get_regularization_loss()
    assert torch.isclose(reg_loss, torch.tensor(0.0))

    # Modify params and check reg loss
    with torch.no_grad():
        model.log_sigma.add_(1.0)

    reg_loss = model.get_regularization_loss()
    assert reg_loss > 0


def test_bspline_template_model():
    num_points = 50
    props = TemplateProps(peak_dist=10.0, sigma=2.0, num_control_points=5)
    model = BSplineTemplateModel(num_points, props)

    # Check parameters
    params = model.get_params()
    assert "sigma1" in params
    assert "sigma2" in params
    assert "amp1" in params
    assert "amp2" in params

    # Check shape
    assert params["sigma1"].shape == (num_points,)
    # Check values (should be uniform initially)
    assert torch.allclose(params["sigma1"], torch.full((num_points,), 2.0), atol=1e-5)


def test_neural_field_template_model():
    props = TemplateProps(peak_dist=10.0, sigma=2.0, neural_hidden_dim=16)
    model = NeuralFieldTemplateModel(props)

    # Dummy coordinates (N, 2)
    coords = torch.randn(10, 2)

    params = model.get_params(coordinates=coords)

    assert "peak_dist" in params
    assert params["peak_dist"].shape == (10,)

    # Check initialization (should be exactly the props values)
    assert torch.allclose(params["peak_dist"], torch.full((10,), 10.0), atol=1e-5)
    assert torch.allclose(params["sigma1"], torch.full((10,), 2.0), atol=1e-5)
