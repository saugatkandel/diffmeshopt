import pytest
import torch
import torch.nn as nn

from diffmeshopt.opt2d.props import (
    BSplineTemplateProps,
    GaussianSplatTemplateProps,
    GridTemplateProps,
    NeuralFieldTemplateProps,
    RegularizerType,
    TemplateProps,
)
from diffmeshopt.opt2d.template import (
    BSplineTemplateModel,
    FixedTemplateModel,
    GaussianSplatTemplateModel,
    GlobalOptimizableTemplateModel,
    GridTemplateModel,
    NeuralFieldTemplateModel,
    PerPointTemplateModel,
    TemplateModelFactory,
)


def test_per_point_template_model():
    num_points = 5
    props = TemplateProps(peak_dist=10.0, sigma=2.0)
    model = TemplateModelFactory.create("per_point", props, num_vertices=num_points)

    # Check parameters exist
    assert isinstance(model.log_excess, nn.Parameter)
    assert isinstance(model.log_sigma, nn.Parameter)

    # Check initial values
    params = model.get_params()
    assert torch.allclose(params["peak_dist"], torch.full((num_points,), 10.0), atol=1e-5)
    assert torch.allclose(params["sigma1"], torch.full((num_points,), 2.0), atol=1e-5)

    # Check subsetting
    indices = torch.tensor([0, 2])
    params_sub = model.get_params(batch_indices=indices)
    assert params_sub["peak_dist"].shape == (2,)

    # Check regularization loss
    # Initially should be 0 since we init at props values
    reg_loss = model.get_regularization_loss()
    assert torch.isclose(
        reg_loss[RegularizerType.TEMPLATE_PARAM_ANCHOR.value], torch.tensor(0.0), atol=1e-5
    )
    assert torch.isclose(
        reg_loss[RegularizerType.TEMPLATE_PARAM_LAPLACIAN.value], torch.tensor(0.0), atol=1e-5
    )

    # Modify params and check reg loss
    with torch.no_grad():
        model.log_sigma.add_(1.0)

    reg_loss = model.get_regularization_loss()
    assert reg_loss[RegularizerType.TEMPLATE_PARAM_ANCHOR.value] > 0


def test_bspline_template_model():
    num_points = 50
    props = BSplineTemplateProps(peak_dist=10.0, sigma=2.0, bspline_num_control_points=5)
    model = TemplateModelFactory.create("bspline", props, num_vertices=num_points)

    # Check parameters
    dummy_coords = torch.stack(
        [torch.linspace(0, 10, num_points), torch.linspace(0, 10, num_points)], dim=1
    )
    params = model.get_params(coordinates=dummy_coords)
    assert "sigma1" in params
    assert "sigma2" in params
    assert "amp1" in params
    assert "amp2" in params

    # Check shape
    assert params["sigma1"].shape == (num_points,)
    # Check values (should be uniform initially)
    assert torch.allclose(params["sigma1"], torch.full((num_points,), props.sigma), atol=1e-5)


def test_neural_field_template_model():
    props = NeuralFieldTemplateProps(peak_dist=10.0, sigma=2.0, neural_hidden_dim=16)
    model = TemplateModelFactory.create("neural", props, image_shape=(100, 100))

    # Dummy coordinates (N, 2)
    coords = torch.randn(10, 2)

    params = model.get_params(coordinates=coords)

    assert "peak_dist" in params
    assert params["peak_dist"].shape == (10,)

    # Check initialization (should be exactly the props values)
    assert torch.allclose(params["peak_dist"], torch.full((10,), 10.0), atol=1e-5)
    assert torch.allclose(params["sigma1"], torch.full((10,), 2.0), atol=1e-5)


def test_global_optimizable_template_model():
    props = TemplateProps(peak_dist=10.0, sigma=2.0)
    model = TemplateModelFactory.create("global", props)

    # Check parameters are learnable
    assert isinstance(model.log_excess, nn.Parameter)
    assert isinstance(model.log_sigma, nn.Parameter)

    params = model.get_params()
    assert params["peak_dist"].requires_grad
    assert params["sigma1"].requires_grad

    # Check initial values
    assert torch.isclose(params["peak_dist"], torch.tensor(10.0), atol=1e-5)
    assert torch.isclose(params["sigma1"], torch.tensor(2.0), atol=1e-5)

    # Check that it returns a single value that broadcasts
    coords = torch.randn(10, 2)
    params_multi = model.get_params(coordinates=coords)
    assert params_multi["peak_dist"].shape == ()  # Should be a scalar
    assert torch.all(params_multi["peak_dist"] == params["peak_dist"])


def test_grid_template_model():
    props = GridTemplateProps(peak_dist=10.0, sigma=2.0, grid_size=8)
    image_shape = (100, 100)
    model = TemplateModelFactory.create("grid", props, image_shape=image_shape)

    assert isinstance(model.grid, nn.Parameter)
    assert model.grid.shape == (1, 5, props.grid_size, props.grid_size)

    # Coords are in world space; model should normalize them for grid_sample
    coords = torch.tensor([[25.0, 25.0], [75.0, 75.0]])  # (N, 2)

    params = model.get_params(coordinates=coords)
    assert "peak_dist" in params
    assert params["peak_dist"].shape == (2,)
    assert params["peak_dist"].requires_grad

    # Check that gradients flow to the grid
    loss = params["peak_dist"].sum()
    loss.backward()
    assert model.grid.grad is not None


def test_gaussian_splat_template_model():
    props = GaussianSplatTemplateProps(peak_dist=10.0, sigma=2.0, splat_num_splats=4)
    image_shape = (100, 100)
    model = TemplateModelFactory.create("splat", props, image_shape=image_shape)

    assert isinstance(model.centers, nn.Parameter)
    assert model.centers.shape == (props.splat_num_splats, 2)
    assert isinstance(model.log_radius, nn.Parameter)
    assert isinstance(model.payloads, nn.Parameter)

    coords = torch.randn(10, 2)
    params = model.get_params(coordinates=coords)

    assert "peak_dist" in params
    assert params["peak_dist"].shape == (10,)
    assert params["peak_dist"].requires_grad

    # Check that gradients flow
    loss = params["peak_dist"].sum()
    loss.backward()
    assert model.centers.grad is not None
    assert model.log_radius.grad is not None
    assert model.payloads.grad is not None
