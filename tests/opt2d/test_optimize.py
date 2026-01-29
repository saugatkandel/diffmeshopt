import pytest
import torch

from diffmeshopt.opt2d.optimize import BSplineContourRefiner, ContourRefiner
from diffmeshopt.opt2d.props import (
    BSplineContourRefinerProps,
    BSplineTemplateProps,
    ContourRefinerProps,
    GaussianSplatTemplateProps,
    GridTemplateProps,
    NeuralFieldTemplateProps,
    TemplateProps,
)
from diffmeshopt.opt2d.template import TemplateModelFactory


@pytest.mark.parametrize("refiner_class", [ContourRefiner, BSplineContourRefiner])
@pytest.mark.parametrize(
    "template_name", ["fixed", "global", "per_point", "bspline", "neural", "grid", "splat"]
)
@pytest.mark.parametrize("template_is_symmetric", [True, False])
def test_refiner_template_combinations(
    synthetic_data, refiner_class, template_name, template_is_symmetric
):
    """
    Integration test for various combinations of Refiners and Template Models.
    Checks if optimization runs and reduces loss for each combination.
    """
    image, initial_contour = synthetic_data
    num_points = len(initial_contour)
    img_size = image.shape[-1]
    # 1. Set up properties for the refiner
    if refiner_class is ContourRefiner:
        props = ContourRefinerProps(
            num_steps=5,
            learning_rate=0.1,
            data_loss_weight=1.0,
            laplacian_loss_weight=0.1,
            edge_length_loss_weight=0.1,
        )
    elif refiner_class is BSplineContourRefiner:
        props = BSplineContourRefinerProps(
            num_steps=5,
            learning_rate=0.1,
            data_loss_weight=1.0,
            laplacian_loss_weight=0.1,
            edge_length_loss_weight=0.1,
            contour_num_control_points=16,
        )
    else:
        pytest.fail(f"Unknown refiner class: {refiner_class}")

    # 2. Set up properties for the template and create the model
    template_props = TemplateProps(
        symmetric=template_is_symmetric,
    )
    if "bspline" in template_name:
        template_props = BSplineTemplateProps(
            **template_props.__dict__, bspline_num_control_points=8
        )
    elif "neural" in template_name:
        template_props = NeuralFieldTemplateProps(**template_props.__dict__, neural_hidden_dim=16)
        # Neural fields can be sensitive to high learning rates in short tests
        if hasattr(props, "learning_rate"):
            props.learning_rate = 0.01
    elif "grid" in template_name:
        template_props = GridTemplateProps(**template_props.__dict__, grid_size=8)
    elif "splat" in template_name:
        template_props = GaussianSplatTemplateProps(**template_props.__dict__, splat_num_splats=4)

    factory_kwargs = {
        "num_vertices": num_points,
        "image_shape": (img_size, img_size),
    }
    template_model = TemplateModelFactory.create(
        template_name, props=template_props, **factory_kwargs
    )

    # 3. Instantiate Refiner
    refiner = refiner_class(initial_contour.clone(), props=props, template_model=template_model)

    # 4. Run optimization and check for convergence
    history = list(refiner.refine(image))

    assert len(history) == props.num_steps
    initial_loss = history[0]["total_loss"]
    final_loss = history[-1]["total_loss"]

    assert initial_loss is not None
    assert final_loss is not None
    # The total loss should decrease, indicating convergence.
    # Add a small tolerance for stochasticity or tricky loss landscapes
    assert final_loss < initial_loss + 1e-5
