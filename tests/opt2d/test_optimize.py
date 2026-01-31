import pytest
import torch

from diffmeshopt.opt2d.optimize import (
    BSplineContourRefiner,
    ContourRefiner,
    GradientSurgeryContourRefiner,
    RBFContourRefiner,
)
from diffmeshopt.opt2d.props import (
    BSplineContourRefinerProps,
    BSplineTemplateProps,
    ContourRefinerProps,
    GaussianSplatTemplateProps,
    GridTemplateProps,
    NeuralFieldTemplateProps,
    RBFContourRefinerProps,
    TemplateProps,
)
from diffmeshopt.opt2d.template import TemplateModelFactory


@pytest.mark.parametrize(
    "refiner_class",
    [
        ContourRefiner,
        BSplineContourRefiner,
        GradientSurgeryContourRefiner,
        RBFContourRefiner,
    ],
)
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
    if refiner_class in (ContourRefiner, GradientSurgeryContourRefiner):
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
    elif refiner_class is RBFContourRefiner:
        props = RBFContourRefinerProps(
            num_steps=5,
            learning_rate=0.1,
            data_loss_weight=1.0,
            rbf_num_control_points=8,
            rbf_kernel_sigma=10.0,
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


def test_gradient_surgery_no_shrinking():
    """
    Verifies that GradientSurgeryContourRefiner does not shrink a circle
    in the absence of data forces, whereas standard ContourRefiner does.
    """
    # Create a circle
    theta = torch.linspace(0, 2 * torch.pi, 51)[:-1]
    radius = 20.0
    initial_contour = torch.stack([radius * torch.sin(theta), radius * torch.cos(theta)], dim=1)

    # Dummy image (flat, so data loss gradient is 0)
    image = torch.zeros(1, 1, 100, 100)

    # 1. Standard Refiner (Shrinking)
    props_std = ContourRefinerProps(
        num_steps=20,
        learning_rate=0.5,
        data_loss_weight=0.0,  # No data force
        laplacian_loss_weight=1.0,  # Strong shrinking force
        edge_length_loss_weight=0.0,
    )
    template_model = TemplateModelFactory.create("fixed", props=TemplateProps())

    refiner_std = ContourRefiner(initial_contour.clone(), props_std, template_model)
    list(refiner_std.refine(image))  # Run

    final_radius_std = torch.norm(refiner_std.contour, dim=1).mean().item()
    assert final_radius_std < radius - 1.0  # Should have shrunk significantly

    # 2. Gradient Surgery Refiner (Non-Shrinking)
    props_gs = ContourRefinerProps(
        num_steps=20,
        learning_rate=0.5,
        data_loss_weight=0.0,
        spacing_loss_weight=5.0,  # Strong spacing force
        fairing_loss_weight=1.0,
    )
    # Note: GradientSurgeryContourRefiner forces laplacian=0 internally

    refiner_gs = GradientSurgeryContourRefiner(initial_contour.clone(), props_gs, template_model)
    list(refiner_gs.refine(image))

    final_radius_gs = torch.norm(refiner_gs.contour, dim=1).mean().item()

    # Should stay roughly the same size
    assert abs(final_radius_gs - radius) < 0.5
    assert final_radius_gs > final_radius_std


def test_rbf_refiner_movement():
    """
    Test that RBF refiner moves the whole contour using sparse control points.
    Verifies that points NOT in the control set still move (solving 'left behind' issue).
    """
    # Create a line of 10 points
    initial_contour = torch.stack(
        [torch.zeros(10), torch.linspace(0, 10, 10)], dim=1
    )  # x=0, y=0..10

    # Use only 2 control points (start and end)
    props = RBFContourRefinerProps(
        rbf_num_control_points=2,
        rbf_kernel_sigma=50.0,  # Large sigma to influence everything
        learning_rate=1.0,
    )
    template_model = TemplateModelFactory.create("fixed", props=TemplateProps())

    refiner = RBFContourRefiner(initial_contour, props, template_model)

    # Manually perturb the weights to simulate optimization
    # Move control points by +1 in x direction
    with torch.no_grad():
        refiner.rbf_weights.add_(1.0)

    # Check the resulting contour
    new_contour = refiner.contour

    # All points should have moved in x, even though we only have 2 control points
    # Because sigma is large, the movement should be roughly uniform +1
    assert torch.all(new_contour[:, 0] > 0.5)
