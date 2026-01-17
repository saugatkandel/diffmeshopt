import torch

from diffmeshopt.opt2d.optimize import BSplineContourRefiner, ContourRefiner


def test_contour_refiner_initialization(synthetic_data, props):
    """Test initialization of ContourRefiner."""
    image, contour_np = synthetic_data
    opt_props, samp_props, temp_props = props

    # Image needs to be numpy for Refiner init (it converts to tensor internally)
    image_np = image.numpy()[0, 0]

    refiner = ContourRefiner(
        image=image_np,
        initial_contour=contour_np,
        optimization_props=opt_props,
        sampling_props=samp_props,
        template_props=temp_props,
    )

    assert isinstance(refiner, ContourRefiner)
    assert refiner.contour.shape == contour_np.shape
    assert isinstance(refiner.contour, torch.Tensor)


def test_contour_refiner_step(synthetic_data, props):
    """Test a single optimization step of ContourRefiner."""
    image, contour_np = synthetic_data
    opt_props, samp_props, temp_props = props
    image_np = image.numpy()[0, 0]

    refiner = ContourRefiner(
        image=image_np,
        initial_contour=contour_np,
        optimization_props=opt_props,
        sampling_props=samp_props,
        template_props=temp_props,
    )

    # Run a step
    losses = refiner.step()

    # Check if losses are returned
    assert "total_loss" in losses
    assert "data_loss" in losses
    assert "laplacian_loss" in losses
    assert "edge_loss" in losses

    # Check if values are floats
    for k, v in losses.items():
        assert isinstance(v, float)


def test_bspline_refiner_step(synthetic_data, props):
    """Test a single optimization step of BSplineContourRefiner."""
    image, contour_np = synthetic_data
    opt_props, samp_props, temp_props = props
    image_np = image.numpy()[0, 0]

    num_eval_points = 60

    refiner = BSplineContourRefiner(
        image=image_np,
        initial_contour=contour_np,
        optimization_props=opt_props,
        sampling_props=samp_props,
        template_props=temp_props,
        num_control_points=10,
        num_eval_points=num_eval_points,
    )

    # Check initialization
    assert refiner.contour.shape == (num_eval_points, 2)

    # Run a step
    losses = refiner.step()

    assert "total_loss" in losses
    assert isinstance(losses["total_loss"], float)


def test_template_optimization(synthetic_data, props):
    """Test ContourRefiner with template optimization enabled."""
    image, contour_np = synthetic_data
    opt_props, samp_props, temp_props = props
    image_np = image.numpy()[0, 0]

    refiner = ContourRefiner(
        image=image_np,
        initial_contour=contour_np,
        optimization_props=opt_props,
        sampling_props=samp_props,
        template_props=temp_props,
        optimize_template=True,
    )

    # Check if template parameters are in optimizer
    # Contour params (1) + Template params (2: log_peak_dist, log_sigma)
    assert len(refiner.optimizer.param_groups[0]["params"]) == 3

    losses = refiner.step()
    assert "sigma_reg" in losses
    assert "shape_loss" in losses
