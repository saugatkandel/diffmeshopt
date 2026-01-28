from dataclasses import dataclass


@dataclass
class ContourRefinerProps:
    """Properties for the ContourRefiner."""

    num_steps: int = 100
    learning_rate: float = 0.1
    # Loss weights
    data_loss_weight: float = 1.0
    laplacian_loss_weight: float = 50.0
    edge_length_loss_weight: float = 10.0
    sigma_reg_loss_weight: float = 1.0
    template_shape_loss_weight: float = 0.5
    template_smooth_loss_weight: float = 10.0
    # Sampling
    profile_length: int = 51
    profile_width: int = 1
    sample_step: float = 1.0
    num_sampled_profiles: int = 256
    # Geometry
    laplacian_window_size: int = 3


@dataclass
class BSplineContourRefinerProps(ContourRefinerProps):
    """Properties for the BSplineContourRefiner."""

    contour_num_control_points: int = 64
    # B-spline regularization is on control points, so weights can be smaller
    laplacian_loss_weight: float = 5.0
    edge_length_loss_weight: float = 1.0


@dataclass
class TemplateProps:
    # Common parameters
    sigma: float = 0.75
    peak_dist: float = 4.5
    amp: float = 1.0
    min_peak_ratio: float = 4.0
    sigma_ratio: float = 1.0
    amp_ratio: float = 1.0
    symmetric: bool = False

    def model_copy(self, update: dict = None):
        if update is None:
            update = {}
        return self.__class__(**{**self.__dict__, **update})


@dataclass
class BSplineTemplateProps(TemplateProps):
    # BSpline Template specific
    bspline_num_control_points: int = 10


@dataclass
class NeuralFieldTemplateProps(TemplateProps):
    # Neural Field Template specific
    neural_hidden_dim: int = 32
    neural_num_layers: int = 2


@dataclass
class GridTemplateProps(TemplateProps):
    # Grid Template specific
    grid_size: int = 32


@dataclass
class GaussianSplatTemplateProps(TemplateProps):
    # Gaussian Splat Template specific
    splat_num_splats: int = 32


@dataclass
class TrainerProps:
    """Properties for the OptimizationTrainer."""

    output_dir: str = "output"
    checkpoint_interval: int = 100
