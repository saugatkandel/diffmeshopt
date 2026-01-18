from pydantic import BaseModel


class TemplateProps(BaseModel):
    peak_dist: float = 6.0
    sigma: float = 0.75
    amp: float = 1.0
    num_samples: int = 21
    num_control_points: int = 10
    neural_hidden_dim: int = 32
    neural_num_layers: int = 2


class SamplingProps(BaseModel):
    num_samples: int = 21
    sample_step: float = 1.0
    width: int = 3
    batch_size: int = 100


class OptimizationProps(BaseModel):
    lr: float = 1.0
    w_data: float = 1.0
    w_laplacian: float = 0.1
    w_edge: float = 0.1
    w_sigma_reg: float = 0.1
    w_template_shape: float = 0.1
    batch_size: int = 100
