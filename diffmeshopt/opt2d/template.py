import torch
import torch.nn as nn

from diffmeshopt.opt2d.props import TemplateProps


class TemplateModel(nn.Module):
    def __init__(self, props: TemplateProps):
        super().__init__()
        self.props = props
        # Register initial values to keep them on the correct device
        self.register_buffer("peak_dist_init", torch.tensor(float(props.peak_dist)))
        self.register_buffer("sigma_init", torch.tensor(float(props.sigma)))

    def get_params(self, batch_indices: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """
        Returns a dictionary of parameters (peak_dist, sigma) for the given indices.
        If batch_indices is None, returns parameters for all points.
        """
        raise NotImplementedError

    def get_regularization_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.peak_dist_init.device)


class FixedTemplateModel(TemplateModel):
    def get_params(self, batch_indices: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        # Return scalar tensors which will be broadcasted by the loss function
        return {
            "peak_dist": self.peak_dist_init,
            "sigma": self.sigma_init,
        }


class PerPointTemplateModel(TemplateModel):
    def __init__(self, num_points: int, props: TemplateProps):
        super().__init__(props)
        # Parameters for each point
        self.log_peak_dist = nn.Parameter(torch.full((num_points,), float(props.peak_dist)).log())
        self.log_sigma = nn.Parameter(torch.full((num_points,), float(props.sigma)).log())

    def get_params(self, batch_indices: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if batch_indices is not None:
            return {
                "peak_dist": self.log_peak_dist[batch_indices].exp(),
                "sigma": self.log_sigma[batch_indices].exp(),
            }
        return {
            "peak_dist": self.log_peak_dist.exp(),
            "sigma": self.log_sigma.exp(),
        }

    def get_regularization_loss(self) -> torch.Tensor:
        # Regularization: Gaussian prior on log(sigma) centered at initialization
        return (self.log_sigma - self.sigma_init.log()).pow(2).mean()
