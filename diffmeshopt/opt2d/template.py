import torch
import torch.nn as nn

from diffmeshopt.opt2d.geometry import get_bspline_matrix
from diffmeshopt.opt2d.props import TemplateProps


class TemplateModel(nn.Module):
    def __init__(self, props: TemplateProps):
        super().__init__()
        self.props = props
        # Register initial values to keep them on the correct device
        self.register_buffer("peak_dist_init", torch.tensor(float(props.peak_dist)))
        self.register_buffer("sigma_init", torch.tensor(float(props.sigma)))

    def get_params(
        self,
        batch_indices: torch.Tensor | None = None,
        coordinates: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Returns a dictionary of parameters (peak_dist, sigma) for the given indices.
        coordinates: (N, 2) tensor of spatial positions (used for Neural Fields).
        If batch_indices is None, returns parameters for all points.
        """
        raise NotImplementedError

    def get_regularization_loss(self) -> dict[str, torch.Tensor]:
        return {}


class FixedTemplateModel(TemplateModel):
    def get_params(
        self,
        batch_indices: torch.Tensor | None = None,
        coordinates: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # Return scalar tensors which will be broadcasted by the loss function
        return {
            "peak_dist": self.peak_dist_init,
            "sigma": self.sigma_init,
        }


class GlobalOptimizableTemplateModel(TemplateModel):
    def __init__(self, props: TemplateProps):
        super().__init__(props)
        # Single scalar parameters for the whole contour
        self.log_peak_dist = nn.Parameter(torch.tensor(float(props.peak_dist)).log())
        self.log_sigma = nn.Parameter(torch.tensor(float(props.sigma)).log())

    def get_params(
        self,
        batch_indices: torch.Tensor | None = None,
        coordinates: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return {
            "peak_dist": self.log_peak_dist.exp(),
            "sigma": self.log_sigma.exp(),
        }

    def get_regularization_loss(self) -> dict[str, torch.Tensor]:
        # Weak prior to stay near initialization
        prior = (self.log_sigma - self.sigma_init.log()).pow(2) + (
            self.log_peak_dist - self.peak_dist_init.log()
        ).pow(2)
        return {"sigma_reg": prior}


class PerPointTemplateModel(TemplateModel):
    def __init__(self, num_points: int, props: TemplateProps):
        super().__init__(props)
        # Parameters for each point
        self.log_peak_dist = nn.Parameter(torch.full((num_points,), float(props.peak_dist)).log())
        self.log_sigma = nn.Parameter(torch.full((num_points,), float(props.sigma)).log())

    def get_params(
        self,
        batch_indices: torch.Tensor | None = None,
        coordinates: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if batch_indices is not None:
            return {
                "peak_dist": self.log_peak_dist[batch_indices].exp(),
                "sigma": self.log_sigma[batch_indices].exp(),
            }
        return {
            "peak_dist": self.log_peak_dist.exp(),
            "sigma": self.log_sigma.exp(),
        }

    def get_regularization_loss(self) -> dict[str, torch.Tensor]:
        # Regularization: Gaussian prior on log(sigma) centered at initialization
        prior = (self.log_sigma - self.sigma_init.log()).pow(2).mean()

        # Smoothness: Penalize changes along the contour
        diff_sigma = self.log_sigma - torch.roll(self.log_sigma, shifts=1, dims=0)
        diff_peak = self.log_peak_dist - torch.roll(self.log_peak_dist, shifts=1, dims=0)
        smoothness = diff_sigma.pow(2).mean() + diff_peak.pow(2).mean()

        return {"sigma_reg": prior, "template_smooth": smoothness}


class BSplineTemplateModel(TemplateModel):
    def __init__(self, num_eval_points: int, props: TemplateProps):
        super().__init__(props)
        self.num_cp = props.num_control_points
        self.num_eval = num_eval_points

        # Precompute B-spline evaluation matrix
        # Shape: (num_eval, num_cp)
        self.register_buffer("M", get_bspline_matrix(self.num_cp, self.num_eval))

        # Initialize control points in log space for positivity
        # Parameters: peak_dist, sigma1, sigma2, amp1, amp2
        self.log_peak_dist_cp = nn.Parameter(
            torch.full((self.num_cp,), float(props.peak_dist)).log()
        )
        self.log_sigma1_cp = nn.Parameter(torch.full((self.num_cp,), float(props.sigma)).log())
        self.log_sigma2_cp = nn.Parameter(torch.full((self.num_cp,), float(props.sigma)).log())
        self.log_amp1_cp = nn.Parameter(torch.full((self.num_cp,), float(props.amp)).log())
        self.log_amp2_cp = nn.Parameter(torch.full((self.num_cp,), float(props.amp)).log())

    def get_params(
        self,
        batch_indices: torch.Tensor | None = None,
        coordinates: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # Evaluate B-splines to get dense parameters
        # (num_eval, num_cp) @ (num_cp,) -> (num_eval,)
        peak_dist = (self.M @ self.log_peak_dist_cp).exp()
        sigma1 = (self.M @ self.log_sigma1_cp).exp()
        sigma2 = (self.M @ self.log_sigma2_cp).exp()
        amp1 = (self.M @ self.log_amp1_cp).exp()
        amp2 = (self.M @ self.log_amp2_cp).exp()

        if batch_indices is not None:
            return {
                "peak_dist": peak_dist[batch_indices],
                "sigma1": sigma1[batch_indices],
                "sigma2": sigma2[batch_indices],
                "amp1": amp1[batch_indices],
                "amp2": amp2[batch_indices],
            }
        return {
            "peak_dist": peak_dist,
            "sigma1": sigma1,
            "sigma2": sigma2,
            "amp1": amp1,
            "amp2": amp2,
        }


class NeuralFieldTemplateModel(TemplateModel):
    def __init__(self, props: TemplateProps):
        super().__init__(props)
        # Coordinate-based MLP: (x, y) -> (peak_dist, sigma1, sigma2, amp1, amp2)
        layers = []
        in_dim = 2
        for _ in range(props.neural_num_layers):
            layers.append(nn.Linear(in_dim, props.neural_hidden_dim))
            layers.append(nn.ReLU())
            in_dim = props.neural_hidden_dim

        self.net = nn.Sequential(*layers)
        # Output 5 parameters
        self.head = nn.Linear(in_dim, 5)

        # Initialize head to zero so we start exactly at the initial props values
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def get_params(
        self,
        batch_indices: torch.Tensor | None = None,
        coordinates: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if coordinates is None:
            raise ValueError("NeuralFieldTemplateModel requires coordinates")

        # Normalize coordinates roughly to help optimization?
        # For now, we assume the MLP can handle the scale or we rely on normalization elsewhere.
        # Using raw coordinates.
        out = self.head(self.net(coordinates))

        # Base values in log space
        base_peak = self.peak_dist_init.log()
        base_sigma = self.sigma_init.log()
        base_amp = torch.tensor(self.props.amp, device=out.device).log()

        # Apply learned residuals in log space (ensures positivity)
        # out: [d_peak, d_s1, d_s2, d_a1, d_a2]
        return {
            "peak_dist": (base_peak + out[:, 0]).exp(),
            "sigma1": (base_sigma + out[:, 1]).exp(),
            "sigma2": (base_sigma + out[:, 2]).exp(),
            "amp1": (base_amp + out[:, 3]).exp(),
            "amp2": (base_amp + out[:, 4]).exp(),
        }
