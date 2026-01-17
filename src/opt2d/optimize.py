import numpy as np
import torch
import torch.nn as nn

from src.opt2d.geometry import get_bspline_matrix
from src.opt2d.loss import ContourLoss
from src.opt2d.props import OptimizationProps, SamplingProps, TemplateProps
from src.opt2d.sampling import sample_profiles_stochastic
from src.opt2d.template import FixedTemplateModel, PerPointTemplateModel


class ContourRefinerBase(nn.Module):
    def __init__(
        self,
        image: np.ndarray,
        num_vertices: int,
        optimization_props: OptimizationProps = None,
        template_props: TemplateProps = None,
        sampling_props: SamplingProps | None = None,
        laplacian_window_size: int = 1,
        optimize_template: bool = False,
    ):
        super().__init__()
        self.register_buffer("image", torch.from_numpy(image).float())

        self.optimization_props = optimization_props or OptimizationProps()
        self.sampling_props = sampling_props or SamplingProps()
        self.template_props = template_props or TemplateProps()

        self.loss_fn = ContourLoss(
            optimization_props=self.optimization_props,
            template_props=self.template_props,
            laplacian_window_size=laplacian_window_size,
        )

        if optimize_template:
            self.template_model = PerPointTemplateModel(num_vertices, self.template_props)
        else:
            self.template_model = FixedTemplateModel(self.template_props)

    @property
    def contour(self) -> torch.Tensor:
        raise NotImplementedError

    def _get_optimizer_params(self) -> list:
        raise NotImplementedError

    def configure_optimizer(self):
        lr = self.optimization_props.lr
        # Combine contour parameters with template parameters
        params = self._get_optimizer_params() + list(self.template_model.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

    def _forward_propagate(self):
        # --- Data Loss (Stochastic) ---
        profiles, sub_indices = sample_profiles_stochastic(
            self.image,
            self.contour,
            sampling_props=self.sampling_props,
        )

        # Get template parameters for the sampled points
        template_params = self.template_model.get_params(sub_indices)

        # Calculate main losses
        losses = self.loss_fn(
            profiles,
            self.contour,  # Use the full contour for geometric regularization
            **template_params,
        )

        # Add template regularization loss
        sigma_reg = self.template_model.get_regularization_loss()
        w_sigma_reg = getattr(self.optimization_props, "w_sigma_reg", 1.0)

        losses["sigma_reg"] = w_sigma_reg * sigma_reg
        losses["total_loss"] += losses["sigma_reg"]

        losses["total_loss"].backward()
        return losses

    def step(self) -> dict[str, float]:
        self.optimizer.zero_grad()

        losses = self._forward_propagate()
        self.optimizer.step()
        return {k: v.item() for k, v in losses.items()}


class ContourRefiner(ContourRefinerBase):
    def __init__(
        self,
        image: np.ndarray,
        initial_contour: np.ndarray,
        optimization_props: OptimizationProps = None,
        template_props: TemplateProps = None,
        sampling_props: SamplingProps | None = None,
        laplacian_window_size: int = 1,
        optimize_template: bool = False,
    ):
        super().__init__(
            image=image,
            num_vertices=len(initial_contour),
            optimization_props=optimization_props,
            template_props=template_props,
            sampling_props=sampling_props,
            laplacian_window_size=laplacian_window_size,
            optimize_template=optimize_template,
        )
        self._contour = nn.Parameter(torch.from_numpy(initial_contour).float())
        self.configure_optimizer()

    @property
    def contour(self) -> torch.Tensor:
        return self._contour

    def _get_optimizer_params(self) -> list:
        return [self._contour]

    def update_contour(self, contour_np: np.ndarray) -> None:
        """
        Updates the contour state in-place and clears gradients.
        Assumes the number of vertices remains constant.
        """
        with torch.no_grad():
            self._contour.copy_(torch.from_numpy(contour_np).float().to(self._contour.device))
            self._contour.grad = None


class BSplineContourRefiner(ContourRefinerBase):
    def __init__(
        self,
        image: np.ndarray,
        initial_contour: np.ndarray,
        optimization_props: OptimizationProps = None,
        template_props: TemplateProps = None,
        sampling_props: SamplingProps | None = None,
        num_control_points: int = 40,
        num_eval_points: int = 200,
        laplacian_window_size: int = 1,
        optimize_template: bool = False,
    ):
        super().__init__(
            image=image,
            num_vertices=num_eval_points,
            optimization_props=optimization_props,
            template_props=template_props,
            sampling_props=sampling_props,
            laplacian_window_size=laplacian_window_size,
            optimize_template=optimize_template,
        )

        # 1. Fit initial control points to the initial contour
        # We create a temporary matrix for the initial contour length
        M_init = get_bspline_matrix(num_control_points, len(initial_contour))
        target = torch.from_numpy(initial_contour).float()
        # Solve linear system M_init @ P = V for P (control points)
        # P = (M^T M)^-1 M^T V
        # Using least squares
        initial_cp = torch.linalg.lstsq(M_init, target).solution

        self.control_points = nn.Parameter(initial_cp)

        # 2. Precompute evaluation matrix for the desired resolution
        self.register_buffer("M_eval", get_bspline_matrix(num_control_points, num_eval_points))

        self.configure_optimizer()

    @property
    def contour(self):
        # Generate dense contour from control points
        return self.M_eval @ self.control_points

    def _get_optimizer_params(self) -> list:
        return [self.control_points]
