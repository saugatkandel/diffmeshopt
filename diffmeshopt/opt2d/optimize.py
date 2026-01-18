import abc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffmeshopt.opt2d.geometry import get_bspline_derivative_matrix, get_bspline_matrix
from diffmeshopt.opt2d.loss import ContourLoss
from diffmeshopt.opt2d.props import OptimizationProps, SamplingProps, TemplateProps
from diffmeshopt.opt2d.sampling import sample_profiles_stochastic
from diffmeshopt.opt2d.template import (
    BSplineTemplateModel,
    FixedTemplateModel,
    GlobalOptimizableTemplateModel,
    NeuralFieldTemplateModel,
    PerPointTemplateModel,
)


class ContourRefinerBase(nn.Module):
    def __init__(
        self,
        image: np.ndarray,
        num_vertices: int,
        optimization_props: OptimizationProps = None,
        template_props: TemplateProps = None,
        sampling_props: SamplingProps | None = None,
        laplacian_window_size: int = 1,
        template_mode: str = "fixed",
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

        if template_mode == "per_point":
            self.template_model = PerPointTemplateModel(num_vertices, self.template_props)
        elif template_mode == "global":
            self.template_model = GlobalOptimizableTemplateModel(self.template_props)
        elif template_mode == "fixed":
            self.template_model = FixedTemplateModel(self.template_props)
        elif template_mode == "bspline":
            self.template_model = BSplineTemplateModel(num_vertices, self.template_props)
        elif template_mode == "neural":
            self.template_model = NeuralFieldTemplateModel(self.template_props)
        else:
            raise ValueError(f"Unknown template_mode: {template_mode}")

    @property
    @abc.abstractmethod
    def contour(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def points_for_regularization(self) -> torch.Tensor:
        return self.contour

    @property
    def normals(self) -> torch.Tensor:
        """Returns unit normals at each contour point."""
        # Default: Finite differences
        coords = self.contour
        next_coords = torch.roll(coords, shifts=-1, dims=0)
        prev_coords = torch.roll(coords, shifts=1, dims=0)

        # Central difference tangents
        tangents = next_coords - prev_coords
        # Rotate 90 degrees: (x, y) -> (-y, x)
        normals = torch.stack([-tangents[:, 1], tangents[:, 0]], dim=1)
        return F.normalize(normals, dim=-1)

    @abc.abstractmethod
    def _get_optimizer_params(self) -> list:
        raise NotImplementedError

    def configure_optimizer(self):
        lr = self.optimization_props.lr
        # Combine contour parameters with template parameters
        params = self._get_optimizer_params() + list(self.template_model.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

    def _forward_propagate(self):
        # --- Data Loss (Stochastic) ---
        profiles, sub_indices, valid_mask = sample_profiles_stochastic(
            self.image,
            self.contour,
            sampling_props=self.sampling_props,
            normals=self.normals,
        )

        # Get coordinates for the sampled points (needed for NeuralFieldTemplateModel)
        batch_coords = self.contour[sub_indices]

        # Get template parameters for the sampled points
        template_params = self.template_model.get_params(sub_indices, coordinates=batch_coords)

        # Calculate main losses
        losses = self.loss_fn(
            profiles,
            self.points_for_regularization,
            **template_params,
            mask=valid_mask,
        )

        # Add template regularization loss
        reg_losses = self.template_model.get_regularization_loss()
        w_sigma_reg = getattr(self.optimization_props, "w_sigma_reg", 1.0)
        w_template_smooth = getattr(self.optimization_props, "w_template_smooth", 1.0)

        if "sigma_reg" in reg_losses:
            losses["sigma_reg"] = w_sigma_reg * reg_losses["sigma_reg"]
            losses["total_loss"] += losses["sigma_reg"]
        if "template_smooth" in reg_losses:
            losses["template_smooth"] = w_template_smooth * reg_losses["template_smooth"]
            losses["total_loss"] += losses["template_smooth"]

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
        template_mode: str = "fixed",
    ):
        super().__init__(
            image=image,
            num_vertices=len(initial_contour),
            optimization_props=optimization_props,
            template_props=template_props,
            sampling_props=sampling_props,
            laplacian_window_size=laplacian_window_size,
            template_mode=template_mode,
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
        template_mode: str = "fixed",
    ):
        super().__init__(
            image=image,
            num_vertices=num_eval_points,
            optimization_props=optimization_props,
            template_props=template_props,
            sampling_props=sampling_props,
            laplacian_window_size=laplacian_window_size,
            template_mode=template_mode,
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

        # 3. Precompute derivative matrix for analytical normals
        self.register_buffer(
            "M_deriv", get_bspline_derivative_matrix(num_control_points, num_eval_points)
        )
        self.configure_optimizer()

    @property
    def contour(self):
        # Generate dense contour from control points
        return self.M_eval @ self.control_points

    @property
    def points_for_regularization(self) -> torch.Tensor:
        return self.control_points

    @property
    def normals(self) -> torch.Tensor:
        # Analytical tangents: M_deriv @ P
        tangents = self.M_deriv @ self.control_points
        normals = torch.stack([-tangents[:, 1], tangents[:, 0]], dim=1)
        return F.normalize(normals, dim=-1)

    def _get_optimizer_params(self) -> list:
        return [self.control_points]
