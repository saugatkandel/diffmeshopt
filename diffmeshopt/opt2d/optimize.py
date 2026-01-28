import abc
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffmeshopt.opt2d.geometry import get_bspline_derivative_matrix, get_bspline_matrix
from diffmeshopt.opt2d.loss import ContourLoss
from diffmeshopt.opt2d.props import BSplineContourRefinerProps, ContourRefinerProps
from diffmeshopt.opt2d.sampling import sample_profiles_stochastic
from diffmeshopt.opt2d.template import BaseTemplateModel


class ContourRefinerBase(nn.Module, abc.ABC):
    """Abstract base class for contour refinement."""

    def __init__(
        self,
        props: ContourRefinerProps,
        template_model: BaseTemplateModel,
    ):
        super().__init__()
        self.props = props
        self.template_model = template_model

        self.loss_fn = ContourLoss(
            data_loss_weight=props.data_loss_weight,
            laplacian_loss_weight=props.laplacian_loss_weight,
            edge_length_loss_weight=props.edge_length_loss_weight,
            sigma_reg_loss_weight=props.sigma_reg_loss_weight,
            template_shape_loss_weight=props.template_shape_loss_weight,
            template_smooth_loss_weight=props.template_smooth_loss_weight,
            template_props=template_model.props,
            num_samples=props.profile_length,
            sample_step=props.sample_step,
            laplacian_window_size=props.laplacian_window_size,
        )

        self.optimizer = None
        self._initial_state = None

    @property
    @abc.abstractmethod
    def contour(self) -> torch.Tensor:
        """The evaluated contour points. Must be implemented by subclasses."""
        pass

    @property
    @abc.abstractmethod
    def points_for_regularization(self) -> torch.Tensor:
        """The points to apply geometric regularization to (e.g., vertices or control points)."""
        pass

    @property
    def normals(self) -> torch.Tensor:
        """Returns unit normals at each contour point. Default is finite differences."""
        coords = self.contour
        next_coords = torch.roll(coords, shifts=-1, dims=0)
        prev_coords = torch.roll(coords, shifts=1, dims=0)
        tangents = next_coords - prev_coords
        normals = torch.stack([-tangents[:, 1], tangents[:, 0]], dim=1)
        return F.normalize(normals, dim=-1)

    def create_optimizer(self) -> torch.optim.Optimizer:
        """Creates the optimizer configured for this refiner."""
        return torch.optim.Adam(self.parameters(), lr=self.props.learning_rate)

    def configure_optimizer(self):
        self.optimizer = self.create_optimizer()

    def compute_losses(self, image: torch.Tensor):
        profiles, sub_indices, valid_mask = sample_profiles_stochastic(
            image,
            self.contour,
            profile_length=self.props.profile_length,
            profile_width=self.props.profile_width,
            sample_step=self.props.sample_step,
            num_samples=self.props.num_sampled_profiles,
            normals=self.normals,
        )

        template_params = self.template_model.get_params(
            batch_indices=sub_indices, coordinates=self.contour
        )

        reg_losses = self.template_model.get_regularization_loss()

        losses = self.loss_fn(
            profiles,
            self.points_for_regularization,
            **template_params,
            mask=valid_mask,
            reg_losses=reg_losses,
        )

        return losses

    def step(self, image: torch.Tensor) -> dict[str, float]:
        if self.optimizer is None:
            self.configure_optimizer()
        self.optimizer.zero_grad()

        losses = self.compute_losses(image)
        losses["total_loss"].backward()
        self.optimizer.step()
        return {k: v.item() for k, v in losses.items()}

    def refine(self, image: torch.Tensor):
        """Generator that yields optimization history for a number of steps."""
        if self.optimizer is None:
            self.configure_optimizer()

        for step in range(self.props.num_steps):
            losses = self.step(image)
            history_item = {
                "step": step,
                "contour": self.contour.detach().clone(),
                **losses,
            }
            yield history_item

    def export_state(self) -> dict:
        """
        Exports the current state of the contour and template parameters as numpy arrays.
        """
        with torch.no_grad():
            contour_np = self.contour.detach().cpu().numpy()
            params_torch = self.template_model.get_params(coordinates=self.contour)
            params_np = {
                k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
                for k, v in params_torch.items()
            }
            return {
                "contour": contour_np,
                "template_params": params_np,
                "mode": self.template_model.mode.value if self.template_model.mode else "unknown",
            }

    def capture_initial_state(self):
        """Captures the current state dict as the initial state."""
        self._initial_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}

    def reset(self):
        """Resets the parameters to the captured initial state."""
        if self._initial_state is not None:
            self.load_state_dict(self._initial_state)
            self.optimizer = None
        else:
            logging.warning("reset() called but no initial state was captured.")


class ContourRefiner(ContourRefinerBase):
    """Refines a 2D contour by optimizing its vertex positions directly."""

    def __init__(
        self,
        initial_contour: torch.Tensor,
        props: ContourRefinerProps,
        template_model: BaseTemplateModel,
    ):
        super().__init__(props, template_model)
        self.contour_param = nn.Parameter(initial_contour.clone())
        self.capture_initial_state()

    @property
    def contour(self) -> torch.Tensor:
        return self.contour_param

    @property
    def points_for_regularization(self) -> torch.Tensor:
        return self.contour_param


class BSplineContourRefiner(ContourRefinerBase):
    """Refines a 2D contour by optimizing B-spline control points."""

    def __init__(
        self,
        initial_contour: torch.Tensor,
        props: BSplineContourRefinerProps,
        template_model: BaseTemplateModel,
    ):
        super().__init__(props, template_model)
        num_control_points = props.contour_num_control_points
        num_eval_points = len(initial_contour)

        logging.info(
            f"Initializing BSplineContourRefiner with {num_control_points} control points, {num_eval_points} eval points."
        )

        # Fit initial control points to the initial contour
        M_init = get_bspline_matrix(num_control_points, len(initial_contour))
        target = initial_contour.float()
        initial_cp = torch.linalg.lstsq(M_init, target).solution
        self.control_points = nn.Parameter(initial_cp)

        # Precompute evaluation and derivative matrices
        self.register_buffer(
            "M_eval", get_bspline_matrix(num_control_points, num_eval_points), persistent=False
        )
        self.register_buffer(
            "M_deriv",
            get_bspline_derivative_matrix(num_control_points, num_eval_points),
            persistent=False,
        )
        self.capture_initial_state()

    @property
    def contour(self):
        return self.M_eval @ self.control_points

    @property
    def points_for_regularization(self) -> torch.Tensor:
        return self.control_points

    @property
    def normals(self) -> torch.Tensor:
        """Analytical normals from B-spline derivatives."""
        tangents = self.M_deriv @ self.control_points
        normals = torch.stack([-tangents[:, 1], tangents[:, 0]], dim=1)
        return F.normalize(normals, dim=-1)
