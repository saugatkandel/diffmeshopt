import abc
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffmeshopt.opt2d.config import DataLossType, RegularizerType, TemplateProps


class BiGaussianBaseLoss(nn.Module, abc.ABC):
    """Abstract base class for profile-matching data losses.

    Handles spatial coordinate setup, bi-Gaussian template generation,
    asymmetry penalties, and masked loss reduction.
    """

    def __init__(
        self,
        template_props: TemplateProps | None = None,
        num_samples: int = 51,
        sample_step: float = 1.0,
        center_symmetry_weight: float = 0.0,
    ):
        super().__init__()

        if template_props is None:
            template_props = TemplateProps()
        self.peak_dist = template_props.peak_dist
        self.sigma = template_props.sigma
        self.center_symmetry_weight = center_symmetry_weight

        # Initialize buffers
        self.register_buffer("x", torch.zeros(num_samples))
        self.register_buffer("template", torch.zeros(num_samples))

        # Setup coordinate system and initial template
        self.update_sampling(num_samples, sample_step)

    def update_sampling(self, num_samples: int, sample_step: float):
        """Update internal coordinate system and template for a new sampling resolution."""
        self.profile_len = num_samples
        self.sample_step = sample_step

        x = (
            torch.arange(num_samples, dtype=torch.float32) - (num_samples - 1) / 2.0
        ) * sample_step
        self.register_buffer("x", x)

        template = self.get_bigaussian_profile(x, self.peak_dist, self.sigma)
        self.register_buffer("template", template)

    @staticmethod
    def get_bigaussian_profile(
        x: torch.Tensor,
        peak_dist: float | torch.Tensor,
        sigma: float | torch.Tensor | None = None,
        sigma1: float | torch.Tensor | None = None,
        sigma2: float | torch.Tensor | None = None,
        amp1: float | torch.Tensor | None = None,
        amp2: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generates the raw BiGaussian intensity profile."""
        if sigma1 is None:
            sigma1 = sigma if sigma is not None else 1.0
        if sigma2 is None:
            sigma2 = sigma if sigma is not None else 1.0
        if amp1 is None:
            amp1 = 1.0
        if amp2 is None:
            amp2 = 1.0

        def _ensure_dim(t):
            if isinstance(t, torch.Tensor) and t.ndim == 1:
                return t.unsqueeze(-1)
            return t

        peak_dist = _ensure_dim(peak_dist)
        sigma1 = _ensure_dim(sigma1)
        sigma2 = _ensure_dim(sigma2)
        amp1 = _ensure_dim(amp1)
        amp2 = _ensure_dim(amp2)

        mu1 = -peak_dist / 2
        mu2 = peak_dist / 2

        template = amp1 * torch.exp(-((x - mu1) ** 2) / (2 * sigma1**2)) + amp2 * torch.exp(
            -((x - mu2) ** 2) / (2 * sigma2**2)
        )
        return template

    @abc.abstractmethod
    def forward_metric(self, profiles: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
        """Compute the core similarity loss per profile."""
        pass

    def compute_asymmetry_loss(self, profiles: torch.Tensor) -> torch.Tensor:
        """Computes penalty for asymmetric profiles around origin x=0.

        If the contour is correctly centered on a bilayer, the left half
        and flipped right half of the sampled profile will match.
        """
        # Flip profile along spatial dimension K
        profiles_flipped = torch.flip(profiles, dims=[-1])

        # Measure difference between profile and its reverse reflection
        asymmetry = torch.mean(torch.abs(profiles - profiles_flipped), dim=-1)
        return asymmetry

    def forward(
        self,
        profiles: torch.Tensor,
        peak_dist: torch.Tensor | None = None,
        sigma: torch.Tensor | None = None,
        sigma1: torch.Tensor | None = None,
        sigma2: torch.Tensor | None = None,
        amp1: torch.Tensor | None = None,
        amp2: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        if peak_dist is not None:
            template = self.get_bigaussian_profile(
                self.x,
                peak_dist=peak_dist,
                sigma=sigma,
                sigma1=sigma1,
                sigma2=sigma2,
                amp1=amp1,
                amp2=amp2,
            )
        else:
            template = self.template

        template = torch.atleast_2d(template)

        # 1. Base profile match loss (Wasserstein or Correlation)
        match_loss = self.forward_metric(profiles, template)

        # 2. Symmetry constraint around origin
        if self.center_symmetry_weight > 0.0:
            asym_loss = self.compute_asymmetry_loss(profiles)
            total_loss = match_loss + (self.center_symmetry_weight * asym_loss)
        else:
            total_loss = match_loss

        # 3. Masked reduction
        if mask is not None:
            if mask.dtype == torch.bool:
                mask = mask.float()
            return (total_loss * mask).sum() / (mask.sum() + 1e-8)

        return total_loss.mean()


class BiGaussianCorrelationLoss(BiGaussianBaseLoss):
    """Normalized cross-correlation data loss.

    Measures pattern correlation between sampled profiles and the template.
    Optimal for fine snapping near convergence (high frequency sensitivity).
    """

    def forward_metric(self, profiles: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
        # Standardize template
        t_mean = template.mean(dim=-1, keepdim=True)
        t_std = template.std(dim=-1, keepdim=True, unbiased=False)
        template_norm = (template - t_mean) / (t_std + 1e-8)

        # Standardize sampled profiles
        p_mean = profiles.mean(dim=-1, keepdim=True)
        p_std = profiles.std(dim=-1, keepdim=True, unbiased=False)
        profiles_norm = (profiles - p_mean) / (p_std + 1e-8)

        # Calculate 1 - correlation
        correlation = (profiles_norm * template_norm).mean(dim=-1)
        return 1.0 - correlation


class BiGaussianWassersteinLoss(BiGaussianBaseLoss):
    """1D Optimal Transport (Earth Mover's Distance) data loss.

    Measures the spatial work required to transform sampled intensity CDFs into template CDFs.
    Provides smooth, continuous gradients across large positional initial errors.
    """

    def __init__(
        self,
        template_props: TemplateProps | None = None,
        num_samples: int = 51,
        sample_step: float = 1.0,
        temperature: float = 1.0,
    ):
        super().__init__(
            template_props=template_props,
            num_samples=num_samples,
            sample_step=sample_step,
        )
        self.temperature = temperature

    def forward_metric(self, profiles: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
        # 1. Convert profiles and template into 1D probability distributions
        p_sample = torch.softmax(profiles / self.temperature, dim=-1)
        p_template = torch.softmax(template / self.temperature, dim=-1)

        # 2. Compute Cumulative Distribution Functions (CDFs)
        cdf_sample = torch.cumsum(p_sample, dim=-1)
        cdf_template = torch.cumsum(p_template, dim=-1)

        # 3. 1D Wasserstein distance is the L1 distance between CDFs
        # Scale by sample_step so the loss is denominated in pixel distance units
        return torch.sum(torch.abs(cdf_sample - cdf_template), dim=-1) * self.sample_step


class LaplacianSmoothingLoss(nn.Module):
    def __init__(self, window_size: int = 3, mode: str = "full"):
        super().__init__()
        self.window_size = window_size
        self.mode = mode

        # Create Gaussian kernel for Laplacian: v_i - weighted_mean(neighbors)
        # Weights decay with distance from center (Gaussian)
        k = window_size
        kernel_size = 2 * k + 1

        # Coordinate grid centered at 0: [-k, ..., 0, ..., k]
        x_grid = torch.arange(-k, k + 1, dtype=torch.float32)

        # Gaussian weights: exp(-x^2 / (2*sigma^2))
        sigma = max(1.0, window_size / 2.0)
        weights = torch.exp(-(x_grid**2) / (2 * sigma**2))
        weights[k] = 0.0  # Zero out center
        weights = weights / weights.sum()  # Normalize

        # Laplacian kernel: 1 at center, -weights elsewhere
        kernel = -weights
        kernel[k] = 1.0
        kernel = kernel.view(1, 1, -1)

        self.register_buffer("kernel", kernel)

    def forward(
        self,
        x: torch.Tensor,
        normals: torch.Tensor | None = None,
        edges: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Calculates the Laplacian smoothing loss.
        x: (N, C) tensor.
        edges: (2, E) LongTensor of vertex indices. If provided, computes Graph Laplacian.
        """
        if edges is not None:
            # Graph Laplacian for general meshes (3D surfaces)
            # L_i = x_i - mean(neighbors_i)
            row, col = edges

            # Sum neighbors: out[i] = sum(x[j]) for j in neighbors(i)
            neighbor_sum = torch.zeros_like(x)
            neighbor_sum.index_add_(0, row, x[col])

            # Degree: out[i] = count(j)
            degree = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
            degree.index_add_(0, row, torch.ones_like(row, dtype=x.dtype))

            # Mean
            neighbor_mean = neighbor_sum / (degree.unsqueeze(-1) + 1e-8)
            laplacian = x - neighbor_mean

            C = x.shape[-1]
        else:
            # 1D Convolution for contours/curves
            if x.ndim == 1:
                x = x.unsqueeze(-1)

            # x: (N, C)
            C = x.shape[-1]

            # (N, C) -> (1, C, N) for conv1d
            x_in = x.permute(1, 0).unsqueeze(0)

            # Circular padding
            x_pad = F.pad(x_in, (self.window_size, self.window_size), mode="circular")

            # Convolution
            weight = self.kernel.expand(C, -1, -1)
            laplacian = F.conv1d(x_pad, weight, groups=C)

            # (1, C, N) -> (N, C)
            laplacian = laplacian.squeeze(0).permute(1, 0)

        if self.mode == "tangential":
            # Project onto tangent to regularize distribution without shrinking
            if normals is not None:
                if C not in (2, 3):
                    raise ValueError(f"Tangential Laplacian expects 2D or 3D vectors, got C={C}.")
                # Detach normals to ensure we don't optimize the projection direction itself
                normals = normals.detach()
                # 3D-generalizable formulation: L_tangential = L - (L . n)n
                # laplacian: (N, 2), normals: (N, 2)
                normal_comp = (laplacian * normals).sum(dim=-1, keepdim=True) * normals
                tangential_laplacian = laplacian - normal_comp
                return (tangential_laplacian**2).sum(dim=-1).mean()

            v_next = torch.roll(x, shifts=-1, dims=0)
            v_prev = torch.roll(x, shifts=1, dims=0)
            tangents = F.normalize(v_next - v_prev, dim=-1, eps=1e-8)
            proj = (laplacian * tangents).sum(dim=-1)
            return (proj**2).mean()

        if self.mode == "curvature_consistency":
            # Penalize variance of curvature (magnitude of laplacian)
            # Detach mean to prevent global expansion/shrinking bias
            k = torch.norm(laplacian, dim=-1)
            mean_k = k.mean().detach()
            # Normalize by mean_k^2 for scale invariance (Coefficient of Variation)
            return ((k - mean_k) ** 2).mean() / (mean_k**2 + 1e-8)

        # Minimize the magnitude of the Laplacian vectors
        loss = (laplacian**2).sum(dim=-1).mean()
        return loss


class EdgeLengthConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, contour: torch.Tensor, edges: torch.Tensor | None = None) -> torch.Tensor:
        """
        Penalizes the variance of edge lengths to encourage uniform vertex distribution.
        contour: (N, C) vertices.
        edges: (2, E) indices of edges. If None, assumes closed loop 1D topology.
        """
        if edges is not None:
            v0 = contour[edges[0]]
            v1 = contour[edges[1]]
            edge_lengths = torch.norm(v0 - v1, dim=-1)
        else:
            v_next = torch.roll(contour, shifts=-1, dims=0)
            edge_lengths = torch.norm(contour - v_next, dim=-1)

        # Minimize variance: mean((l - mean_l)^2)
        # return torch.var(edge_lengths)
        # Detach mean to prevent shrinking bias and normalize for scale invariance
        mean_l = edge_lengths.mean().detach()
        return ((edge_lengths - mean_l) ** 2).mean() / (mean_l**2 + 1e-8)


class NormalConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, normals: torch.Tensor, edges: torch.Tensor | None = None) -> torch.Tensor:
        """
        Penalizes the angle between adjacent normals (Fairing term).
        normals: (N, C) unit normals.
        edges: (2, E) indices of adjacent vertices/faces. If None, assumes closed loop 1D topology.
        """
        if edges is not None:
            n0 = normals[edges[0]]
            n1 = normals[edges[1]]
            dot = (n0 * n1).sum(dim=-1)
        else:
            n_next = torch.roll(normals, shifts=-1, dims=0)
            dot = (normals * n_next).sum(dim=-1)

        # Clamp for numerical stability
        dot = torch.clamp(dot, -1.0, 1.0)
        return (1.0 - dot).mean()


class TemplateShapeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        profiles: torch.Tensor,
        template: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is not None:
            if mask.sum() == 0:
                return torch.tensor(0.0, device=profiles.device)
            profiles = profiles[mask.bool()]

        if profiles.shape[0] == 0:
            return torch.tensor(0.0, device=profiles.device)

        mean_profile = profiles.mean(dim=0)
        mp_mean = mean_profile.mean()
        mp_std = mean_profile.std(unbiased=False)
        mean_profile_norm = (mean_profile - mp_mean) / (mp_std + 1e-8)

        return F.l1_loss(mean_profile_norm.expand_as(template), template)


class ContourLoss(nn.Module):
    """Combined loss for contour refinement with data fidelity and regularization.

    Architecture:
    - Data loss: Measures match between image profiles and template (always weight=1.0)
    - Contour geometry regularizers: Smooth/regularize vertex positions
    - Template parameter regularizers: Regularize learned template parameters

    Weight management:
    - Weights are stored as buffers (enable save/load and device transfer)
    - Weights can be adapted during optimization (see AdaptiveRegularizationProps)
    - Raw (unweighted) losses stored in self._raw_losses for adaptive computation

    Workflow:
    1. ContourRefiner calls loss_fn.forward(profiles, vertices, template_params, ...)
    2. This method computes all losses (data + regularizers)
    3. Returns weighted losses for backprop
    4. Raw losses available in self._raw_losses for weight adaptation
    """

    def __init__(
        self,
        template_props: TemplateProps | None = None,
        num_samples: int = 51,
        sample_step: float = 1.0,
        data_loss_type: DataLossType = DataLossType.BIGAUSSIAN_CORRELATION,
        laplacian_window_size: int = 3,
        laplacian_mode: str = "full",
        shape_loss_weight: float = 1.0,
        initial_regularization_weights: dict[str, float] | None = None,
    ):

        super().__init__()
        logging.info("Initializing ContourLoss")

        # Dynamic buffer registration: automatically creates buffers for all regularizers
        # This eliminates manual synchronization while using the correct PyTorch primitive
        # Buffers (not Parameters) are semantically correct for hyperparameters/weights

        # Override weights dict: user-provided weights override defaults via initial_regularization_weights
        # keys should match RegularizerType values (e.g. "contour_laplacian")
        weight_overrides = {}
        if initial_regularization_weights:
            for k, v in initial_regularization_weights.items():
                # Try to match string key to RegularizerType
                try:
                    reg_type = RegularizerType(k)
                    weight_overrides[reg_type] = v
                except ValueError:
                    logging.warning(
                        f"ContourLoss received unknown argument/weight: '{k}'. Ignoring."
                    )

        # Dynamically register buffer for each regularizer (automatically synced with RegularizerType)
        for reg_type in RegularizerType:
            weight_value = weight_overrides.get(reg_type, 0.0)
            buffer_name = f"w_{reg_type.value}"
            self.register_buffer(buffer_name, torch.tensor(weight_value, dtype=torch.float32))

        # Register shape loss weight explicitly (it is part of data term, not a regularizer)
        self.register_buffer("w_shape", torch.tensor(shape_loss_weight, dtype=torch.float32))
        print("Shape loss weight", self.w_shape)

        # Storage for raw (unweighted) losses for adaptive weight computation
        self._raw_losses: dict[str, torch.Tensor] = {}

        # Loss function instances
        # Note: These are intentionally created explicitly (not dynamically) for clarity:
        # - Easy to understand what losses exist
        # - Easy to configure (each may need different parameters)
        # - No performance benefit to dynamic creation
        # - Only weights need dynamic registration (to ensure sync with RegularizerType)
        if data_loss_type == DataLossType.BIGAUSSIAN_CORRELATION:
            self.data_loss_fn = BiGaussianCorrelationLoss(
                template_props=template_props, num_samples=num_samples, sample_step=sample_step
            )
        elif data_loss_type == DataLossType.BIGAUSSIAN_WASSERSTEIN:
            self.data_loss_fn = BiGaussianWassersteinLoss(
                template_props=template_props, num_samples=num_samples, sample_step=sample_step
            )
        else:
            raise ValueError(f"Unsupported data loss type: {data_loss_type}")
        self.laplacian_loss_fn = LaplacianSmoothingLoss(
            window_size=laplacian_window_size, mode=laplacian_mode
        )
        self.edge_loss_fn = EdgeLengthConsistencyLoss()
        self.spacing_loss_fn = LaplacianSmoothingLoss(
            window_size=laplacian_window_size, mode="tangential"
        )
        self.fairing_loss_fn = NormalConsistencyLoss()
        self.shape_loss_fn = TemplateShapeLoss()

    def get_weight(self, reg_type) -> torch.Tensor:
        """Get weight buffer for a regularizer.

        Args:
            reg_type: RegularizerType enum or string value

        Returns:
            Weight tensor (buffer)
        """
        if isinstance(reg_type, RegularizerType):
            key = reg_type.value
        else:
            key = reg_type

        buffer_name = f"w_{key}"
        return getattr(self, buffer_name)

    def set_weight(self, reg_type, value: float) -> None:
        """Set weight buffer for a regularizer (for adaptive adjustment).

        Args:
            reg_type: RegularizerType enum or string value
            value: New weight value
        """
        if isinstance(reg_type, RegularizerType):
            key = reg_type.value
        else:
            key = reg_type

        buffer_name = f"w_{key}"
        getattr(self, buffer_name).fill_(value)

    def forward(
        self,
        profiles: torch.Tensor,
        points_for_reg: torch.Tensor,
        peak_dist: torch.Tensor | None = None,
        sigma: torch.Tensor | None = None,
        sigma1: torch.Tensor | None = None,
        sigma2: torch.Tensor | None = None,
        amp1: torch.Tensor | None = None,
        amp2: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        vertices: torch.Tensor | None = None,
        normals: torch.Tensor | None = None,
        edges: torch.Tensor | None = None,
        reg_losses: dict[str, torch.Tensor]
        | None = None,  # From template_model.get_regularization_loss()
    ) -> dict[str, torch.Tensor]:
        """Compute combined loss with data fidelity and regularization.

        Args:
            profiles: Sampled intensity profiles from image (N, profile_length)
            points_for_reg: Points to regularize (vertices or control points)
            peak_dist, sigma, sigma1, sigma2, amp1, amp2: Template parameters from template_model
            mask: Valid sample mask (for handling boundary/edge cases)
            vertices: Actual contour vertices (may differ from points_for_reg for B-spline/RBF)
            normals: Contour normals for tangential/normal regularizers
            edges: Mesh connectivity for graph Laplacian (optional)
            reg_losses: Template regularization losses from template_model.get_regularization_loss()
                       Expected keys: "template_param_anchor", "template_param_laplacian"

        Returns:
            Dictionary with total_loss and all component losses (weighted).
        """
        # Compute all raw (unweighted) losses
        data_loss = self.data_loss_fn(
            profiles, peak_dist, sigma, sigma1, sigma2, amp1, amp2, mask=mask
        )

        shape_loss = torch.tensor(0.0, device=profiles.device)
        if peak_dist is not None:
            template = self.data_loss_fn.get_bigaussian_profile(
                self.data_loss_fn.x, peak_dist, sigma, sigma1, sigma2, amp1, amp2
            )
            shape_loss = self.shape_loss_fn(profiles, template, mask=mask)

        contour_laplacian_loss = self.laplacian_loss_fn(points_for_reg, edges=edges)
        edge_length_loss = self.edge_loss_fn(points_for_reg, edges=edges)

        geom_target = vertices if vertices is not None else points_for_reg

        tangential_laplacian_loss = torch.tensor(0.0, device=profiles.device)
        if self.get_weight(RegularizerType.TANGENTIAL_LAPLACIAN) > 0:
            tangential_laplacian_loss = self.spacing_loss_fn(
                geom_target, normals=normals, edges=edges
            )

        normal_consistency_loss = torch.tensor(0.0, device=profiles.device)
        if self.get_weight(RegularizerType.NORMAL_CONSISTENCY) > 0 and normals is not None:
            normal_consistency_loss = self.fairing_loss_fn(normals, edges=edges)

        # Store raw losses for adaptive weight computation
        # Keys must match RegularizerType enum values (plus "data" which is special)
        self._raw_losses = {
            "correlation": data_loss,
            "shape": shape_loss,
            RegularizerType.CONTOUR_LAPLACIAN.value: contour_laplacian_loss,
            RegularizerType.EDGE_LENGTH.value: edge_length_loss,
            RegularizerType.TANGENTIAL_LAPLACIAN.value: tangential_laplacian_loss,
            RegularizerType.NORMAL_CONSISTENCY.value: normal_consistency_loss,
        }

        # Dynamically merge template regularization losses
        if reg_losses is not None:
            for k, v in reg_losses.items():
                # Only include valid regularizers to avoid issues in adaptive weight update
                try:
                    RegularizerType(k)
                    self._raw_losses[k] = v
                except ValueError:
                    logging.warning(f"Unknown regularizer key in reg_losses: '{k}'. Ignoring.")

        # Compute weighted losses and total
        # Data Term = Correlation Loss + Weighted Shape Loss
        # Correlation loss always has weight=1.0

        weighted_shape_loss = self.w_shape * shape_loss
        total_data_loss = data_loss + weighted_shape_loss
        total_loss = total_data_loss

        # Return weighted losses
        results = {
            "data_loss": total_data_loss,  # Combined data term
            "correlation_loss": data_loss,
            "shape_loss": weighted_shape_loss,
        }

        # Dynamically compute total loss and populate results for all regularizers
        for reg_type in RegularizerType:
            key = reg_type.value
            # Use get() with default 0.0 to ensure all regularizers appear in results
            # even if not computed (e.g. template losses for FixedTemplateModel)
            raw_loss = self._raw_losses.get(key, torch.tensor(0.0, device=profiles.device))
            # print(f"{reg_type} weight: {self.get_weight(reg_type)}")
            weighted_loss = self.get_weight(reg_type) * raw_loss
            total_loss = total_loss + weighted_loss
            results[f"{key}_loss"] = weighted_loss

        results["total_loss"] = total_loss
        return results
