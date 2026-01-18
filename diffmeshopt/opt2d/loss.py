import torch
import torch.nn as nn
import torch.nn.functional as F

from diffmeshopt.opt2d.props import OptimizationProps, TemplateProps


class BiGaussianLoss(nn.Module):
    def __init__(self, template_props: TemplateProps | None = None):
        """
        peak_dist: Distance between the two Gaussian peaks in pixels.
        sigma: Sigma of each Gaussian (width parameter).
        profile_len: Length of the sampled profile vector.
        """
        super().__init__()

        if template_props is None:
            template_props = TemplateProps()
        self.peak_dist = template_props.peak_dist
        self.sigma = template_props.sigma
        self.profile_len = template_props.num_samples

        # Create template
        # Center is 0. Range is roughly [-profile_len/2, profile_len/2]
        x = torch.arange(self.profile_len, dtype=torch.float32) - (self.profile_len - 1) / 2.0

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
        """
        Generates the raw BiGaussian intensity profile.
        """
        # Defaults
        if sigma1 is None:
            sigma1 = sigma if sigma is not None else 1.0
        if sigma2 is None:
            sigma2 = sigma if sigma is not None else 1.0
        if amp1 is None:
            amp1 = 1.0
        if amp2 is None:
            amp2 = 1.0

        # Handle broadcasting for batch optimization
        # We assume inputs are either scalars or (N,) tensors.
        # We need (N, 1) for broadcasting against x (L,).
        def _ensure_dim(t):
            if isinstance(t, torch.Tensor) and t.ndim == 1:
                return t.unsqueeze(-1)
            return t

        peak_dist = _ensure_dim(peak_dist)
        sigma1 = _ensure_dim(sigma1)
        sigma2 = _ensure_dim(sigma2)
        amp1 = _ensure_dim(amp1)
        amp2 = _ensure_dim(amp2)

        # Peaks at +/- peak_dist / 2
        mu1 = -peak_dist / 2
        mu2 = peak_dist / 2

        template = amp1 * torch.exp(-((x - mu1) ** 2) / (2 * sigma1**2)) + amp2 * torch.exp(
            -((x - mu2) ** 2) / (2 * sigma2**2)
        )

        # Normalize template so that correlation is 1.0 for perfect match
        t_mean = template.mean(dim=-1, keepdim=True)
        t_std = template.std(dim=-1, keepdim=True)
        template = (template - t_mean) / (t_std + 1e-8)
        return template

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
        # profiles: (N, K)
        # Normalize profiles
        mean = profiles.mean(dim=-1, keepdim=True)
        std = profiles.std(dim=-1, keepdim=True)
        profiles_norm = (profiles - mean) / (std + 1e-8)

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

        # Cross correlation
        correlation = (profiles_norm * template).mean(dim=-1)

        # We want to maximize correlation, so minimize 1 - correlation
        loss = 1.0 - correlation

        if mask is not None:
            if mask.dtype == torch.bool:
                mask = mask.float()
            # Compute weighted mean (avoid division by zero)
            return (loss * mask).sum() / (mask.sum() + 1e-8)

        return loss.mean()


class LaplacianSmoothingLoss(nn.Module):
    def __init__(self, window_size: int = 1):
        super().__init__()
        self.window_size = window_size

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

    def forward(self, contour: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Laplacian smoothing loss for a closed 2D contour.
        Penalizes the deviation of each vertex from the average of its neighbors.
        This acts as a regularizer to keep the contour smooth (minimizing curvature).

        window_size determines how many neighbors on each side are considered.
        Calculates Laplacian smoothing loss using 1D convolution.
        """
        # contour: (N, 2) -> (1, 2, N) for conv1d
        x = contour.permute(1, 0).unsqueeze(0)

        # Circular padding
        x_pad = F.pad(x, (self.window_size, self.window_size), mode="circular")

        # Convolution (groups=2 applies same kernel to x and y independently)
        weight = self.kernel.expand(2, -1, -1)
        laplacian = F.conv1d(x_pad, weight, groups=2)

        # (1, 2, N) -> (N, 2)
        laplacian = laplacian.squeeze(0).permute(1, 0)

        # Minimize the magnitude of the Laplacian vectors
        loss = (laplacian**2).sum(dim=-1).mean()
        return loss


class EdgeLengthConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, contour: torch.Tensor) -> torch.Tensor:
        """
        Penalizes the variance of edge lengths to encourage uniform vertex distribution.
        """
        v_next = torch.roll(contour, shifts=-1, dims=0)
        edge_lengths = torch.norm(contour - v_next, dim=-1)

        # Minimize variance: mean((l - mean_l)^2)
        return torch.var(edge_lengths)


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
        mp_std = mean_profile.std()
        mean_profile_norm = (mean_profile - mp_mean) / (mp_std + 1e-8)

        return F.mse_loss(mean_profile_norm, template)


class ContourLoss(nn.Module):
    def __init__(
        self,
        optimization_props: OptimizationProps,
        template_props: TemplateProps,
        laplacian_window_size: int = 1,
    ):
        super().__init__()
        self.w_data = optimization_props.w_data
        self.w_laplacian = optimization_props.w_laplacian
        self.w_edge = optimization_props.w_edge
        self.w_sigma_reg = getattr(optimization_props, "w_sigma_reg", 1.0)
        self.w_template_shape = getattr(optimization_props, "w_template_shape", 0.1)

        self.data_loss_fn = BiGaussianLoss(template_props=template_props)
        self.laplacian_loss_fn = LaplacianSmoothingLoss(window_size=laplacian_window_size)
        self.edge_loss_fn = EdgeLengthConsistencyLoss()
        self.shape_loss_fn = TemplateShapeLoss()

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
    ) -> dict[str, torch.Tensor]:
        # Data Loss: Cross-correlation with template
        # If peak_dist/sigma are provided (from TemplateModel), they are used.
        data_loss = self.data_loss_fn(
            profiles, peak_dist, sigma, sigma1, sigma2, amp1, amp2, mask=mask
        )

        shape_loss = torch.tensor(0.0, device=profiles.device)
        if peak_dist is not None:
            # Shape Loss: Match the shape of the consensus (mean) profile to the template.
            # This acts as a constraint to keep dynamic templates grounded to the data mean.
            template = self.data_loss_fn.get_bigaussian_profile(
                self.data_loss_fn.x, peak_dist, sigma, sigma1, sigma2, amp1, amp2
            )
            shape_loss = self.shape_loss_fn(profiles, template, mask=mask)

        laplacian_loss = self.laplacian_loss_fn(points_for_reg)
        edge_loss = self.edge_loss_fn(points_for_reg)

        total_loss = (
            self.w_data * data_loss
            + self.w_laplacian * laplacian_loss
            + self.w_edge * edge_loss
            + self.w_template_shape * shape_loss
        )

        return {
            "total_loss": total_loss,
            "data_loss": self.w_data * data_loss,
            "laplacian_loss": self.w_laplacian * laplacian_loss,
            "edge_loss": self.w_edge * edge_loss,
            "shape_loss": self.w_template_shape * shape_loss,
        }
