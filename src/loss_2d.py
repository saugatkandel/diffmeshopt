import torch
import torch.nn as nn
import torch.nn.functional as F

from src.props_2d import TemplateProps


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
        self.profile_len = template_props.profile_len

        # Create template
        # Center is 0. Range is roughly [-profile_len/2, profile_len/2]
        x = torch.arange(self.profile_len, dtype=torch.float32) - (self.profile_len - 1) / 2.0

        template = self.get_bigaussian_profile(x, self.peak_dist, self.sigma)
        self.register_buffer("template", template)

    @staticmethod
    def get_bigaussian_profile(x: torch.Tensor, peak_dist: float, sigma: float) -> torch.Tensor:
        """
        Generates the raw BiGaussian intensity profile.
        """
        # Peaks at +/- peak_dist / 2
        mu1 = -peak_dist / 2
        mu2 = peak_dist / 2

        template = torch.exp(-((x - mu1) ** 2) / (2 * sigma**2)) + torch.exp(
            -((x - mu2) ** 2) / (2 * sigma**2)
        )

        # Normalize template so that correlation is 1.0 for perfect match
        t_mean = template.mean()
        t_std = template.std()
        template = (template - t_mean) / (t_std + 1e-8)
        return template

    def forward(self, profiles: torch.Tensor) -> torch.Tensor:
        # profiles: (N, K)
        # Normalize profiles
        mean = profiles.mean(dim=-1, keepdim=True)
        std = profiles.std(dim=-1, keepdim=True)
        profiles_norm = (profiles - mean) / (std + 1e-8)

        # Cross correlation
        correlation = (profiles_norm * self.template).mean(dim=-1)

        # We want to maximize correlation, so minimize 1 - correlation
        loss = 1 - correlation
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
