import torch
import torch.nn as nn


class BiGaussianLoss(nn.Module):
    def __init__(self, peak_dist=6.0, sigma=1.0, profile_len=21):
        """
        peak_dist: Distance between the two Gaussian peaks in pixels.
        sigma: Sigma of each Gaussian (width parameter).
        profile_len: Length of the sampled profile vector.
        """
        super().__init__()
        self.peak_dist = peak_dist
        self.sigma = sigma
        self.profile_len = profile_len

        # Create template
        # Center is 0. Range is roughly [-profile_len/2, profile_len/2]
        x = torch.arange(profile_len, dtype=torch.float32) - (profile_len - 1) / 2

        template = self.get_bigaussian_profile(x, peak_dist, sigma)
        self.register_buffer("template", template)

    @staticmethod
    def get_bigaussian_profile(x, peak_dist, sigma):
        """
        Generates the raw BiGaussian intensity profile.
        """
        # Peaks at +/- peak_dist / 2
        mu1 = -peak_dist / 2
        mu2 = peak_dist / 2

        template = torch.exp(-((x - mu1) ** 2) / (2 * sigma**2)) + torch.exp(
            -((x - mu2) ** 2) / (2 * sigma**2)
        )
        return template

    def forward(self, profiles):
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
    def __init__(self):
        super().__init__()

    def forward(self, contour):
        """
        Calculates the Laplacian smoothing loss for a closed 2D contour.
        Penalizes the deviation of each vertex from the average of its neighbors.
        This acts as a regularizer to keep the contour smooth (minimizing curvature).
        """
        # contour: (N, 2)
        # Neighbors for closed loop
        v_prev = torch.roll(contour, shifts=1, dims=0)
        v_next = torch.roll(contour, shifts=-1, dims=0)

        # The Laplacian is the vector from the vertex to the average of its neighbors
        # L_i = v_i - (v_{i-1} + v_{i+1}) / 2
        laplacian = contour - (v_prev + v_next) / 2.0

        # Minimize the magnitude of the Laplacian vectors
        loss = (laplacian**2).sum(dim=-1).mean()
        return loss


class EdgeLengthConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, contour):
        """
        Penalizes the variance of edge lengths to encourage uniform vertex distribution.
        """
        v_next = torch.roll(contour, shifts=-1, dims=0)
        edge_lengths = torch.norm(contour - v_next, dim=-1)

        # Minimize variance: mean((l - mean_l)^2)
        return torch.var(edge_lengths)
