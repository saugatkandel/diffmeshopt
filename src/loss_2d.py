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

        # Peaks at +/- peak_dist / 2
        mu1 = -peak_dist / 2
        mu2 = peak_dist / 2

        template = torch.exp(-((x - mu1) ** 2) / (2 * sigma**2)) + torch.exp(
            -((x - mu2) ** 2) / (2 * sigma**2)
        )

        # Normalize template to zero mean, unit variance for correlation
        template = (template - template.mean()) / (template.std() + 1e-8)
        self.register_buffer("template", template)

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
