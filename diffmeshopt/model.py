import torch
import torch.nn as nn
from pytorch3d.structures import Meshes


class MeshRefinementModel(nn.Module):
    def __init__(self, verts, faces, initial_gaussian_params=None):
        """
        A model to refine a mesh and its associated per-vertex parameters.

        Args:
            verts: Initial vertex positions.
            faces: Mesh faces.
            initial_gaussian_params (dict, optional): A dictionary with initial
                values for the Gaussian parameters. If None, default values are used.
        """
        super().__init__()
        self.verts = nn.Parameter(verts)
        self.register_buffer("faces", faces)

        num_verts = verts.shape[0]

        if initial_gaussian_params is None:
            # Default initialization
            initial_gaussian_params = {
                "mean1": torch.full((num_verts,), -1.0),
                "mean2": torch.full((num_verts,), 1.0),
                "log_sigma1": torch.full((num_verts,), 0.0),  # log(1.0)
                "log_sigma2": torch.full((num_verts,), 0.0),  # log(1.0)
                "log_weight1": torch.full((num_verts,), -0.693),  # log(0.5)
                "log_weight2": torch.full((num_verts,), -0.693),  # log(0.5)
            }

        self.mean1 = nn.Parameter(initial_gaussian_params["mean1"])
        self.mean2 = nn.Parameter(initial_gaussian_params["mean2"])
        self.log_sigma1 = nn.Parameter(initial_gaussian_params["log_sigma1"])
        self.log_sigma2 = nn.Parameter(initial_gaussian_params["log_sigma2"])
        self.log_weight1 = nn.Parameter(initial_gaussian_params["log_weight1"])
        self.log_weight2 = nn.Parameter(initial_gaussian_params["log_weight2"])

    def forward(self):
        """
        Returns the refined mesh and the Gaussian parameters.
        """
        mesh = Meshes(verts=[self.verts], faces=[self.faces])

        # Exponentiate logs to ensure sigmas and weights are positive
        gaussian_params = {
            "mean1": self.mean1,
            "mean2": self.mean2,
            "sigma1": torch.exp(self.log_sigma1),
            "sigma2": torch.exp(self.log_sigma2),
            "weight1": torch.exp(self.log_weight1),
            "weight2": torch.exp(self.log_weight2),
        }

        return mesh, gaussian_params
