import torch
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import mesh_laplacian_smoothing

def sample_volume_along_normals(
    mesh: Meshes,
    volume: torch.Tensor,
    num_samples: int,
    sample_distance: float
):
    """
    Sample intensity values from a 3D volume along the vertex normals of a mesh.

    Args:
        mesh: A PyTorch3D Meshes object.
        volume: A 3D torch tensor representing the volume data.
        num_samples: The number of points to sample along each normal.
        sample_distance: The total distance to sample along the normal (half in each direction).

    Returns:
        A tensor of shape (num_vertices, num_samples) with the sampled intensity profiles.
    """
    verts = mesh.verts_list()[0]
    # Ensure normals are computed, and they are per-vertex
    mesh.verts_normals_list()
    normals = mesh.verts_normals_packed()

    # Create sampling points along normals
    line_points = torch.linspace(-sample_distance / 2, sample_distance / 2, num_samples, device=verts.device)
    sample_points = verts.unsqueeze(1) + normals.unsqueeze(1) * line_points.view(1, -1, 1)

    # Normalize sample points to be in the range [-1, 1] for grid_sample
    # Assuming volume coordinates are from 0 to D-1, H-1, W-1
    # and we need to map them to [-1, 1]
    volume_shape = torch.tensor(volume.shape, device=verts.device, dtype=torch.float32)
    sample_points_normalized = (sample_points / (volume_shape - 1)) * 2 - 1
    
    # grid_sample expects input in (N, D, H, W, 3) format for 3D
    # our sample_points_normalized is (V, S, 3), we need to reshape it
    # to (1, V, S, 1, 3) for 3D sampling (D=V, H=S, W=1)
    # The volume needs to be in (N, C, D, H, W) format, so we add batch and channel dims.
    volume_unsqueezed = volume.unsqueeze(0).unsqueeze(0)
    sample_points_grid = sample_points_normalized.view(1, -1, num_samples, 1, 3)

    # Sample the volume
    sampled_values = F.grid_sample(
        volume_unsqueezed,
        sample_points_grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )

    # Reshape back to (num_vertices, num_samples)
    return sampled_values.view(-1, num_samples)

def sum_of_two_gaussians(x, mean1, sigma1, mean2, sigma2, weight1, weight2):
    """
    Compute the sum of two Gaussians.
    x is a 1D tensor of positions.
    The other parameters can be per-vertex.
    """
    gauss1 = weight1 * torch.exp(-0.5 * ((x - mean1) / sigma1)**2)
    gauss2 = weight2 * torch.exp(-0.5 * ((x - mean2) / sigma2)**2)
    return gauss1 + gauss2

def gaussian_prior_loss(
    mesh: Meshes,
    volume: torch.Tensor,
    gaussian_params: dict,
    num_samples: int = 20,
    sample_distance: float = 10.0,
    smoothness_weight: float = 0.1
):
    """
    A loss function that encourages the intensity profile along the mesh normals
    to follow a sum of two Gaussians.
    """
    # Sample intensity profiles along vertex normals
    intensity_profiles = sample_volume_along_normals(
        mesh, volume, num_samples, sample_distance
    )

    # Generate the expected profiles from the sum of two Gaussians
    x = torch.linspace(-sample_distance / 2, sample_distance / 2, num_samples, device=mesh.device)
    
    # The gaussian_params are per-vertex, so they need to be unsqueezed for broadcasting
    expected_profiles = sum_of_two_gaussians(
        x.unsqueeze(0),
        gaussian_params["mean1"].unsqueeze(1),
        gaussian_params["sigma1"].unsqueeze(1),
        gaussian_params["mean2"].unsqueeze(1),
        gaussian_params["sigma2"].unsqueeze(1),
        gaussian_params["weight1"].unsqueeze(1),
        gaussian_params["weight2"].unsqueeze(1)
    )

    # Reconstruction loss
    reconstruction_loss = F.mse_loss(intensity_profiles, expected_profiles)

    # Smoothness loss for the Gaussian parameters
    # We can use laplacian smoothing on the parameter fields
    verts = mesh.verts_list()[0]
    faces = mesh.faces_list()[0]
    temp_mesh = Meshes(verts=[verts], faces=[faces])
    
    smoothness_loss = 0
    for param_name, param_values in gaussian_params.items():
        # We need to create a "mesh" where the vertex features are the parameters
        # and compute the laplacian smoothing on these features.
        # However, mesh_laplacian_smoothing works on vertex positions.
        # A simpler way is to compute the laplacian of the parameter field on the mesh graph.
        # Pytorch3d does not directly expose this for arbitrary fields.
        # We can implement a simple version of it.
        # Or we can just apply laplacian smoothing to a dummy mesh where the vertex positions are the parameters.
        # Let's try to implement a simple version.
        
        # A simple approximation for laplacian smoothing on a field `f` is sum_neighbors((f_j - f_i)^2)
        # We can get neighbors from the mesh edges.
        edges = temp_mesh.edges_packed()
        v_i = param_values[edges[:, 0]]
        v_j = param_values[edges[:, 1]]
        smoothness_loss += ((v_i - v_j)**2).mean()

    total_loss = reconstruction_loss + smoothness_weight * smoothness_loss
    return total_loss
