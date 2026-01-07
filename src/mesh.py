from skimage import measure
import numpy as np
import torch
from pytorch3d.structures import Meshes

def segmentation_to_mesh(segmentation, level=0.5):
    """
    Convert a 3D segmentation to a mesh using marching cubes.
    Returns vertices and faces as torch tensors, and a PyTorch3D Meshes object.
    """
    # Ensure segmentation is a numpy array on the CPU
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()

    # Marching cubes
    verts, faces, _, _ = measure.marching_cubes(segmentation, level)
    
    # Convert to torch tensors
    verts = torch.from_numpy(verts.copy()).float()
    faces = torch.from_numpy(faces.copy()).long()
    
    # Create a PyTorch3D Meshes object
    mesh = Meshes(verts=[verts], faces=[faces])

    return verts, faces, mesh
