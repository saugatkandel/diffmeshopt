import torch
import trimesh


def save_mesh(mesh, path):
    """
    Save a PyTorch3D mesh to a file using trimesh.
    """
    verts = mesh.verts_list()[0].detach().cpu().numpy()
    faces = mesh.faces_list()[0].detach().cpu().numpy()

    trimesh_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    trimesh_mesh.export(path)
