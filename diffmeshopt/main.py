import os

import torch
from tqdm import tqdm

from diffmeshopt.data import load_segmentation, preprocess_segmentation
from diffmeshopt.loss import boundary_loss
from diffmeshopt.mesh import segmentation_to_mesh
from diffmeshopt.model import MeshRefinementModel
from diffmeshopt.prior import gaussian_prior_loss
from diffmeshopt.utils import save_mesh


def main():
    """
    Main function to run the mesh refinement pipeline.

    Note: Please run `python src/generate_sample_data.py` first to generate the sample data.
    """
    # Configuration
    data_path = "data/sphere.zarr"
    output_dir = "output"
    label = 1
    learning_rate = 1e-3
    num_iterations = 100
    prior_weight = 1.0

    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}.")
        print("Please run `python src/generate_sample_data.py` to generate the sample data.")
        return

    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load and preprocess data
    print("Loading and preprocessing data...")
    segmentation = load_segmentation(data_path)
    binary_mask = preprocess_segmentation(segmentation, label)

    # Convert segmentation to mesh
    print("Converting segmentation to mesh...")
    verts, faces, initial_mesh = segmentation_to_mesh(binary_mask)
    save_mesh(initial_mesh, os.path.join(output_dir, "initial_mesh.obj"))

    # Create model and optimizer
    print("Creating model and optimizer...")
    model = MeshRefinementModel(verts, faces)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Optimization loop
    print("Optimizing mesh...")
    for i in tqdm(range(num_iterations)):
        optimizer.zero_grad()
        refined_mesh, gaussian_params = model()
        prior = gaussian_prior_loss(refined_mesh, segmentation, gaussian_params)
        loss = boundary_loss(refined_mesh) + prior_weight * prior
        loss.backward()
        optimizer.step()

    # Save final mesh
    print("Saving final mesh...")
    final_mesh, _ = model()
    save_mesh(final_mesh, os.path.join(output_dir, "refined_mesh.obj"))
    print(f"Refined mesh saved to {os.path.join(output_dir, 'refined_mesh.obj')}")


if __name__ == "__main__":
    main()
