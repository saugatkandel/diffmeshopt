import os
import shutil

import numpy as np
import ome_zarr.writer
import zarr


def generate_sphere(size=64, radius=20):
    """
    Generate a 3D numpy array of a sphere.
    """
    center = size // 2
    x, y, z = np.ogrid[:size, :size, :size]

    sphere = (x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2 < radius**2
    return sphere.astype(np.uint8)


def save_ome_zarr(array: np.ndarray, path: str):
    """
    Save a numpy array as an OME-Zarr store.
    """
    if os.path.exists(path):
        shutil.rmtree(path)  # zarr doesn't overwrite, so remove old store
    store = zarr.DirectoryStore(path)
    root_group = zarr.group(store=store, overwrite=True)
    ome_zarr.writer.write_image(image=array, group=root_group, axes="zyx")


if __name__ == "__main__":
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    file_path = os.path.join(data_dir, "sphere.zarr")
    sphere = generate_sphere()
    save_ome_zarr(sphere, file_path)
    print(f"Generated and saved {file_path}")
