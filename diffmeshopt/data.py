import torch
import zarr


def load_segmentation(path):
    """
    Load a 3D segmentation from an OME-Zarr store.
    Returns a torch tensor.
    """
    store = zarr.open(path, mode="r")
    # OME-Zarr writers typically store the highest resolution in a '0' key
    array = store["0"][:]
    return torch.from_numpy(array)


def preprocess_segmentation(segmentation, label):
    """
    Binarize the segmentation for a given label.
    """
    return (segmentation == label).float()
