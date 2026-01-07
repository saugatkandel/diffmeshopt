import SimpleITK as sitk
import numpy as np
import torch


def load_segmentation(path):
    """
    Load a 3D segmentation from a file.
    Returns a torch tensor.
    """
    image = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(image)
    return torch.from_numpy(array)


def preprocess_segmentation(segmentation, label):
    """
    Binarize the segmentation for a given label.
    """
    return (segmentation == label).float()
