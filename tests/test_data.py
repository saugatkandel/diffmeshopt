import torch
import numpy as np
import SimpleITK as sitk
import os
from src.data import load_segmentation, preprocess_segmentation


def test_load_segmentation():
    # Create a dummy numpy array
    dummy_array = np.random.randint(0, 5, size=(10, 10, 10), dtype=np.uint8)

    # Create a dummy NIfTI file
    dummy_image = sitk.GetImageFromArray(dummy_array)
    test_file = "test_seg.nii.gz"
    sitk.WriteImage(dummy_image, test_file)

    # Load the segmentation
    loaded_seg = load_segmentation(test_file)

    # Check if the loaded segmentation is a torch tensor
    assert isinstance(loaded_seg, torch.Tensor)

    # Check if the content is the same
    assert torch.equal(torch.from_numpy(dummy_array), loaded_seg)

    # Clean up the dummy file
    os.remove(test_file)


def test_preprocess_segmentation():
    # Create a dummy segmentation
    dummy_seg = torch.tensor([[[0, 1, 2], [1, 2, 0], [2, 0, 1]]])

    # Preprocess for label 1
    binary_mask = preprocess_segmentation(dummy_seg, 1)

    # Expected mask
    expected_mask = torch.tensor([[[0, 1, 0], [1, 0, 0], [0, 0, 1]]]).float()

    # Check if the binary mask is correct
    assert torch.equal(binary_mask, expected_mask)
