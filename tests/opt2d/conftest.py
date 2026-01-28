import pytest
import torch

from diffmeshopt.opt2d.generate_2d_data import generate_synthetic_data

# Use a fixed seed for reproducibility in synthetic data generation
torch.manual_seed(42)


@pytest.fixture(scope="module")
def synthetic_data():
    """
    Generate a single synthetic image and initial contour for all tests in this module.
    This is a moderately expensive operation, so we use module scope.
    """
    # Call the function with arguments it accepts
    img_np, initial_contour_np, _ = generate_synthetic_data(
        shape=(64, 64), radius=20, center=(32, 32)
    )
    # Convert numpy arrays to torch tensors for the refiners
    img = torch.from_numpy(img_np.copy()).float().unsqueeze(0).unsqueeze(0)
    initial_contour = torch.from_numpy(initial_contour_np.copy()).float()

    return img, initial_contour
