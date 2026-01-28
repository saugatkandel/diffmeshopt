import logging
from pathlib import Path

import click
import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import measure


def generate_synthetic_data(shape=(256, 256), radius=60, center=(128, 128)):
    """
    Generates a synthetic 2D image with a bi-Gaussian membrane profile.
    """
    logging.info("Generating synthetic data...")
    image = np.zeros(shape, dtype=np.float32)
    y, x = np.ogrid[: shape[0], : shape[1]]

    # Ground truth membrane (circle)
    dist_from_center = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    dist_from_membrane = dist_from_center - radius

    # Bi-Gaussian profile: two peaks at +/- width/2
    width = 4.0
    sigma = 1.0

    # Profile function: exp(-(d - w/2)^2) + exp(-(d + w/2)^2)
    profile = np.exp(-((dist_from_membrane - width / 2) ** 2) / (2 * sigma**2)) + np.exp(
        -((dist_from_membrane + width / 2) ** 2) / (2 * sigma**2)
    )

    image += profile

    # Add noise
    noise = np.random.normal(0, 0.1, shape).astype(np.float32)
    image += noise

    # Blur (PSF)
    image = gaussian_filter(image, sigma=1.0)

    # Generate initial contour (perturbed circle)
    # Create a rough mask
    rough_radius = radius * 0.95  # Shrunk slightly
    mask = dist_from_center <= rough_radius

    contours = measure.find_contours(mask, 0.5)
    initial_contour = max(contours, key=len)  # (N, 2) array of (row, col)

    # Ground truth contour for evaluation
    gt_contours = measure.find_contours(dist_from_center <= radius, 0.5)
    gt_contour = max(gt_contours, key=len)

    return image, initial_contour, gt_contour


def load_real_data(path):
    """
    Loads real data from pickle file.
    Expected content: Dictionary with keys "tomo_avg30A", "organelle", "membrane"
    """
    logging.info(f"Loading real data from {path}...")
    data = joblib.load(path)

    tomo_slice = data["tomo_avg30A"]
    organelle_seg = data["organelle"]
    membrane_seg = data["membrane"]

    # Normalize image
    tomo_slice = (tomo_slice - np.mean(tomo_slice)) / (np.std(tomo_slice) + 1e-8)

    # Extract contour from organelle segmentation (rough initialization)
    # organelle_seg is assumed to be a binary mask or label map
    contours = measure.find_contours(organelle_seg, 0.5)
    if not contours:
        raise ValueError("No contours found in organelle segmentation")

    # Assume the largest contour is the organelle
    initial_contour = max(contours, key=len)

    # For real data, we might not have a perfect ground truth line,
    # but we can return the membrane_seg as a reference if needed.
    # For now, we'll just return an empty array for GT to indicate it's real data.
    return tomo_slice, initial_contour, np.zeros((0, 2))


def trim_data(image, contour, gt=None, margin=50):
    """
    Trims the image around the contour with a given margin.
    Returns cropped image and updated coordinates.
    """
    if margin <= 0 or contour is None:
        return image, contour, gt

    logging.info(f"Trimming image with margin {margin}...")
    # Contour is (N, 2) -> (row, col)
    min_row, min_col = np.floor(np.min(contour, axis=0)).astype(int)
    max_row, max_col = np.ceil(np.max(contour, axis=0)).astype(int)

    # Apply margin and clip to image bounds
    row_start = max(0, min_row - margin)
    row_end = min(image.shape[0], max_row + margin)
    col_start = max(0, min_col - margin)
    col_end = min(image.shape[1], max_col + margin)

    # Crop
    image = image[row_start:row_end, col_start:col_end]

    # Update coordinates (on copies)
    contour = contour.copy()
    contour[:, 0] -= row_start
    contour[:, 1] -= col_start

    if gt is not None and len(gt) > 0:
        gt = gt.copy()
        gt[:, 0] -= row_start
        gt[:, 1] -= col_start

    return image, contour, gt


@click.command()
@click.option(
    "--real-path",
    type=click.Path(path_type=Path),
    default="/workspace/diffmeshopt/data/20289/denoised/data_slice123.pkl",
    help="Path to real data file.",
)
@click.option("--synthetic", is_flag=True, help="Force synthetic data generation.")
@click.option("--visualize", is_flag=True, help="Save a visualization of the data.")
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default="/workspace/diffmeshopt/data/2d_training_data.pkl",
    help="Output path for the generated data.",
)
@click.option(
    "--trim-margin",
    type=int,
    default=50,
    help="Margin in pixels to trim around the segmentation.",
)
def main(real_path, synthetic, visualize, output, trim_margin):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    output.parent.mkdir(parents=True, exist_ok=True)

    image = None
    contour = None
    gt = None

    if not synthetic and real_path.exists():
        try:
            image, contour, gt = load_real_data(real_path)
            logging.info("Successfully loaded real data.")
        except Exception as e:
            logging.warning(f"Failed to load real data: {e}. Falling back to synthetic.")
    elif not synthetic:
        logging.info(f"Real data not found at {real_path}.")

    if image is None:
        image, contour, gt = generate_synthetic_data()

    # Trim image around contour if requested
    image, contour, gt = trim_data(image, contour, gt, trim_margin)

    joblib.dump({"image": image, "contour": contour, "gt": gt}, output)
    logging.info(f"Data saved to {output}")
    logging.info(f"Image shape: {image.shape}")
    logging.info(f"Contour shape: {contour.shape}")

    if visualize:
        vis_path = output.with_suffix(".png")
        logging.info(f"Saving visualization to {vis_path}...")
        plt.figure(figsize=(10, 10))
        plt.imshow(image, cmap="gray")
        plt.plot(contour[:, 1], contour[:, 0], "r-", linewidth=2, label="Initial Contour")
        if gt is not None and len(gt) > 0:
            plt.plot(gt[:, 1], gt[:, 0], "g--", linewidth=2, label="Ground Truth")
        plt.legend()
        plt.title("2D Data Visualization")
        plt.savefig(vis_path)
        plt.close()


if __name__ == "__main__":
    main()
