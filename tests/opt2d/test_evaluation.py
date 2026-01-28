import pytest
import torch

from diffmeshopt.opt2d.evaluation import compute_contour_metrics


@pytest.fixture
def contour_a():
    """A simple square contour."""
    return torch.tensor(
        [
            [10.0, 10.0],
            [20.0, 10.0],
            [20.0, 20.0],
            [10.0, 20.0],
        ],
        dtype=torch.float32,
    )


@pytest.fixture
def contour_b_translated(contour_a):
    """Contour A translated by (2, 3)."""
    return contour_a + torch.tensor([2.0, 3.0])


@pytest.fixture
def contour_c_outlier(contour_a):
    """Contour A with one point moved far away."""
    c = contour_a.clone()
    c[2] = torch.tensor([50.0, 50.0])
    return c


def test_compute_metrics_identical(contour_a):
    """Metrics for identical contours should be zero."""
    metrics = compute_contour_metrics(contour_a, contour_a)
    assert isinstance(metrics, dict)
    assert metrics["mean_dist"] == pytest.approx(0.0)
    assert metrics["hausdorff_dist"] == pytest.approx(0.0)


def test_compute_metrics_translated(contour_a, contour_b_translated):
    """Test metrics for a simple translated contour."""
    metrics = compute_contour_metrics(contour_a, contour_b_translated)
    expected_dist = (2**2 + 3**2) ** 0.5
    assert metrics["mean_dist"] == pytest.approx(expected_dist)
    assert metrics["hausdorff_dist"] == pytest.approx(expected_dist)


def test_compute_metrics_hausdorff(contour_a, contour_c_outlier):
    """Test Hausdorff distance with an outlier point."""
    metrics = compute_contour_metrics(contour_a, contour_c_outlier)
    expected_hausdorff = ((50.0 - 20.0) ** 2 + (50.0 - 20.0) ** 2) ** 0.5
    assert metrics["hausdorff_dist"] == pytest.approx(expected_hausdorff)
    assert metrics["mean_dist"] < metrics["hausdorff_dist"]
