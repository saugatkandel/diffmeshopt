import numpy as np

from diffmeshopt.opt2d.generate_2d_data import trim_data


def test_trim_data_center():
    """Test trimming when contour is well inside the image."""
    image = np.zeros((100, 100))
    # Square contour from (40, 40) to (60, 60)
    contour = np.array([[40, 40], [60, 60]], dtype=float)
    margin = 10

    # Expected bounds:
    # min=40, max=60
    # start = 40 - 10 = 30
    # end = 60 + 10 = 70
    # size = 70 - 30 = 40
    trimmed_img, trimmed_cnt, _ = trim_data(image, contour, margin=margin)

    assert trimmed_img.shape == (40, 40)

    # Shift should be -30
    expected_cnt = contour - 30
    assert np.allclose(trimmed_cnt, expected_cnt)


def test_trim_data_boundary():
    """Test trimming when contour is near the image boundary (clipping)."""
    image = np.zeros((100, 100))
    # Contour near top-left
    contour = np.array([[5, 5], [15, 15]], dtype=float)
    margin = 10

    # Expected bounds:
    # min=5, max=15
    # start = 5 - 10 = -5 -> clipped to 0
    # end = 15 + 10 = 25
    # size = 25 - 0 = 25
    trimmed_img, trimmed_cnt, _ = trim_data(image, contour, margin=margin)

    assert trimmed_img.shape == (25, 25)
    # Shift should be 0 because start was clipped to 0
    assert np.allclose(trimmed_cnt, contour)


def test_trim_data_no_op():
    image = np.zeros((100, 100))
    contour = np.array([[40, 40]], dtype=float)
    # Margin 0 should return original
    img, cnt, _ = trim_data(image, contour, margin=0)
    assert img.shape == (100, 100)
    assert np.allclose(cnt, contour)
