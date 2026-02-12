"""
Unit tests for diffmeshopt.opt2d.vis
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from diffmeshopt.opt2d import vis
from diffmeshopt.opt2d.config import (
    ContourRefinerProps,
    RBFContourRefinerProps,
    TemplateProps,
)
from diffmeshopt.opt2d.refiner import RBFContourRefiner
from diffmeshopt.opt2d.template import FixedTemplateModel


@pytest.fixture
def mock_plt():
    with patch("diffmeshopt.opt2d.vis.plt") as mock:
        # Default behavior: return a single mock axis, which is the most common case.
        # This covers subplots() and subplots(1, 1).
        mock.subplots.return_value = (MagicMock(), MagicMock())
        yield mock


@pytest.fixture
def synthetic_data():
    H, W = 100, 100
    image = np.zeros((H, W), dtype=np.float32)
    # Create a simple square contour
    contour = np.array([[40, 40], [40, 60], [60, 60], [60, 40]], dtype=np.float32)
    return image, contour


def test_plot_prior_and_landscape_from_contour_defaults(mock_plt, synthetic_data):
    image, contour = synthetic_data
    # This function expects subplots to return two axes for unpacking.
    # Configure the mock specifically for this test's expectation.
    mock_plt.subplots.return_value = (MagicMock(), [MagicMock(), MagicMock()])
    # Should run without errors using default props
    vis.plot_prior_and_landscape_from_contour(image, contour)
    assert mock_plt.show.called


@patch("diffmeshopt.opt2d.vis.plot_prior_and_landscape_from_profiles")
@patch("diffmeshopt.opt2d.vis.sampling.sample_profiles_stochastic")
def test_plot_prior_and_landscape_props_propagation(
    mock_sample, mock_plot_profiles, mock_plt, synthetic_data
):
    image, contour = synthetic_data
    refiner_props = ContourRefinerProps(
        profile_length=12, profile_width=3, sample_step=0.5, num_sampled_profiles=7
    )
    template_props = TemplateProps(peak_dist=5.0)

    # Mock return of sample_profiles_stochastic
    # Returns: profiles, sub_indices, valid_mask
    # Profiles shape should match (num_samples, profile_length)
    mock_sample.return_value = (torch.zeros((7, 12)), torch.zeros(7), torch.zeros(7))

    vis.plot_prior_and_landscape_from_contour(
        image, contour, refiner_props=refiner_props, template_props=template_props
    )

    # Check if sampling was called with correct props
    mock_sample.assert_called_once()
    _, kwargs = mock_sample.call_args
    assert kwargs["profile_length"] == 12
    assert kwargs["profile_width"] == 3
    assert kwargs["sample_step"] == 0.5
    assert kwargs["num_samples"] == 7

    # Check if plotting was called with correct template props
    mock_plot_profiles.assert_called_once()
    _, kwargs_plot = mock_plot_profiles.call_args
    assert kwargs_plot["template_props"] == template_props


def test_plot_contour_normals_defaults(mock_plt, synthetic_data):
    image, contour = synthetic_data
    vis.plot_contour_normals(image, contour)
    assert mock_plt.show.called


@patch("diffmeshopt.opt2d.vis.sampling._get_stratified_indices")
def test_plot_contour_normals_with_props_stochastic(mock_get_indices, mock_plt, synthetic_data):
    image, contour = synthetic_data
    refiner_props = ContourRefinerProps(profile_length=15, num_sampled_profiles=5)

    # Mock return value for indices to control how many normals are plotted
    mock_get_indices.return_value = torch.tensor([0, 2])

    vis.plot_contour_normals(image, contour, stochastic=True, refiner_props=refiner_props)

    # Check if _get_stratified_indices was called with correct num_lines
    mock_get_indices.assert_called_once()
    args, _ = mock_get_indices.call_args
    assert args[0] == len(contour)
    assert args[1] == refiner_props.num_sampled_profiles

    # Check that plot was called for the contour and each normal
    ax = mock_plt.subplots.return_value[1]
    assert ax.plot.call_count == 1 + len(mock_get_indices.return_value)

    assert mock_plt.show.called


def test_plot_contour_normals_with_props_non_stochastic(mock_plt, synthetic_data):
    image, contour = synthetic_data
    refiner_props = ContourRefinerProps(profile_length=15, num_sampled_profiles=3)

    vis.plot_contour_normals(image, contour, stochastic=False, refiner_props=refiner_props)

    # Check that plot was called for the contour and the correct number of normals
    # num_lines_to_plot = min(num_sampled_profiles, len(contour)) = min(3, 4) = 3
    ax = mock_plt.subplots.return_value[1]
    assert ax.plot.call_count == 1 + 3

    assert mock_plt.show.called


def test_plot_profile_statistics_defaults(mock_plt):
    profiles = np.random.rand(10, 20).astype(np.float32)
    vis.plot_profile_statistics(profiles)
    assert mock_plt.show.called


@patch("diffmeshopt.opt2d.vis.BiGaussianLoss.get_bigaussian_profile")
def test_plot_profile_statistics_with_props(mock_get_profile, mock_plt):
    profiles = np.random.rand(10, 20).astype(np.float32)
    template_props = TemplateProps(peak_dist=2.0, sigma=0.8)
    mock_get_profile.return_value = torch.zeros(profiles.shape[1])

    vis.plot_profile_statistics(profiles, template_props=template_props)
    assert mock_plt.show.called

    # Check if get_bigaussian_profile was called with correct props
    mock_get_profile.assert_called_once()
    _, kwargs = mock_get_profile.call_args
    assert kwargs["peak_dist"] == template_props.peak_dist
    assert kwargs["sigma"] == template_props.sigma


def test_plot_bspline_basis(mock_plt):
    vis.plot_bspline_basis(num_cp=5, num_eval=20)
    assert mock_plt.show.called


def test_compare_bspline_basis_functions(mock_plt):
    configs = [5, 10]
    # This function expects subplots to return multiple axes, so we
    # configure the mock to return a list of axes.
    mock_plt.subplots.return_value = (MagicMock(), [MagicMock() for _ in configs])
    vis.compare_bspline_basis_functions(configs=configs, num_eval=20)
    assert mock_plt.show.called


def test_plot_parameter_curves(mock_plt):
    params = {"p1": torch.randn(10), "p2": torch.randn(10)}
    vis.plot_parameter_curves(params)
    assert mock_plt.show.called


def test_plot_rbf_deformation(mock_plt):
    init_c = np.zeros((10, 2))
    final_c = np.zeros((10, 2))
    cp = np.zeros((5, 2))
    w = np.zeros((5, 2))
    vis.plot_rbf_deformation(init_c, final_c, cp, w)
    assert mock_plt.show.called


def test_refiner_visualize_rbf_field(mock_plt):
    """Test the integration method on the refiner."""
    initial_contour = torch.rand((10, 2))
    props = RBFContourRefinerProps(rbf_num_control_points=5)
    template = FixedTemplateModel(TemplateProps())
    refiner = RBFContourRefiner(initial_contour, props, template)

    refiner.visualize_rbf_field()
    assert mock_plt.show.called
