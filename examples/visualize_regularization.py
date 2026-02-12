"""
Script to visualize the effects of different regularization strategies.

Experiment Design:
------------------
Demonstrates specific regularization artifacts and solutions using isolated test cases.

1. Vertex Shrinking: Shows how standard Laplacian smoothing collapses volume,
   while Tangential Smoothing preserves it.
2. B-Spline Corner Cutting: Shows how regularizing control points incorrectly can
   bias the curve shape.
3. RBF Weight Decay: Shows how weight decay controls the stiffness of the deformation.

This script demonstrates:
1. Vertex Refiner: The "Shrinking" problem with standard Laplacian smoothing vs.
   the volume preservation of Tangential Smoothing.
2. B-Spline Refiner: The "Corner Cutting" problem when shrinking forces are applied
   to control points vs. correct regularization.
3. RBF Refiner: The effect of Weight Decay on deformation stiffness.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from diffmeshopt.opt2d.config import (
    BSplineContourRefinerProps,
    ContourRefinerProps,
    RBFContourRefinerProps,
    RegularizerType,
    TemplateProps,
)
from diffmeshopt.opt2d.refiner import (
    BSplineContourRefiner,
    ContourRefiner,
    RBFContourRefiner,
)
from diffmeshopt.opt2d.regularizer_recipes import (
    TANGENTIAL_SMOOTHING_RBF,
    TANGENTIAL_SMOOTHING_VERTEX,
)
from diffmeshopt.opt2d.template import TemplateModelFactory


def run_optimization(refiner, image, steps=50):
    """Helper to run optimization loop."""
    for _ in range(steps):
        refiner.step(image)
    return refiner.contour.detach().cpu().numpy()


def visualize_vertex_shrinking():
    print("Visualizing Vertex Refiner: Shrinking vs Tangential Smoothing...")
    # Create a circle, flat image (no data term to isolate regularization forces)
    theta = torch.linspace(0, 2 * np.pi, 100)[:-1]
    radius = 30.0
    initial_contour = torch.stack(
        [radius * torch.sin(theta) + 50, radius * torch.cos(theta) + 50], dim=1
    )
    image = torch.zeros(1, 1, 100, 100)  # Flat image

    template = TemplateModelFactory.create("fixed", TemplateProps())

    # Case 1: Standard Laplacian Smoothing (Causes Shrinking)
    props_shrink = ContourRefinerProps(
        learning_rate=1.0,
        profile_length=21,
        initial_loss_weights={
            "data_loss": 0.0,
            RegularizerType.CONTOUR_LAPLACIAN.value: 1.0,  # Strong shrinking force
            RegularizerType.TANGENTIAL_LAPLACIAN.value: 0.0,
            RegularizerType.CONTOUR_ANCHOR.value: 0.0,
        },
    )
    refiner_shrink = ContourRefiner(initial_contour.clone(), props_shrink, template)
    final_shrink = run_optimization(refiner_shrink, image)

    # Case 2: Tangential Smoothing (Preserves Volume)
    loss_weights_tan = TANGENTIAL_SMOOTHING_VERTEX.copy()
    loss_weights_tan[RegularizerType.CONTOUR_ANCHOR.value] = 0.0
    loss_weights_tan[RegularizerType.NORMAL_CONSISTENCY.value] = 1.0

    props_tan = ContourRefinerProps(
        learning_rate=1.0,
        profile_length=21,
        initial_loss_weights=loss_weights_tan,
    )
    refiner_tan = ContourRefiner(initial_contour.clone(), props_tan, template)
    final_tan = run_optimization(refiner_tan, image)

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Close loops for plotting
    init_c = np.vstack([initial_contour.numpy(), initial_contour.numpy()[0]])
    shrink_c = np.vstack([final_shrink, final_shrink[0]])
    tan_c = np.vstack([final_tan, final_tan[0]])

    ax[0].set_title("Standard Laplacian\n(Shrinking)")
    ax[0].plot(init_c[:, 1], init_c[:, 0], "k--", label="Initial")
    ax[0].plot(shrink_c[:, 1], shrink_c[:, 0], "r-", label="Final")
    ax[0].set_xlim(0, 100)
    ax[0].set_ylim(0, 100)
    ax[0].legend()
    ax[0].set_aspect("equal")
    ax[0].grid(True, alpha=0.3)

    ax[1].set_title("Tangential Smoothing\n(Volume Preserved)")
    ax[1].plot(init_c[:, 1], init_c[:, 0], "k--", label="Initial")
    ax[1].plot(tan_c[:, 1], tan_c[:, 0], "g-", label="Final")
    ax[1].set_xlim(0, 100)
    ax[1].set_ylim(0, 100)
    ax[1].legend()
    ax[1].set_aspect("equal")
    ax[1].grid(True, alpha=0.3)

    plt.suptitle("Effect of Regularization on Vertex Refiner (No Data Term)")
    plt.tight_layout()
    plt.show()


def visualize_bspline_corner_cutting():
    print("Visualizing B-Spline Refiner: Corner Cutting...")
    # Square contour
    points = []
    for i in range(20):
        points.append([30, 30 + i * 2])
    for i in range(20):
        points.append([30 + i * 2, 70])
    for i in range(20):
        points.append([70, 70 - i * 2])
    for i in range(20):
        points.append([70 - i * 2, 30])
    initial_contour = torch.tensor(points, dtype=torch.float32)

    image = torch.zeros(1, 1, 100, 100)
    template = TemplateModelFactory.create("fixed", TemplateProps())

    # Case 1: Contour Laplacian on Control Points (Corner Cutting)
    props_bad = BSplineContourRefinerProps(
        contour_num_control_points=16,
        learning_rate=0.5,
        profile_length=21,
        initial_loss_weights={
            "data_loss": 0.0,
            RegularizerType.CONTOUR_LAPLACIAN.value: 2.0,  # Shrinks CP polygon
            RegularizerType.TANGENTIAL_LAPLACIAN.value: 0.0,
            RegularizerType.CONTOUR_ANCHOR.value: 0.0,
        },
    )
    refiner_bad = BSplineContourRefiner(initial_contour.clone(), props_bad, template)
    final_bad = run_optimization(refiner_bad, image)

    # Close loops
    init_c = np.vstack([initial_contour.numpy(), initial_contour.numpy()[0]])
    bad_c = np.vstack([final_bad, final_bad[0]])

    # Plot
    plt.figure(figsize=(6, 6))
    plt.title("B-Spline Corner Cutting\n(caused by shrinking control points)")
    plt.plot(init_c[:, 1], init_c[:, 0], "k--", label="Initial (Square)")
    plt.plot(bad_c[:, 1], bad_c[:, 0], "r-", label="Refined (Rounded)")
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.legend()
    plt.gca().set_aspect("equal")
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    plt.show()


def visualize_rbf_weight_decay():
    print("Visualizing RBF Refiner: Weight Decay (Stiffness)...")
    # Line contour
    initial_contour = torch.stack(
        [torch.full((20,), 16.0), torch.linspace(0, 100, 20)], dim=1
    )  # Vertical line

    # Create a dummy image with a target line shifted by 20 pixels
    # This creates a strong data force pulling the contour to the right
    image = torch.zeros(1, 1, 100, 100)
    image[:, :, :, 20:] = 1.0  # Step function

    template = TemplateModelFactory.create("fixed", TemplateProps())

    results = []
    weights = [0.0, 0.1, 1.0, 10.0]

    for w in weights:
        loss_weights = TANGENTIAL_SMOOTHING_RBF.copy()
        loss_weights[RegularizerType.RBF_WEIGHT_DECAY.value] = w

        props = RBFContourRefinerProps(
            rbf_num_control_points=5,
            learning_rate=0.5,
            profile_length=21,
            initial_loss_weights=loss_weights,
            rbf_kernel_sigma=20.0,
        )
        refiner = RBFContourRefiner(initial_contour.clone(), props, template)
        # Manually pull weights to simulate data force if we don't run full optim
        # Or just run optimization
        # Let's run optimization against a zero image but initialize weights
        # Actually, let's just optimize.
        final = run_optimization(refiner, image, steps=30)
        results.append((w, final))

    plt.figure(figsize=(8, 6))
    plt.title("Effect of RBF Weight Decay on Deformation")
    plt.plot(initial_contour[:, 1], initial_contour[:, 0], "k--", label="Initial", linewidth=2)
    plt.axvline(36.0, color="gray", linestyle=":", label="Target Data")  # Target at 16+20=36
    for w, final in results:
        plt.plot(final[:, 1], final[:, 0], label=f"Weight={w}")
    plt.legend()
    plt.xlim(-10, 50)
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    visualize_vertex_shrinking()
    visualize_bspline_corner_cutting()
    visualize_rbf_weight_decay()
