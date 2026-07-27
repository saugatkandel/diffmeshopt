from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from matplotlib import pyplot as plt

import diffmeshopt.opt2d.config as config_module
import diffmeshopt.opt2d.evaluation as evaluation
from diffmeshopt.opt2d import vis
from diffmeshopt.opt2d.config import (
    DataLossType,
    RBFContourRefinerProps,
    TemplateProps,
)
from diffmeshopt.opt2d.refiner import RefinerFactory
from diffmeshopt.opt2d.template import TemplateModelFactory
from diffmeshopt.opt2d.trainer import OptimizationTrainer, TrainerConfig

DEFAULT_RBF_REFINER_PROPS = RBFContourRefinerProps(
    learning_rate=5e-1,
    data_loss_type=DataLossType.BIGAUSSIAN_CORRELATION,
    profile_length=51,
    profile_width=5,
    sample_step=1.0,
    num_sampled_profiles=256,
    laplacian_window_size=5,
    shape_loss_weight=0.1,
    center_symmetry_weight=0.0,
    initial_regularization_weights={
        "contour_laplacian": 0.0,
        "edge_length": 0.0,
        "normal_consistency": 0.0,
        "tangential_laplacian": 0.0,
        "contour_anchor": 0.0,
        "rbf_weight_decay": 0.001,
    },
    rbf_num_control_points=40,
    rbf_kernel_sigma=0.0,
)

DEFAULT_TEMPLATE_PROPS = TemplateProps(
    symmetric=True,
    sigma=0.75,
    peak_dist=4.5,
    min_peak_ratio=3.0,
    sigma_ratio=1.0,
    amp_ratio=1.0,
    smoothness_window_size=5,
)


def load_perturbation_dataset(dataset_path: str | Path) -> dict[str, Any]:
    return joblib.load(dataset_path)


def create_trainer_from_manual_config(
    dataset: dict[str, Any],
    output_dir: str | Path,
    device: str,
    refiner_props: config_module.ContourRefinerProps,
    template_props: config_module.TemplateProps,
    run_id: str,
    *,
    sample_key: str = "shrink_5_perturb_3",
    seed: int | None = 145,
    refiner_kind: str = "rbf",
    template_kind: str = "bspline",
    max_steps: int = 50000,
):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    output_dir = Path(output_dir)
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    sample = dataset[sample_key]
    original = dataset["original"]

    init_contour = sample["contour"]
    image = sample["image"]

    gt_contour = original["contour"].copy()
    gt_contour[:, 0] += original["row_start"] - sample["row_start"]
    gt_contour[:, 1] += original["col_start"] - sample["col_start"]

    image_t = torch.from_numpy(image).float().to(device)
    contour_t = torch.from_numpy(init_contour).float().to(device)
    gt_contour_t = torch.from_numpy(gt_contour).float().to(device)

    print("Computing contour metrics between GT and initial contour...")
    print(evaluation.compute_contour_metrics(gt_contour_t, contour_t))

    template_model = TemplateModelFactory.create(
        template_kind,
        template_props,
        num_vertices=len(init_contour),
        image_shape=image.shape,
    ).to(device)

    refiner = RefinerFactory.create(refiner_kind, contour_t, refiner_props, template_model).to(
        device
    )
    trainer_config = TrainerConfig(
        output_dir=run_dir,
        image=image,
        gt_contour=gt_contour,
        max_steps=max_steps,
        save_interval=2000,
        log_interval=100,
        use_tensorboard=False,
        save_images=True,
        enable_progress_bar=True,
    )

    trainer = OptimizationTrainer(refiner, trainer_config)

    return (
        trainer,
        run_dir,
        image_t,
        contour_t,
        gt_contour_t,
    )


def analyze_trainer_results(
    trainer: OptimizationTrainer, contour_t: torch.Tensor, image_t: torch.Tensor
):
    from IPython.display import display

    display(trainer.plot_image())
    display(trainer.plot_metrics())

    metrics_from_init = evaluation.compute_contour_metrics(trainer.refiner.contour, contour_t)
    chamfer_from_init = metrics_from_init["mean_dist"]
    print("Chamfer from init", chamfer_from_init)
    tmp_coords = trainer.refiner.contour
    tmp_coords = tmp_coords.detach().cpu()
    template_params = trainer.refiner.template_model.get_params(coordinates=tmp_coords)
    (
        template_params["peak_dist"].mean(),
        template_params["peak_dist"].std(),
        template_params["sigma1"].mean(),
        template_params["sigma1"].std(),
    )

    mean_peak_dist = template_params["peak_dist"].detach().numpy().mean()
    mean_sigma1 = template_params["sigma1"].detach().numpy().mean()
    mean_sigma2 = template_params["sigma2"].detach().numpy().mean()

    print(
        f"Mean Peak Dist: {mean_peak_dist:.2f}, Mean Sigma1: {mean_sigma1:.2f}, Mean Sigma2: {mean_sigma2:.2f}"
    )
    plt.figure()
    plt.plot(template_params["peak_dist"].detach().numpy())
    plt.plot(template_params["sigma1"].detach().numpy())
    plt.axhline(
        y=mean_peak_dist,
        color="b",
        linestyle="dashed",
        label=f"Mean Peak Dist: {mean_peak_dist:.2f}",
    )
    plt.axhline(
        y=mean_sigma1, color="g", linestyle="dashed", label=f"Mean Sigma1: {mean_sigma1:.2f}"
    )
    plt.legend()
    plt.show()

    trainer.refiner.visualize_rbf_field()
    plt.show()

    tmp_profiles, *_ = trainer.refiner.sample_image_features(image_t)
    tmp_profiles = tmp_profiles.detach()
    trainer.refiner.visualize_profile_statistics(image_t.cpu())
    plt.show()
    trainer.refiner.visualize_contour(image_t.detach())
    vis.plot_contour_normals(image_t, tmp_coords.detach().numpy())
    plt.show()
