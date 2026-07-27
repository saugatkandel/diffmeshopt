"""
Hyperparameter search for 2D contour refinement.

This script performs a hierarchical search over refiner types, template models,
and regularization parameters using Optuna. It evaluates robustness by
progressively testing on unperturbed data, then small perturbations, then large perturbations.
"""

import json
import logging
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import click
import joblib
import numpy as np
import optuna
import polars as pl
import torch

from diffmeshopt.opt2d.config import (
    AdaptiveRegularizationProps,
    BSplineContourRefinerProps,
    BSplineTemplateProps,
    ContourRefinerProps,
    RBFContourRefinerProps,
    RegularizerConfig,
    RegularizerDefaults,
    RegularizerType,
    TemplateProps,
)
from diffmeshopt.opt2d.evaluation import compute_contour_metrics
from diffmeshopt.opt2d.refiner import (
    BSplineContourRefiner,
    ContourRefiner,
    RBFContourRefiner,
)
from diffmeshopt.opt2d.template import TemplateModelFactory, TemplateType
from diffmeshopt.opt2d.trainer import OptimizationTrainer, TrainerConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Centralized defaults for fixed parameters in the search
SEARCH_DEFAULTS = {
    "optimization": {
        "num_steps": 2000,
        "learning_rate": 0.05,
    },
    "template": {
        "sigma": 0.75,
        "peak_dist": 4.0,
        "sigma_ratio": 1.0,
        "amp_ratio": 1.0,
        "min_peak_ratio": 2.0,
    },
    "adaptive": {
        "update_interval": 10,
        "warmup_steps": 50,
    },
    "shape_loss_weight": 1.0,
}


class ExperimentRunner:
    """
    Manages the execution of hyperparameter search experiments for 2D contour refinement.

    This class handles dataset loading, curriculum definition, model construction,
    and the execution of individual optimization trials.

    Attributes:
        dataset_path (Path): Path to the dataset file.
        output_dir (Path): Directory where results and artifacts are saved.
        device (str): The compute device to use ('cpu' or 'cuda').
    """

    def __init__(self, dataset_path: str, output_dir: str, device: str = "cpu"):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        # Setup file logging
        file_handler = logging.FileHandler(self.output_dir / "search.log")
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logging.getLogger().addHandler(file_handler)

        # Load dataset
        logger.info(f"Loading dataset from {self.dataset_path}")
        self.dataset = joblib.load(self.dataset_path)

        # Define curriculum stages
        self.stages = [
            ["original"],
            ["shrink_5_perturb_3", "expand_5_perturb_3"],
            ["shrink_10_perturb_5", "expand_10_perturb_5"],
        ]

    def _build_props(self, params: dict) -> tuple[ContourRefinerProps, TemplateProps, str]:
        """
        Construct property objects from flat search parameters.

        Args:
            params (dict): Dictionary of hyperparameters sampled by Optuna.

        Returns:
            tuple: A tuple containing (ContourRefinerProps, TemplateProps, template_mode_string).
        """
        # Create a copy of the dictionary to track usage
        params_tracker = params.copy()

        def use(key, default=None):
            """Pop key from p if present, else retrieve from original params."""
            if key in params_tracker:
                return params_tracker.pop(key)

            return params.get(key, default)

        # Mark metadata as used
        use("trial_id")

        # 1. Regularization Configuration
        reg_defaults = RegularizerDefaults.get_defaults()

        # If adaptive, we set target ratios in defaults.
        # If static, we set static weights in initial_regularization_weights.
        initial_weights = {}

        # Map search params to RegularizerTypes
        reg_map = {
            "laplacian": [RegularizerType.CONTOUR_LAPLACIAN],
            "edge": [RegularizerType.EDGE_LENGTH],
            "normal": [RegularizerType.NORMAL_CONSISTENCY],
            "tangential": [RegularizerType.TANGENTIAL_LAPLACIAN],
            # "shape": [RegularizerType.TEMPLATE_SHAPE],
            "anchor": [
                RegularizerType.ANCHOR_SIGMA_RATIO,
                RegularizerType.ANCHOR_AMP_RATIO,
            ],
            "contour_anchor": [RegularizerType.CONTOUR_ANCHOR],
            # We use a single smoothing weight for all template parameters.
            # Justification: Regularization losses are computed in log-space (relative changes).
            # A 10% change in sigma incurs the same penalty as a 10% change in peak_dist,
            # making the single weight scale-invariant and applicable across different parameter types.
            "smooth_param": [
                RegularizerType.SMOOTH_SIGMA,
                RegularizerType.SMOOTH_PEAK_DIST,
                RegularizerType.SMOOTH_SIGMA_RATIO,
                RegularizerType.SMOOTH_AMP_RATIO,
            ],
        }

        reg_mode = use("reg_mode")

        for name, reg_types in reg_map.items():
            for reg_type in reg_types:
                if reg_mode == "adaptive":
                    # Update default config with searched ratio
                    ratio = use(f"ratio_{name}")
                    ratio = 0.1 if ratio is None else ratio
                    print(f"Setting adaptive ratio for {reg_type.value}: {ratio}")
                    # Keep static weight as 0.0 or 1.0 (doesn't matter much if adaptive is on,
                    # but good to have a starting point)
                    reg_defaults.regularizers[reg_type] = RegularizerConfig(
                        static_weight=1.0, target_ratio=ratio
                    )
                else:
                    # Static mode: set initial weight directly
                    weight = use(f"w_{name}")
                    initial_weights[reg_type.value] = weight
                    # Disable adaptation for this reg
                    reg_defaults.regularizers[reg_type] = RegularizerConfig(
                        static_weight=weight,
                        target_ratio=0.0,  # 0.0 effectively disables if logic checks > 0
                    )

        # Adaptive Config
        adaptive_props = AdaptiveRegularizationProps(
            enabled=(reg_mode == "adaptive"),
            update_interval=SEARCH_DEFAULTS["adaptive"]["update_interval"],
            warmup_steps=SEARCH_DEFAULTS["adaptive"]["warmup_steps"],
        )

        # 2. Refiner Props
        common_args = {
            "num_steps": use("num_steps", SEARCH_DEFAULTS["optimization"]["num_steps"]),
            "learning_rate": use(
                "learning_rate", SEARCH_DEFAULTS["optimization"]["learning_rate"]
            ),
            "initial_regularization_weights": initial_weights,
            "adaptive_reg": adaptive_props,
            "_reg_defaults": reg_defaults,
            "shape_loss_weight": use("shape_loss_weight", SEARCH_DEFAULTS["shape_loss_weight"]),
        }

        refiner_mode = use("refiner")

        if refiner_mode == "vertex":
            refiner_props = ContourRefinerProps(**common_args)
        elif refiner_mode == "bspline":
            refiner_props = BSplineContourRefinerProps(
                contour_num_control_points=use("num_cp"), **common_args
            )
        elif refiner_mode == "rbf":
            refiner_props = RBFContourRefinerProps(
                rbf_num_control_points=use("num_cp"),
                rbf_kernel_sigma=use("rbf_sigma"),
                **common_args,
            )
        else:
            raise ValueError(f"Unknown refiner: {refiner_mode}")

        # 3. Template Props
        t_mode = use("template")
        t_sym = use("symmetric")

        # Always consume smoothness_window_size if present
        smooth_win = use("smoothness_window_size", 3)

        # min peak ratio
        min_peak_ratio = use("min_peak_ratio", SEARCH_DEFAULTS["template"]["min_peak_ratio"])

        t_args = {
            "symmetric": t_sym,
            "sigma": SEARCH_DEFAULTS["template"]["sigma"],
            "peak_dist": SEARCH_DEFAULTS["template"]["peak_dist"],
            "sigma_ratio": SEARCH_DEFAULTS["template"]["sigma_ratio"],
            "amp_ratio": SEARCH_DEFAULTS["template"]["amp_ratio"],
            "min_peak_ratio": min_peak_ratio,
            "smoothness_window_size": smooth_win,
        }

        if t_mode == "global":
            template_props = TemplateProps(**t_args)
        elif t_mode == "per_point":
            template_props = TemplateProps(**t_args)
        elif t_mode == "bspline":
            template_props = BSplineTemplateProps(
                bspline_num_control_points=max(4, use("num_cp") // 4),  # Heuristic
                **t_args,
            )
        elif t_mode == "neural":
            template_props = TemplateProps(**t_args)
        elif t_mode == "fixed":
            template_props = TemplateProps(**t_args)
        else:
            raise ValueError(f"Unknown template: {t_mode}")

        # Log usage
        unused_keys = sorted(params_tracker.keys())
        used_keys = sorted(set(params.keys()) - set(params_tracker.keys()))

        logger.info(f"Used params: {used_keys}")
        logger.info(f"Unused params: {unused_keys}")

        return refiner_props, template_props, t_mode

    def run_single_case(self, case_name: str, params: dict) -> float:
        """
        Run optimization for a single data case.

        Args:
            case_name (str): The key of the case in the dataset.
            params (dict): The hyperparameters for this run.

        Returns:
            float: The final Chamfer Distance metric. Returns float('inf') if optimization fails.
        """
        logger.info(f"Running case: {case_name}")
        logger.info(f"Parameters: {json.dumps(params, default=str)}")

        data = self.dataset[case_name]
        image = data["image"]
        init_contour = data["contour"]

        # Use unperturbed contour as ground truth
        gt_contour = None
        if "original" in self.dataset:
            gt_contour = self.dataset["original"]["contour"]

        # Prepare Tensors
        image_t = torch.from_numpy(image).float().to(self.device)
        contour_t = torch.from_numpy(init_contour).float().to(self.device)

        # Build Model
        refiner_props, template_props, t_mode = self._build_props(params)

        # Template
        template_model = TemplateModelFactory.create(
            t_mode,
            template_props,
            num_vertices=len(init_contour),
            image_shape=image.shape,
        ).to(self.device)

        # Refiner
        if params["refiner"] == "vertex":
            refiner = ContourRefiner(contour_t, refiner_props, template_model)
        elif params["refiner"] == "bspline":
            refiner = BSplineContourRefiner(contour_t, refiner_props, template_model)
        elif params["refiner"] == "rbf":
            refiner = RBFContourRefiner(contour_t, refiner_props, template_model)

        refiner.to(self.device)

        # Trainer (Minimal configuration for speed)
        # We use a temporary directory for this specific run
        run_dir = self.output_dir / "trials" / f"trial_{params['trial_id']}_{case_name}"

        config = TrainerConfig(
            output_dir=run_dir,
            image=image,
            gt_contour=gt_contour,
            max_steps=SEARCH_DEFAULTS["optimization"]["num_steps"],
            save_interval=SEARCH_DEFAULTS["optimization"]["num_steps"],  # Only save at end
            log_interval=500,
            use_tensorboard=False,  # Disable TB to save IO
            save_images=False,
            enable_progress_bar=False,  # Disable progress bar for parallel runs
        )

        trainer = OptimizationTrainer(refiner, config, run_dir=run_dir)

        # Run
        try:
            trainer.fit(steps_to_run=SEARCH_DEFAULTS["optimization"]["num_steps"])
        except Exception as e:
            logger.error(f"Training crashed for {case_name}: {e}")
            return float("inf")

        # Evaluate
        final_contour = refiner.contour.detach().cpu().numpy()

        # Check for NaNs or explosion
        if np.isnan(final_contour).any() or np.max(np.abs(final_contour)) > max(image.shape) * 2:
            logger.error("Nans or explotion in final contour.")
            return float("inf")

        # Compute Metric (Chamfer Distance to GT)
        # We assume GT is available in the dataset. If not, we can't compute this.
        if gt_contour is None or len(gt_contour) == 0:
            logger.error(
                f"GT contour for case '{case_name}' is missing (checked 'original' case)."
            )
            return float("inf")

        gt_contour_t = torch.from_numpy(gt_contour).float().to(self.device)
        metrics = compute_contour_metrics(refiner.contour, gt_contour_t)
        if not metrics:
            logger.error("Metrics does not exist.")
            return float("inf")
        chamfer = metrics["mean_dist"]

        logger.info(
            f"Trial {params['trial_id']} | Case: {case_name} | "
            f"Chamfer: {chamfer:.4f} | Status: Success"
        )
        # ------------------------------
        return chamfer

    def verify_run(self) -> bool:
        """
        Performs a sample run with default parameters to verify correctness.

        This method picks the first available case and runs the optimization loop
        with a standard set of parameters. It is useful for debugging and ensuring
        the environment is set up correctly before launching a full search.

        Returns:
            bool: True if the run completes and produces a valid score, False otherwise.
        """
        logger.info("Starting verification run...")

        # Pick a case (try to find one from the first stage)
        case_name = None
        for stage in self.stages:
            for name in stage:
                if name in self.dataset:
                    case_name = name
                    break
            if case_name:
                break

        if not case_name:
            logger.error("No valid cases found in dataset for verification.")
            return False

        logger.info(f"Verifying on case: {case_name}")

        # Define default parameters

        default_params = {
            "trial_id": "verify",
            "refiner": "vertex",
            "template": "per_point",
            "symmetric": True,
            "reg_mode": "static",
            "num_cp": 50,
            "rbf_sigma": 1.0,
            "w_laplacian": 0.0,
            "w_edge": 0.0,
            "w_normal": 50.0,
            "w_tangential": 5.0,
            "w_shape": 1.0,
            "w_anchor": 0.0,
            "w_smooth_param": 50.0,
            "min_peak_ratio": 2.0,
        }

        try:
            score = self.run_single_case(case_name, default_params)
            logger.info(f"Verification run finished. Score: {score}")
            return not np.isinf(score)
        except Exception:
            logger.error("Verification run failed with exception.")
            traceback.print_exc()
            return False

    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function.

        Samples hyperparameters and evaluates them on the curriculum of cases.
        Implements pruning to stop unpromising trials early.

        Args:
            trial (optuna.Trial): The current Optuna trial object.

        Returns:
            float: The objective value (mean Chamfer Distance) to minimize.
        """

        # 1. Sample Hyperparameters
        params = {
            "trial_id": trial.number,
            "refiner": trial.suggest_categorical("refiner", ["vertex", "bspline", "rbf"]),
            "template": trial.suggest_categorical(
                "template", ["global", "bspline", "neural", "per_point"]
            ),
            "symmetric": trial.suggest_categorical("symmetric", [True, False]),
            "reg_mode": trial.suggest_categorical("reg_mode", ["static", "adaptive"]),
            # Model Specifics
            "num_cp": trial.suggest_int("num_cp", 20, 150),
            "rbf_sigma": trial.suggest_float("rbf_sigma", 0.5, 10.0),
            "smoothness_window_size": trial.suggest_categorical(
                "smoothness_window_size", [1, 3, 5]
            ),
        }

        # Weights / Ratios
        reg_names = [
            "laplacian",
            "edge",
            "normal",
            "tangential",
            "shape",
            "anchor",
            "smooth_param",
        ]
        for name in reg_names:
            if params["reg_mode"] == "static":
                if trial.suggest_categorical(f"use_w_{name}", [True, False]):
                    params[f"w_{name}"] = trial.suggest_float(f"w_{name}", 1e-4, 10.0, log=True)
                else:
                    params[f"w_{name}"] = 0.0
            else:
                if trial.suggest_categorical(f"use_ratio_{name}", [True, False]):
                    params[f"ratio_{name}"] = trial.suggest_float(
                        f"ratio_{name}", 1e-4, 1.0, log=True
                    )
                else:
                    params[f"ratio_{name}"] = 0.0

        # 2. Hierarchical Execution
        total_score = 0.0

        # Thresholds for pruning (Chamfer distance in pixels)
        # If error is > 5 pixels after refinement, it's a failure
        FAILURE_THRESHOLD = 5.0

        for stage_idx, stage_cases in enumerate(self.stages):
            stage_scores = []
            for case_name in stage_cases:
                if case_name not in self.dataset:
                    logger.warning(f"Case {case_name} not found in dataset. Skipping.")
                    continue

                score = self.run_single_case(case_name, params)

                # Check for failure
                if score > FAILURE_THRESHOLD or np.isinf(score):
                    logger.info(
                        f"Trial {trial.number} failed at stage {stage_idx} ({case_name}). Score: {score}"
                    )
                    # Prune
                    raise optuna.TrialPruned(f"Failed at stage {stage_idx} with score {score}")

                stage_scores.append(score)

            if not stage_scores:
                continue

            avg_stage_score = np.mean(stage_scores)
            total_score += avg_stage_score

            # Report intermediate result to Optuna
            trial.report(total_score, step=stage_idx)

            # Standard Optuna pruning (based on intermediate values compared to other trials)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return total_score


@click.command()
@click.option("--dataset", required=True, help="Path to perturbed dataset .pkl")
@click.option("--output", required=True, help="Output directory for study")
@click.option("--n-trials", default=500, help="Number of trials")
@click.option("--n-jobs", default=10, help="Number of parallel jobs")
@click.option("--device", default="cuda", help="Device (cpu/cuda)")
@click.option("--verify", is_flag=True, help="Run a sample verification case and exit.")
def main(dataset, output, n_trials, n_jobs, device, verify):
    """Hierarchical Hyperparameter Search"""
    runner = ExperimentRunner(dataset, output, device)

    if verify:
        if runner.verify_run():
            logger.info("Verification successful.")
            sys.exit(0)
        else:
            logger.error("Verification failed.")
            sys.exit(1)

    # Define storage (SQLite for persistence and parallel coordination)
    storage_url = f"sqlite:///{output}/study.db"

    # Sampler: TPESampler is Bayesian-like and handles conditional parameters well
    sampler = optuna.samplers.TPESampler(multivariate=True)

    study = optuna.create_study(
        study_name="contour_refinement_search",
        storage=storage_url,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
    )

    if n_jobs > 1:
        logger.info(f"Launching {n_jobs} parallel workers for {n_trials} total trials...")
        processes = []
        trials_per_job = n_trials // n_jobs
        remainder = n_trials % n_jobs

        for i in range(n_jobs):
            t = trials_per_job + (1 if i < remainder else 0)
            if t == 0:
                continue

            cmd = [
                sys.executable,
                "-m",
                "diffmeshopt.opt2d.hyperparameter_search",
                "--dataset",
                str(dataset),
                "--output",
                str(output),
                "--n-trials",
                str(t),
                "--n-jobs",
                "1",
                "--device",
                device,
            ]
            processes.append(subprocess.Popen(cmd))

        try:
            for p in processes:
                p.wait()
        except KeyboardInterrupt:
            logger.info("Interrupted. Terminating workers...")
            for p in processes:
                p.terminate()
    else:
        logger.info("Starting optimization...")
        try:
            study.optimize(runner.objective, n_trials=n_trials, n_jobs=1)
        except KeyboardInterrupt:
            logger.info("Optimization interrupted.")

    # Save results
    logger.info("Saving results...")
    trials_df = study.trials_dataframe()
    pl_df = pl.from_pandas(trials_df)
    pl_df.write_parquet(Path(output) / "results.parquet")

    # Save metadata
    metadata = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": len(study.trials),
        "dataset_path": str(dataset),
        "device": device,
        "study_name": study.study_name,
    }
    with open(Path(output) / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("Best params:")
    print(study.best_params)
    print(f"Best value: {study.best_value}")


if __name__ == "__main__":
    main()
