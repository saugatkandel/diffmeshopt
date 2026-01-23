import json
import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from diffmeshopt.opt2d.evaluation import (
    compute_contour_metrics,
    compute_gt_distance_map,
    compute_metrics_from_map,
)
from diffmeshopt.opt2d.optimize import ContourRefinerBase


class OptimizationTrainer:
    """
    Manages the optimization loop, logging, and checkpointing for ContourRefiner.
    """

    def __init__(
        self,
        refiner: ContourRefinerBase,
        output_dir: str | Path,
        gt_contour: np.ndarray | None = None,
        image: np.ndarray | None = None,
        num_iterations: int = 1000,
        save_interval: int = 100,
        log_interval: int = 10,
        log_image_interval: int = 200,
        use_tensorboard: bool = False,
    ):
        self.refiner = refiner
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gt_contour = gt_contour
        self.image = image
        self.num_iterations = num_iterations
        self.save_interval = save_interval
        self.log_interval = log_interval
        self.log_image_interval = log_image_interval
        self.use_tensorboard = use_tensorboard
        self.writer = None
        self.metric_history = []

        if self.use_tensorboard:
            if SummaryWriter is None:
                logging.warning(
                    "torch.utils.tensorboard not found. Disabling tensorboard logging."
                )
                self.use_tensorboard = False
            else:
                self.writer = SummaryWriter(log_dir=str(self.output_dir / "logs"))

    def fit(self):
        """Runs the optimization loop."""
        self.refiner.train()  # Ensure module is in training mode

        gt_contour_tensor = None
        if self.gt_contour is not None:
            device = next(self.refiner.parameters()).device
            gt_contour_tensor = torch.from_numpy(self.gt_contour).float().to(device)

        # Precompute distance map for fast evaluation
        gt_distance_map = None
        if self.gt_contour is not None and self.image is not None:
            gt_distance_map = compute_gt_distance_map(self.gt_contour, self.image.shape)
        if gt_distance_map is not None:
            gt_distance_map = gt_distance_map.to(next(self.refiner.parameters()).device)

        iterator = tqdm(range(1, self.num_iterations + 1), desc="Optimizing")

        for step in iterator:
            # Perform one optimization step
            metrics = self.refiner.step()

            # Evaluation metrics
            if step % self.log_interval == 0:
                if gt_distance_map is not None:
                    # Fast path: O(N) lookup
                    eval_metrics = compute_metrics_from_map(
                        self.refiner.contour.detach(), gt_distance_map
                    )
                elif gt_contour_tensor is not None:
                    # Slow path: O(N*M) pairwise distance
                    eval_metrics = compute_contour_metrics(
                        self.refiner.contour.detach(), gt_contour_tensor
                    )
                else:
                    eval_metrics = {}
                metrics.update(eval_metrics)

            # Logging
            if step % self.log_interval == 0:
                # Format float values for progress bar
                postfix = {
                    k: f"{v:.4f}" if isinstance(v, float) else v
                    for k, v in metrics.items()
                    if k in ["total_loss", "mean_dist", "p95_dist"]
                }
                iterator.set_postfix(postfix)
                if self.use_tensorboard:
                    for k, v in metrics.items():
                        if isinstance(v, torch.Tensor):
                            v = v.item()
                        if isinstance(v, (int, float, np.number)):
                            self.writer.add_scalar(k, v, step)

                # Save metrics to JSON history
                log_entry = {"step": step}
                for k, v in metrics.items():
                    if hasattr(v, "item"):
                        log_entry[k] = v.item()
                    else:
                        log_entry[k] = v
                self.metric_history.append(log_entry)
                with open(self.output_dir / "metrics.json", "w") as f:
                    json.dump(self.metric_history, f, indent=2)

            # Image Logging
            if self.image is not None and step % self.log_image_interval == 0:
                self.log_optimization_image(step)

            # Checkpointing
            if step % self.save_interval == 0:
                self.save_checkpoint(step)

        # Final save
        self.save_checkpoint(self.num_iterations, final=True)

        if self.use_tensorboard:
            self.writer.close()

    def log_optimization_image(self, step: int):
        """Logs the current contour overlaid on the image."""
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(self.image, cmap="gray")

        # Plot current contour
        contour = self.refiner.contour.detach().cpu().numpy()
        # Assuming contour is (N, 2) -> (y, x) for plotting (x, y)
        ax.plot(contour[:, 1], contour[:, 0], "r-", linewidth=2, label="Current")

        if self.gt_contour is not None:
            ax.plot(self.gt_contour[:, 1], self.gt_contour[:, 0], "g--", linewidth=1, label="GT")

        ax.legend()
        ax.set_title(f"Step {step}")

        if self.use_tensorboard:
            self.writer.add_figure("optimization_vis", fig, global_step=step)

        if not self.use_tensorboard:
            vis_dir = self.output_dir / "vis"
            vis_dir.mkdir(exist_ok=True)
            fig.savefig(vis_dir / f"step_{step:05d}.png")

        plt.close(fig)

    def save_checkpoint(self, step: int, final: bool = False):
        """Saves the current state of the refiner."""
        state = self.refiner.export_state()
        filename = "final.pkl" if final else f"checkpoint_{step:05d}.pkl"
        save_path = self.output_dir / filename

        joblib.dump(state, save_path)
        logging.info(f"Saved checkpoint to {save_path}")
