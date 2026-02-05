import dataclasses as dt
import logging
import os
import sys
from pathlib import Path
from typing import Any, Self

import joblib
import lightning.pytorch as pl_lightning
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import Logger, TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_only
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffmeshopt.opt2d.config import RegularizerType

torch.set_float32_matmul_precision("medium")
# --- 1. DATA LAYER: Atomic Parquet Logger ---


@dt.dataclass(frozen=True)
class TrainerConfig:
    """Strictly typed configuration for the Optimization session."""

    output_dir: str | Path
    image: np.ndarray
    gt_contour: np.ndarray | None = None
    max_steps: int = -1
    min_steps: int | None = None
    max_epochs: int = -1
    min_epochs: int | None = None
    save_interval: int = 50
    log_interval: int = 10
    log_image_interval: int = 200
    use_tensorboard: bool = True
    calc_chamfer: bool = True
    calc_hausdorff: bool = False
    calc_p95: bool = False
    save_images: bool = True
    enable_progress_bar: bool = True


class AtomicParquetLogger(Logger):
    def __init__(self, save_dir: Path, name: str = "metrics", flush_interval: int = 50):
        super().__init__()
        self._name = name
        self._save_dir = Path(save_dir)

        self._flush_interval = flush_interval
        self.history: list[dict[str, Any]] = self._load_existing()

    @property
    def log_file(self) -> Path:
        return Path(self.log_dir) / "metrics.parquet"

    @property
    def tmp_file(self) -> Path:
        return self.log_file.with_suffix(".parquet.tmp")

    def _load_existing(self) -> list[dict[str, Any]]:
        if self.log_file.exists():
            try:
                return pl.read_parquet(self.log_file).to_dicts()
            except Exception as e:
                logging.warning(f"Logger: Could not reload history: {e}")
        return []

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return "1.0"

    @property
    def log_dir(self) -> str:
        """Lightning internally expects log_dir to return a string path."""

        return str(self._save_dir / self._name)

    @log_dir.setter
    def log_dir(self, value: str):
        """FIX: Adding a setter prevents 'AttributeError: can't set attribute'."""
        self._save_dir = Path(value).parent
        self._name = Path(value).name

    @rank_zero_only
    def log_metrics(self, metrics: dict[str, Any], step: int | None):
        if step is None:
            return
        self.history.append({"step": step, **metrics})
        if len(self.history) % self._flush_interval == 0:
            self.save()

    @rank_zero_only
    def log_hyperparams(self, params: dict[str, Any]) -> None:
        """Saves hyperparameters to a separate file to keep metrics.parquet clean."""
        hp_path = self.log_dir / "hyperparams.parquet"
        try:
            pl.DataFrame([params]).write_parquet(hp_path)
        except Exception as e:
            logging.error(f"Failed to log hyperparameters: {e}")

    @rank_zero_only
    def save(self) -> None:
        if not self.history:
            return
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        # Atomic Write Pattern
        pl.DataFrame(self.history).write_parquet(self.tmp_file)
        os.replace(self.tmp_file, self.log_file)

    @rank_zero_only
    def finalize(self, status: str) -> None:
        self.save()


# --- 2. LOGIC LAYER: Lightning Module ---


class ContourLightningModule(pl_lightning.LightningModule):
    def __init__(
        self,
        refiner: nn.Module,
        image: np.ndarray,
        gt_contour: np.ndarray | None = None,
        image_shape: tuple[int, int] | None = None,
        log_interval: int = 10,
        calc_chamfer: bool = True,
        calc_hausdorff: bool = False,
        calc_p95: bool = False,
    ):
        super().__init__()
        self.refiner = refiner
        self.log_interval = log_interval
        self.calc_chamfer = calc_chamfer
        self.calc_hausdorff = calc_hausdorff
        self.calc_p95 = calc_p95

        # 1. Register main image
        self.register_buffer("image_ref", torch.from_numpy(image).float(), persistent=False)

        # 2. Register GT Buffers as empty placeholders first (Fixes KeyError)
        # persistent=True: Small contour data saved in checkpoint
        # persistent=False: Large distance map stays in VRAM only
        self.register_buffer("gt_contour_ref", torch.empty(0), persistent=True)
        self.register_buffer("gt_distance_map", torch.empty(0), persistent=False)

        # 3. Populate buffers only if data is provided
        if gt_contour is not None and len(gt_contour) > 0:
            # Updating existing buffer content is safe
            self.gt_contour_ref = torch.from_numpy(gt_contour).float()

            if image_shape:
                from diffmeshopt.opt2d.evaluation import compute_gt_distance_map

                dist_map = compute_gt_distance_map(gt_contour, image_shape)
                if dist_map is not None:
                    self.gt_distance_map = dist_map

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        # Sync step counter for adaptive logic
        self.refiner.step_counter = self.global_step

        # Use unified forward pass logic (handles sampling, adaptive weights, and loss)
        losses = self.refiner.forward_pass(self.image_ref)

        for k, v in losses.items():
            self.log(k, v, prog_bar=(k == "total_loss"), on_step=True, on_epoch=False)

        if self.global_step % self.log_interval == 0:
            self._eval_metrics()

        # Log adaptive weights if enabled
        if (
            hasattr(self.refiner, "adaptive_config")
            and self.refiner.adaptive_config is not None
            and self.refiner.adaptive_config.enabled
        ):
            self._log_adaptive_weights()

        return losses["total_loss"]

    def _log_adaptive_weights(self):
        """Log current regularization weights for monitoring."""
        # Iterate over all regularizers defined in the system
        for reg_type in RegularizerType:
            # Use get_weight() which handles the buffer naming internally
            if hasattr(self.refiner.loss_fn, "get_weight"):
                weight_val = self.refiner.loss_fn.get_weight(reg_type).item()
                self.log(f"weight/{reg_type.value}", weight_val, on_step=True, on_epoch=False)

    def _eval_metrics(self) -> None:
        from diffmeshopt.opt2d.evaluation import compute_contour_metrics, compute_metrics_from_map

        with torch.no_grad():
            curr = self.refiner.contour

            # Use .numel() to check if the buffer is populated (size > 0)
            if self.gt_distance_map.numel() > 0:
                metrics = compute_metrics_from_map(
                    curr,
                    self.gt_distance_map,
                    calc_chamfer=self.calc_chamfer,
                    calc_hausdorff=self.calc_hausdorff,
                    calc_p95=self.calc_p95,
                )
            elif self.gt_contour_ref.numel() > 0:
                metrics = compute_contour_metrics(curr, self.gt_contour_ref)
            else:
                metrics = {}

            if metrics:
                self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=False)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.refiner.create_optimizer()


# --- 3. CALLBACKS: Visualization ---


class ImageLoggerCallback(Callback):
    def __init__(
        self,
        image: np.ndarray,
        gt_contour: np.ndarray | None,
        log_interval: int,
        output_dir: Path,
        save_images: bool = True,
    ):
        super().__init__()
        self.image = image
        self.gt_contour = gt_contour
        self.log_interval = log_interval
        self.vis_dir = output_dir / "vis"
        self.save_images = save_images
        if self.save_images:
            self.vis_dir.mkdir(parents=True, exist_ok=True)
        self.num_crops = 16  # Grid of 4x4
        self.crop_size = 60  # Size of each crop in pixels
        self.crop_indices = None  # Indices of vertices to center crops on
        self.init_contour = None

    def on_train_start(self, trainer, pl_module):
        import random

        with torch.no_grad():
            self.init_contour = pl_module.refiner.contour.detach().cpu().numpy()

        # Select a fixed set of vertices to center the crops on for the entire run
        if len(self.init_contour) > 0:
            indices = list(range(len(self.init_contour)))
            if len(indices) >= self.num_crops:
                self.crop_indices = random.sample(indices, self.num_crops)
            else:
                # If contour is small, just repeat random points
                self.crop_indices = [random.choice(indices) for _ in range(self.num_crops)]
        else:
            self.crop_indices = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.log_interval == 0:
            self._plot_and_save(pl_module, trainer.global_step, trainer)

    def create_figure(self, pl_module):
        """Creates the default grid of cropped contour visualizations."""
        from diffmeshopt.opt2d import vis

        if self.init_contour is None:
            self.on_train_start(None, pl_module)

        if not self.crop_indices:
            return None

        # --- 1. Get current contour and template parameters ---
        with torch.no_grad():
            contour_tensor = pl_module.refiner.contour
            curr_contour = contour_tensor.detach().cpu().numpy()

            # Get full template params to draw peak/boundary lines
            params = pl_module.refiner.template_model.get_params(coordinates=contour_tensor)
            params_np = {
                k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
                for k, v in params.items()
            }

        return vis.plot_contour_crops(
            image=self.image,
            contour=curr_contour,
            crop_indices=self.crop_indices,
            init_contour=self.init_contour,
            gt_contour=self.gt_contour,
            crop_size=self.crop_size,
            template_params=params_np,
            num_cols=4,
        )

    def _plot_and_save(self, pl_module, step, trainer):
        fig = self.create_figure(pl_module)
        if fig is None:
            return
        try:
            for logger in trainer.loggers:
                if isinstance(logger, TensorBoardLogger):
                    logger.experiment.add_figure("vis/grid_view", fig, global_step=step)

            if self.save_images:
                fig.savefig(self.vis_dir / f"step_{step:05d}.png", bbox_inches="tight")
        finally:
            plt.close(fig)


class StepWindowCallback(Callback):
    """
    Dynamically sets max_steps for the current run based on requested steps.
    Placed first in callback list to ensure ProgressBars see the updated max_steps.
    """

    def __init__(self, steps_to_run: int):
        self.steps_to_run = steps_to_run
        self.offsest = 0

    def on_train_start(self, trainer, pl_module):
        self.offset = trainer.global_step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Stop if we have covered the requested distance
        if (trainer.global_step - self.offset) >= self.steps_to_run:
            trainer.should_stop = True


class LiteTQDM(TQDMProgressBar):
    """A text-only TQDM bar that avoids Jupyter widget deadlocks."""

    def __init__(self, refresh_rate: int = 10) -> None:
        super().__init__(refresh_rate=refresh_rate, process_position=0)

    def init_train_tqdm(self) -> tqdm:
        """Initializes the standard text-based tqdm bar."""
        # Safety check for the starting index
        start_idx = self.trainer.global_step if self.trainer else 0

        return tqdm(
            desc=self.train_description,
            initial=start_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=False,
            ncols=80,
            ascii=True,
            gui=False,
            file=sys.stdout,
        )


# --- 4. ORCHESTRATION: Optimization Trainer ---


class OptimizationTrainer:
    def __init__(
        self,
        refiner: nn.Module,
        config: TrainerConfig,
        run_dir: str | Path | None = None,
    ):
        self.config = config
        if run_dir is not None:
            self.output_dir = Path(run_dir)
        else:
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.output_dir = Path(config.output_dir) / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Output directory set to: {self.output_dir}")

        self.refiner = refiner

        self.model = ContourLightningModule(
            refiner,
            self.config.image,
            self.config.gt_contour,
            self.config.image.shape if self.config.image is not None else None,
            log_interval=self.config.log_interval,
            calc_chamfer=self.config.calc_chamfer,
            calc_hausdorff=self.config.calc_hausdorff,
            calc_p95=self.config.calc_p95,
        )

        self.trainer = pl_lightning.Trainer(
            default_root_dir=self.output_dir,
            max_steps=self.config.max_steps,
            max_epochs=self.config.max_epochs,
            min_steps=self.config.min_steps,
            min_epochs=self.config.min_epochs,
            callbacks=self._setup_callbacks(),
            logger=self._setup_loggers(),
            log_every_n_steps=self.config.log_interval,
            accelerator="auto",
            devices=1,
            enable_model_summary=False,
            enable_progress_bar=self.config.enable_progress_bar,
        )

    def _setup_callbacks(self) -> list[Callback]:
        cbs = [
            ModelCheckpoint(
                dirpath=self.output_dir,
                save_last=True,
                every_n_train_steps=self.config.save_interval,
                filename="ckpt_{step:05d}",
                auto_insert_metric_name=False,
            ),
        ]
        if self.config.enable_progress_bar:
            cbs.insert(0, LiteTQDM(refresh_rate=10))

        if self.config.image is not None:
            cbs.append(
                ImageLoggerCallback(
                    self.config.image,
                    self.config.gt_contour,
                    self.config.log_image_interval,
                    self.output_dir,
                    save_images=self.config.save_images,
                )
            )
        return cbs

    def _setup_loggers(self) -> list[Logger]:
        loggers: list[Logger] = [AtomicParquetLogger(self.output_dir)]
        if self.config.use_tensorboard:
            loggers.append(TensorBoardLogger(self.output_dir, name="tb", version=0))
        return loggers

    def fit(self, steps_to_run: int = 100):
        """
        Runs exactly N steps. Automatically resumes if a checkpoint exists.
        """
        logging.info(f"Starting fit for {steps_to_run} steps.")
        ckpt_path = self.output_dir / "last.ckpt"
        resume_path = str(ckpt_path) if ckpt_path.exists() else None

        # 1. Setup the window limiter
        limiter = StepWindowCallback(steps_to_run)
        # Insert at 0 to ensure it runs before RichProgressBar (which reads max_steps)
        self.trainer.callbacks.insert(0, limiter)
        self.trainer.should_stop = False

        # 2. Infinite data stream
        loader = DataLoader(range(10_000_000), batch_size=None, num_workers=0)

        try:
            self.trainer.fit(self.model, train_dataloaders=loader, ckpt_path=resume_path)
        except KeyboardInterrupt:
            logging.info("Paused.")
        finally:
            # Clean up
            if limiter in self.trainer.callbacks:
                self.trainer.callbacks.remove(limiter)
            self._finalize_run()

    def _finalize_run(self) -> None:
        """Exports the final refined state as a CPU/Numpy friendly dictionary."""
        state = self.model.refiner.export_state()
        clean_state = {
            k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
            for k, v in state.items()
        }

        final_path = self.output_dir / "final.pkl"
        joblib.dump(clean_state, final_path)
        logging.info(f"Final state saved to {final_path}")

    def reset_refiner(self) -> None:
        """Resets the refiner to its initial state."""
        reset_fn = getattr(self.refiner, "reset", None)
        if reset_fn is not None:
            reset_fn()
            logging.info("Refiner parameters reset to initialization state.")
        else:
            logging.warning("Refiner does not support reset().")

    def plot_metrics(self):
        """Plots the metrics history."""
        logger = next(
            (l for l in self.trainer.loggers if isinstance(l, AtomicParquetLogger)), None
        )
        if not logger or not logger.history:
            print("No metrics available.")
            return None

        df = pl.DataFrame(logger.history)
        metrics = [c for c in df.columns if c not in ("step", "epoch")]

        geo_keys = {"mean_dist", "hausdorff_dist", "p95_dist"}

        # Filter losses: exclude regularizers with weight 0
        loss_cols = []
        for c in metrics:
            if c in geo_keys:
                continue
            if c in ("total_loss", "data_loss"):
                loss_cols.append(c)
                continue

            # Check regularizer weight
            if c.endswith("_loss"):
                try:
                    reg_type = RegularizerType(c[:-5])
                    if self.refiner.loss_fn.get_weight(reg_type).item() == 0:
                        continue
                except (ValueError, AttributeError):
                    pass
            loss_cols.append(c)

        geo_cols = [c for c in metrics if c in geo_keys]

        markers = ["o", "s", "^", "v", "D", "x", "+"]

        def _plot_on_ax(ax, cols, title):
            for i, m in enumerate(cols):
                data = df.select(["step", m]).drop_nulls()
                if not data.is_empty():
                    marker = markers[i % len(markers)]
                    ax.plot(data["step"], data[m], label=m, marker=marker, markersize=4, alpha=0.7)
            ax.set_xlabel("Step")
            ax.set_yscale("log")
            ax.set_title(title)
            ax.legend()
            ax.grid(True, which="both", linestyle="--", alpha=0.7)

        if geo_cols and loss_cols:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
            _plot_on_ax(axes[0], loss_cols, "Losses")
            _plot_on_ax(axes[1], geo_cols, "Geometric Metrics")
        else:
            fig, ax = plt.subplots(figsize=(6, 5))
            cols = loss_cols if loss_cols else geo_cols
            title = "Losses" if loss_cols else "Geometric Metrics"
            _plot_on_ax(ax, cols, title)

        plt.close(fig)
        return fig

    def plot_image(self):
        """Plots the default grid of cropped contour visualizations."""
        cb = next((c for c in self.trainer.callbacks if isinstance(c, ImageLoggerCallback)), None)
        if cb:
            fig = cb.create_figure(self.model)
            if fig is not None:
                # Close the figure created by the callback to prevent double display
                plt.close(fig)
            return fig
        print("ImageLoggerCallback not found.")
        return None

    def plot_full_view(
        self, plot_normals: bool = False, save_path: str | Path | None = None
    ) -> Any:
        """
        Generates a large, detailed plot of the full contour state.

        Args:
            plot_normals (bool): If True, overlays the yellow normal lines. Defaults to False.
            save_path (str | Path | None): Optional path to save the figure. If provided,
                                           the figure is saved and not displayed.

        Returns:
            matplotlib.figure.Figure or None: The figure object for display in notebooks,
                                              or None if `save_path` is provided.
        """
        fig, ax = plt.subplots(figsize=(12, 12))

        # Use the refiner's high-level visualization function for core plotting
        self.refiner.visualize_contour(
            image=torch.from_numpy(self.config.image).float(),
            ax=ax,
            stochastic=False,  # Show full contour state
            plot_normals=plot_normals,
            title=f"Refinement @ Step {self.trainer.global_step}",
        )

        # Add GT and initial contours for context, sourcing them from the logger callback
        cb = next((c for c in self.trainer.callbacks if isinstance(c, ImageLoggerCallback)), None)
        if cb:
            if cb.gt_contour is not None and len(cb.gt_contour) > 0:
                ax.plot(cb.gt_contour[:, 1], cb.gt_contour[:, 0], "k:", label="GT")
            if cb.init_contour is not None:
                ax.plot(cb.init_contour[:, 1], cb.init_contour[:, 0], "r--", label="Initial")

        # Consolidate legend to avoid duplicates
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
            plt.close(fig)
            return None

        plt.close(fig)  # Prevent double-display in notebooks
        return fig
