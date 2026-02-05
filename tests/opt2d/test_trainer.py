"""
Unit tests for the OptimizationTrainer.
Mocks external dependencies (refiner) to test loop logic and checkpointing.
"""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from diffmeshopt.opt2d.refiner import ContourRefinerBase
from diffmeshopt.opt2d.trainer import OptimizationTrainer, TrainerConfig


@pytest.fixture
def mock_refiner():
    """Creates a mock refiner compatible with the Lightning-based trainer."""
    refiner = MagicMock(spec=ContourRefinerBase)
    refiner.compute_losses.return_value = {"total_loss": torch.tensor(1.0, requires_grad=True)}
    refiner.parameters.return_value = [torch.nn.Parameter(torch.randn(1))]
    refiner.create_optimizer.return_value = torch.optim.SGD(
        refiner.parameters.return_value, lr=0.1
    )
    refiner.contour = torch.zeros(10, 2)
    return refiner


@pytest.fixture
def temp_output_dir():
    """Creates a temporary directory for output."""
    temp_dir = Path(tempfile.mkdtemp())
    yield str(temp_dir)
    shutil.rmtree(str(temp_dir))


def test_trainer_init(mock_refiner, temp_output_dir):
    """Test trainer initialization."""
    config = TrainerConfig(output_dir=temp_output_dir, image=np.zeros((10, 10)), max_steps=10)
    trainer = OptimizationTrainer(refiner=mock_refiner, config=config)
    # Trainer creates a timestamped subdirectory
    assert trainer.output_dir.parent == Path(temp_output_dir)
    assert trainer.refiner == mock_refiner


def test_trainer_checkpointing(mock_refiner, temp_output_dir):
    """Test that the trainer saves checkpoints."""
    config = TrainerConfig(
        output_dir=temp_output_dir, image=np.zeros((10, 10)), save_interval=2, max_steps=5
    )
    trainer = OptimizationTrainer(refiner=mock_refiner, config=config)

    # Mock the actual fitting process
    trainer.trainer.fit = MagicMock()
    trainer.fit(steps_to_run=5)

    # Check that ModelCheckpoint was created with the right params
    checkpoint_cb = next(
        (cb for cb in trainer.trainer.callbacks if isinstance(cb, ModelCheckpoint)), None
    )
    assert checkpoint_cb is not None
    assert Path(checkpoint_cb.dirpath) == trainer.output_dir

    # Check for saved files
    # The test doesn't actually run, so we can't check for files.
    # We check that the callback is configured correctly.
    trainer.trainer.fit.assert_called_once()
