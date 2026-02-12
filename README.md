# Differentiable Mesh Optimization (2D Prototype)

This project implements a differentiable optimization framework for refining 2D segmentation contours using a bi-Gaussian intensity prior. It is a prototype for a future 3D mesh refinement tool for cryo-ET segmentation.

## A Vibe-Coding Project

Note that this project is approximately 90% "vibe-coded," with a human providing high-level instructions and a coding assistant implementing the details.

## Project Structure

The core logic is located in `diffmeshopt/opt2d/`.

*   `diffmeshopt/opt2d/optimize.py`: Contains the `ContourRefiner` and `BSplineContourRefiner` classes that drive the optimization.
*   `diffmeshopt/opt2d/loss.py`: Implements the `BiGaussianLoss` (data term) and geometric regularizers (`LaplacianSmoothingLoss`, `EdgeLengthConsistencyLoss`).
*   `diffmeshopt/opt2d/template.py`: Defines various parameterizations for the membrane intensity profile (Fixed, Global, Per-Point, B-Spline, Neural Field).
*   `diffmeshopt/opt2d/sampling.py`: Handles differentiable image sampling along contour normals.
*   `diffmeshopt/opt2d/generate_2d_data.py`: Scripts to generate synthetic data or load real cryo-ET slices.
*   `diffmeshopt/opt2d/vis.py`: Visualization utilities.
*   `diffmeshopt/opt2d/trainer.py`: Manages the training loop, logging, and checkpointing.

The `examples/` directory contains standalone scripts for analyzing specific components (e.g., `analyze_contour_anchor.py`, `compare_refiners.py`).

## Project Documentation

For a detailed breakdown of the project's design, goals, and current status, please see:
*   **`plan_2d.md`**: Outlines the implementation and testing strategy for the 2D prototype.
*   **`PROJECT_STATUS.md`**: A living document tracking the current state, design decisions, and next steps.

## Setup and Installation

This project uses `pixi` for dependency management, as defined in `pyproject.toml`. To set up the environment and install all dependencies (including the local package in editable mode), run:

```bash
pixi install
```

## Usage

1.  **Generate Data**:
    ```bash
    python diffmeshopt/opt2d/generate_2d_data.py
    ```
    This creates `data/2d_training_data.pkl`.

2.  **Run Optimization**:
    See `notebooks/optimize_2d.ipynb` for an interactive example of loading data, configuring the refiner, and visualizing the optimization process.

## Running Tests

To run the unit tests, use the following command:
```bash
pytest tests/opt2d/
```
This will execute the tests in the `tests/` directory using `pytest`.
