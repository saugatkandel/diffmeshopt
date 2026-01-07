# Differentiable Mesh Optimization for 3D Segmentation Refinement

This project is a proof-of-concept for refining 3D segmentation boundaries using a self-supervised learning framework with differentiable surface meshes.

## Project Structure

```
.
├── data/
│   └── .gitignore
├── output/
│   └── .gitignore
├── src/
│   ├── data.py
│   ├── generate_sample_data.py
│   ├── loss.py
│   ├── main.py
│   ├── mesh.py
│   ├── model.py
│   └── utils.py
├── tests/
│   ├── __init__.py
│   └── test_data.py
├── pixi.toml
└── README.md
```

- `src/`: Contains the core Python source code.
- `data/`: To store input data (e.g., NIfTI files).
- `output/`: To store the output meshes.
- `tests/`: Contains unit tests.
- `pixi.toml`: Project configuration and dependencies for the Pixi package manager.

## Setup and Installation

This project uses [Pixi](https://pixi.sh/) to manage dependencies.

1.  **Install Pixi:**
    Follow the instructions on the [official website](https://pixi.sh/docs/latest/installation).

2.  **Install Dependencies:**
    Once Pixi is installed, open a terminal in the project root and run:
    ```bash
    pixi install
    ```
    This will create a virtual environment and install all the necessary libraries specified in the `pixi.toml` file.

## Usage

1.  **Generate Sample Data:**
    Before running the main application, you need to generate a sample 3D segmentation. Run the following command from the project root:
    ```bash
    pixi run python src/generate_sample_data.py
    ```
    This will create a `sphere.nii.gz` file in the `data/` directory.

2.  **Run the Refinement:**
    To run the mesh refinement pipeline, execute the main script:
    ```bash
    pixi run start
    ```
    This command is a shortcut for `pixi run python src/main.py`. The script will:
    - Load the segmentation from `data/sphere.nii.gz`.
    - Create an initial mesh and save it as `output/initial_mesh.obj`.
    - Refine the mesh using a self-supervised optimization loop.
    - Save the final refined mesh as `output/refined_mesh.obj`.

## Running Tests

To run the unit tests, use the following command:
```bash
pixi run test
```
This will execute the tests in the `tests/` directory using `pytest`.
