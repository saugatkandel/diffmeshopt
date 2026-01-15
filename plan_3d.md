# 3D Implementation Plan

## 1. Data Structures
*   **Mesh**: Use `trimesh` for loading/saving (OBJ/STL) and `pytorch3d` (if available) or custom tensors for optimization.
    *   Vertices: $V \in \mathbb{R}^{N \times 3}$
    *   Faces: $F \in \mathbb{Z}^{M \times 3}$
*   **Volume**: 3D Tomogram tensor $I \in \mathbb{R}^{1 \times 1 \times D \times H \times W}$.

## 2. Core Components (Porting from 2D)
*   **Normals**: Compute vertex normals by averaging adjacent face normals.
*   **Sampling**:
    *   Use `torch.nn.functional.grid_sample` with 5D input.
    *   Construct sampling grids: For each vertex $v$, generate a local coordinate system $(n, t, b)$ (normal, tangent, bitangent).
    *   **Rectangular/Prism Sampling**: Sample a small grid in the $t, b$ plane at every step along $n$ to replicate the 2D rectangular averaging. This improves SNR.
*   **Regularization**:
    *   Laplacian smoothing (cotangent or uniform).
    *   Edge length variance.
    *   Surface consistency checks (prevent self-intersections).

## 3. Architecture
*   **Input**: 3D Volume + Initial Mesh.
*   **Encoder**: 3D CNN or local patch sampling.
*   **Decoder**: GCN (Graph Convolutional Network) over the mesh connectivity to predict scalar shifts along normals.

## 4. Execution Steps
1.  Create `src/generate_3d_data.py` (or load real 3D crop).
2.  Create `src/optimize_3d.py` implementing the geometric backend.
3.  Verify 3D sampling and loss gradients.