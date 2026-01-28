import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import cdist
from skimage.draw import polygon_perimeter


def _to_numpy(x: np.ndarray | torch.Tensor | list) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asanyarray(x)


def _to_tensor(x: np.ndarray | torch.Tensor | list, device=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x.float()
    else:
        t = torch.tensor(x, dtype=torch.float32)
    if device is not None:
        t = t.to(device)
    return t


def compute_contour_metrics(
    contour: np.ndarray | torch.Tensor, gt_contour: np.ndarray | torch.Tensor
) -> dict[str, float]:
    """
    Computes geometric metrics comparing the optimized contour to the ground truth.
    NOTE: This computes bidirectional metrics. Currently, we prefer unidirectional
    metrics using the distance map for training efficiency, but this is kept for
    potential future use.

    Args:
        contour: (N, 2) numpy array or torch tensor of vertices.
        gt_contour: (M, 2) numpy array or torch tensor of ground truth vertices.

    Returns:
        Dictionary containing 'mean_dist' and 'hausdorff_dist'.
    """
    if gt_contour is None or len(gt_contour) == 0:
        return {}

    # If both are torch tensors, stay in torch to potentially use GPU
    if isinstance(contour, torch.Tensor) and isinstance(gt_contour, torch.Tensor):
        return _compute_metrics_torch(contour, gt_contour)

    # Otherwise fallback to numpy/scipy
    return _compute_metrics_numpy(_to_numpy(contour), _to_numpy(gt_contour))


def _compute_metrics_torch(contour: torch.Tensor, gt_contour: torch.Tensor) -> dict[str, float]:
    # Ensure same device
    if contour.device != gt_contour.device:
        gt_contour = gt_contour.to(contour.device)

    # Compute pairwise distances
    dists = torch.cdist(contour.float(), gt_contour.float())

    # Distance from each point in contour to nearest point in gt
    d1 = dists.min(dim=1).values
    # Distance from each point in gt to nearest point in contour
    d2 = dists.min(dim=0).values

    mean_dist = (d1.mean() + d2.mean()) / 2.0
    hausdorff_dist = torch.max(d1.max(), d2.max())

    # 95th percentile Hausdorff distance
    h95_1 = torch.quantile(d1, 0.95)
    h95_2 = torch.quantile(d2, 0.95)
    p95_dist = torch.max(h95_1, h95_2)

    return {
        "mean_dist": mean_dist.item(),
        "hausdorff_dist": hausdorff_dist.item(),
        "p95_dist": p95_dist.item(),
    }


def _compute_metrics_numpy(contour: np.ndarray, gt_contour: np.ndarray) -> dict[str, float]:
    # Compute pairwise distances
    dists = cdist(contour, gt_contour, metric="euclidean")

    # Distance from each point in contour to nearest point in gt
    d1 = dists.min(axis=1)
    # Distance from each point in gt to nearest point in contour
    d2 = dists.min(axis=0)

    mean_dist = (np.mean(d1) + np.mean(d2)) / 2.0
    hausdorff_dist = max(np.max(d1), np.max(d2))

    # 95th percentile Hausdorff distance
    h95_1 = np.percentile(d1, 95)
    h95_2 = np.percentile(d2, 95)
    p95_dist = max(h95_1, h95_2)

    return {
        "mean_dist": float(mean_dist),
        "hausdorff_dist": float(hausdorff_dist),
        "p95_dist": float(p95_dist),
    }


def compute_metrics_from_map(
    contour: torch.Tensor | np.ndarray,
    distance_map: torch.Tensor,
    calc_chamfer: bool = True,
    calc_hausdorff: bool = False,
    calc_p95: bool = False,
) -> dict[str, float]:
    """
    Computes metrics using a precomputed distance map (Distance Transform) of the GT.
    This is O(N) and avoids pairwise distance calculation.
    NOTE: This is unidirectional (Contour -> GT).

    Args:
        contour: (N, 2) tensor of (row, col) coordinates.
        distance_map: (H, W) tensor where value is distance to nearest GT point.
    """
    if len(contour) == 0:
        return {}

    # Ensure contour is a tensor on the same device as distance_map
    contour = _to_tensor(contour, device=distance_map.device)

    H, W = distance_map.shape[-2:]

    # contour: (N, 2) -> (y, x)
    # grid_sample expects (x, y) in [-1, 1]
    y = contour[:, 0]
    x = contour[:, 1]

    norm_x = (x / (W - 1)) * 2 - 1
    norm_y = (y / (H - 1)) * 2 - 1

    # (1, 1, N, 2)
    grid = torch.stack([norm_x, norm_y], dim=-1).view(1, 1, -1, 2)

    # Sample distances
    # distance_map should be (1, 1, H, W) or (H, W)
    if distance_map.ndim == 2:
        dm = distance_map.view(1, 1, H, W)
    elif distance_map.ndim == 3:
        dm = distance_map.unsqueeze(0)
    else:
        dm = distance_map

    dists = F.grid_sample(dm, grid, align_corners=True, padding_mode="border")

    # (1, 1, 1, N) -> (N,)
    dists = dists.view(-1)

    metrics = {}
    if calc_chamfer:
        metrics["mean_dist"] = dists.mean().item()
    if calc_hausdorff:
        metrics["hausdorff_dist"] = dists.max().item()
    if calc_p95:
        metrics["p95_dist"] = torch.quantile(dists, 0.95).item()

    return metrics


def compute_gt_distance_map(
    gt_contour: np.ndarray | torch.Tensor, image_shape: tuple[int, ...]
) -> torch.Tensor | None:
    """Precomputes the Euclidean Distance Transform of the GT contour."""
    if gt_contour is None or len(gt_contour) == 0:
        return None

    gt_contour_np = _to_numpy(gt_contour)

    H, W = image_shape[:2]
    mask = np.ones((H, W), dtype=bool)

    # Rasterize GT contour
    r = np.round(gt_contour_np[:, 0]).astype(int)
    c = np.round(gt_contour_np[:, 1]).astype(int)

    # Clip coordinates to image bounds to avoid errors
    r = np.clip(r, 0, H - 1)
    c = np.clip(c, 0, W - 1)

    rr, cc = polygon_perimeter(r, c, shape=(H, W), clip=True)
    mask[rr, cc] = 0

    dt = distance_transform_edt(mask)
    return torch.from_numpy(dt).float()
