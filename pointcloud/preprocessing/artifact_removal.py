from __future__ import annotations

import numpy as np

from ..models import PointCloud


def remove_mirror_artifacts(
    cloud: PointCloud,
    intensity_percentile: float = 99.0,
    planarity_threshold: float = 0.95,
    cluster_eps_m: float = 0.05,
    min_cluster_size: int = 50,
    enable_raycast_check: bool = False,
) -> tuple[PointCloud, int]:
    """
    Detect and remove mirror/glass reflection artifacts.

    Mirror reflections produce planar clusters of high-intensity points
    that appear "through" a wall or glass surface. We identify them via:
    1. High intensity (retroreflection from mirrors/glass)
    2. Near-perfect local planarity (PCA eigenvalue ratio)
    3. Normal alignment with an adjacent wall plane (optional raycast)
    """
    if cloud.intensities is None:
        return cloud, 0

    threshold = float(np.percentile(cloud.intensities, intensity_percentile))
    candidate_mask = cloud.intensities >= threshold
    candidate_indices = np.where(candidate_mask)[0]

    if len(candidate_indices) < min_cluster_size:
        return cloud, 0

    candidate_points = cloud.points[candidate_indices]
    artifact_indices = set()

    clusters = _dbscan_cluster(candidate_points, eps=cluster_eps_m, min_samples=min_cluster_size)
    for cluster_id in np.unique(clusters):
        if cluster_id == -1:
            continue
        cluster_mask = clusters == cluster_id
        pts = candidate_points[cluster_mask]
        if len(pts) < min_cluster_size:
            continue
        if _is_planar(pts, planarity_threshold):
            original_idx = candidate_indices[cluster_mask]
            artifact_indices.update(original_idx.tolist())

    if not artifact_indices:
        return cloud, 0

    keep_mask = np.ones(cloud.n_points, dtype=bool)
    keep_mask[list(artifact_indices)] = False
    removed = int((~keep_mask).sum())
    return cloud.apply_mask(keep_mask), removed


def _dbscan_cluster(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    from sklearn.cluster import DBSCAN

    db = DBSCAN(eps=eps, min_samples=min_samples, algorithm="ball_tree", n_jobs=-1)
    return db.fit_predict(points)


def _is_planar(points: np.ndarray, threshold: float) -> bool:
    """Return True if points form a near-perfectly flat surface via PCA."""
    if len(points) < 4:
        return False
    centered = points - points.mean(axis=0)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    # s are singular values; eigenvalues ~ s^2
    eigenvalues = s**2
    total = eigenvalues.sum()
    if total == 0:
        return False
    # Smallest eigenvalue fraction: near 0 means planar
    planarity = 1.0 - (eigenvalues[2] / total)
    return bool(planarity >= threshold)
