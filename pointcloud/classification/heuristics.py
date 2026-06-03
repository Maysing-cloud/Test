from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..models import PointCloud, PointLabel


@dataclass
class GeometricFeatures:
    height_above_ground: np.ndarray  # (N,) float32
    normal_inclination: np.ndarray   # (N,) float32, degrees from vertical
    planarity: np.ndarray            # (N,) float32 [0,1]
    linearity: np.ndarray            # (N,) float32 [0,1]
    sphericity: np.ndarray           # (N,) float32 [0,1]
    roughness: np.ndarray            # (N,) float32


def extract_features(
    cloud: PointCloud,
    neighborhood_radius_m: float = 0.2,
    ground_z_percentile: float = 5.0,
) -> GeometricFeatures:
    """Extract per-point geometric features using local neighborhoods."""
    points = cloud.points
    n = len(points)

    # Ground plane height: low percentile of Z values
    ground_z = float(np.percentile(points[:, 2], ground_z_percentile))
    height_above_ground = (points[:, 2] - ground_z).astype(np.float32)

    # Normal-based features
    if cloud.normals is not None:
        normals = cloud.normals
        # Inclination from vertical: angle between normal and Z-axis
        cos_angle = np.clip(np.abs(normals[:, 2]), 0.0, 1.0)
        normal_inclination = np.degrees(np.arccos(cos_angle)).astype(np.float32)
    else:
        normal_inclination = np.full(n, 45.0, dtype=np.float32)

    # Local covariance features via KD-tree neighborhood
    planarity, linearity, sphericity, roughness = _covariance_features(
        points, neighborhood_radius_m
    )

    return GeometricFeatures(
        height_above_ground=height_above_ground,
        normal_inclination=normal_inclination,
        planarity=planarity,
        linearity=linearity,
        sphericity=sphericity,
        roughness=roughness,
    )


def _covariance_features(
    points: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute planarity, linearity, sphericity, roughness from local PCA."""
    from scipy.spatial import KDTree

    n = len(points)
    planarity = np.zeros(n, dtype=np.float32)
    linearity = np.zeros(n, dtype=np.float32)
    sphericity = np.zeros(n, dtype=np.float32)
    roughness = np.zeros(n, dtype=np.float32)

    tree = KDTree(points)
    indices = tree.query_ball_point(points, r=radius, workers=-1)

    for i, nbr_idx in enumerate(indices):
        if len(nbr_idx) < 4:
            sphericity[i] = 1.0
            continue
        nbr = points[nbr_idx]
        centered = nbr - nbr.mean(axis=0)
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        e = np.sort(s**2)[::-1]  # eigenvalues descending
        total = e.sum()
        if total < 1e-12:
            sphericity[i] = 1.0
            continue
        e1, e2, e3 = e[0] / total, e[1] / total, e[2] / total
        linearity[i] = float((e1 - e2) / (e1 + 1e-9))
        planarity[i] = float((e2 - e3) / (e1 + 1e-9))
        sphericity[i] = float(e3 / (e1 + 1e-9))
        # Roughness: std of point distances to the local mean plane
        roughness[i] = float(np.sqrt(e[2] / len(nbr_idx)))

    return planarity, linearity, sphericity, roughness


def classify_by_rules(
    cloud: PointCloud,
    features: GeometricFeatures,
    ground_height_max_m: float = 0.15,
    ceiling_height_min_fraction: float = 0.85,
    wall_planarity_min: float = 0.7,
    wall_verticality_max_deg: float = 20.0,
    vegetation_roughness_min: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assign PointLabel to each point using heuristic rules.
    Returns (labels, confidence) arrays.
    """
    n = cloud.n_points
    labels = np.full(n, int(PointLabel.UNCLASSIFIED), dtype=np.int8)
    confidence = np.full(n, 0.5, dtype=np.float32)

    h = features.height_above_ground
    inc = features.normal_inclination
    plan = features.planarity
    rough = features.roughness

    total_height = float(h.max() - h.min()) if h.max() > h.min() else 1.0

    # GROUND: low height, roughly horizontal normal
    ground_mask = (h <= ground_height_max_m) & (inc <= 25.0)
    labels[ground_mask] = int(PointLabel.GROUND)
    confidence[ground_mask] = 0.85

    # CEILING: top fraction of height range, horizontal normal
    ceiling_z = float(h.min() + ceiling_height_min_fraction * total_height)
    ceiling_mask = (h >= ceiling_z) & (inc <= 25.0)
    labels[ceiling_mask] = int(PointLabel.CEILING)
    confidence[ceiling_mask] = 0.80

    # WALL: high planarity, nearly vertical normal
    wall_mask = (
        (plan >= wall_planarity_min)
        & (inc >= (90.0 - wall_verticality_max_deg))
        & (labels == int(PointLabel.UNCLASSIFIED))
    )
    labels[wall_mask] = int(PointLabel.WALL)
    confidence[wall_mask] = np.clip(plan[wall_mask], 0.6, 0.95).astype(np.float32)

    # VEGETATION: high roughness, not already classified
    vegetation_mask = (rough >= vegetation_roughness_min) & (labels == int(PointLabel.UNCLASSIFIED))
    labels[vegetation_mask] = int(PointLabel.VEGETATION)
    confidence[vegetation_mask] = 0.65

    # FURNITURE: mid-height, not wall/ground/ceiling/vegetation
    furniture_mask = (
        (h > ground_height_max_m)
        & (h < ceiling_z)
        & (labels == int(PointLabel.UNCLASSIFIED))
    )
    labels[furniture_mask] = int(PointLabel.FURNITURE)
    confidence[furniture_mask] = 0.55

    # Remaining unclassified → CLUTTER
    clutter_mask = labels == int(PointLabel.UNCLASSIFIED)
    labels[clutter_mask] = int(PointLabel.CLUTTER)
    confidence[clutter_mask] = 0.40

    return labels, confidence


def refine_region(
    cloud: PointCloud,
    labels: np.ndarray,
    confidence: np.ndarray,
    region_type: str,
    center: list[float],
    half_extents_or_radius: float,
    target_label: Optional[str] = None,
    override_confidence: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-classify points in a spatial region (AABB or sphere)."""
    c = np.array(center, dtype=np.float64)
    pts = cloud.points

    if region_type == "sphere":
        dist = np.linalg.norm(pts - c, axis=1)
        region_mask = dist <= half_extents_or_radius
    else:  # aabb
        r = half_extents_or_radius
        region_mask = (
            (np.abs(pts[:, 0] - c[0]) <= r)
            & (np.abs(pts[:, 1] - c[1]) <= r)
            & (np.abs(pts[:, 2] - c[2]) <= r)
        )

    if target_label is not None:
        label_map = {l.name.lower(): int(l) for l in PointLabel}
        label_val = label_map.get(target_label.lower(), int(PointLabel.CLUTTER))
        labels = labels.copy()
        labels[region_mask] = label_val

    if override_confidence is not None:
        confidence = confidence.copy()
        confidence[region_mask] = float(override_confidence)

    return labels, confidence


def smooth_boundaries(
    cloud: PointCloud,
    labels: np.ndarray,
    confidence: np.ndarray,
    iterations: int = 3,
    radius_m: float = 0.15,
    label_pairs: Optional[list[list[str]]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Majority-vote smoothing to reduce label noise at class boundaries."""
    from scipy.spatial import KDTree

    tree = KDTree(cloud.points)
    indices = tree.query_ball_point(cloud.points, r=radius_m, workers=-1)
    new_labels = labels.copy()

    for _ in range(iterations):
        for i, nbr_idx in enumerate(indices):
            if len(nbr_idx) < 3:
                continue
            nbr_labels = new_labels[nbr_idx]
            # Majority vote weighted by confidence
            nbr_conf = confidence[nbr_idx]
            scores: dict[int, float] = {}
            for lbl, conf in zip(nbr_labels, nbr_conf):
                scores[int(lbl)] = scores.get(int(lbl), 0.0) + float(conf)
            best = max(scores, key=lambda k: scores[k])
            new_labels[i] = best

    return new_labels, confidence
