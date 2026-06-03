from __future__ import annotations

import numpy as np

from ..models import PointCloud


def _to_o3d(cloud: PointCloud):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud.points)
    if cloud.normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(cloud.normals)
    if cloud.rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(cloud.rgb.astype(np.float64) / 255.0)
    return pcd


def _apply_inlier_indices(cloud: PointCloud, indices: list[int]) -> PointCloud:
    idx = np.array(indices)
    return PointCloud(
        points=cloud.points[idx],
        intensities=cloud.intensities[idx] if cloud.intensities is not None else None,
        rgb=cloud.rgb[idx] if cloud.rgb is not None else None,
        normals=cloud.normals[idx] if cloud.normals is not None else None,
        source_path=cloud.source_path,
        metadata=dict(cloud.metadata),
    )


def statistical_outlier_removal(
    cloud: PointCloud, nb_neighbors: int = 20, std_ratio: float = 2.0
) -> tuple[PointCloud, int]:
    pcd = _to_o3d(cloud)
    _, inlier_idx = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    removed = cloud.n_points - len(inlier_idx)
    return _apply_inlier_indices(cloud, inlier_idx), removed


def radius_outlier_removal(
    cloud: PointCloud, nb_points: int = 6, radius: float = 0.1
) -> tuple[PointCloud, int]:
    pcd = _to_o3d(cloud)
    _, inlier_idx = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    removed = cloud.n_points - len(inlier_idx)
    return _apply_inlier_indices(cloud, inlier_idx), removed
