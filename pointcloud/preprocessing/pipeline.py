from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..models import PointCloud
from .range_filter import range_filter
from .artifact_removal import remove_mirror_artifacts
from .noise_filter import statistical_outlier_removal, radius_outlier_removal


@dataclass
class PreprocessingConfig:
    max_range_m: float = 100.0
    min_range_m: float = 0.05
    sor_nb_neighbors: int = 20
    sor_std_ratio: float = 2.0
    radius_nb_points: int = 6
    radius_m: float = 0.1
    voxel_downsample_m: Optional[float] = None
    artifact_intensity_percentile: float = 99.0
    artifact_planarity_threshold: float = 0.95
    artifact_cluster_eps_m: float = 0.05
    artifact_min_cluster_size: int = 50
    estimate_normals: bool = True
    normal_radius_m: float = 0.3
    normal_max_nn: int = 30


class PreprocessingPipeline:
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()

    def run(self, cloud: PointCloud) -> tuple[PointCloud, dict]:
        cfg = self.config
        stats: dict = {"input_points": cloud.n_points}

        cloud, n = range_filter(cloud, cfg.min_range_m, cfg.max_range_m)
        stats["range_removed"] = n

        cloud, n = remove_mirror_artifacts(
            cloud,
            intensity_percentile=cfg.artifact_intensity_percentile,
            planarity_threshold=cfg.artifact_planarity_threshold,
            cluster_eps_m=cfg.artifact_cluster_eps_m,
            min_cluster_size=cfg.artifact_min_cluster_size,
        )
        stats["artifact_removed"] = n

        cloud, n = statistical_outlier_removal(cloud, cfg.sor_nb_neighbors, cfg.sor_std_ratio)
        stats["sor_removed"] = n

        cloud, n = radius_outlier_removal(cloud, cfg.radius_nb_points, cfg.radius_m)
        stats["ror_removed"] = n

        if cfg.voxel_downsample_m:
            cloud, after = self._voxel_downsample(cloud, cfg.voxel_downsample_m)
            stats["after_downsample"] = after

        if cfg.estimate_normals:
            cloud = self._estimate_normals(cloud, cfg.normal_radius_m, cfg.normal_max_nn)

        stats["output_points"] = cloud.n_points
        return cloud, stats

    def _voxel_downsample(self, cloud: PointCloud, voxel_size: float) -> tuple[PointCloud, int]:
        import open3d as o3d
        import numpy as np

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud.points)
        downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
        pts = numpy.asarray(downsampled.points, dtype=np.float64)
        new_cloud = PointCloud(
            points=pts,
            source_path=cloud.source_path,
            metadata=dict(cloud.metadata),
        )
        return new_cloud, len(pts)

    def _estimate_normals(self, cloud: PointCloud, radius: float, max_nn: int) -> PointCloud:
        import open3d as o3d
        import numpy as np

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud.points)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)
        cloud.normals = np.asarray(pcd.normals, dtype=np.float64)
        return cloud
