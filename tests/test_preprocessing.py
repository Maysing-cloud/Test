import numpy as np
import pytest

from pointcloud.models import PointCloud
from pointcloud.preprocessing.range_filter import range_filter
from pointcloud.preprocessing.artifact_removal import remove_mirror_artifacts, _is_planar


def make_cloud(n: int = 1000, seed: int = 0) -> PointCloud:
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-5, 5, (n, 3)).astype(np.float64)
    intensities = rng.uniform(0.0, 1.0, n).astype(np.float32)
    return PointCloud(points=pts, intensities=intensities)


class TestRangeFilter:
    def test_removes_far_points(self):
        cloud = make_cloud(1000)
        # Inject some far points
        cloud.points[0] = [200, 0, 0]
        cloud.points[1] = [0, 200, 0]
        filtered, removed = range_filter(cloud, min_m=0.0, max_m=100.0)
        assert removed == 2
        assert filtered.n_points == 998

    def test_removes_near_points(self):
        cloud = make_cloud(500)
        cloud.points[0] = [0.001, 0, 0]
        filtered, removed = range_filter(cloud, min_m=0.05, max_m=100.0)
        assert removed >= 1

    def test_preserves_intensities(self):
        cloud = make_cloud(200)
        cloud.points[0] = [999, 0, 0]
        filtered, _ = range_filter(cloud, min_m=0.0, max_m=100.0)
        assert filtered.intensities is not None
        assert len(filtered.intensities) == filtered.n_points

    def test_no_removal_within_range(self):
        rng = np.random.default_rng(1)
        pts = rng.uniform(1.0, 5.0, (100, 3)).astype(np.float64)
        cloud = PointCloud(points=pts)
        filtered, removed = range_filter(cloud, min_m=0.5, max_m=10.0)
        assert removed == 0
        assert filtered.n_points == 100


class TestIsPlanar:
    def test_planar_points(self):
        pts = np.array([[i, j, 0.0] for i in range(10) for j in range(10)], dtype=np.float64)
        assert _is_planar(pts, threshold=0.90) is True

    def test_non_planar_points(self):
        rng = np.random.default_rng(5)
        pts = rng.uniform(-1, 1, (200, 3)).astype(np.float64)
        assert _is_planar(pts, threshold=0.90) is False

    def test_too_few_points(self):
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        assert _is_planar(pts, threshold=0.90) is False


class TestArtifactRemoval:
    def test_no_artifacts_if_no_intensity(self):
        cloud = PointCloud(points=np.random.randn(100, 3))
        filtered, removed = remove_mirror_artifacts(cloud)
        assert removed == 0
        assert filtered.n_points == 100

    def test_detects_planar_high_intensity_cluster(self):
        rng = np.random.default_rng(42)
        # Normal points: moderate intensity, random positions
        normal_pts = rng.uniform(-5, 5, (500, 3)).astype(np.float64)
        normal_int = rng.uniform(0.1, 0.5, 500).astype(np.float32)

        # Mirror artifact: flat plane, very high intensity
        grid = np.array([[i * 0.02, j * 0.02, 0.0] for i in range(15) for j in range(15)], dtype=np.float64)
        mirror_int = np.full(len(grid), 0.99, dtype=np.float32)

        pts = np.vstack([normal_pts, grid])
        intensities = np.concatenate([normal_int, mirror_int])
        cloud = PointCloud(points=pts, intensities=intensities)

        filtered, removed = remove_mirror_artifacts(
            cloud,
            intensity_percentile=95.0,
            planarity_threshold=0.90,
            cluster_eps_m=0.1,
            min_cluster_size=50,
        )
        assert removed > 0
