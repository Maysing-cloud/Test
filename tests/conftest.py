import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pointcloud.models import PointCloud, ClassifiedCloud, PointLabel


def make_room_cloud(n_per_surface: int = 500, seed: int = 42) -> PointCloud:
    """Synthetic indoor room point cloud: floor, 4 walls, ceiling."""
    rng = np.random.default_rng(seed)
    parts = []

    # Floor: z ~ 0, XY in [0, 5]
    xy = rng.uniform(0, 5, (n_per_surface, 2))
    z = rng.normal(0, 0.01, n_per_surface)
    floor = np.stack([xy[:, 0], xy[:, 1], z], axis=1)
    parts.append(floor)

    # Ceiling: z ~ 2.7
    z = rng.normal(2.7, 0.01, n_per_surface)
    ceiling = np.stack([xy[:, 0], xy[:, 1], z], axis=1)
    parts.append(ceiling)

    # Walls
    for x_fixed, y_range, x_range, y_fixed, axis in [
        (0.0, None, None, None, "x0"),
        (5.0, None, None, None, "x5"),
        (None, 0.0, None, None, "y0"),
        (None, 5.0, None, None, "y5"),
    ]:
        if axis in ("x0", "x5"):
            x = rng.normal(0.0 if axis == "x0" else 5.0, 0.01, n_per_surface)
            y = rng.uniform(0, 5, n_per_surface)
        else:
            y = rng.normal(0.0 if axis == "y0" else 5.0, 0.01, n_per_surface)
            x = rng.uniform(0, 5, n_per_surface)
        z = rng.uniform(0, 2.7, n_per_surface)
        parts.append(np.stack([x, y, z], axis=1))

    points = np.vstack(parts)
    intensities = rng.uniform(0.1, 0.9, len(points)).astype(np.float32)
    return PointCloud(points=points, intensities=intensities)


@pytest.fixture
def room_cloud() -> PointCloud:
    return make_room_cloud()


@pytest.fixture
def classified_room(room_cloud) -> ClassifiedCloud:
    n = room_cloud.n_points
    labels = np.zeros(n, dtype=np.int8)
    # Assign ground, ceiling, walls by Z heuristic
    z = room_cloud.points[:, 2]
    labels[z < 0.15] = int(PointLabel.GROUND)
    labels[z > 2.55] = int(PointLabel.CEILING)
    mid = (z >= 0.15) & (z <= 2.55)
    labels[mid] = int(PointLabel.WALL)
    confidence = np.full(n, 0.80, dtype=np.float32)
    return ClassifiedCloud(cloud=room_cloud, labels=labels, confidence=confidence, iteration=1)
