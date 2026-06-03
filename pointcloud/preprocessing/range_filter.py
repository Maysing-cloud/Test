from __future__ import annotations

import numpy as np

from ..models import PointCloud


def range_filter(cloud: PointCloud, min_m: float, max_m: float) -> tuple[PointCloud, int]:
    """Remove points outside [min_m, max_m] distance from the scanner origin."""
    origin = np.array(cloud.metadata.get("scanner_origin", [0.0, 0.0, 0.0]))
    distances = np.linalg.norm(cloud.points - origin, axis=1)
    mask = (distances >= min_m) & (distances <= max_m)
    removed = int((~mask).sum())
    return cloud.apply_mask(mask), removed
