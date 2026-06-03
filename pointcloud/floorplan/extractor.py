from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from pointcloud.models import ClassifiedCloud, PointLabel
from .geometry import alpha_shape, simplify_polygon


@dataclass
class FloorPlanConfig:
    grid_resolution_m: float = 0.05
    wall_height_min_m: float = 0.3
    wall_height_max_m: float = 3.0
    morphology_close_iter: int = 3
    rdp_tolerance_m: float = 0.05
    alpha: float = 0.5
    min_door_width_m: float = 0.7
    max_door_width_m: float = 2.5


@dataclass
class FloorPlan:
    outer_boundary: Optional[object] = None       # shapely Polygon
    room_polygons: list = field(default_factory=list)
    wall_lines: list = field(default_factory=list)
    door_openings: list = field(default_factory=list)
    grid_origin: tuple[float, float] = (0.0, 0.0)
    grid_resolution_m: float = 0.05


class FloorPlanExtractor:
    def __init__(self, config: Optional[FloorPlanConfig] = None):
        self.config = config or FloorPlanConfig()

    def extract(self, classified_cloud: ClassifiedCloud) -> FloorPlan:
        cfg = self.config
        pts = classified_cloud.cloud.points
        labels = classified_cloud.labels

        wall_mask = (labels == int(PointLabel.WALL)) & (
            (pts[:, 2] >= cfg.wall_height_min_m) & (pts[:, 2] <= cfg.wall_height_max_m)
        )
        ground_mask = labels == int(PointLabel.GROUND)

        wall_pts_2d = pts[wall_mask, :2]
        ground_pts_2d = pts[ground_mask, :2]

        occupancy, origin = self._build_occupancy_grid(wall_pts_2d, cfg.grid_resolution_m)
        occupancy = self._morphological_close(occupancy, cfg.morphology_close_iter)
        wall_lines = self._extract_wall_lines(occupancy, origin, cfg.grid_resolution_m, cfg.rdp_tolerance_m)
        room_polygons = self._extract_room_polygons(ground_pts_2d, cfg.alpha, cfg.rdp_tolerance_m)
        outer = room_polygons[0] if room_polygons else None
        door_openings = self._detect_door_openings(wall_lines, cfg.min_door_width_m, cfg.max_door_width_m)

        return FloorPlan(
            outer_boundary=outer,
            room_polygons=room_polygons,
            wall_lines=wall_lines,
            door_openings=door_openings,
            grid_origin=tuple(origin),
            grid_resolution_m=cfg.grid_resolution_m,
        )

    def _build_occupancy_grid(
        self, wall_pts_2d: np.ndarray, resolution: float
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(wall_pts_2d) == 0:
            return np.zeros((10, 10), dtype=np.uint8), np.array([0.0, 0.0])
        origin = wall_pts_2d.min(axis=0) - resolution
        indices = ((wall_pts_2d - origin) / resolution).astype(int)
        max_idx = indices.max(axis=0) + 2
        grid = np.zeros((max_idx[1], max_idx[0]), dtype=np.uint8)
        grid[indices[:, 1], indices[:, 0]] = 1
        return grid, origin

    def _morphological_close(self, grid: np.ndarray, iterations: int) -> np.ndarray:
        from scipy.ndimage import binary_closing

        struct = np.ones((3, 3), dtype=bool)
        result = grid.astype(bool)
        for _ in range(iterations):
            result = binary_closing(result, structure=struct)
        return result.astype(np.uint8)

    def _extract_wall_lines(
        self, grid: np.ndarray, origin: np.ndarray, resolution: float, tolerance: float
    ) -> list:
        from skimage import measure
        from shapely.geometry import LineString

        contours = measure.find_contours(grid.astype(float), 0.5)
        lines = []
        for contour in contours:
            # Convert grid indices back to world coordinates
            world_pts = contour[:, ::-1] * resolution + origin
            coords = [tuple(pt) for pt in world_pts]
            if len(coords) < 2:
                continue
            ls = LineString(coords)
            ls = simplify_polygon(ls, tolerance)
            lines.append(ls)
        return lines

    def _extract_room_polygons(
        self, ground_pts_2d: np.ndarray, alpha: float, tolerance: float
    ) -> list:
        if len(ground_pts_2d) < 4:
            return []
        from shapely.geometry import MultiPolygon, Polygon

        shape = alpha_shape(ground_pts_2d, alpha=alpha)
        shape = simplify_polygon(shape, tolerance)

        if shape.is_empty:
            return []
        if isinstance(shape, Polygon):
            return [shape]
        if isinstance(shape, MultiPolygon):
            return sorted(shape.geoms, key=lambda g: g.area, reverse=True)
        return [shape]

    def _detect_door_openings(
        self, wall_lines: list, min_width: float, max_width: float
    ) -> list:
        """
        Detect door-sized gaps in wall lines.
        Gaps are segments between consecutive line endpoints that are within [min_width, max_width].
        """
        from shapely.geometry import LineString

        door_lines = []
        for line in wall_lines:
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                p1 = np.array(coords[i])
                p2 = np.array(coords[i + 1])
                dist = float(np.linalg.norm(p2 - p1))
                if min_width <= dist <= max_width:
                    door_lines.append(LineString([p1.tolist(), p2.tolist()]))
        return door_lines
