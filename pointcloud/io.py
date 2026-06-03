from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .models import PointCloud


def load(path: str | Path) -> PointCloud:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".las", ".laz"):
        return _load_las(path)
    if suffix in (".ply", ".pcd", ".xyz"):
        return _load_open3d(path)
    raise ValueError(f"Unsupported format: {suffix}")


def save(cloud: PointCloud, path: str | Path) -> None:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".las", ".laz"):
        _save_las(cloud, path)
    elif suffix in (".ply", ".pcd"):
        _save_open3d(cloud, path)
    else:
        raise ValueError(f"Unsupported format: {suffix}")


def save_classified(cloud: PointCloud, labels: np.ndarray, path: str | Path) -> None:
    """Save point cloud with per-point classification labels (LAS format)."""
    path = Path(path)
    import laspy

    header = laspy.LasHeader(point_format=6, version="1.4")
    las = laspy.LasData(header=header)
    las.x = cloud.points[:, 0]
    las.y = cloud.points[:, 1]
    las.z = cloud.points[:, 2]
    las.classification = labels.astype(np.uint8)
    if cloud.intensities is not None:
        las.intensity = (cloud.intensities * 65535).clip(0, 65535).astype(np.uint16)
    las.write(str(path))


def _load_las(path: Path) -> PointCloud:
    import laspy

    las = laspy.read(str(path))
    points = np.stack([las.x, las.y, las.z], axis=1).astype(np.float64)

    intensities: Optional[np.ndarray] = None
    if hasattr(las, "intensity"):
        intensities = las.intensity.astype(np.float32) / 65535.0

    rgb: Optional[np.ndarray] = None
    if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
        r = las.red.astype(np.float32) / 65535.0 * 255
        g = las.green.astype(np.float32) / 65535.0 * 255
        b = las.blue.astype(np.float32) / 65535.0 * 255
        rgb = np.stack([r, g, b], axis=1).astype(np.uint8)

    meta: dict = {
        "format": "las",
        "point_count": len(points),
        "file": str(path),
    }
    if hasattr(las.header, "offsets"):
        meta["offsets"] = las.header.offsets.tolist()

    return PointCloud(points=points, intensities=intensities, rgb=rgb, source_path=str(path), metadata=meta)


def _save_las(cloud: PointCloud, path: Path) -> None:
    import laspy

    header = laspy.LasHeader(point_format=6, version="1.4")
    las = laspy.LasData(header=header)
    las.x = cloud.points[:, 0]
    las.y = cloud.points[:, 1]
    las.z = cloud.points[:, 2]
    if cloud.intensities is not None:
        las.intensity = (cloud.intensities * 65535).clip(0, 65535).astype(np.uint16)
    las.write(str(path))


def _load_open3d(path: Path) -> PointCloud:
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(path))
    points = np.asarray(pcd.points, dtype=np.float64)

    normals: Optional[np.ndarray] = None
    if pcd.has_normals():
        normals = np.asarray(pcd.normals, dtype=np.float64)

    rgb: Optional[np.ndarray] = None
    if pcd.has_colors():
        rgb = (np.asarray(pcd.colors) * 255).astype(np.uint8)

    return PointCloud(
        points=points,
        normals=normals,
        rgb=rgb,
        source_path=str(path),
        metadata={"format": path.suffix.lstrip("."), "point_count": len(points)},
    )


def _save_open3d(cloud: PointCloud, path: Path) -> None:
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud.points)
    if cloud.normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(cloud.normals)
    if cloud.rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(cloud.rgb.astype(np.float64) / 255.0)
    o3d.io.write_point_cloud(str(path), pcd)
