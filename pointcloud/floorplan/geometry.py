from __future__ import annotations

import numpy as np


def alpha_shape(points_2d: np.ndarray, alpha: float) -> object:
    """
    Compute the alpha-shape (concave hull) of a 2D point set.
    Returns a shapely geometry (Polygon or MultiPolygon).
    """
    from scipy.spatial import Delaunay
    from shapely.geometry import MultiPolygon, Polygon
    from shapely.ops import unary_union

    if len(points_2d) < 4:
        from shapely.geometry import MultiPoint
        return MultiPoint(points_2d.tolist()).convex_hull

    tri = Delaunay(points_2d)
    triangles = points_2d[tri.simplices]

    # Filter by circumradius < 1/alpha
    polys = []
    for t in triangles:
        r = _circumradius(t[0], t[1], t[2])
        if r < 1.0 / alpha:
            polys.append(Polygon(t))

    if not polys:
        from shapely.geometry import MultiPoint
        return MultiPoint(points_2d.tolist()).convex_hull

    return unary_union(polys)


def _circumradius(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = np.linalg.norm(b - a)
    bc = np.linalg.norm(c - b)
    ca = np.linalg.norm(a - c)
    area = abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2.0
    if area < 1e-10:
        return float("inf")
    return float(ab * bc * ca / (4.0 * area))


def simplify_polygon(geom, tolerance: float) -> object:
    """Ramer-Douglas-Peucker simplification via shapely."""
    return geom.simplify(tolerance, preserve_topology=True)
