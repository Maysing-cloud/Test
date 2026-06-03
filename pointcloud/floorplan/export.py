from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .extractor import FloorPlan


class FloorPlanExporter:
    def to_dxf(self, floor_plan: FloorPlan, output_path: str | Path) -> None:
        import ezdxf

        doc = ezdxf.new(dxfversion="R2010")
        msp = doc.modelspace()

        for layer_name, color in [("WALLS", 7), ("ROOMS", 3), ("DOORS", 4)]:
            doc.layers.add(name=layer_name, color=color)

        for line in floor_plan.wall_lines:
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                msp.add_line(coords[i], coords[i + 1], dxfattribs={"layer": "WALLS"})

        for poly in floor_plan.room_polygons:
            pts = list(poly.exterior.coords)
            if len(pts) >= 2:
                msp.add_lwpolyline(pts, dxfattribs={"layer": "ROOMS", "closed": True})

        for door in floor_plan.door_openings:
            coords = list(door.coords)
            msp.add_line(coords[0], coords[1], dxfattribs={"layer": "DOORS"})

        doc.saveas(str(output_path))

    def to_svg(self, floor_plan: FloorPlan, output_path: str | Path) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPolygon
        from matplotlib.collections import PatchCollection

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_aspect("equal")

        for line in floor_plan.wall_lines:
            xs, ys = line.xy
            ax.plot(xs, ys, color="gray", linewidth=1.0, label="_")

        for poly in floor_plan.room_polygons:
            xs, ys = poly.exterior.xy
            ax.fill(xs, ys, alpha=0.1, color="blue")
            ax.plot(xs, ys, color="blue", linewidth=0.5)

        for door in floor_plan.door_openings:
            xs, ys = door.xy
            ax.plot(xs, ys, color="green", linewidth=2.0)

        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_title("Floor Plan")
        fig.savefig(str(output_path), format="svg", bbox_inches="tight")
        plt.close(fig)

    def to_geojson(self, floor_plan: FloorPlan, output_path: str | Path) -> None:
        features = []

        for poly in floor_plan.room_polygons:
            features.append({
                "type": "Feature",
                "geometry": poly.__geo_interface__,
                "properties": {"layer": "room"},
            })

        for line in floor_plan.wall_lines:
            features.append({
                "type": "Feature",
                "geometry": line.__geo_interface__,
                "properties": {"layer": "wall"},
            })

        for door in floor_plan.door_openings:
            features.append({
                "type": "Feature",
                "geometry": door.__geo_interface__,
                "properties": {"layer": "door"},
            })

        fc = {"type": "FeatureCollection", "features": features}
        with open(str(output_path), "w") as f:
            json.dump(fc, f, indent=2)
