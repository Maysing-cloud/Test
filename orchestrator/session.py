from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from pointcloud.models import PointCloud, ClassifiedCloud, PointLabel, LABEL_NAMES


@dataclass
class ClassificationSession:
    cloud: PointCloud
    current_classified_cloud: Optional[ClassifiedCloud] = None
    iteration: int = 0
    _features: object = field(default=None, repr=False)  # GeometricFeatures cache

    def cloud_summary_json(self) -> str:
        """Compact JSON summary of the point cloud for LLM context."""
        pts = self.cloud.points
        summary = {
            "n_points": self.cloud.n_points,
            "bbox": {
                "x": [round(float(pts[:, 0].min()), 3), round(float(pts[:, 0].max()), 3)],
                "y": [round(float(pts[:, 1].min()), 3), round(float(pts[:, 1].max()), 3)],
                "z": [round(float(pts[:, 2].min()), 3), round(float(pts[:, 2].max()), 3)],
            },
            "height_range_m": round(float(pts[:, 2].max() - pts[:, 2].min()), 3),
            "has_intensity": self.cloud.intensities is not None,
            "has_rgb": self.cloud.rgb is not None,
            "has_normals": self.cloud.normals is not None,
        }
        if self.cloud.intensities is not None:
            summary["intensity_stats"] = {
                "min": round(float(self.cloud.intensities.min()), 4),
                "max": round(float(self.cloud.intensities.max()), 4),
                "mean": round(float(self.cloud.intensities.mean()), 4),
            }
        return json.dumps(summary, indent=2)

    def classified_summary_json(self) -> str:
        """JSON summary of the current classification state."""
        if self.current_classified_cloud is None:
            return json.dumps({"status": "not classified yet"})
        cc = self.current_classified_cloud
        labels = cc.labels
        confidence = cc.confidence
        coverage = {}
        for lbl in PointLabel:
            mask = labels == int(lbl)
            count = int(mask.sum())
            if count > 0:
                coverage[LABEL_NAMES[int(lbl)]] = {
                    "count": count,
                    "fraction": round(float(count / len(labels)), 4),
                    "mean_confidence": round(float(confidence[mask].mean()), 4),
                }
        return json.dumps({
            "iteration": cc.iteration,
            "n_points": len(labels),
            "overall_confidence": round(float(confidence.mean()), 4),
            "label_coverage": coverage,
        }, indent=2)
