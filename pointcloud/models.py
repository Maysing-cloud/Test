from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np


class PointLabel(IntEnum):
    UNCLASSIFIED = 0
    GROUND = 1
    WALL = 2
    VEGETATION = 3
    CEILING = 4
    FURNITURE = 5
    CLUTTER = 6
    ARTIFACT = 7


LABEL_COLORS: dict[PointLabel, tuple[int, int, int]] = {
    PointLabel.UNCLASSIFIED: (255, 0, 0),
    PointLabel.GROUND: (139, 115, 85),
    PointLabel.WALL: (200, 200, 200),
    PointLabel.VEGETATION: (34, 139, 34),
    PointLabel.CEILING: (230, 230, 255),
    PointLabel.FURNITURE: (205, 133, 63),
    PointLabel.CLUTTER: (128, 128, 128),
    PointLabel.ARTIFACT: (255, 165, 0),
}

LABEL_NAMES: dict[int, str] = {int(l): l.name.lower() for l in PointLabel}


@dataclass
class PointCloud:
    points: np.ndarray  # (N, 3) float64 XYZ
    intensities: Optional[np.ndarray] = None  # (N,) float32
    rgb: Optional[np.ndarray] = None  # (N, 3) uint8
    normals: Optional[np.ndarray] = None  # (N, 3) float64
    source_path: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        return len(self.points)

    def apply_mask(self, mask: np.ndarray) -> "PointCloud":
        """Return a new PointCloud filtered by a boolean mask."""
        return PointCloud(
            points=self.points[mask],
            intensities=self.intensities[mask] if self.intensities is not None else None,
            rgb=self.rgb[mask] if self.rgb is not None else None,
            normals=self.normals[mask] if self.normals is not None else None,
            source_path=self.source_path,
            metadata=dict(self.metadata),
        )


@dataclass
class ClassifiedCloud:
    cloud: PointCloud
    labels: np.ndarray  # (N,) int8
    confidence: np.ndarray  # (N,) float32 [0, 1]
    iteration: int = 0
    classifier_rationale: str = ""


@dataclass
class SuspiciousRegion:
    center: list[float]  # [x, y, z]
    radius: float
    issue: str


@dataclass
class ClassificationMetrics:
    overall_confidence: float
    label_coverage: dict[str, float]
    low_confidence_fraction: float
    boundary_coherence_score: float
    geometric_consistency_score: float
    suspicious_regions: list[SuspiciousRegion]
    passed: bool
    feedback_text: str


@dataclass
class ValidationReport:
    iteration: int
    metrics: ClassificationMetrics
    refinement_hints: list[str]
    stop_requested: bool
