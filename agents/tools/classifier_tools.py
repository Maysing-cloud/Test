from __future__ import annotations

import json
from typing import Any

import numpy as np

from pointcloud.classification.heuristics import (
    extract_features,
    classify_by_rules,
    refine_region,
    smooth_boundaries,
    GeometricFeatures,
)
from pointcloud.models import ClassifiedCloud, PointLabel, LABEL_NAMES


CLASSIFIER_TOOL_DEFINITIONS = [
    {
        "name": "extract_geometric_features",
        "description": (
            "Extract per-point geometric features used for classification: "
            "height above ground, surface normal inclination from vertical, "
            "local planarity, linearity, sphericity, and roughness. "
            "Returns a JSON summary of feature distributions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "neighborhood_radius_m": {
                    "type": "number",
                    "description": "Neighborhood radius in meters for local feature computation.",
                    "default": 0.2,
                },
                "ground_z_percentile": {
                    "type": "number",
                    "description": "Percentile of Z values used as ground plane height.",
                    "default": 5.0,
                },
            },
        },
    },
    {
        "name": "classify_by_rules",
        "description": (
            "Apply heuristic rules to produce an initial classification. "
            "Categories: ground, wall, vegetation, ceiling, furniture, clutter. "
            "Confidence is derived from feature strength per rule. "
            "Must call extract_geometric_features first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ground_height_max_m": {
                    "type": "number",
                    "description": "Max height above ground plane for GROUND class.",
                    "default": 0.15,
                },
                "ceiling_height_min_fraction": {
                    "type": "number",
                    "description": "Fraction of total height range above which points are CEILING.",
                    "default": 0.85,
                },
                "wall_planarity_min": {
                    "type": "number",
                    "description": "Minimum planarity score for WALL class.",
                    "default": 0.7,
                },
                "wall_verticality_max_deg": {
                    "type": "number",
                    "description": "Max deviation from vertical (degrees) for WALL normals.",
                    "default": 20.0,
                },
                "vegetation_roughness_min": {
                    "type": "number",
                    "description": "Minimum roughness for VEGETATION class.",
                    "default": 0.05,
                },
            },
        },
    },
    {
        "name": "refine_region",
        "description": (
            "Re-classify all points within a spatial region (AABB or sphere) "
            "to a specific label. Use this to act on Validator feedback for "
            "specific suspicious regions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "region_type": {
                    "type": "string",
                    "enum": ["aabb", "sphere"],
                    "description": "Shape of the refinement region.",
                },
                "center": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                    "description": "XYZ center of the region.",
                },
                "half_extents_or_radius": {
                    "type": "number",
                    "description": "Half-extent (AABB) or radius (sphere) in meters.",
                },
                "target_label": {
                    "type": "string",
                    "description": "Label to assign: ground, wall, vegetation, ceiling, furniture, clutter.",
                },
                "override_confidence": {
                    "type": "number",
                    "description": "Optional fixed confidence value for re-classified points.",
                },
            },
            "required": ["region_type", "center", "half_extents_or_radius"],
        },
    },
    {
        "name": "smooth_boundaries",
        "description": (
            "Apply majority-vote smoothing to reduce label noise at class boundaries. "
            "Uses confidence-weighted voting in a local neighborhood."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "iterations": {
                    "type": "integer",
                    "description": "Number of smoothing passes.",
                    "default": 3,
                },
                "radius_m": {
                    "type": "number",
                    "description": "Neighborhood radius in meters for voting.",
                    "default": 0.15,
                },
            },
        },
    },
    {
        "name": "finalize_classification",
        "description": "Signal that classification is complete. Call this last.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


class ClassifierToolExecutor:
    def __init__(self, session):
        self.session = session
        self._features: GeometricFeatures | None = None
        self._labels: np.ndarray | None = None
        self._confidence: np.ndarray | None = None

    def dispatch(self, name: str, inputs: dict) -> dict:
        if name == "extract_geometric_features":
            return self._extract_geometric_features(**inputs)
        if name == "classify_by_rules":
            return self._classify_by_rules(**inputs)
        if name == "refine_region":
            return self._refine_region(**inputs)
        if name == "smooth_boundaries":
            return self._smooth_boundaries(**inputs)
        if name == "finalize_classification":
            return self._finalize_classification()
        return {"error": f"Unknown tool: {name}"}

    def _extract_geometric_features(
        self,
        neighborhood_radius_m: float = 0.2,
        ground_z_percentile: float = 5.0,
    ) -> dict:
        self._features = extract_features(
            self.session.cloud,
            neighborhood_radius_m=neighborhood_radius_m,
            ground_z_percentile=ground_z_percentile,
        )
        f = self._features
        return {
            "status": "ok",
            "n_points": self.session.cloud.n_points,
            "height_stats": {
                "min": round(float(f.height_above_ground.min()), 3),
                "max": round(float(f.height_above_ground.max()), 3),
                "p25": round(float(np.percentile(f.height_above_ground, 25)), 3),
                "p75": round(float(np.percentile(f.height_above_ground, 75)), 3),
            },
            "normal_inclination_mean_deg": round(float(f.normal_inclination.mean()), 2),
            "planarity_mean": round(float(f.planarity.mean()), 4),
            "roughness_p90": round(float(np.percentile(f.roughness, 90)), 5),
        }

    def _classify_by_rules(
        self,
        ground_height_max_m: float = 0.15,
        ceiling_height_min_fraction: float = 0.85,
        wall_planarity_min: float = 0.7,
        wall_verticality_max_deg: float = 20.0,
        vegetation_roughness_min: float = 0.05,
    ) -> dict:
        if self._features is None:
            return {"error": "Call extract_geometric_features first."}
        self._labels, self._confidence = classify_by_rules(
            self.session.cloud,
            self._features,
            ground_height_max_m=ground_height_max_m,
            ceiling_height_min_fraction=ceiling_height_min_fraction,
            wall_planarity_min=wall_planarity_min,
            wall_verticality_max_deg=wall_verticality_max_deg,
            vegetation_roughness_min=vegetation_roughness_min,
        )
        return self._label_summary()

    def _refine_region(
        self,
        region_type: str,
        center: list,
        half_extents_or_radius: float,
        target_label: str | None = None,
        override_confidence: float | None = None,
    ) -> dict:
        if self._labels is None:
            return {"error": "Call classify_by_rules first."}
        before_counts = self._count_labels()
        self._labels, self._confidence = refine_region(
            self.session.cloud,
            self._labels,
            self._confidence,
            region_type=region_type,
            center=center,
            half_extents_or_radius=half_extents_or_radius,
            target_label=target_label,
            override_confidence=override_confidence,
        )
        after_counts = self._count_labels()
        changed = {
            k: after_counts.get(k, 0) - before_counts.get(k, 0)
            for k in set(list(before_counts) + list(after_counts))
            if after_counts.get(k, 0) != before_counts.get(k, 0)
        }
        return {"status": "ok", "label_changes": changed}

    def _smooth_boundaries(self, iterations: int = 3, radius_m: float = 0.15) -> dict:
        if self._labels is None:
            return {"error": "Call classify_by_rules first."}
        self._labels, self._confidence = smooth_boundaries(
            self.session.cloud,
            self._labels,
            self._confidence,
            iterations=iterations,
            radius_m=radius_m,
        )
        return {"status": "ok", "message": f"Smoothing applied ({iterations} iterations)."}

    def _finalize_classification(self) -> dict:
        if self._labels is None:
            return {"error": "No classification to finalize."}
        self.session.current_classified_cloud = ClassifiedCloud(
            cloud=self.session.cloud,
            labels=self._labels,
            confidence=self._confidence,
            iteration=self.session.iteration,
        )
        return {
            "status": "finalized",
            **self._label_summary(),
        }

    def _label_summary(self) -> dict:
        if self._labels is None:
            return {}
        coverage = {}
        for lbl in PointLabel:
            mask = self._labels == int(lbl)
            count = int(mask.sum())
            if count > 0:
                coverage[LABEL_NAMES[int(lbl)]] = {
                    "count": count,
                    "fraction": round(float(count / len(self._labels)), 4),
                    "mean_confidence": round(float(self._confidence[mask].mean()), 4),
                }
        return {
            "overall_confidence": round(float(self._confidence.mean()), 4),
            "label_coverage": coverage,
        }

    def _count_labels(self) -> dict[str, int]:
        if self._labels is None:
            return {}
        return {
            LABEL_NAMES[int(lbl)]: int((self._labels == int(lbl)).sum())
            for lbl in PointLabel
        }
