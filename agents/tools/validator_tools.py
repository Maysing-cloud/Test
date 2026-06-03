from __future__ import annotations

import base64
import io
import json
from typing import Any

import numpy as np

from pointcloud.models import ClassifiedCloud, PointLabel, LABEL_NAMES, SuspiciousRegion, ClassificationMetrics, ValidationReport


VALIDATOR_TOOL_DEFINITIONS = [
    {
        "name": "compute_confidence_distribution",
        "description": "Returns histogram of per-point confidence values and fraction below threshold.",
        "input_schema": {
            "type": "object",
            "properties": {
                "low_confidence_threshold": {
                    "type": "number",
                    "description": "Confidence below this value is considered low quality.",
                    "default": 0.5,
                },
            },
        },
    },
    {
        "name": "check_geometric_consistency",
        "description": (
            "Check that classified regions obey physical constraints: "
            "ground points lie in a near-horizontal plane, "
            "wall points form vertical planes, "
            "ceiling points are at top of height range."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "detect_boundary_discontinuities",
        "description": (
            "Find spatial locations where label transitions are implausibly abrupt, "
            "e.g. ground-to-ceiling without wall. Returns suspicious transition zones."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "min_cluster_size": {
                    "type": "integer",
                    "description": "Minimum cluster size to report.",
                    "default": 50,
                },
            },
        },
    },
    {
        "name": "compute_label_coverage",
        "description": "Returns fraction of points per label and flags missing expected labels.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "sample_visual_cross_section",
        "description": (
            "Render a 2D cross-section slice of the classified cloud as a base64 PNG. "
            "Useful for visual inspection of classification quality."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "axis": {
                    "type": "string",
                    "enum": ["z", "x", "y"],
                    "description": "Axis perpendicular to the cross-section plane.",
                    "default": "z",
                },
                "slice_position_fraction": {
                    "type": "number",
                    "description": "Position of the slice as fraction of axis range [0,1].",
                    "default": 0.5,
                },
                "slice_thickness_m": {
                    "type": "number",
                    "description": "Thickness of the slice in meters.",
                    "default": 0.1,
                },
            },
        },
    },
    {
        "name": "submit_validation_report",
        "description": "Finalise and submit the structured ValidationReport. Call this last.",
        "input_schema": {
            "type": "object",
            "properties": {
                "overall_confidence": {
                    "type": "number",
                    "description": "Overall classification quality score [0,1].",
                },
                "geometric_consistency_score": {
                    "type": "number",
                    "description": "Geometric plausibility score [0,1].",
                    "default": 0.0,
                },
                "boundary_coherence_score": {
                    "type": "number",
                    "description": "Class boundary smoothness score [0,1].",
                    "default": 0.0,
                },
                "passed": {
                    "type": "boolean",
                    "description": "True if quality thresholds are met.",
                },
                "feedback_text": {
                    "type": "string",
                    "description": "Human-readable assessment for the Classifier Agent.",
                },
                "refinement_hints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific actionable hints for the Classifier Agent.",
                },
                "stop_requested": {
                    "type": "boolean",
                    "description": "True when quality target is reached and no more iterations needed.",
                },
                "suspicious_regions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "center": {"type": "array", "items": {"type": "number"}},
                            "radius": {"type": "number"},
                            "issue": {"type": "string"},
                        },
                        "required": ["center", "radius", "issue"],
                    },
                    "default": [],
                },
            },
            "required": ["overall_confidence", "passed", "feedback_text", "stop_requested"],
        },
    },
]

QUALITY_THRESHOLDS = {
    "min_overall_confidence": 0.72,
    "max_low_confidence_fraction": 0.12,
    "min_geometric_consistency": 0.80,
    "min_boundary_coherence": 0.75,
}


class ValidatorToolExecutor:
    def __init__(self, session):
        self.session = session
        self._report: ValidationReport | None = None
        self._consistency_score: float = 0.0
        self._coherence_score: float = 0.0
        self._low_conf_fraction: float = 0.0
        self._suspicious_regions: list[SuspiciousRegion] = []

    def dispatch(self, name: str, inputs: dict) -> dict:
        if name == "compute_confidence_distribution":
            return self._compute_confidence_distribution(**inputs)
        if name == "check_geometric_consistency":
            return self._check_geometric_consistency()
        if name == "detect_boundary_discontinuities":
            return self._detect_boundary_discontinuities(**inputs)
        if name == "compute_label_coverage":
            return self._compute_label_coverage()
        if name == "sample_visual_cross_section":
            return self._sample_visual_cross_section(**inputs)
        if name == "submit_validation_report":
            return self._submit_validation_report(**inputs)
        return {"error": f"Unknown tool: {name}"}

    def _compute_confidence_distribution(self, low_confidence_threshold: float = 0.5) -> dict:
        cc = self.session.current_classified_cloud
        conf = cc.confidence
        counts, edges = np.histogram(conf, bins=10, range=(0.0, 1.0))
        low_frac = float((conf < low_confidence_threshold).mean())
        self._low_conf_fraction = low_frac
        return {
            "mean_confidence": round(float(conf.mean()), 4),
            "std_confidence": round(float(conf.std()), 4),
            "low_confidence_fraction": round(low_frac, 4),
            "threshold": low_confidence_threshold,
            "histogram": {
                f"{edges[i]:.1f}-{edges[i+1]:.1f}": int(counts[i])
                for i in range(len(counts))
            },
        }

    def _check_geometric_consistency(self) -> dict:
        cc = self.session.current_classified_cloud
        pts = cc.cloud.points
        labels = cc.labels
        normals = cc.cloud.normals
        results = {}
        score_sum = 0.0
        checks = 0

        # Ground: should be near-horizontal (normal z component ~ 1)
        gnd = labels == int(PointLabel.GROUND)
        if gnd.sum() > 0 and normals is not None:
            gnd_normals = np.abs(normals[gnd, 2])
            frac_horizontal = float((gnd_normals > 0.9).mean())
            results["ground_horizontal"] = round(frac_horizontal, 4)
            score_sum += frac_horizontal
            checks += 1

        # Walls: should be near-vertical (normal z component ~ 0)
        wall = labels == int(PointLabel.WALL)
        if wall.sum() > 0 and normals is not None:
            wall_normals = np.abs(normals[wall, 2])
            frac_vertical = float((wall_normals < 0.2).mean())
            results["wall_vertical"] = round(frac_vertical, 4)
            score_sum += frac_vertical
            checks += 1

        # Ceiling: should be in the top 20% of height
        ceil = labels == int(PointLabel.CEILING)
        if ceil.sum() > 0:
            z_max = pts[:, 2].max()
            z_range = pts[:, 2].max() - pts[:, 2].min()
            ceiling_threshold = z_max - 0.2 * z_range
            frac_top = float((pts[ceil, 2] >= ceiling_threshold).mean())
            results["ceiling_at_top"] = round(frac_top, 4)
            score_sum += frac_top
            checks += 1

        overall = float(score_sum / checks) if checks > 0 else 0.5
        self._consistency_score = overall
        results["overall_score"] = round(overall, 4)
        return results

    def _detect_boundary_discontinuities(self, min_cluster_size: int = 50) -> dict:
        from scipy.spatial import KDTree

        cc = self.session.current_classified_cloud
        pts = cc.cloud.points
        labels = cc.labels

        # Find points where neighbors have very different labels (boundary points)
        tree = KDTree(pts)
        pairs = tree.query_pairs(r=0.3)
        boundary_pts: set[int] = set()
        invalid_transitions = {
            (int(PointLabel.GROUND), int(PointLabel.CEILING)),
            (int(PointLabel.CEILING), int(PointLabel.GROUND)),
        }
        for i, j in pairs:
            li, lj = int(labels[i]), int(labels[j])
            if (li, lj) in invalid_transitions or (lj, li) in invalid_transitions:
                boundary_pts.add(i)
                boundary_pts.add(j)

        suspicious_regions = []
        if len(boundary_pts) >= min_cluster_size:
            bp_array = pts[list(boundary_pts)]
            center = bp_array.mean(axis=0).tolist()
            radius = float(np.linalg.norm(bp_array - bp_array.mean(axis=0), axis=1).max())
            region = SuspiciousRegion(
                center=[round(c, 3) for c in center],
                radius=round(radius, 3),
                issue="Abrupt label transition (e.g. ground-ceiling without wall)",
            )
            suspicious_regions.append(region)
            self._suspicious_regions.extend(suspicious_regions)

        n_boundary = len(boundary_pts)
        coherence = max(0.0, 1.0 - n_boundary / max(len(pts), 1))
        self._coherence_score = float(coherence)

        return {
            "boundary_point_count": n_boundary,
            "coherence_score": round(coherence, 4),
            "suspicious_regions": [
                {"center": r.center, "radius": r.radius, "issue": r.issue}
                for r in suspicious_regions
            ],
        }

    def _compute_label_coverage(self) -> dict:
        cc = self.session.current_classified_cloud
        labels = cc.labels
        n = len(labels)
        coverage = {}
        for lbl in PointLabel:
            count = int((labels == int(lbl)).sum())
            if count > 0:
                coverage[LABEL_NAMES[int(lbl)]] = {
                    "count": count,
                    "fraction": round(float(count / n), 4),
                }
        missing = [
            LABEL_NAMES[int(l)] for l in [PointLabel.GROUND, PointLabel.WALL]
            if int(l) not in [int(lbl) for lbl in PointLabel if (labels == int(lbl)).any()]
        ]
        return {"coverage": coverage, "missing_expected_labels": missing}

    def _sample_visual_cross_section(
        self,
        axis: str = "z",
        slice_position_fraction: float = 0.5,
        slice_thickness_m: float = 0.1,
    ) -> dict:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from pointcloud.models import LABEL_COLORS

        cc = self.session.current_classified_cloud
        pts = cc.cloud.points
        labels = cc.labels

        ax_idx = {"x": 0, "y": 1, "z": 2}[axis]
        ax_vals = pts[:, ax_idx]
        pos = float(ax_vals.min() + slice_position_fraction * (ax_vals.max() - ax_vals.min()))
        mask = (ax_vals >= pos - slice_thickness_m / 2) & (ax_vals <= pos + slice_thickness_m / 2)

        if mask.sum() == 0:
            return {"error": "No points in cross-section slice."}

        slice_pts = pts[mask]
        slice_labels = labels[mask]

        ax1, ax2 = [i for i in range(3) if i != ax_idx]
        fig, ax = plt.subplots(figsize=(8, 6))
        for lbl in PointLabel:
            lbl_mask = slice_labels == int(lbl)
            if lbl_mask.sum() == 0:
                continue
            color = tuple(c / 255.0 for c in LABEL_COLORS[lbl])
            ax.scatter(
                slice_pts[lbl_mask, ax1],
                slice_pts[lbl_mask, ax2],
                c=[color],
                s=0.5,
                label=LABEL_NAMES[int(lbl)],
            )
        ax.legend(markerscale=10, loc="upper right")
        ax.set_xlabel(["x", "y", "z"][ax1])
        ax.set_ylabel(["x", "y", "z"][ax2])
        ax.set_title(f"Cross-section at {axis}={pos:.2f}m")
        ax.set_aspect("equal")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        return {"image_base64_png": b64, "n_points_in_slice": int(mask.sum())}

    def _submit_validation_report(
        self,
        overall_confidence: float,
        passed: bool,
        feedback_text: str,
        stop_requested: bool,
        geometric_consistency_score: float = 0.0,
        boundary_coherence_score: float = 0.0,
        refinement_hints: list[str] | None = None,
        suspicious_regions: list[dict] | None = None,
    ) -> dict:
        cc = self.session.current_classified_cloud
        conf = cc.confidence
        labels = cc.labels
        n = len(labels)

        coverage = {
            LABEL_NAMES[int(lbl)]: round(float((labels == int(lbl)).sum() / n), 4)
            for lbl in PointLabel
            if (labels == int(lbl)).sum() > 0
        }

        regions = []
        if suspicious_regions:
            for r in suspicious_regions:
                regions.append(SuspiciousRegion(
                    center=r["center"],
                    radius=r["radius"],
                    issue=r["issue"],
                ))
        regions.extend(self._suspicious_regions)

        metrics = ClassificationMetrics(
            overall_confidence=overall_confidence,
            label_coverage=coverage,
            low_confidence_fraction=self._low_conf_fraction,
            boundary_coherence_score=boundary_coherence_score or self._coherence_score,
            geometric_consistency_score=geometric_consistency_score or self._consistency_score,
            suspicious_regions=regions,
            passed=passed,
            feedback_text=feedback_text,
        )

        self._report = ValidationReport(
            iteration=cc.iteration,
            metrics=metrics,
            refinement_hints=refinement_hints or [],
            stop_requested=stop_requested,
        )
        self.session.last_validation_report = self._report
        return {"status": "report_submitted", "passed": passed, "stop_requested": stop_requested}
