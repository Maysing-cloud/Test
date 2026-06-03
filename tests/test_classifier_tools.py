import numpy as np
import pytest

from pointcloud.models import PointCloud, PointLabel
from orchestrator.session import ClassificationSession
from agents.tools.classifier_tools import ClassifierToolExecutor
from tests.conftest import make_room_cloud


@pytest.fixture
def session():
    cloud = make_room_cloud(n_per_surface=200)
    # Estimate normals synthetically
    pts = cloud.points
    normals = np.zeros_like(pts)
    # Floor and ceiling: z-normals
    normals[pts[:, 2] < 0.2, 2] = 1.0
    normals[pts[:, 2] > 2.5, 2] = -1.0
    # Walls: x- or y-normals
    mid = (pts[:, 2] >= 0.2) & (pts[:, 2] <= 2.5)
    normals[mid & (pts[:, 0] < 0.1), 0] = -1.0
    normals[mid & (pts[:, 0] > 4.9), 0] = 1.0
    normals[mid & (pts[:, 1] < 0.1), 1] = -1.0
    normals[mid & (pts[:, 1] > 4.9), 1] = 1.0
    mid_zero = np.linalg.norm(normals[mid], axis=1) == 0
    normals[np.where(mid)[0][mid_zero], 0] = 1.0
    # Normalize
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    cloud.normals = normals / norms
    return ClassificationSession(cloud=cloud)


@pytest.fixture
def executor(session):
    return ClassifierToolExecutor(session)


class TestExtractFeatures:
    def test_returns_feature_summary(self, executor):
        result = executor.dispatch("extract_geometric_features", {})
        assert "height_stats" in result
        assert "planarity_mean" in result
        assert executor._features is not None

    def test_height_range_positive(self, executor):
        result = executor.dispatch("extract_geometric_features", {})
        h = result["height_stats"]
        assert h["max"] > h["min"]
        assert h["max"] >= 2.0  # room is ~2.7m tall


class TestClassifyByRules:
    def test_requires_features_first(self, executor):
        result = executor.dispatch("classify_by_rules", {})
        assert "error" in result

    def test_classifies_all_points(self, executor, session):
        executor.dispatch("extract_geometric_features", {})
        result = executor.dispatch("classify_by_rules", {})
        assert "overall_confidence" in result
        coverage = result["label_coverage"]
        total = sum(v["count"] for v in coverage.values())
        assert total == session.cloud.n_points

    def test_ground_and_wall_present(self, executor):
        executor.dispatch("extract_geometric_features", {})
        result = executor.dispatch("classify_by_rules", {})
        coverage = result["label_coverage"]
        assert "ground" in coverage
        assert "wall" in coverage or "ceiling" in coverage


class TestRefineRegion:
    def test_relabels_sphere_region(self, executor, session):
        executor.dispatch("extract_geometric_features", {})
        executor.dispatch("classify_by_rules", {})
        center = session.cloud.points.mean(axis=0).tolist()
        result = executor.dispatch("refine_region", {
            "region_type": "sphere",
            "center": center,
            "half_extents_or_radius": 0.5,
            "target_label": "clutter",
        })
        assert "label_changes" in result


class TestFinalizeClassification:
    def test_sets_classified_cloud_on_session(self, executor, session):
        executor.dispatch("extract_geometric_features", {})
        executor.dispatch("classify_by_rules", {})
        result = executor.dispatch("finalize_classification", {})
        assert result["status"] == "finalized"
        assert session.current_classified_cloud is not None
        assert session.current_classified_cloud.labels is not None
