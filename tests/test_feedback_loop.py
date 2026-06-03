"""
Integration tests for the feedback loop orchestrator using mocked agents.
No Anthropic API calls are made.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from pointcloud.models import (
    PointCloud, ClassifiedCloud, PointLabel,
    ValidationReport, ClassificationMetrics, SuspiciousRegion
)
from orchestrator.session import ClassificationSession
from orchestrator.feedback_loop import IterativeFeedbackOrchestrator, FeedbackLoopConfig
from tests.conftest import make_room_cloud


def make_classified(cloud: PointCloud, confidence: float, iteration: int) -> ClassifiedCloud:
    n = cloud.n_points
    labels = np.full(n, int(PointLabel.WALL), dtype=np.int8)
    labels[:n // 3] = int(PointLabel.GROUND)
    labels[n // 3: 2 * n // 3] = int(PointLabel.CEILING)
    conf = np.full(n, confidence, dtype=np.float32)
    return ClassifiedCloud(cloud=cloud, labels=labels, confidence=conf, iteration=iteration)


def make_report(
    iteration: int,
    confidence: float,
    passed: bool,
    stop: bool,
) -> ValidationReport:
    metrics = ClassificationMetrics(
        overall_confidence=confidence,
        label_coverage={"ground": 0.3, "wall": 0.4, "ceiling": 0.3},
        low_confidence_fraction=0.05,
        boundary_coherence_score=0.85,
        geometric_consistency_score=0.88,
        suspicious_regions=[],
        passed=passed,
        feedback_text="Good classification.",
    )
    return ValidationReport(
        iteration=iteration,
        metrics=metrics,
        refinement_hints=[],
        stop_requested=stop,
    )


@pytest.fixture
def room_cloud():
    return make_room_cloud(n_per_surface=100)


class TestFeedbackLoopStopping:
    def test_stops_on_validator_request(self, room_cloud, tmp_path):
        classifier = MagicMock()
        validator = MagicMock()
        session = ClassificationSession(cloud=room_cloud)

        classifier.session = session
        validator.session = session

        classified = make_classified(room_cloud, 0.85, 1)
        classifier.classify.return_value = classified
        validator.validate.return_value = make_report(1, 0.85, passed=True, stop=True)

        cfg = FeedbackLoopConfig(
            max_iterations=5, min_iterations=1,
            save_intermediate=False,
        )
        orch = IterativeFeedbackOrchestrator(classifier, validator, cfg)
        result = orch.run(room_cloud)

        assert result.best_report.stop_requested is True
        assert classifier.classify.call_count == 1
        assert validator.validate.call_count == 1

    def test_stops_at_max_iterations(self, room_cloud, tmp_path):
        classifier = MagicMock()
        validator = MagicMock()
        session = ClassificationSession(cloud=room_cloud)
        classifier.session = session
        validator.session = session

        def classify_side_effect(*args, **kwargs):
            session.iteration = getattr(session, "_iter_count", 0) + 1
            session._iter_count = session.iteration
            return make_classified(room_cloud, 0.60, session.iteration)

        classifier.classify.side_effect = classify_side_effect
        validator.validate.side_effect = lambda cc: make_report(cc.iteration, 0.60 + cc.iteration * 0.01, False, False)

        cfg = FeedbackLoopConfig(
            max_iterations=3, min_iterations=1, save_intermediate=False,
            convergence_confidence_delta=0.0,  # disable convergence stopping
        )
        orch = IterativeFeedbackOrchestrator(classifier, validator, cfg)
        result = orch.run(room_cloud)

        assert classifier.classify.call_count == 3
        assert len(result.all_history) == 3

    def test_picks_best_iteration(self, room_cloud):
        classifier = MagicMock()
        validator = MagicMock()
        session = ClassificationSession(cloud=room_cloud)
        classifier.session = session
        validator.session = session

        confidences = [0.60, 0.90, 0.70]
        call_count = [0]

        def classify_side_effect(*args, **kwargs):
            call_count[0] += 1
            return make_classified(room_cloud, confidences[call_count[0] - 1], call_count[0])

        def validate_side_effect(cc):
            c = confidences[cc.iteration - 1]
            return make_report(cc.iteration, c, passed=(c >= 0.80), stop=False)

        classifier.classify.side_effect = classify_side_effect
        validator.validate.side_effect = validate_side_effect

        cfg = FeedbackLoopConfig(max_iterations=3, min_iterations=3, save_intermediate=False)
        orch = IterativeFeedbackOrchestrator(classifier, validator, cfg)
        result = orch.run(room_cloud)

        # Best should be iteration 2 (confidence 0.90)
        assert result.best_report.metrics.overall_confidence == 0.90
