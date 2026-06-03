from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from pointcloud.models import PointCloud, ClassifiedCloud, ValidationReport
from pointcloud import io as pc_io
from orchestrator.session import ClassificationSession
from agents.classifier_agent import ClassifierAgent
from agents.validator_agent import ValidatorAgent


@dataclass
class FeedbackLoopConfig:
    max_iterations: int = 5
    min_iterations: int = 1
    convergence_confidence_delta: float = 0.02
    target_overall_confidence: float = 0.80
    save_intermediate: bool = True
    intermediate_dir: str = "./runs"


@dataclass
class OrchestratorResult:
    best_classified_cloud: ClassifiedCloud
    best_report: ValidationReport
    all_history: list[tuple[ClassifiedCloud, ValidationReport]]

    def summary(self) -> dict:
        return {
            "total_iterations": len(self.all_history),
            "best_iteration": self.best_classified_cloud.iteration,
            "best_confidence": round(self.best_report.metrics.overall_confidence, 4),
            "passed": self.best_report.metrics.passed,
            "stop_reason": "validator" if self.best_report.stop_requested else "convergence_or_max",
        }


class IterativeFeedbackOrchestrator:
    def __init__(
        self,
        classifier: ClassifierAgent,
        validator: ValidatorAgent,
        config: Optional[FeedbackLoopConfig] = None,
    ):
        self.classifier = classifier
        self.validator = validator
        self.config = config or FeedbackLoopConfig()

    def run(self, preprocessed_cloud: PointCloud) -> OrchestratorResult:
        cfg = self.config
        session = ClassificationSession(cloud=preprocessed_cloud)
        self.classifier.session = session
        self.validator.session = session

        if cfg.save_intermediate:
            Path(cfg.intermediate_dir).mkdir(parents=True, exist_ok=True)

        history: list[tuple[ClassifiedCloud, ValidationReport]] = []
        report: Optional[ValidationReport] = None

        for iteration in range(1, cfg.max_iterations + 1):
            print(f"\n[Iteration {iteration}/{cfg.max_iterations}] Running ClassifierAgent...")
            classified = self.classifier.classify(validation_report=report)
            classified.iteration = iteration

            print(f"[Iteration {iteration}] Running ValidatorAgent...")
            report = self.validator.validate(classified)
            history.append((classified, report))

            conf = report.metrics.overall_confidence
            print(
                f"[Iteration {iteration}] confidence={conf:.4f}, "
                f"passed={report.metrics.passed}, stop={report.stop_requested}"
            )

            if cfg.save_intermediate:
                self._save_intermediate(classified, report, iteration, cfg.intermediate_dir)

            if iteration >= cfg.min_iterations:
                if report.stop_requested:
                    print("  → Validator requested stop — quality target reached.")
                    break
                if iteration > 1:
                    prev_conf = history[-2][1].metrics.overall_confidence
                    delta = conf - prev_conf
                    if delta < cfg.convergence_confidence_delta:
                        print(f"  → Convergence detected (Δ={delta:.4f} < {cfg.convergence_confidence_delta}). Stopping.")
                        break

        best_idx = max(range(len(history)), key=lambda i: history[i][1].metrics.overall_confidence)
        best_cloud, best_report = history[best_idx]

        return OrchestratorResult(
            best_classified_cloud=best_cloud,
            best_report=best_report,
            all_history=history,
        )

    def _save_intermediate(
        self, classified: ClassifiedCloud, report: ValidationReport, iteration: int, out_dir: str
    ) -> None:
        base = Path(out_dir) / f"iter_{iteration:02d}"
        base.mkdir(parents=True, exist_ok=True)

        pc_io.save_classified(
            classified.cloud,
            classified.labels,
            base / "classified.las",
        )

        report_dict = {
            "iteration": report.iteration,
            "passed": report.metrics.passed,
            "stop_requested": report.stop_requested,
            "overall_confidence": report.metrics.overall_confidence,
            "geometric_consistency_score": report.metrics.geometric_consistency_score,
            "boundary_coherence_score": report.metrics.boundary_coherence_score,
            "low_confidence_fraction": report.metrics.low_confidence_fraction,
            "label_coverage": report.metrics.label_coverage,
            "feedback_text": report.metrics.feedback_text,
            "refinement_hints": report.refinement_hints,
            "suspicious_regions": [
                {"center": r.center, "radius": r.radius, "issue": r.issue}
                for r in report.metrics.suspicious_regions
            ],
        }
        with open(base / "report.json", "w") as f:
            json.dump(report_dict, f, indent=2)
