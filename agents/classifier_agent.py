from __future__ import annotations

import json
import os
from typing import Optional

from pointcloud.models import ClassifiedCloud, ValidationReport
from .tools.classifier_tools import CLASSIFIER_TOOL_DEFINITIONS, ClassifierToolExecutor


class ClassifierAgent:
    """
    Uses Claude to classify LiDAR point cloud points into semantic categories.
    Runs an agentic tool-use loop until finalize_classification is called.
    """

    MODEL = os.environ.get("CLASSIFIER_MODEL", "claude-opus-4-5")
    MAX_TOKENS = 4096

    def __init__(self, client, session):
        self.client = client
        self.session = session

    def classify(self, validation_report: Optional[ValidationReport] = None) -> ClassifiedCloud:
        self.session.iteration += 1
        executor = ClassifierToolExecutor(self.session)

        system_prompt = self._build_system_prompt(validation_report)
        messages = [
            {
                "role": "user",
                "content": (
                    "Please classify this point cloud. Here is the summary:\n\n"
                    + self.session.cloud_summary_json()
                    + "\n\nProceed step by step: extract features, classify, optionally refine "
                    "suspicious regions, smooth boundaries, then finalize."
                ),
            }
        ]

        while True:
            response = self.client.messages.create(
                model=self.MODEL,
                max_tokens=self.MAX_TOKENS,
                system=system_prompt,
                tools=CLASSIFIER_TOOL_DEFINITIONS,
                messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                break

            tool_results = []
            done = False
            for block in response.content:
                if block.type == "tool_use":
                    result = executor.dispatch(block.name, block.input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        }
                    )
                    if block.name == "finalize_classification":
                        done = True

            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            if done:
                break

        return self.session.current_classified_cloud

    def _build_system_prompt(self, report: Optional[ValidationReport]) -> str:
        base = (
            "You are a LiDAR point cloud classification specialist. "
            "Classify points into: ground, wall, vegetation, ceiling, furniture, clutter.\n\n"
            "Use tools in this order:\n"
            "1. extract_geometric_features — understand the data\n"
            "2. classify_by_rules — initial classification with appropriate thresholds\n"
            "3. refine_region — fix specific problem areas if needed\n"
            "4. smooth_boundaries — reduce label noise\n"
            "5. finalize_classification — signal completion\n\n"
            "Choose thresholds based on the feature distributions returned by step 1. "
            "For indoor scans, ceiling_height_min_fraction is typically 0.80–0.90. "
            "For outdoor scans, use 0.95+.\n"
        )
        if report:
            hints = "\n".join(f"  - {h}" for h in report.refinement_hints)
            suspicious = ""
            if report.metrics.suspicious_regions:
                regions = report.metrics.suspicious_regions
                suspicious = "\n\nSuspicious regions requiring refinement:\n" + "\n".join(
                    f"  - center={r.center}, radius={r.radius}m: {r.issue}"
                    for r in regions
                )
            base += (
                f"\n\n## Validator Feedback (iteration {report.iteration})\n"
                f"{report.metrics.feedback_text}\n\n"
                f"Refinement hints:\n{hints}"
                f"{suspicious}\n\n"
                "Use refine_region() to address each suspicious region before finalizing."
            )
        return base
