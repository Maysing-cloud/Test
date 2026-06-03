from __future__ import annotations

import json
import os
from typing import Optional

from pointcloud.models import ClassifiedCloud, ValidationReport
from .tools.validator_tools import VALIDATOR_TOOL_DEFINITIONS, ValidatorToolExecutor, QUALITY_THRESHOLDS


class ValidatorAgent:
    """
    Uses Claude to evaluate the quality of a classification and produce structured feedback.
    """

    MODEL = os.environ.get("VALIDATOR_MODEL", "claude-haiku-4-5")
    MAX_TOKENS = 2048

    def __init__(self, client, session):
        self.client = client
        self.session = session

    def validate(self, classified_cloud: ClassifiedCloud) -> ValidationReport:
        self.session.current_classified_cloud = classified_cloud
        executor = ValidatorToolExecutor(self.session)

        system_prompt = (
            "You are a LiDAR classification quality evaluator. "
            "Evaluate the classification quality systematically using the tools provided.\n\n"
            "Required steps:\n"
            "1. compute_confidence_distribution (threshold=0.5)\n"
            "2. check_geometric_consistency\n"
            "3. detect_boundary_discontinuities\n"
            "4. compute_label_coverage\n"
            "5. (Optional) sample_visual_cross_section if more insight is needed\n"
            "6. submit_validation_report — include actionable refinement_hints and "
            "set stop_requested=true only if ALL quality thresholds are met:\n"
            f"   - overall_confidence >= {QUALITY_THRESHOLDS['min_overall_confidence']}\n"
            f"   - low_confidence_fraction <= {QUALITY_THRESHOLDS['max_low_confidence_fraction']}\n"
            f"   - geometric_consistency_score >= {QUALITY_THRESHOLDS['min_geometric_consistency']}\n"
            f"   - boundary_coherence_score >= {QUALITY_THRESHOLDS['min_boundary_coherence']}\n"
        )

        messages = [
            {
                "role": "user",
                "content": (
                    "Evaluate this classification result:\n\n"
                    + self.session.classified_summary_json()
                    + "\n\nProvide detailed feedback and specific refinement hints."
                ),
            }
        ]

        while True:
            response = self.client.messages.create(
                model=self.MODEL,
                max_tokens=self.MAX_TOKENS,
                system=system_prompt,
                tools=VALIDATOR_TOOL_DEFINITIONS,
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
                    if block.name == "submit_validation_report":
                        done = True

            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            if done:
                break

        if executor._report is None:
            raise RuntimeError("ValidatorAgent did not call submit_validation_report.")
        return executor._report
