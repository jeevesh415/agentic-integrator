"""Prediction Memory — tracks world model prediction accuracy over time."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger("agentic_integrator.memory")


@dataclass
class PredictionRecord:
    """Record of a single prediction and its outcome."""

    action: str
    predicted_score: float
    actual_accuracy: float
    action_type: str = ""


class PredictionMemory:
    """
    Tracks prediction accuracy over time to enable adaptive confidence.

    Records how well the world model predicted outcomes for different
    action types. This information is used by the planner to decide
    when simulation is worthwhile vs when to skip it.
    """

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.records: List[PredictionRecord] = []
        self.records_by_type: Dict[str, List[PredictionRecord]] = defaultdict(list)
        self._episode_records: List[PredictionRecord] = []

    def record(
        self,
        action: str,
        predicted_score: float,
        actual_accuracy: float,
    ) -> None:
        """Record a prediction result."""
        action_type = self._classify_action(action)

        rec = PredictionRecord(
            action=action[:200],
            predicted_score=predicted_score,
            actual_accuracy=actual_accuracy,
            action_type=action_type,
        )

        self.records.append(rec)
        self.records_by_type[action_type].append(rec)
        self._episode_records.append(rec)

        # Keep window bounded
        if len(self.records) > self.window_size * 2:
            self.records = self.records[-self.window_size:]

        logger.debug(
            f"Prediction recorded: type={action_type}, "
            f"predicted={predicted_score:.2f}, actual={actual_accuracy:.2f}"
        )

    def get_accuracy_for_type(self, action_type: str) -> float:
        """Get average prediction accuracy for an action type."""
        records = self.records_by_type.get(action_type, [])
        if not records:
            return 0.5  # Default for unknown types

        recent = records[-self.window_size:]
        return sum(r.actual_accuracy for r in recent) / len(recent)

    def get_overall_accuracy(self) -> float:
        """Get overall prediction accuracy."""
        if not self.records:
            return 0.5
        recent = self.records[-self.window_size:]
        return sum(r.actual_accuracy for r in recent) / len(recent)

    def should_simulate(self, action_type: str) -> bool:
        """
        Determine if world model simulation is worthwhile for this action type.

        If the world model consistently predicts well for this type,
        simulation adds value. If predictions are random (accuracy ~0.5),
        it's better to skip simulation to save compute.
        """
        accuracy = self.get_accuracy_for_type(action_type)
        records = self.records_by_type.get(action_type, [])

        # Need at least 5 records to make a judgment
        if len(records) < 5:
            return True  # Default to simulating

        # If accuracy is consistently high, simulation is valuable
        if accuracy > 0.65:
            return True

        # If accuracy is near random, skip simulation
        if accuracy < 0.55 and len(records) >= 10:
            logger.info(
                f"Skipping simulation for '{action_type}': "
                f"accuracy={accuracy:.2f} (near random)"
            )
            return False

        return True

    def reset_episode(self) -> None:
        """Reset episode-level tracking (called between tasks)."""
        self._episode_records = []

    def _classify_action(self, action: str) -> str:
        """Classify action type for memory tracking."""
        action_lower = action.lower()

        if "click" in action_lower:
            return "click"
        if "typewrite" in action_lower or "write(" in action_lower:
            return "type"
        if "scroll" in action_lower:
            return "scroll"
        if "hotkey" in action_lower or "press(" in action_lower:
            return "keyboard"
        if "drag" in action_lower:
            return "drag"
        if "code_agent" in action_lower:
            return "code"
        return "other"

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_records": len(self.records),
            "episode_records": len(self._episode_records),
            "overall_accuracy": self.get_overall_accuracy(),
            "accuracy_by_type": {
                action_type: self.get_accuracy_for_type(action_type)
                for action_type in self.records_by_type
            },
        }
