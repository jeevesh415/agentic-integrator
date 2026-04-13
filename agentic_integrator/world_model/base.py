"""Abstract base class for World Models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from PIL import Image


@dataclass
class SimulationResult:
    """Result of a world model simulation for a single candidate action."""

    action: str
    predicted_state_description: str
    task_progress_score: float  # 0.0 to 1.0 — how much this advances the task
    confidence: float  # 0.0 to 1.0 — how confident the world model is
    safety_score: float = 1.0  # 0.0 to 1.0 — 1.0 = safe, 0.0 = dangerous
    is_irreversible: bool = False
    risk_description: str = ""
    reasoning: str = ""

    @property
    def composite_score(self) -> float:
        """Weighted composite score for ranking candidates."""
        return (
            self.task_progress_score * 0.5
            + self.safety_score * 0.3
            + self.confidence * 0.2
        )


@dataclass
class PlanningResult:
    """Result of the full planning process."""

    selected_action: str
    selected_simulation: SimulationResult
    all_simulations: List[SimulationResult]
    planning_skipped: bool = False  # True if adaptive confidence bypassed simulation
    skip_reason: str = ""


class WorldModel(ABC):
    """
    Abstract base class for world models.

    A world model predicts the outcome of an action given the current state.
    This enables the agent to "imagine" what will happen before committing
    to an action — the key to human-like navigation.
    """

    @abstractmethod
    def simulate(
        self,
        screenshot: Image.Image,
        action: str,
        task_description: str,
        action_history: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> SimulationResult:
        """
        Simulate the outcome of taking an action in the current state.

        Args:
            screenshot: Current screen state as PIL Image
            action: The candidate action to simulate (e.g., pyautogui code)
            task_description: The overall task the agent is trying to complete
            action_history: List of previously taken actions
            context: Additional context (e.g., previous predictions, reflections)

        Returns:
            SimulationResult with predicted outcome and scores
        """
        ...

    @abstractmethod
    def batch_simulate(
        self,
        screenshot: Image.Image,
        actions: List[str],
        task_description: str,
        action_history: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[SimulationResult]:
        """
        Simulate multiple actions in parallel for efficiency.

        Args:
            screenshot: Current screen state
            actions: List of candidate actions to simulate
            task_description: The overall task
            action_history: Previously taken actions
            context: Additional context

        Returns:
            List of SimulationResults, one per action
        """
        ...
