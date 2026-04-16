"""Enhanced Worker — extends Agent-S3 Worker with world model planning.

Requires gui_agents (Agent-S3) to be installed for live screen execution.
All other components (memory, vision, world model) work without it.
"""

from __future__ import annotations

import logging
import textwrap
from typing import Any, Dict, List, Optional, Tuple

try:
    from gui_agents.s3.agents.grounding import ACI
    from gui_agents.s3.agents.worker import Worker
    from gui_agents.s3.memory.procedural_memory import PROCEDURAL_MEMORY
    from gui_agents.s3.utils.common_utils import call_llm_safe, parse_code_from_string
    _AGENT_S3_AVAILABLE = True
except ImportError:
    _AGENT_S3_AVAILABLE = False
    ACI = None  # type: ignore
    Worker = object  # type: ignore — fallback base class
    call_llm_safe = None
    parse_code_from_string = None

from agentic_integrator.world_model.planner import WorldModelPlanner
from agentic_integrator.enhanced_worker.verification import PredictionVerifier

logger = logging.getLogger("agentic_integrator.enhanced_worker")


CANDIDATE_GENERATION_PROMPT = textwrap.dedent("""\
    You are an expert GUI automation agent. Given the current screenshot and task,
    generate {num_candidates} DIFFERENT candidate actions that could advance the task.

    Task: {task_description}

    Previous actions:
    {action_history}

    For each candidate, provide a different approach or target element.
    Candidates should be meaningfully different (not just minor variations).

    Format your response as a numbered list:
    1. <pyautogui action code>
    2. <pyautogui action code>
    3. <pyautogui action code>

    Each action should be valid Python pyautogui code.
""")


class EnhancedWorker(Worker):
    """
    Enhanced Worker that uses a World Model Planner for action selection.

    Extends Agent-S3's Worker with:
    1. Multi-candidate action generation
    2. World model simulation and ranking
    3. Prediction verification after execution
    4. Adaptive confidence-based simulation skipping
    """

    def __init__(
        self,
        worker_engine_params: Dict,
        grounding_agent: ACI,
        planner: WorldModelPlanner,
        platform: str = "ubuntu",
        max_trajectory_length: int = 8,
        enable_reflection: bool = True,
    ):
        super().__init__(
            worker_engine_params=worker_engine_params,
            grounding_agent=grounding_agent,
            platform=platform,
            max_trajectory_length=max_trajectory_length,
            enable_reflection=enable_reflection,
        )

        self.planner = planner
        self.verifier = PredictionVerifier()
        self.last_planning_result = None
        self._world_model_active = True

    def generate_next_action(
        self, instruction: str, obs: Dict
    ) -> Tuple[Optional[Dict], List[str]]:
        """
        Generate the next action with world model planning.

        This overrides Agent-S3's Worker.generate_next_action to add:
        1. Multi-candidate generation
        2. World model simulation and selection
        3. Post-action verification
        """
        # Step 0: Verify previous prediction (if any)
        if self.last_planning_result and self.turn_count > 0:
            self._verify_previous_prediction(obs)

        # Step 1: Let the base Worker generate its primary action
        # (This includes reflection, screenshot analysis, etc.)
        base_info, base_actions = super().generate_next_action(instruction, obs)

        if not self._world_model_active or not base_actions:
            return base_info, base_actions

        # Step 2: Extract the primary action and generate additional candidates
        primary_action = base_actions[0] if base_actions else ""

        # Check if this is a code agent call (don't simulate those)
        if "call_code_agent" in primary_action:
            logger.debug("Code agent call detected — skipping world model")
            return base_info, base_actions

        # Step 3: Generate alternative candidates
        try:
            candidates = self._generate_candidates(
                instruction, obs, primary_action, num_candidates=self.planner.num_candidates
            )
        except Exception as e:
            logger.warning(f"Candidate generation failed: {e}. Using primary action only.")
            candidates = [primary_action]

        # Step 4: Run world model planner
        try:
            screenshot = obs.get("screenshot")
            if screenshot is None:
                logger.warning("No screenshot in observation — skipping world model")
                return base_info, base_actions

            from PIL import Image
            from io import BytesIO

            if isinstance(screenshot, bytes):
                screenshot = Image.open(BytesIO(screenshot))
            elif isinstance(screenshot, str):
                screenshot = Image.open(screenshot)

            action_history = [
                h.get("action", "") for h in self.worker_history
            ]

            planning_result = self.planner.plan(
                screenshot=screenshot,
                candidates=candidates,
                task_description=instruction,
                action_history=action_history,
                worker_confidence=self._estimate_confidence(base_info),
            )

            self.last_planning_result = planning_result

            # Use the planned action
            selected_action = planning_result.selected_action
            sim = planning_result.selected_simulation

            logger.info(
                f"World model selected action "
                f"(progress={sim.task_progress_score:.2f}, "
                f"safety={sim.safety_score:.2f}): "
                f"{selected_action[:80]}..."
            )

            # Update info with world model data
            if base_info is None:
                base_info = {}
            base_info["world_model"] = {
                "selected_score": sim.composite_score,
                "candidates_evaluated": len(planning_result.all_simulations),
                "planning_skipped": planning_result.planning_skipped,
                "predicted_state": sim.predicted_state_description[:200],
            }

            return base_info, [selected_action]

        except Exception as e:
            logger.error(f"World model planning failed: {e}. Falling back to base action.")
            self.last_planning_result = None
            return base_info, base_actions

    def _generate_candidates(
        self,
        instruction: str,
        obs: Dict,
        primary_action: str,
        num_candidates: int = 3,
    ) -> List[str]:
        """
        Generate alternative candidate actions beyond the primary.

        Uses the LLM to propose N different approaches to the current step.
        """
        if num_candidates <= 1:
            return [primary_action]

        # Always include the primary action as candidate 0
        candidates = [primary_action]

        # Ask the LLM for alternatives
        action_history = "\n".join(
            f"  Step {i + 1}: {h.get('action', '')[:80]}"
            for i, h in enumerate(self.worker_history[-5:])
        )

        prompt = CANDIDATE_GENERATION_PROMPT.format(
            num_candidates=num_candidates - 1,
            task_description=instruction,
            action_history=action_history or "(No previous actions)",
        )

        try:
            response = call_llm_safe(self.generator_agent, prompt, obs.get("screenshot"))
            if response:
                # Parse numbered list
                import re

                lines = response.strip().split("\n")
                for line in lines:
                    match = re.match(r"\d+\.\s*(.+)", line.strip())
                    if match:
                        code = parse_code_from_string(match.group(1))
                        if code and code != primary_action:
                            candidates.append(code)
                    if len(candidates) >= num_candidates:
                        break
        except Exception as e:
            logger.debug(f"Could not generate alternative candidates: {e}")

        return candidates[:num_candidates]

    def _estimate_confidence(self, info: Optional[Dict]) -> float:
        """Estimate confidence from the worker's info dict."""
        if info is None:
            return 0.5

        # Check if reflection indicated uncertainty
        reflection = info.get("reflection", "")
        if any(kw in str(reflection).lower() for kw in ("unsure", "unclear", "not sure", "might", "maybe")):
            return 0.3

        # Check if the action verification passed
        if info.get("verification_passed", True):
            return 0.7

        return 0.5

    def _verify_previous_prediction(self, obs: Dict) -> None:
        """Compare the previous prediction against the actual current state."""
        if not self.last_planning_result:
            return

        try:
            prediction = self.last_planning_result.selected_simulation
            accuracy = self.verifier.verify(
                predicted_description=prediction.predicted_state_description,
                actual_observation=obs,
            )

            self.planner.prediction_memory.record(
                action=prediction.action,
                predicted_score=prediction.task_progress_score,
                actual_accuracy=accuracy,
            )

            logger.debug(
                f"Prediction verification: accuracy={accuracy:.2f} "
                f"(predicted progress={prediction.task_progress_score:.2f})"
            )
        except Exception as e:
            logger.debug(f"Prediction verification failed: {e}")
