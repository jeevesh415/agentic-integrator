"""LLM-based World Model — uses LLMs as implicit world models (WebDreamer-style)."""

from __future__ import annotations

import base64
import json
import logging
import re
from io import BytesIO
from typing import Any, Dict, List, Optional

from PIL import Image

from agentic_integrator.world_model.base import SimulationResult, WorldModel
from agentic_integrator.utils.prompts import (
    WORLD_MODEL_SYSTEM_PROMPT,
    WORLD_MODEL_SIMULATE_PROMPT,
    WORLD_MODEL_BATCH_SIMULATE_PROMPT,
)

logger = logging.getLogger("agentic_integrator.world_model")


def _image_to_base64(image: Image.Image, max_size: int = 1024) -> str:
    """Convert PIL Image to base64 string, resizing if needed."""
    if max(image.size) > max_size:
        image = image.copy()
        image.thumbnail((max_size, max_size), Image.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _parse_simulation_response(response: str, action: str) -> SimulationResult:
    """Parse LLM response into a SimulationResult."""
    # Extract JSON from response (handle markdown code blocks)
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if json_match:
        data = json.loads(json_match.group(1))
    else:
        # Try parsing the whole response as JSON
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            # Fallback: extract scores from text
            logger.warning("Could not parse world model response as JSON, using defaults")
            data = {}

    return SimulationResult(
        action=action,
        predicted_state_description=data.get("predicted_state", "Unknown outcome"),
        task_progress_score=float(data.get("task_progress_score", 0.5)),
        confidence=float(data.get("confidence", 0.5)),
        safety_score=float(data.get("safety_score", 1.0)),
        is_irreversible=bool(data.get("is_irreversible", False)),
        risk_description=data.get("risk_description", ""),
        reasoning=data.get("reasoning", ""),
    )


class LLMWorldModel(WorldModel):
    """
    LLM-based World Model — the LLM imagines action outcomes.

    Inspired by WebDreamer: "Is Your LLM Secretly a World Model of the Internet?"
    The LLM encodes vast knowledge about how GUIs work, making it an effective
    implicit world model for screen navigation.
    """

    def __init__(self, engine_params: Dict[str, Any]):
        self.engine_params = engine_params
        self._total_simulations = 0
        self._init_client()

    def _init_client(self):
        """Initialize the LLM client based on engine params."""
        engine_type = self.engine_params.get("engine_type", "openai")

        if engine_type == "openai":
            from openai import OpenAI

            self.client = OpenAI(
                api_key=self.engine_params.get("api_key"),
                base_url=self.engine_params.get("base_url") or None,
            )
        elif engine_type == "anthropic":
            from anthropic import Anthropic

            self.client = Anthropic(
                api_key=self.engine_params.get("api_key"),
            )
        else:
            # Fallback to OpenAI-compatible
            from openai import OpenAI

            self.client = OpenAI(
                api_key=self.engine_params.get("api_key"),
                base_url=self.engine_params.get("base_url") or None,
            )

        self.engine_type = engine_type
        self.model = self.engine_params.get("model", "gpt-4o")

    def _call_llm(
        self, system_prompt: str, user_content: list, temperature: float = 0.3
    ) -> str:
        """Call the LLM with vision support."""
        if self.engine_type == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}],
            )
            return response.content[0].text
        else:
            # OpenAI-compatible
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            return response.choices[0].message.content

    def simulate(
        self,
        screenshot: Image.Image,
        action: str,
        task_description: str,
        action_history: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> SimulationResult:
        """Simulate a single action outcome using the LLM."""
        self._total_simulations += 1

        img_b64 = _image_to_base64(screenshot)
        history_str = "\n".join(
            f"  Step {i + 1}: {a}" for i, a in enumerate(action_history[-5:])
        )

        prompt = WORLD_MODEL_SIMULATE_PROMPT.format(
            task_description=task_description,
            action_history=history_str or "(No previous actions)",
            candidate_action=action,
        )

        user_content = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}", "detail": "high"}},
            {"type": "text", "text": prompt},
        ]

        try:
            response = self._call_llm(WORLD_MODEL_SYSTEM_PROMPT, user_content)
            result = _parse_simulation_response(response, action)
            logger.debug(
                f"Simulation for '{action[:50]}...': "
                f"progress={result.task_progress_score:.2f}, "
                f"safety={result.safety_score:.2f}, "
                f"confidence={result.confidence:.2f}"
            )
            return result
        except Exception as e:
            logger.error(f"World model simulation failed: {e}")
            return SimulationResult(
                action=action,
                predicted_state_description="Simulation failed",
                task_progress_score=0.5,
                confidence=0.3,
                safety_score=0.8,
                reasoning=f"Error: {str(e)}",
            )

    def batch_simulate(
        self,
        screenshot: Image.Image,
        actions: List[str],
        task_description: str,
        action_history: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[SimulationResult]:
        """
        Simulate multiple actions. Uses a single LLM call for efficiency
        when possible, falling back to individual calls.
        """
        if len(actions) <= 1:
            return [
                self.simulate(screenshot, a, task_description, action_history, context)
                for a in actions
            ]

        self._total_simulations += len(actions)
        img_b64 = _image_to_base64(screenshot)
        history_str = "\n".join(
            f"  Step {i + 1}: {a}" for i, a in enumerate(action_history[-5:])
        )

        actions_str = "\n".join(
            f"  Candidate {i + 1}: {a}" for i, a in enumerate(actions)
        )

        prompt = WORLD_MODEL_BATCH_SIMULATE_PROMPT.format(
            task_description=task_description,
            action_history=history_str or "(No previous actions)",
            candidate_actions=actions_str,
            num_candidates=len(actions),
        )

        user_content = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}", "detail": "high"}},
            {"type": "text", "text": prompt},
        ]

        try:
            response = self._call_llm(WORLD_MODEL_SYSTEM_PROMPT, user_content)
            # Parse array of results
            json_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", response, re.DOTALL)
            if json_match:
                results_data = json.loads(json_match.group(1))
            else:
                results_data = json.loads(response)

            results = []
            for i, data in enumerate(results_data):
                action = actions[i] if i < len(actions) else data.get("action", "")
                results.append(
                    SimulationResult(
                        action=action,
                        predicted_state_description=data.get("predicted_state", ""),
                        task_progress_score=float(data.get("task_progress_score", 0.5)),
                        confidence=float(data.get("confidence", 0.5)),
                        safety_score=float(data.get("safety_score", 1.0)),
                        is_irreversible=bool(data.get("is_irreversible", False)),
                        risk_description=data.get("risk_description", ""),
                        reasoning=data.get("reasoning", ""),
                    )
                )
            return results

        except Exception as e:
            logger.warning(f"Batch simulation failed, falling back to individual: {e}")
            return [
                self.simulate(screenshot, a, task_description, action_history, context)
                for a in actions
            ]

    def get_stats(self) -> Dict[str, int]:
        return {"total_simulations": self._total_simulations}
