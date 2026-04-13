"""Main AgenticIntegrator — orchestrates Agent-S3 with World Model planning."""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from gui_agents.s3.agents.agent_s import AgentS3
from gui_agents.s3.agents.grounding import ACI
from gui_agents.s3.core.engine import LMMEngineOpenAI, LMMEngineAnthropic

from agentic_integrator.world_model.llm_world_model import LLMWorldModel
from agentic_integrator.world_model.planner import WorldModelPlanner
from agentic_integrator.world_model.safety_gate import SafetyGate
from agentic_integrator.enhanced_worker.worker import EnhancedWorker
from agentic_integrator.memory.prediction_memory import PredictionMemory

logger = logging.getLogger("agentic_integrator")


@dataclass
class IntegratorConfig:
    """Configuration for the Agentic Integrator."""

    # LLM provider settings
    provider: str = "openai"
    model: str = "gpt-4o"
    model_url: str = ""
    model_api_key: str = ""
    model_temperature: Optional[float] = None

    # Grounding model settings
    ground_provider: str = "huggingface"
    ground_url: str = "http://localhost:8080"
    ground_model: str = "ui-tars-1.5-7b"
    grounding_width: int = 1920
    grounding_height: int = 1080

    # World model settings
    world_model_candidates: int = 3
    world_model_provider: str = ""  # defaults to main provider
    world_model_model: str = ""  # defaults to main model
    safety_gate_enabled: bool = True
    adaptive_confidence: bool = True

    # Agent-S3 settings
    max_trajectory_length: int = 8
    enable_reflection: bool = True
    enable_local_env: bool = False

    # Platform
    platform: str = field(default_factory=lambda: platform.system().lower())


class AgenticIntegrator:
    """
    Orchestrates Agent-S3 with a World Model for human-like screen navigation.

    The agent imagines outcomes before acting:
    1. Generates N candidate actions per step
    2. Simulates each via the world model
    3. Ranks by task progress + safety
    4. Executes the best action
    5. Verifies prediction vs reality
    """

    def __init__(self, config: Optional[IntegratorConfig] = None, **kwargs):
        if config is None:
            config = IntegratorConfig(**kwargs)
        self.config = config

        # Build engine params
        self.engine_params = self._build_engine_params()

        # Initialize grounding agent (ACI)
        self.grounding_agent = self._build_grounding_agent()

        # Initialize prediction memory
        self.prediction_memory = PredictionMemory()

        # Initialize world model
        wm_engine_params = self._build_world_model_engine_params()
        self.world_model = LLMWorldModel(engine_params=wm_engine_params)

        # Initialize safety gate
        self.safety_gate = SafetyGate(
            engine_params=wm_engine_params,
            enabled=config.safety_gate_enabled,
        )

        # Initialize world model planner
        self.planner = WorldModelPlanner(
            world_model=self.world_model,
            safety_gate=self.safety_gate,
            prediction_memory=self.prediction_memory,
            num_candidates=config.world_model_candidates,
            adaptive_confidence=config.adaptive_confidence,
        )

        # Initialize enhanced worker (replaces Agent-S3's default Worker)
        self.worker = EnhancedWorker(
            worker_engine_params=self.engine_params,
            grounding_agent=self.grounding_agent,
            planner=self.planner,
            platform=config.platform,
            max_trajectory_length=config.max_trajectory_length,
            enable_reflection=config.enable_reflection,
        )

        # Build Agent-S3 with enhanced worker
        self.agent = AgentS3(
            worker_engine_params=self.engine_params,
            grounding_agent=self.grounding_agent,
            platform=config.platform,
            max_trajectory_length=config.max_trajectory_length,
            enable_reflection=config.enable_reflection,
        )
        # Override the executor with our enhanced worker
        self.agent.executor = self.worker

        logger.info(
            f"AgenticIntegrator initialized: "
            f"model={config.model}, "
            f"candidates={config.world_model_candidates}, "
            f"safety_gate={config.safety_gate_enabled}, "
            f"adaptive={config.adaptive_confidence}"
        )

    def _build_engine_params(self) -> Dict[str, Any]:
        """Build engine parameters for the main LLM."""
        params = {
            "engine_type": self.config.provider,
            "model": self.config.model,
        }
        if self.config.model_url:
            params["base_url"] = self.config.model_url
        if self.config.model_api_key:
            params["api_key"] = self.config.model_api_key
        if self.config.model_temperature is not None:
            params["temperature"] = self.config.model_temperature
        return params

    def _build_world_model_engine_params(self) -> Dict[str, Any]:
        """Build engine parameters for the world model LLM (can be separate)."""
        params = dict(self.engine_params)
        if self.config.world_model_provider:
            params["engine_type"] = self.config.world_model_provider
        if self.config.world_model_model:
            params["model"] = self.config.world_model_model
        return params

    def _build_grounding_agent(self) -> ACI:
        """Build the ACI grounding agent."""
        return ACI()

    def predict(self, instruction: str, observation: Dict) -> Tuple[Dict, List[str]]:
        """
        Generate the next action with world model planning.

        Args:
            instruction: Natural language task description
            observation: Current UI state (screenshot, etc.)

        Returns:
            Tuple of (info dict, list of action strings)
        """
        return self.agent.predict(instruction, observation)

    def reset(self) -> None:
        """Reset agent state for a new task."""
        self.agent.reset()
        # Re-inject our enhanced worker
        self.agent.executor = self.worker
        self.prediction_memory.reset_episode()
        logger.info("Agent reset for new task")

    def get_stats(self) -> Dict[str, Any]:
        """Get world model performance statistics."""
        return {
            "prediction_memory": self.prediction_memory.get_stats(),
            "planner": self.planner.get_stats(),
            "safety_gate": self.safety_gate.get_stats(),
        }
