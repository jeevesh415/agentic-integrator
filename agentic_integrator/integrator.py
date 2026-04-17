"""
Main AgenticIntegrator — orchestrates Agent-S3 with World Model planning.

Agent-S3 (gui_agents) is an optional runtime dependency. When installed,
the full pipeline runs: LLM → World Model → Safety Gate → Agent-S3 executor.
When not installed, the standalone components (memory, vision, world model)
remain fully usable — you just won't have the GUI execution layer.

Install Agent-S3:
    pip install gui-agents   # or: pip install -e path/to/Agent-S
"""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Agent-S3 is an optional runtime dependency (requires local screen access).
# All other components are standalone.
try:
    from gui_agents.s3.agents.agent_s import AgentS3
    from gui_agents.s3.agents.grounding import ACI
    from gui_agents.s3.core.engine import LMMEngineOpenAI, LMMEngineAnthropic
    _AGENT_S3_AVAILABLE = True
except ImportError:
    _AGENT_S3_AVAILABLE = False
    AgentS3 = None  # type: ignore
    ACI = None  # type: ignore

from agentic_integrator.world_model.llm_world_model import LLMWorldModel
from agentic_integrator.world_model.planner import WorldModelPlanner
from agentic_integrator.world_model.safety_gate import SafetyGate
from agentic_integrator.memory.prediction_memory import PredictionMemory
from agentic_integrator.memory.unified_controller import UnifiedMemoryController
from agentic_integrator.memory.sparse_distributed_memory import SparseDistributedMemory
from agentic_integrator.memory.experience_replay import RewardWeightedReplay as ExperienceReplay
from agentic_integrator.memory.prediction_memory import PredictionMemory

# Vision — frontier layer
from agentic_integrator.vision.dense_visual_field import DenseVisualField
from agentic_integrator.vision.optical_flow_tracker import TemporalIdentityTracker
from agentic_integrator.vision.appearance_navigator import AppearanceNavigator, VisualTarget

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
    world_model_model: str = ""     # defaults to main model
    safety_gate_enabled: bool = True
    adaptive_confidence: bool = True

    # Memory settings
    embedding_dim: int = 128
    episodic_capacity: int = 10_000
    hdc_dim: int = 10_000

    # Vision — frontier settings
    dense_patch_size: int = 32
    dense_embed_dim: int = 256
    nav_smoothing_sigma: float = 4.0
    flow_max_tracks: int = 150

    # Agent-S3 settings (only used when gui_agents is installed)
    max_trajectory_length: int = 8
    enable_reflection: bool = True
    enable_local_env: bool = False

    # Platform
    platform: str = field(default_factory=lambda: platform.system().lower())


class AgenticIntegrator:
    """
    Full agentic pipeline: World Model + Memory + Vision + (optional) GUI execution.

    When gui_agents (Agent-S3) is installed:
        LLM → World Model Planning → Safety Gate → Agent-S3 Worker → GUI action

    When gui_agents is NOT installed:
        All memory, vision, and world model components are still fully usable.
        Only the live screen execution layer is unavailable.

    Architecture:
        1. Five memory systems (Hierarchical, Holographic, HDC, SDM, Experience Replay)
        2. World model simulates actions before execution
        3. Safety gate blocks dangerous/irreversible actions
        4. (Optional) Agent-S3 executes on real screens
    """

    def __init__(self, config: Optional[IntegratorConfig] = None, **kwargs):
        if config is None:
            config = IntegratorConfig(**kwargs)
        self.config = config

        # ── Memory systems (orchestrated via UnifiedMemoryController) ──────
        # UnifiedMemoryController creates and owns all memory subsystems.
        # Access them via controller.hierarchical, .holographic, .hdc, etc.
        self.memory = UnifiedMemoryController(
            embedding_dim=config.embedding_dim,
            hdc_dim=config.hdc_dim,
            episodic_capacity=config.episodic_capacity,
        )
        # Convenience references to internal systems
        self.hierarchical_memory = self.memory.hierarchical
        self.holographic_memory  = self.memory.holographic
        self.hdc_memory          = self.memory.hyperdimensional
        self.prediction_memory   = self.memory.prediction
        self.sdm                 = SparseDistributedMemory()
        self.experience_replay   = ExperienceReplay()

        # ── Frontier Vision Layer ────────────────────────────────────────────
        # Coordinate-free screen understanding — no bounding boxes
        self.dense_field = DenseVisualField(
            patch_size=config.dense_patch_size,
            embed_dim=config.dense_embed_dim,
        )
        self.flow_tracker = TemporalIdentityTracker(
            max_tracks=config.flow_max_tracks,
        )
        self.navigator = AppearanceNavigator(
            patch_size=config.dense_patch_size,
            embed_dim=config.dense_embed_dim,
            max_tracks=config.flow_max_tracks,
            smoothing_sigma=config.nav_smoothing_sigma,
        )

        # ── World model ──────────────────────────────────────────────────────
        engine_params = self._build_engine_params()
        wm_engine_params = self._build_world_model_engine_params(engine_params)

        self.world_model = LLMWorldModel(engine_params=wm_engine_params)
        self.safety_gate = SafetyGate(
            engine_params=wm_engine_params,
            enabled=config.safety_gate_enabled,
        )
        self.planner = WorldModelPlanner(
            world_model=self.world_model,
            safety_gate=self.safety_gate,
            prediction_memory=self.prediction_memory,
            num_candidates=config.world_model_candidates,
            adaptive_confidence=config.adaptive_confidence,
        )

        # ── Agent-S3 execution layer (optional) ─────────────────────────────
        if _AGENT_S3_AVAILABLE:
            from agentic_integrator.enhanced_worker.worker import EnhancedWorker
            self._engine_params = engine_params
            self.grounding_agent = ACI()
            self.worker = EnhancedWorker(
                worker_engine_params=engine_params,
                grounding_agent=self.grounding_agent,
                planner=self.planner,
                platform=config.platform,
                max_trajectory_length=config.max_trajectory_length,
                enable_reflection=config.enable_reflection,
            )
            self.agent = AgentS3(
                worker_engine_params=engine_params,
                grounding_agent=self.grounding_agent,
                platform=config.platform,
                max_trajectory_length=config.max_trajectory_length,
                enable_reflection=config.enable_reflection,
            )
            self.agent.executor = self.worker
            logger.info(f"AgenticIntegrator initialized with Agent-S3 execution layer")
        else:
            self.grounding_agent = None
            self.worker = None
            self.agent = None
            logger.info(
                "AgenticIntegrator initialized (standalone mode — gui_agents not installed). "
                "Memory, vision, and world model components are available. "
                "Install gui-agents for live screen execution."
            )

        logger.info(
            f"Config: model={config.model}, "
            f"candidates={config.world_model_candidates}, "
            f"safety_gate={config.safety_gate_enabled}, "
            f"embedding_dim={config.embedding_dim}, "
            f"hdc_dim={config.hdc_dim}"
        )

    def _build_engine_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {
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

    def _build_world_model_engine_params(self, base: Dict[str, Any]) -> Dict[str, Any]:
        params = dict(base)
        if self.config.world_model_provider:
            params["engine_type"] = self.config.world_model_provider
        if self.config.world_model_model:
            params["model"] = self.config.world_model_model
        return params

    def predict(self, instruction: str, observation: Dict) -> Tuple[Dict, List[str]]:
        """
        Generate the next action with world model planning.
        Requires gui_agents to be installed for live execution.
        """
        if self.agent is None:
            raise RuntimeError(
                "Agent-S3 not available. Install gui-agents: pip install gui-agents"
            )
        return self.agent.predict(instruction, observation)

    def reset(self) -> None:
        """Reset agent state for a new task."""
        if self.agent is not None:
            self.agent.reset()
            self.agent.executor = self.worker
        self.prediction_memory.reset_episode()
        logger.info("Agent reset for new task")

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics across all components."""
        return {
            "prediction_memory": self.prediction_memory.get_stats(),
            "planner": self.planner.get_stats(),
            "safety_gate": self.safety_gate.get_stats(),
            "navigator": self.navigator.get_stats(),
            "agent_s3_available": _AGENT_S3_AVAILABLE,
        }
