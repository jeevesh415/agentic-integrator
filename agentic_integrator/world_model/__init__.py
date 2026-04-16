"""World Model module — simulation and planning for GUI actions."""

from agentic_integrator.world_model.base import WorldModel, SimulationResult, PlanningResult
from agentic_integrator.world_model.safety_gate import SafetyGate, SafetyCheckResult

# LLMWorldModel and WorldModelPlanner have heavier deps (openai/anthropic)
# Import them lazily or directly when needed


def __getattr__(name):
    if name == "LLMWorldModel":
        from agentic_integrator.world_model.llm_world_model import LLMWorldModel
        return LLMWorldModel
    if name in ("WorldModelPlanner", "Planner"):
        from agentic_integrator.world_model.planner import WorldModelPlanner
        return WorldModelPlanner
    raise AttributeError(f"module has no attribute {name!r}")


__all__ = [
    "WorldModel", "SimulationResult", "PlanningResult",
    "SafetyGate", "SafetyCheckResult",
    "LLMWorldModel", "WorldModelPlanner", "Planner",
]
