"""Agentic Integrator — Agent-S3 + World Model for Human-Like Screen Navigation."""

__version__ = "0.1.0"

# Lazy imports to avoid requiring gui_agents at import time.
# Standalone modules (world_model, memory, utils) work without gui_agents.
# Only AgenticIntegrator and EnhancedWorker require gui_agents at runtime.


def __getattr__(name):
    """Lazy import to defer gui_agents dependency."""
    if name == "AgenticIntegrator":
        from agentic_integrator.integrator import AgenticIntegrator
        return AgenticIntegrator
    if name == "WorldModel":
        from agentic_integrator.world_model.base import WorldModel
        return WorldModel
    if name == "LLMWorldModel":
        from agentic_integrator.world_model.llm_world_model import LLMWorldModel
        return LLMWorldModel
    if name == "WorldModelPlanner":
        from agentic_integrator.world_model.planner import WorldModelPlanner
        return WorldModelPlanner
    raise AttributeError(f"module 'agentic_integrator' has no attribute {name!r}")


__all__ = ["AgenticIntegrator", "WorldModel", "LLMWorldModel", "WorldModelPlanner"]
