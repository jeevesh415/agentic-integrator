"""
Advanced Memory Systems for GUI Agents.

Three specialized memory architectures working together:
- HierarchicalMemory: Episodic → Semantic → Procedural (human-like abstraction)
- HolographicMemory: Distributed associative recall via HRR (content-addressable)
- HyperDimensionalMemory: HD Computing for one-shot visual pattern matching
- UnifiedMemoryController: Orchestrates all three + prediction memory
- PredictionMemory: Tracks world model prediction accuracy
"""

from agentic_integrator.memory.prediction_memory import PredictionMemory

# Lazy imports for heavy deps (numpy required at runtime)
def __getattr__(name):
    if name == "HierarchicalMemory":
        from agentic_integrator.memory.hierarchical_memory import HierarchicalMemory
        return HierarchicalMemory
    if name == "HolographicMemory":
        from agentic_integrator.memory.holographic_memory import HolographicMemory
        return HolographicMemory
    if name == "HyperDimensionalMemory":
        from agentic_integrator.memory.hyperdimensional_memory import HyperDimensionalMemory
        return HyperDimensionalMemory
    if name == "UnifiedMemoryController":
        from agentic_integrator.memory.unified_controller import UnifiedMemoryController
        return UnifiedMemoryController
    raise AttributeError(f"module has no attribute {name!r}")


__all__ = [
    "PredictionMemory",
    "HierarchicalMemory",
    "HolographicMemory",
    "HyperDimensionalMemory",
    "UnifiedMemoryController",
]
