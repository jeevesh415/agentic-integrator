"""
Advanced Memory Systems for GUI Agents.

Five specialized memory architectures working together:
- HierarchicalMemory: Episodic → Semantic → Procedural (Tulving / ACT-R)
- HolographicMemory: Distributed associative recall via HRR (Plate 1995)
- HyperDimensionalMemory: HD Computing for one-shot visual pattern matching (Kanerva 2009)
- SparseDistributedMemory: Auto-associative pattern completion from noisy input (Kanerva 1988)
- ExperienceReplay: Prioritized (PER) + Hindsight (HER) replay buffer
- UnifiedMemoryController: Orchestrates all systems + prediction memory
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
    if name == "SparseDistributedMemory":
        from agentic_integrator.memory.sparse_distributed_memory import SparseDistributedMemory
        return SparseDistributedMemory
    if name in ("ExperienceReplay", "RewardWeightedReplay"):
        from agentic_integrator.memory.experience_replay import RewardWeightedReplay
        return RewardWeightedReplay
    if name == "UnifiedMemoryController":
        from agentic_integrator.memory.unified_controller import UnifiedMemoryController
        return UnifiedMemoryController
    raise AttributeError(f"module has no attribute {name!r}")


__all__ = [
    "PredictionMemory",
    "HierarchicalMemory",
    "HolographicMemory",
    "HyperDimensionalMemory",
    "SparseDistributedMemory",
    "ExperienceReplay",        # alias for RewardWeightedReplay
    "RewardWeightedReplay",
    "UnifiedMemoryController",
]
