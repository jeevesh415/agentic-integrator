"""Enhanced Worker module — Agent-S3 Worker with World Model integration.

NOTE: Requires gui_agents (Agent-S3) to be installed.
"""


def __getattr__(name):
    if name == "EnhancedWorker":
        from agentic_integrator.enhanced_worker.worker import EnhancedWorker
        return EnhancedWorker
    if name == "PredictionVerifier":
        from agentic_integrator.enhanced_worker.verification import PredictionVerifier
        return PredictionVerifier
    raise AttributeError(f"module has no attribute {name!r}")


__all__ = ["EnhancedWorker", "PredictionVerifier"]
