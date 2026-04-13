"""
Unified Memory Controller — Orchestrates all three memory systems.

Routes queries to the appropriate memory tier:
- Hierarchical: For temporal/causal reasoning ("what happened before X?")
- Holographic: For associative recall ("where was the red button?")  
- Hyper-dimensional: For one-shot pattern matching ("find similar screens")

Also handles cross-memory consolidation and memory lifecycle.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agentic_integrator.memory.hierarchical_memory import (
    HierarchicalMemory, EpisodicRecord, SemanticConcept, ProceduralSkill,
)
from agentic_integrator.memory.holographic_memory import (
    HolographicMemory, HolographicTrace,
)
from agentic_integrator.memory.hyperdimensional_memory import (
    HyperDimensionalMemory, HDMemoryItem,
)
from agentic_integrator.memory.prediction_memory import PredictionMemory

logger = logging.getLogger("agentic_integrator.memory.controller")


class UnifiedMemoryController:
    """
    Orchestrates Hierarchical, Holographic, and Hyper-Dimensional memories.
    
    Acts as the "memory cortex" — routes storage and retrieval to the
    right memory system based on the nature of the information.
    
    Storage routing:
    - Raw experiences (screenshot + action + outcome) → Hierarchical (episodic)
    - Structured UI knowledge (element + features) → Holographic (HRR)
    - Visual patterns for matching → Hyper-Dimensional (HDC)
    - Action predictions → Prediction Memory
    
    Retrieval routing:
    - "What happened when I did X?" → Hierarchical (temporal)
    - "Where is the Save button?" → Holographic (associative)
    - "Have I seen this screen before?" → Hyper-Dimensional (similarity)
    - "How well did I predict this?" → Prediction Memory
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        hrr_dim: int = 1024,
        hdc_dim: int = 10000,
        episodic_capacity: int = 10000,
        semantic_capacity: int = 2000,
        procedural_capacity: int = 500,
        holographic_capacity: int = 5000,
        hdc_capacity: int = 10000,
    ):
        # Initialize all memory systems
        self.hierarchical = HierarchicalMemory(
            episodic_capacity=episodic_capacity,
            semantic_capacity=semantic_capacity,
            procedural_capacity=procedural_capacity,
            embedding_dim=embedding_dim,
        )
        
        self.holographic = HolographicMemory(
            dim=hrr_dim,
            capacity=holographic_capacity,
        )
        
        self.hyperdimensional = HyperDimensionalMemory(
            dim=hdc_dim,
            capacity=hdc_capacity,
        )
        
        self.prediction = PredictionMemory()
        
        # Cross-memory links
        self._episode_to_holo: Dict[str, str] = {}  # episodic_id → holographic_trace_id
        self._episode_to_hd: Dict[str, str] = {}    # episodic_id → hd_item_id
        
        self._total_operations = 0
    
    def store_experience(
        self,
        visual_embedding: np.ndarray,
        action: str,
        outcome: str,
        task: str,
        app: str = "",
        reward: float = 0.5,
        ui_elements: Optional[List[Dict[str, str]]] = None,
        role_fillers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """
        Store a complete experience across all memory systems.
        
        This is the primary storage API — one call stores into all three
        memory systems simultaneously with appropriate encodings.
        
        Args:
            visual_embedding: Compressed visual state vector
            action: Action taken
            outcome: What happened
            task: Task context
            app: Application name
            reward: How useful this was (0-1)
            ui_elements: List of UI elements for HDC encoding
            role_fillers: Structured role-filler pairs for HRR encoding
            
        Returns:
            Dictionary of IDs from each memory system
        """
        self._total_operations += 1
        ids = {}
        
        # 1. Hierarchical: Store episodic memory
        episode = EpisodicRecord(
            timestamp=time.time(),
            visual_embedding=visual_embedding,
            action_taken=action,
            outcome_description=outcome,
            task_context=task,
            reward_signal=reward,
            emotional_valence=reward * 2 - 1,  # map 0-1 to -1 to 1
            app_context=app,
        )
        ep_id = self.hierarchical.store_episode(episode)
        ids["episodic"] = ep_id
        
        # 2. Holographic: Store as role-filler binding
        if role_fillers is None:
            role_fillers = {
                "action": action[:50],
                "outcome": outcome[:50],
                "task": task[:30],
            }
            if app:
                role_fillers["app"] = app
        
        holo_id = self.holographic.encode_experience(role_fillers, context_tag=task)
        ids["holographic"] = holo_id
        self._episode_to_holo[ep_id] = holo_id
        
        # 3. Hyper-dimensional: Store visual scene
        if ui_elements:
            hd_id = self.hyperdimensional.encode_visual_scene(
                ui_elements,
                scene_metadata={"task": task, "action": action, "app": app},
            )
            ids["hyperdimensional"] = hd_id
            self._episode_to_hd[ep_id] = hd_id
        
        # 4. Prediction: Record if this was a predicted action
        # (handled separately via store_prediction)
        
        return ids
    
    def recall_by_situation(
        self,
        visual_embedding: np.ndarray,
        task: str,
        query_features: Optional[Dict[str, str]] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Recall memories relevant to the current situation.
        
        Queries all memory systems in parallel and returns combined results.
        
        Returns:
            {
                "episodes": [...],      # Past experiences in similar situations
                "concepts": [...],       # Abstracted patterns
                "skills": [...],         # Known action sequences
                "associations": [...],   # Holographic associations
                "similar_scenes": [...], # HDC visual matches
            }
        """
        self._total_operations += 1
        
        results = {
            "episodes": [],
            "concepts": [],
            "skills": [],
            "associations": [],
            "similar_scenes": [],
        }
        
        # Hierarchical: episodic recall
        results["episodes"] = self.hierarchical.recall_episodes(
            visual_embedding, top_k=top_k, task_filter=task
        )
        
        # Hierarchical: semantic recall
        results["concepts"] = self.hierarchical.recall_concepts(
            visual_embedding, top_k=3
        )
        
        # Hierarchical: procedural recall
        if results["concepts"]:
            preconds = [c.description for c in results["concepts"]]
            results["skills"] = self.hierarchical.recall_skills(preconds, task, top_k=3)
        
        # Holographic: associative recall
        if query_features:
            holo_results = self.holographic.recall_by_cue(query_features, top_k=top_k)
            results["associations"] = [
                {"trace_id": t.trace_id, "components": t.raw_components, "score": s}
                for t, s in holo_results
            ]
        
        # Hyper-dimensional: visual similarity
        if query_features:
            hd_results = self.hyperdimensional.query(query_features, top_k=top_k)
            results["similar_scenes"] = [
                {"item_id": item.item_id, "metadata": item.metadata, "score": s}
                for item, s in hd_results
            ]
        
        return results
    
    def store_prediction(
        self,
        action: str,
        predicted_score: float,
        actual_accuracy: float,
    ) -> None:
        """Store a prediction result in prediction memory."""
        self.prediction.record(action, predicted_score, actual_accuracy)
    
    def get_prediction_guidance(self, action_type: str) -> Dict[str, Any]:
        """Get guidance from prediction memory."""
        return {
            "should_simulate": self.prediction.should_simulate(action_type),
            "accuracy_for_type": self.prediction.get_accuracy_for_type(action_type),
            "overall_accuracy": self.prediction.get_overall_accuracy(),
        }
    
    def reset_episode(self) -> None:
        """Reset episode-level state across all memory systems."""
        self.prediction.reset_episode()
        self._episode_to_holo.clear()
        self._episode_to_hd.clear()
    
    def apply_decay(self) -> None:
        """Apply temporal decay across all memory systems."""
        self.holographic.apply_decay()
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_operations": self._total_operations,
            "hierarchical": self.hierarchical.get_stats(),
            "holographic": self.holographic.get_stats(),
            "hyperdimensional": self.hyperdimensional.get_stats(),
            "prediction": self.prediction.get_stats(),
            "cross_links": {
                "episode_to_holo": len(self._episode_to_holo),
                "episode_to_hd": len(self._episode_to_hd),
            },
        }
