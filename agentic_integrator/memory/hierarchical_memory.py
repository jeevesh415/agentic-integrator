"""
Advanced Hierarchical Memory System.

Three-tier architecture inspired by human cognitive science:
- L1 Episodic Memory: Raw sensory experiences (screenshots, actions, outcomes)
- L2 Semantic Memory: Abstracted knowledge (UI patterns, app behaviors, workflows)
- L3 Procedural Memory: Learned action sequences (macros, skill chains)

Each layer abstracts from the one below, enabling:
- Fast retrieval via semantic similarity
- Generalization across tasks
- Skill transfer between applications
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("agentic_integrator.memory.hierarchical")


@dataclass
class EpisodicRecord:
    """L1: A single episodic memory — one moment in time."""
    timestamp: float
    visual_embedding: np.ndarray  # compressed visual state
    action_taken: str
    outcome_description: str
    task_context: str
    reward_signal: float  # 0-1, how useful this action was
    emotional_valence: float  # -1 to 1 (frustration to satisfaction)
    app_context: str = ""  # which application
    ui_element_hash: str = ""  # hash of the UI element interacted with
    
    @property
    def memory_id(self) -> str:
        return hashlib.md5(
            f"{self.timestamp}:{self.action_taken}:{self.task_context}".encode()
        ).hexdigest()[:12]


@dataclass
class SemanticConcept:
    """L2: An abstracted semantic concept derived from episodic memories."""
    concept_id: str
    name: str
    description: str
    prototype_embedding: np.ndarray  # centroid of related episodic embeddings
    related_episodic_ids: List[str] = field(default_factory=list)
    confidence: float = 0.5
    usage_count: int = 0
    last_accessed: float = 0.0
    category: str = ""  # "ui_pattern", "app_behavior", "workflow_step"
    
    # Generalization features
    invariant_features: Dict[str, Any] = field(default_factory=dict)
    variable_features: List[str] = field(default_factory=list)
    
    def access(self):
        self.usage_count += 1
        self.last_accessed = time.time()


@dataclass
class ProceduralSkill:
    """L3: A learned action sequence — a skill or macro."""
    skill_id: str
    name: str
    description: str
    preconditions: List[str]  # visual/semantic conditions to activate
    action_sequence: List[str]  # ordered actions
    postconditions: List[str]  # expected outcomes
    success_rate: float = 0.0
    execution_count: int = 0
    avg_duration: float = 0.0
    
    # Skill composition
    sub_skills: List[str] = field(default_factory=list)  # nested skill IDs
    parent_skills: List[str] = field(default_factory=list)
    
    # Adaptation
    context_adaptations: Dict[str, List[str]] = field(default_factory=dict)


class HierarchicalMemory:
    """
    Three-tier hierarchical memory system.
    
    Inspired by Tulving's memory taxonomy and ACT-R cognitive architecture.
    
    L1 (Episodic) → raw experiences, high detail, fast decay
    L2 (Semantic) → abstracted patterns, moderate detail, slow decay
    L3 (Procedural) → action skills, low detail, almost permanent
    
    Consolidation flows upward: episodic → semantic → procedural
    Retrieval flows downward: procedural → semantic → episodic
    """
    
    def __init__(
        self,
        episodic_capacity: int = 10000,
        semantic_capacity: int = 2000,
        procedural_capacity: int = 500,
        consolidation_threshold: int = 5,  # episodes before abstraction
        embedding_dim: int = 512,
        decay_rate: float = 0.01,
    ):
        self.episodic_capacity = episodic_capacity
        self.semantic_capacity = semantic_capacity
        self.procedural_capacity = procedural_capacity
        self.consolidation_threshold = consolidation_threshold
        self.embedding_dim = embedding_dim
        self.decay_rate = decay_rate
        
        # Memory stores
        self.episodic: List[EpisodicRecord] = []
        self.semantic: Dict[str, SemanticConcept] = {}
        self.procedural: Dict[str, ProceduralSkill] = {}
        
        # Indices for fast retrieval
        self._episodic_by_task: Dict[str, List[int]] = defaultdict(list)
        self._episodic_by_app: Dict[str, List[int]] = defaultdict(list)
        self._semantic_by_category: Dict[str, List[str]] = defaultdict(list)
        
        # Consolidation tracking
        self._unconsolidated_count: Dict[str, int] = defaultdict(int)
        
        # Stats
        self._total_stores = 0
        self._total_retrievals = 0
        self._consolidations = 0
    
    # ─── L1: Episodic Memory ─────────────────────────────────
    
    def store_episode(self, episode: EpisodicRecord) -> str:
        """Store a raw episodic memory."""
        self._total_stores += 1
        
        # Capacity management: remove oldest low-reward episodes
        if len(self.episodic) >= self.episodic_capacity:
            self._evict_episodic()
        
        idx = len(self.episodic)
        self.episodic.append(episode)
        
        # Index
        self._episodic_by_task[episode.task_context].append(idx)
        if episode.app_context:
            self._episodic_by_app[episode.app_context].append(idx)
        
        # Track for consolidation
        self._unconsolidated_count[episode.task_context] += 1
        
        # Auto-consolidate if threshold reached
        if self._unconsolidated_count[episode.task_context] >= self.consolidation_threshold:
            self._consolidate_to_semantic(episode.task_context)
        
        return episode.memory_id
    
    def recall_episodes(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        task_filter: Optional[str] = None,
        app_filter: Optional[str] = None,
        min_reward: float = 0.0,
    ) -> List[EpisodicRecord]:
        """Recall episodic memories by visual/embedding similarity."""
        self._total_retrievals += 1
        
        candidates = list(range(len(self.episodic)))
        
        # Apply filters
        if task_filter:
            candidates = [i for i in candidates if self.episodic[i].task_context == task_filter]
        if app_filter:
            candidates = [i for i in candidates if self.episodic[i].app_context == app_filter]
        if min_reward > 0:
            candidates = [i for i in candidates if self.episodic[i].reward_signal >= min_reward]
        
        if not candidates:
            return []
        
        # Compute similarities
        scored = []
        for idx in candidates:
            ep = self.episodic[idx]
            sim = self._cosine_similarity(query_embedding, ep.visual_embedding)
            
            # Apply recency decay
            age = time.time() - ep.timestamp
            recency_bonus = np.exp(-self.decay_rate * age / 3600)  # decay per hour
            
            # Reward-weighted score
            score = sim * 0.5 + ep.reward_signal * 0.3 + recency_bonus * 0.2
            scored.append((score, idx))
        
        scored.sort(reverse=True)
        return [self.episodic[idx] for _, idx in scored[:top_k]]
    
    # ─── L2: Semantic Memory ─────────────────────────────────
    
    def store_concept(self, concept: SemanticConcept) -> None:
        """Store or update a semantic concept."""
        if concept.concept_id in self.semantic:
            existing = self.semantic[concept.concept_id]
            # Merge: update prototype embedding via running average
            n = existing.usage_count
            existing.prototype_embedding = (
                existing.prototype_embedding * n + concept.prototype_embedding
            ) / (n + 1)
            existing.usage_count += 1
            existing.related_episodic_ids.extend(concept.related_episodic_ids)
            existing.confidence = min(1.0, existing.confidence + 0.05)
        else:
            if len(self.semantic) >= self.semantic_capacity:
                self._evict_semantic()
            self.semantic[concept.concept_id] = concept
            self._semantic_by_category[concept.category].append(concept.concept_id)
    
    def recall_concepts(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
        category_filter: Optional[str] = None,
    ) -> List[SemanticConcept]:
        """Recall semantic concepts by embedding similarity."""
        self._total_retrievals += 1
        
        candidates = list(self.semantic.values())
        if category_filter:
            cids = self._semantic_by_category.get(category_filter, [])
            candidates = [self.semantic[cid] for cid in cids if cid in self.semantic]
        
        if not candidates:
            return []
        
        scored = []
        for concept in candidates:
            sim = self._cosine_similarity(query_embedding, concept.prototype_embedding)
            # Boost by confidence and usage
            score = sim * 0.6 + concept.confidence * 0.2 + min(concept.usage_count / 100, 0.2)
            scored.append((score, concept))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        for _, concept in scored[:top_k]:
            concept.access()
        
        return [c for _, c in scored[:top_k]]
    
    # ─── L3: Procedural Memory ───────────────────────────────
    
    def store_skill(self, skill: ProceduralSkill) -> None:
        """Store a learned procedural skill."""
        if skill.skill_id in self.procedural:
            existing = self.procedural[skill.skill_id]
            # Update success rate via running average
            n = existing.execution_count
            existing.success_rate = (existing.success_rate * n + skill.success_rate) / (n + 1)
            existing.execution_count += 1
        else:
            if len(self.procedural) >= self.procedural_capacity:
                self._evict_procedural()
            self.procedural[skill.skill_id] = skill
    
    def recall_skills(
        self,
        precondition_descriptions: List[str],
        task_context: str = "",
        top_k: int = 3,
    ) -> List[ProceduralSkill]:
        """Recall procedural skills by matching preconditions."""
        self._total_retrievals += 1
        
        scored = []
        for skill in self.procedural.values():
            # Match preconditions
            match_score = self._precondition_match(
                skill.preconditions, precondition_descriptions
            )
            # Weight by success rate
            score = match_score * 0.6 + skill.success_rate * 0.4
            scored.append((score, skill))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        return [s for _, s in scored[:top_k]]
    
    # ─── Consolidation: Episodic → Semantic → Procedural ────
    
    def _consolidate_to_semantic(self, task_context: str) -> None:
        """Abstract episodic memories into semantic concepts."""
        self._consolidations += 1
        indices = self._episodic_by_task.get(task_context, [])
        if len(indices) < self.consolidation_threshold:
            return
        
        recent_indices = indices[-self.consolidation_threshold:]
        episodes = [self.episodic[i] for i in recent_indices]
        
        # Compute centroid embedding
        embeddings = np.array([ep.visual_embedding for ep in episodes])
        centroid = embeddings.mean(axis=0)
        
        # Extract invariant features (common across episodes)
        actions = [ep.action_taken for ep in episodes]
        common_action = max(set(actions), key=actions.count) if actions else ""
        
        avg_reward = np.mean([ep.reward_signal for ep in episodes])
        
        concept_id = f"concept_{task_context}_{self._consolidations}"
        concept = SemanticConcept(
            concept_id=concept_id,
            name=f"Pattern: {task_context}",
            description=f"Abstracted from {len(episodes)} episodes. Common action: {common_action}",
            prototype_embedding=centroid,
            related_episodic_ids=[ep.memory_id for ep in episodes],
            confidence=min(1.0, avg_reward),
            category="ui_pattern",
            invariant_features={"common_action": common_action, "avg_reward": avg_reward},
        )
        
        self.store_concept(concept)
        self._unconsolidated_count[task_context] = 0
        
        logger.debug(
            f"Consolidated {len(episodes)} episodes → concept '{concept_id}' "
            f"(confidence={concept.confidence:.2f})"
        )
    
    def consolidate_to_procedural(
        self,
        concept_ids: List[str],
        skill_name: str,
        preconditions: List[str],
        postconditions: List[str],
    ) -> Optional[ProceduralSkill]:
        """Consolidate semantic concepts into a procedural skill."""
        concepts = [self.semantic[cid] for cid in concept_ids if cid in self.semantic]
        if not concepts:
            return None
        
        # Extract action sequences from episodic memories linked to these concepts
        action_sequence = []
        for concept in concepts:
            for ep_id in concept.related_episodic_ids[-3:]:
                for ep in self.episodic:
                    if ep.memory_id == ep_id:
                        action_sequence.append(ep.action_taken)
                        break
        
        skill = ProceduralSkill(
            skill_id=f"skill_{skill_name}_{int(time.time())}",
            name=skill_name,
            description=f"Skill from {len(concepts)} concepts, {len(action_sequence)} actions",
            preconditions=preconditions,
            action_sequence=action_sequence,
            postconditions=postconditions,
            success_rate=np.mean([c.confidence for c in concepts]),
        )
        
        self.store_skill(skill)
        return skill
    
    # ─── Eviction ────────────────────────────────────────────
    
    def _evict_episodic(self):
        """Remove lowest-value episodic memories."""
        if not self.episodic:
            return
        scored = []
        now = time.time()
        for i, ep in enumerate(self.episodic):
            age = now - ep.timestamp
            value = ep.reward_signal * 0.6 + np.exp(-self.decay_rate * age / 3600) * 0.4
            scored.append((value, i))
        scored.sort()
        # Remove bottom 10%
        remove_count = max(1, len(self.episodic) // 10)
        remove_indices = set(idx for _, idx in scored[:remove_count])
        self.episodic = [ep for i, ep in enumerate(self.episodic) if i not in remove_indices]
        # Rebuild indices
        self._rebuild_episodic_indices()
    
    def _evict_semantic(self):
        """Remove least-accessed semantic concepts."""
        if not self.semantic:
            return
        concepts = sorted(
            self.semantic.values(),
            key=lambda c: c.last_accessed
        )
        remove_count = max(1, len(concepts) // 10)
        for c in concepts[:remove_count]:
            del self.semantic[c.concept_id]
    
    def _evict_procedural(self):
        """Remove lowest success-rate procedural skills."""
        if not self.procedural:
            return
        skills = sorted(self.procedural.values(), key=lambda s: s.success_rate)
        remove_count = max(1, len(skills) // 10)
        for s in skills[:remove_count]:
            del self.procedural[s.skill_id]
    
    def _rebuild_episodic_indices(self):
        self._episodic_by_task.clear()
        self._episodic_by_app.clear()
        for i, ep in enumerate(self.episodic):
            self._episodic_by_task[ep.task_context].append(i)
            if ep.app_context:
                self._episodic_by_app[ep.app_context].append(i)
    
    # ─── Utilities ───────────────────────────────────────────
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    @staticmethod
    def _precondition_match(skill_preconds: List[str], current_preconds: List[str]) -> float:
        if not skill_preconds:
            return 0.5
        matches = sum(1 for sp in skill_preconds if any(sp.lower() in cp.lower() for cp in current_preconds))
        return matches / len(skill_preconds)
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "episodic_count": len(self.episodic),
            "semantic_count": len(self.semantic),
            "procedural_count": len(self.procedural),
            "total_stores": self._total_stores,
            "total_retrievals": self._total_retrievals,
            "consolidations": self._consolidations,
        }
