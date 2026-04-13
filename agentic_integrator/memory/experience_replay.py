"""
Reward-Weighted Experience Replay for GUI Agents.

Inspired by neuroscience research on memory replay during sleep:
- Prioritized Experience Replay (Schaul et al. 2016)
- Reward-Weighted Regression (Peters & Schaal 2007)  
- Hindsight Experience Replay (Andrychowicz et al. 2017)

Key insight: Not all experiences are equally valuable for learning.
High-reward and surprising experiences should be replayed more often.

Beyond SOTA contributions:
- UI-specific priority scoring (task completion > click success > scroll)
- Curriculum replay: easy → hard action sequences
- Counterfactual replay: "what if I had done X instead?"
- Surprise-based priority: unexpected outcomes get replayed more
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("agentic_integrator.memory.experience_replay")


@dataclass
class Experience:
    """A single experience for replay."""
    experience_id: str
    state_embedding: np.ndarray
    action: str
    reward: float
    next_state_embedding: np.ndarray
    task_context: str
    
    # Replay metadata
    priority: float = 1.0
    td_error: float = 0.0  # Temporal difference error (surprise signal)
    times_replayed: int = 0
    last_replayed: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    # UI-specific
    action_type: str = ""  # click, type, scroll, navigate, etc.
    was_successful: bool = True
    task_completed: bool = False
    
    # Counterfactual
    alternative_actions: List[str] = field(default_factory=list)
    alternative_outcomes: List[float] = field(default_factory=list)


class PrioritySumTree:
    """
    Sum-tree data structure for O(log N) prioritized sampling.
    
    Each leaf stores a priority value. Internal nodes store sums.
    Enables efficient sampling proportional to priority.
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self._data: List[Optional[Experience]] = [None] * capacity
        self._write_pointer = 0
        self._size = 0
    
    @property
    def total_priority(self) -> float:
        return float(self._tree[0])
    
    def add(self, priority: float, experience: Experience) -> None:
        """Add an experience with priority."""
        idx = self._write_pointer + self.capacity - 1
        self._data[self._write_pointer] = experience
        self._update(idx, priority)
        self._write_pointer = (self._write_pointer + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
    
    def update(self, tree_idx: int, priority: float) -> None:
        """Update priority at tree index."""
        self._update(tree_idx, priority)
    
    def sample(self, value: float) -> Tuple[int, float, Experience]:
        """Sample by priority value."""
        idx = self._retrieve(0, value)
        data_idx = idx - self.capacity + 1
        return idx, self._tree[idx], self._data[data_idx]
    
    def _update(self, idx: int, priority: float) -> None:
        change = priority - self._tree[idx]
        self._tree[idx] = priority
        while idx > 0:
            idx = (idx - 1) // 2
            self._tree[idx] += change
    
    def _retrieve(self, idx: int, value: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self._tree):
            return idx
        
        if value <= self._tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self._tree[left])


class RewardWeightedReplay:
    """
    Advanced experience replay for GUI agents.
    
    Combines multiple priority signals:
    1. TD Error (surprise): unexpected outcomes get high priority
    2. Reward magnitude: high-reward actions replayed more
    3. Task completion: experiences leading to task success get boosted
    4. Recency: newer experiences slightly preferred
    5. Difficulty: harder action sequences get curriculum priority
    
    Features:
    - Proportional prioritized sampling (via sum-tree)
    - Importance sampling correction (unbiased updates)
    - Counterfactual replay (imagine alternative actions)
    - Curriculum scheduling (easy → hard replay)
    
    References:
    - Schaul et al. (2016) "Prioritized Experience Replay"
    - Peters & Schaal (2007) "Reward-Weighted Regression"
    - Andrychowicz et al. (2017) "Hindsight Experience Replay"
    """
    
    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,      # priority exponent (0=uniform, 1=full prioritization)
        beta_start: float = 0.4,  # importance sampling start
        beta_end: float = 1.0,    # importance sampling end
        beta_steps: int = 10000,  # steps to anneal beta
        td_error_epsilon: float = 0.01,
        reward_weight: float = 0.3,
        surprise_weight: float = 0.4,
        completion_bonus: float = 2.0,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_steps = beta_steps
        self.td_error_epsilon = td_error_epsilon
        self.reward_weight = reward_weight
        self.surprise_weight = surprise_weight
        self.completion_bonus = completion_bonus
        
        self._tree = PrioritySumTree(capacity)
        self._experience_counter = 0
        self._sample_step = 0
        
        # Difficulty tracking for curriculum
        self._action_type_difficulty: Dict[str, float] = {}
        self._action_type_success_rate: Dict[str, List[bool]] = {}
        
        # Stats
        self._total_stores = 0
        self._total_samples = 0
        self._total_replays = 0
    
    @property
    def beta(self) -> float:
        """Current importance sampling exponent."""
        frac = min(self._sample_step / max(self.beta_steps, 1), 1.0)
        return self.beta_start + frac * (self.beta_end - self.beta_start)
    
    def store(self, experience: Experience) -> None:
        """Store an experience with computed priority."""
        self._total_stores += 1
        self._experience_counter += 1
        
        # Compute priority
        priority = self._compute_priority(experience)
        experience.priority = priority
        
        # Track difficulty
        at = experience.action_type or "unknown"
        if at not in self._action_type_success_rate:
            self._action_type_success_rate[at] = []
        self._action_type_success_rate[at].append(experience.was_successful)
        if len(self._action_type_success_rate[at]) > 100:
            self._action_type_success_rate[at] = self._action_type_success_rate[at][-100:]
        
        # Update difficulty
        success_rate = np.mean(self._action_type_success_rate[at])
        self._action_type_difficulty[at] = 1.0 - success_rate
        
        self._tree.add(priority ** self.alpha, experience)
    
    def sample_batch(
        self,
        batch_size: int = 32,
        curriculum_phase: str = "mixed",  # "easy", "hard", "mixed"
    ) -> Tuple[List[Experience], np.ndarray, List[int]]:
        """
        Sample a prioritized batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            curriculum_phase: "easy" (high success), "hard" (low success), "mixed"
            
        Returns:
            (experiences, importance_weights, tree_indices)
        """
        self._total_samples += 1
        self._sample_step += 1
        
        if self._tree._size == 0:
            return [], np.array([]), []
        
        batch_size = min(batch_size, self._tree._size)
        
        experiences = []
        weights = []
        indices = []
        
        # Segment the priority range
        segment = self._tree.total_priority / batch_size
        
        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            value = np.random.uniform(lo, hi)
            
            try:
                tree_idx, priority, exp = self._tree.sample(value)
                if exp is not None:
                    # Curriculum filtering
                    if curriculum_phase == "easy" and not exp.was_successful:
                        continue
                    if curriculum_phase == "hard" and exp.was_successful:
                        continue
                    
                    # Importance sampling weight
                    prob = priority / max(self._tree.total_priority, 1e-10)
                    is_weight = (1.0 / (self._tree._size * prob + 1e-10)) ** self.beta
                    
                    exp.times_replayed += 1
                    exp.last_replayed = time.time()
                    self._total_replays += 1
                    
                    experiences.append(exp)
                    weights.append(is_weight)
                    indices.append(tree_idx)
            except Exception:
                continue
        
        if not weights:
            return [], np.array([]), []
        
        # Normalize weights
        weights = np.array(weights)
        weights /= weights.max()
        
        return experiences, weights, indices
    
    def update_priorities(
        self,
        indices: List[int],
        td_errors: List[float],
    ) -> None:
        """Update priorities based on new TD errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.td_error_epsilon) ** self.alpha
            self._tree.update(idx, priority)
    
    def generate_counterfactual(
        self,
        experience: Experience,
        alternative_action: str,
        estimated_reward: float,
    ) -> Experience:
        """
        Generate a counterfactual experience.
        
        "What if I had done X instead?" — Hindsight Experience Replay
        """
        counterfactual = Experience(
            experience_id=f"{experience.experience_id}_cf",
            state_embedding=experience.state_embedding.copy(),
            action=alternative_action,
            reward=estimated_reward,
            next_state_embedding=experience.next_state_embedding.copy(),
            task_context=experience.task_context,
            action_type=experience.action_type,
            was_successful=estimated_reward > 0.5,
        )
        
        # Record alternative in original
        experience.alternative_actions.append(alternative_action)
        experience.alternative_outcomes.append(estimated_reward)
        
        return counterfactual
    
    def _compute_priority(self, experience: Experience) -> float:
        """Compute priority score for an experience."""
        priority = self.td_error_epsilon  # base priority
        
        # Surprise signal (TD error)
        priority += abs(experience.td_error) * self.surprise_weight
        
        # Reward magnitude
        priority += abs(experience.reward) * self.reward_weight
        
        # Task completion bonus
        if experience.task_completed:
            priority *= self.completion_bonus
        
        # Difficulty bonus (harder actions get more replay)
        difficulty = self._action_type_difficulty.get(
            experience.action_type or "unknown", 0.5
        )
        priority *= (1 + difficulty * 0.5)
        
        # Failure analysis bonus (learn from mistakes)
        if not experience.was_successful:
            priority *= 1.5
        
        return max(priority, self.td_error_epsilon)
    
    def get_difficulty_stats(self) -> Dict[str, float]:
        """Get difficulty estimates per action type."""
        return dict(self._action_type_difficulty)
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_stores": self._total_stores,
            "total_samples": self._total_samples,
            "total_replays": self._total_replays,
            "buffer_size": self._tree._size,
            "capacity": self.capacity,
            "current_beta": self.beta,
            "action_difficulties": self.get_difficulty_stats(),
        }
