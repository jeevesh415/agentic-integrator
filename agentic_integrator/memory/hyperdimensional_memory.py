"""
Hyper-Dimensional Computing (HDC) Memory.

Based on Kanerva's theory of computing with high-dimensional vectors (10,000D).
Unlike traditional computing, HDC uses:
- Binary/bipolar vectors (+1/-1) of very high dimensionality
- Element-wise operations (XOR, majority) instead of floating-point math
- Inherent noise tolerance and one-shot learning
- Mathematically guaranteed near-orthogonality of random vectors

For GUI agents, this enables:
- One-shot visual pattern learning (see once, recognize forever)
- Robust matching under visual noise (theme changes, resolution shifts)
- Compositional scene encoding (bind objects + attributes + positions)
- Temporal sequence encoding (action chains with order preserved)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("agentic_integrator.memory.hyperdimensional")


# ─── HDC Vector Operations ──────────────────────────────────


class HDCOps:
    """
    Core Hyper-Dimensional Computing operations.
    
    All vectors are bipolar (+1/-1) in R^D where D is typically 10,000.
    """
    
    def __init__(self, dim: int = 10000):
        self.dim = dim
        self._codebook: Dict[str, np.ndarray] = {}
        self._level_vectors: Optional[np.ndarray] = None
    
    def random_hv(self) -> np.ndarray:
        """Generate a random bipolar hypervector."""
        return np.where(
            np.random.random(self.dim) > 0.5,
            np.int8(1), np.int8(-1)
        )
    
    def get_atom(self, name: str) -> np.ndarray:
        """Get or create an atomic hypervector for a concept."""
        if name not in self._codebook:
            self._codebook[name] = self.random_hv()
        return self._codebook[name]
    
    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Bind two hypervectors (element-wise XOR for bipolar = multiplication).
        Result is dissimilar to both inputs.
        bind(A, A) = identity vector (all +1s)
        """
        return (a * b).astype(np.int8)
    
    def bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        """
        Bundle (superpose) multiple hypervectors via majority vote.
        Result is similar to all inputs.
        """
        if not vectors:
            return np.ones(self.dim, dtype=np.int8)
        
        stacked = np.array(vectors)
        summed = stacked.sum(axis=0)
        
        # Majority vote: tie-breaking with random
        result = np.where(summed > 0, np.int8(1), np.int8(-1))
        # Handle ties (sum == 0)
        ties = summed == 0
        if ties.any():
            result[ties] = np.where(
                np.random.random(ties.sum()) > 0.5,
                np.int8(1), np.int8(-1)
            )
        return result
    
    def permute(self, v: np.ndarray, shift: int = 1) -> np.ndarray:
        """
        Permute (cyclic shift) — used for sequence encoding.
        permute(v, n) creates a version of v that is nearly orthogonal
        to v but can be "unpermuted" to recover v.
        """
        return np.roll(v, shift).astype(np.int8)
    
    def inverse_permute(self, v: np.ndarray, shift: int = 1) -> np.ndarray:
        """Inverse permutation."""
        return np.roll(v, -shift).astype(np.int8)
    
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Cosine similarity (normalized Hamming for bipolar).
        Random vectors have expected similarity ≈ 0.
        """
        return float(np.dot(a.astype(np.float32), b.astype(np.float32)) / self.dim)
    
    def encode_sequence(self, items: List[np.ndarray]) -> np.ndarray:
        """
        Encode an ordered sequence preserving position information.
        Uses permutation to encode order:
          seq = Σ permute(item_i, N-i)
        """
        if not items:
            return np.ones(self.dim, dtype=np.int8)
        
        n = len(items)
        shifted = [self.permute(item, n - i) for i, item in enumerate(items)]
        return self.bundle(shifted)
    
    def decode_position(self, sequence_hv: np.ndarray, position: int, total_length: int) -> np.ndarray:
        """Extract the element at a given position from a sequence encoding."""
        shift = total_length - position
        return self.inverse_permute(sequence_hv, shift)
    
    def encode_scalar(self, value: float, min_val: float = 0.0, max_val: float = 1.0, levels: int = 100) -> np.ndarray:
        """
        Encode a continuous scalar value as a hypervector.
        Uses level hypervectors with thermometer encoding for similarity preservation:
        similar values → similar hypervectors.
        """
        if self._level_vectors is None or len(self._level_vectors) != levels:
            self._generate_level_vectors(levels)
        
        # Quantize value to nearest level
        normalized = (value - min_val) / (max_val - min_val + 1e-10)
        level_idx = int(np.clip(normalized * (levels - 1), 0, levels - 1))
        return self._level_vectors[level_idx]
    
    def _generate_level_vectors(self, levels: int):
        """Generate level vectors with preserved similarity structure."""
        self._level_vectors = np.zeros((levels, self.dim), dtype=np.int8)
        # Start with random base
        self._level_vectors[0] = self.random_hv()
        
        # Each level flips a fraction of bits from the previous
        flip_per_level = self.dim // (2 * levels)
        for i in range(1, levels):
            self._level_vectors[i] = self._level_vectors[i - 1].copy()
            flip_indices = np.random.choice(self.dim, flip_per_level, replace=False)
            self._level_vectors[i][flip_indices] *= -1


# ─── Hyper-Dimensional Memory System ────────────────────────


@dataclass
class HDMemoryItem:
    """A single item in hyper-dimensional memory."""
    item_id: str
    hypervector: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    category: str = ""


class HyperDimensionalMemory:
    """
    Hyper-Dimensional Computing memory for GUI agents.
    
    Stores visual scenes, action patterns, and UI layouts as
    high-dimensional bipolar vectors. Enables:
    
    1. **One-shot Visual Learning**: See a UI pattern once → encode as HD vector
       → recognize it forever, even with visual changes (themes, sizes)
    
    2. **Compositional Scene Memory**: Encode entire screens as composition of
       element bindings: bind(TYPE, button) ⊛ bind(COLOR, blue) ⊛ bind(LABEL, "OK")
    
    3. **Temporal Action Memory**: Encode action sequences preserving order
       → recall "what did I do after clicking X?"
    
    4. **Analog Pattern Matching**: Similar UI states → similar vectors
       → robust retrieval under noise
    
    Properties:
    - D = 10,000 dimensions (standard for HDC)
    - Bipolar (+1/-1) vectors
    - Near-orthogonal random vectors (P(sim > 0.1) ≈ 10^-8 for D=10000)
    - One-shot learning (no training needed)
    """
    
    def __init__(
        self,
        dim: int = 10000,
        capacity: int = 10000,
        match_threshold: float = 0.15,
    ):
        self.ops = HDCOps(dim=dim)
        self.capacity = capacity
        self.match_threshold = match_threshold
        
        # Associative memory banks (by category)
        self._banks: Dict[str, List[HDMemoryItem]] = {
            "visual_scene": [],    # Encoded screen states
            "action_pattern": [],  # Encoded action sequences
            "ui_element": [],      # Individual UI element encodings
            "workflow": [],        # Multi-step workflow encodings
            "general": [],
        }
        
        self._item_counter = 0
        self._total_stores = 0
        self._total_queries = 0
        self._total_matches = 0
    
    def encode_visual_scene(
        self,
        elements: List[Dict[str, str]],
        scene_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Encode a visual scene (screen state) as a hyper-dimensional vector.
        
        Args:
            elements: List of UI elements, each described as role-filler pairs.
                Example: [
                    {"type": "button", "label": "OK", "color": "blue", "region": "center"},
                    {"type": "textfield", "label": "Search", "region": "top"},
                    {"type": "menu", "label": "File", "region": "top_left"},
                ]
            scene_metadata: Optional metadata (app name, page, etc.)
            
        Returns:
            item_id
        """
        self._total_stores += 1
        
        # Encode each element as a bound product of its features
        element_vectors = []
        for elem in elements:
            feature_bindings = []
            for role, filler in elem.items():
                role_hv = self.ops.get_atom(f"role_{role}")
                filler_hv = self.ops.get_atom(f"val_{filler}")
                feature_bindings.append(self.ops.bind(role_hv, filler_hv))
            
            # Bundle features of this element
            if feature_bindings:
                element_vectors.append(self.ops.bundle(feature_bindings))
        
        # Bundle all elements into scene vector
        scene_vector = self.ops.bundle(element_vectors) if element_vectors else self.ops.random_hv()
        
        return self._store_item(scene_vector, "visual_scene", scene_metadata or {})
    
    def encode_action_sequence(
        self,
        actions: List[str],
        task_context: str = "",
    ) -> str:
        """
        Encode an ordered action sequence preserving position.
        
        Args:
            actions: List of action descriptions in order
            task_context: What task this sequence belongs to
            
        Returns:
            item_id
        """
        self._total_stores += 1
        
        action_vectors = [self.ops.get_atom(f"action_{a}") for a in actions]
        sequence_vector = self.ops.encode_sequence(action_vectors)
        
        metadata = {"actions": actions, "task": task_context, "length": len(actions)}
        return self._store_item(sequence_vector, "action_pattern", metadata)
    
    def encode_ui_element(
        self,
        element_features: Dict[str, str],
    ) -> str:
        """Encode a single UI element."""
        self._total_stores += 1
        
        feature_bindings = []
        for role, filler in element_features.items():
            role_hv = self.ops.get_atom(f"role_{role}")
            filler_hv = self.ops.get_atom(f"val_{filler}")
            feature_bindings.append(self.ops.bind(role_hv, filler_hv))
        
        element_vector = self.ops.bundle(feature_bindings) if feature_bindings else self.ops.random_hv()
        return self._store_item(element_vector, "ui_element", element_features)
    
    def query(
        self,
        query_features: Dict[str, str],
        category: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Tuple[HDMemoryItem, float]]:
        """
        Query memory with partial feature specification.
        
        Args:
            query_features: Partial role-filler pairs to search for
            category: Optional category filter
            top_k: Number of results
            
        Returns:
            List of (item, similarity) tuples
        """
        self._total_queries += 1
        
        # Encode query
        feature_bindings = []
        for role, filler in query_features.items():
            role_hv = self.ops.get_atom(f"role_{role}")
            filler_hv = self.ops.get_atom(f"val_{filler}")
            feature_bindings.append(self.ops.bind(role_hv, filler_hv))
        
        query_vector = self.ops.bundle(feature_bindings) if feature_bindings else self.ops.random_hv()
        return self.query_by_vector(query_vector, category, top_k)
    
    def query_by_vector(
        self,
        query_vector: np.ndarray,
        category: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Tuple[HDMemoryItem, float]]:
        """Query memory with a raw hypervector."""
        self._total_queries += 1
        
        categories = [category] if category else list(self._banks.keys())
        
        scored = []
        for cat in categories:
            for item in self._banks.get(cat, []):
                sim = self.ops.similarity(query_vector, item.hypervector)
                if sim > self.match_threshold:
                    scored.append((item, sim))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        results = scored[:top_k]
        
        if results:
            self._total_matches += 1
            for item, _ in results:
                item.access_count += 1
        
        return results
    
    def find_similar_scenes(
        self, scene_item_id: str, top_k: int = 5
    ) -> List[Tuple[HDMemoryItem, float]]:
        """Find scenes similar to a stored scene."""
        for item in self._banks["visual_scene"]:
            if item.item_id == scene_item_id:
                return self.query_by_vector(item.hypervector, "visual_scene", top_k)
        return []
    
    def _store_item(
        self,
        hypervector: np.ndarray,
        category: str,
        metadata: Dict[str, Any],
    ) -> str:
        """Store an item in the appropriate memory bank."""
        self._item_counter += 1
        item_id = f"hd_{category}_{self._item_counter}"
        
        item = HDMemoryItem(
            item_id=item_id,
            hypervector=hypervector,
            metadata=metadata,
            category=category,
        )
        
        if category not in self._banks:
            self._banks[category] = []
        
        # Capacity management per bank
        bank = self._banks[category]
        max_per_bank = self.capacity // len(self._banks)
        if len(bank) >= max_per_bank:
            # Remove least accessed
            bank.sort(key=lambda x: x.access_count)
            bank.pop(0)
        
        bank.append(item)
        return item_id
    
    def get_stats(self) -> Dict[str, Any]:
        bank_sizes = {name: len(items) for name, items in self._banks.items()}
        return {
            "total_items": sum(bank_sizes.values()),
            "bank_sizes": bank_sizes,
            "total_stores": self._total_stores,
            "total_queries": self._total_queries,
            "total_matches": self._total_matches,
            "match_rate": self._total_matches / max(self._total_queries, 1),
            "codebook_size": len(self.ops._codebook),
        }
