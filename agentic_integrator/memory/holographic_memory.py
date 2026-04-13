"""
Holographic Reduced Representation (HRR) Memory.

Based on Plate (1995) — "Holographic Reduced Representations" and
Kanerva (2009) — "Hyperdimensional Computing".

Key idea: Encode complex structured knowledge (role-filler bindings)
into fixed-size high-dimensional vectors using circular convolution.

Features:
- Content-addressable recall (retrieve by partial cue)
- Compositional structure (bind roles to values)
- Graceful degradation under noise
- Constant-time retrieval regardless of memory size

Example: Encode "the red button is at the top-left of the settings page"
  → bind(RED, BUTTON) ⊛ bind(LOCATION, TOP_LEFT) ⊛ bind(PAGE, SETTINGS)
  → Single holographic vector that can be queried by any component
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("agentic_integrator.memory.holographic")


class HRROperations:
    """Core Holographic Reduced Representation operations."""
    
    def __init__(self, dim: int = 1024):
        self.dim = dim
        self._symbol_cache: Dict[str, np.ndarray] = {}
    
    def random_vector(self) -> np.ndarray:
        """Generate a random unit vector in the HRR space."""
        v = np.random.randn(self.dim).astype(np.float32)
        return v / np.linalg.norm(v)
    
    def get_symbol(self, name: str) -> np.ndarray:
        """Get or create a symbol vector for a named concept."""
        if name not in self._symbol_cache:
            # Deterministic seeding for consistent symbols
            rng = np.random.RandomState(hash(name) % (2**31))
            v = rng.randn(self.dim).astype(np.float32)
            self._symbol_cache[name] = v / np.linalg.norm(v)
        return self._symbol_cache[name]
    
    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Bind two vectors using circular convolution.
        This creates a new vector that is dissimilar to both inputs
        but can be "unbound" using correlation.
        """
        return np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b))).astype(np.float32)
    
    def unbind(self, bound: np.ndarray, cue: np.ndarray) -> np.ndarray:
        """
        Unbind (correlation) — retrieve the other component from a binding.
        unbind(bind(A, B), A) ≈ B
        """
        # Correlation = convolution with the inverse
        cue_inv = self.approximate_inverse(cue)
        return self.bind(bound, cue_inv)
    
    def approximate_inverse(self, v: np.ndarray) -> np.ndarray:
        """Approximate inverse for circular correlation."""
        return np.roll(v[::-1], 1)
    
    def bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        """
        Bundle (superpose) multiple vectors.
        The result is similar to all inputs — used for set union.
        """
        if not vectors:
            return np.zeros(self.dim, dtype=np.float32)
        result = np.sum(vectors, axis=0)
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm
        return result.astype(np.float32)
    
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two HRR vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


@dataclass
class HolographicTrace:
    """A single holographic memory trace."""
    trace_id: str
    holographic_vector: np.ndarray  # The encoded HRR vector
    raw_components: Dict[str, str]  # role→filler mappings for debugging
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    strength: float = 1.0  # memory strength (decays over time)
    context_tag: str = ""
    
    def access(self):
        self.access_count += 1
        self.strength = min(1.0, self.strength + 0.1)


class HolographicMemory:
    """
    Holographic associative memory using HRR.
    
    Stores UI experiences as holographic traces that can be recalled
    by any partial cue. Perfect for:
    - "Where was that red button?" → query with bind(COLOR, RED) ⊛ bind(TYPE, BUTTON)
    - "What happens when I click Submit?" → query with bind(ACTION, CLICK) ⊛ bind(TARGET, SUBMIT)
    - "How did I navigate to Settings last time?" → query with bind(GOAL, SETTINGS)
    
    The beauty: retrieval is O(N) in traces but O(1) per comparison,
    and gracefully handles noisy/partial cues.
    """
    
    def __init__(
        self,
        dim: int = 1024,
        capacity: int = 5000,
        decay_rate: float = 0.005,
        similarity_threshold: float = 0.15,
    ):
        self.hrr = HRROperations(dim=dim)
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.similarity_threshold = similarity_threshold
        
        self.traces: List[HolographicTrace] = []
        self._trace_counter = 0
        
        # Superposed clean-up memory (sum of all traces, used for content-addressable recall)
        self._cleanup_memory = np.zeros(dim, dtype=np.float32)
        
        # Stats
        self._total_stores = 0
        self._total_recalls = 0
        self._successful_recalls = 0
    
    def encode_experience(
        self,
        role_fillers: Dict[str, str],
        context_tag: str = "",
    ) -> str:
        """
        Encode a UI experience as a holographic trace.
        
        Args:
            role_fillers: Dictionary mapping roles to values.
                Example: {
                    "element_type": "button",
                    "color": "red",
                    "location": "top_left",
                    "page": "settings",
                    "action": "click",
                    "outcome": "dialog_opened"
                }
            context_tag: Optional context label
            
        Returns:
            trace_id
        """
        self._total_stores += 1
        
        # Encode each role-filler pair and bind them
        bindings = []
        for role, filler in role_fillers.items():
            role_vec = self.hrr.get_symbol(f"ROLE_{role}")
            filler_vec = self.hrr.get_symbol(f"FILLER_{filler}")
            bindings.append(self.hrr.bind(role_vec, filler_vec))
        
        # Bundle all bindings into one holographic vector
        holographic_vector = self.hrr.bundle(bindings)
        
        # Create trace
        self._trace_counter += 1
        trace_id = f"holo_{self._trace_counter}"
        trace = HolographicTrace(
            trace_id=trace_id,
            holographic_vector=holographic_vector,
            raw_components=role_fillers,
            context_tag=context_tag,
        )
        
        # Capacity management
        if len(self.traces) >= self.capacity:
            self._evict()
        
        self.traces.append(trace)
        
        # Update cleanup memory
        self._cleanup_memory = self._cleanup_memory + holographic_vector
        
        return trace_id
    
    def recall_by_cue(
        self,
        cue_role_fillers: Dict[str, str],
        top_k: int = 5,
        context_filter: Optional[str] = None,
    ) -> List[Tuple[HolographicTrace, float]]:
        """
        Recall memories using a partial cue.
        
        Args:
            cue_role_fillers: Partial role-filler pairs to search for.
                Example: {"color": "red", "element_type": "button"}
                This will find all memories involving red buttons.
            top_k: Number of results
            context_filter: Optional context tag filter
            
        Returns:
            List of (trace, similarity_score) tuples
        """
        self._total_recalls += 1
        
        # Encode the cue
        bindings = []
        for role, filler in cue_role_fillers.items():
            role_vec = self.hrr.get_symbol(f"ROLE_{role}")
            filler_vec = self.hrr.get_symbol(f"FILLER_{filler}")
            bindings.append(self.hrr.bind(role_vec, filler_vec))
        
        cue_vector = self.hrr.bundle(bindings)
        
        # Search through traces
        scored = []
        for trace in self.traces:
            if context_filter and trace.context_tag != context_filter:
                continue
            
            sim = self.hrr.similarity(cue_vector, trace.holographic_vector)
            # Apply memory strength
            effective_sim = sim * trace.strength
            
            if effective_sim > self.similarity_threshold:
                scored.append((trace, effective_sim))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        
        results = scored[:top_k]
        if results:
            self._successful_recalls += 1
            for trace, _ in results:
                trace.access()
        
        return results
    
    def recall_by_vector(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[HolographicTrace, float]]:
        """Recall by raw vector similarity (for integration with other systems)."""
        self._total_recalls += 1
        scored = []
        for trace in self.traces:
            sim = self.hrr.similarity(query_vector, trace.holographic_vector)
            if sim > self.similarity_threshold:
                scored.append((trace, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
    
    def decode_role(self, trace: HolographicTrace, role: str) -> List[Tuple[str, float]]:
        """
        Decode a specific role from a holographic trace.
        
        Given a trace and a role name, find what filler was bound to that role.
        This is the "unbinding" operation.
        """
        role_vec = self.hrr.get_symbol(f"ROLE_{role}")
        decoded = self.hrr.unbind(trace.holographic_vector, role_vec)
        
        # Compare against known filler symbols
        results = []
        for symbol_name, symbol_vec in self.hrr._symbol_cache.items():
            if symbol_name.startswith("FILLER_"):
                sim = self.hrr.similarity(decoded, symbol_vec)
                if sim > self.similarity_threshold:
                    filler_name = symbol_name[7:]  # Remove "FILLER_" prefix
                    results.append((filler_name, sim))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def apply_decay(self) -> None:
        """Apply temporal decay to all traces."""
        now = time.time()
        for trace in self.traces:
            age = now - trace.timestamp
            trace.strength *= np.exp(-self.decay_rate * age / 3600)
    
    def _evict(self):
        """Remove weakest traces."""
        self.traces.sort(key=lambda t: t.strength * (1 + t.access_count * 0.1))
        remove_count = max(1, len(self.traces) // 10)
        removed = self.traces[:remove_count]
        self.traces = self.traces[remove_count:]
        
        # Update cleanup memory
        for trace in removed:
            self._cleanup_memory -= trace.holographic_vector
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_traces": len(self.traces),
            "total_stores": self._total_stores,
            "total_recalls": self._total_recalls,
            "successful_recalls": self._successful_recalls,
            "recall_rate": self._successful_recalls / max(self._total_recalls, 1),
            "unique_symbols": len(self.hrr._symbol_cache),
        }
