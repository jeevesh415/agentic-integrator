"""
Sparse Distributed Memory (SDM) — Kanerva (1988).

The foundational theory behind hyper-dimensional computing.
A mathematical model of human long-term memory with:
- Hard addresses: binary address space of 2^N possible locations
- Soft addressing: read/write to all locations within Hamming distance
- Graceful degradation: robust to noise, partial cues, interference
- Auto-associative: patterns complete themselves from partial inputs

This is the ORIGINAL architecture from Pentti Kanerva's 1988 book
"Sparse Distributed Memory" (MIT Press). Our implementation goes beyond
by adding:
- Continuous-valued data vectors (not just binary)
- Adaptive radius based on memory utilization
- Temporal weighting for recency effects
- Confidence scoring for retrieval quality
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("agentic_integrator.memory.sdm")


@dataclass
class SDMConfig:
    """Configuration for Sparse Distributed Memory."""
    address_bits: int = 256       # Dimension of binary address space
    num_hard_locations: int = 1000  # Number of physical storage locations
    data_dim: int = 512           # Dimension of stored data vectors
    access_radius: int = 103      # Hamming distance threshold for activation
    adaptive_radius: bool = True  # Auto-adjust radius based on utilization
    min_activated: int = 3        # Minimum locations to activate per read
    max_activated: int = 100      # Maximum locations to activate per read


class SparseDistributedMemory:
    """
    Kanerva's Sparse Distributed Memory — the foundation of brain-like memory.
    
    How it works:
    1. Storage: Random binary "hard locations" are placed in {0,1}^N space
    2. Write: An address activates all hard locations within Hamming distance R
              → data is added to ALL activated locations (distributed storage)
    3. Read:  An address activates nearby locations → data is the SUM of all
              activated locations → threshold to get binary output
    
    Why it matters for GUI agents:
    - Robust to visual noise (theme changes, resolution shifts)
    - Auto-associative: partial UI screenshots can recall full patterns
    - Interference-resistant: thousands of UI patterns stored without conflict
    - Biologically plausible: models how human memory actually works
    
    This goes BEYOND standard SDM with:
    - Real-valued data (not just binary) — stores embeddings directly
    - Temporal decay — recent memories stronger than old ones
    - Confidence scoring — know when retrieval is reliable vs guessing
    - Adaptive radius — adjusts to memory utilization
    
    References:
    - Kanerva P. (1988) "Sparse Distributed Memory" MIT Press
    - Kanerva P. (2009) "Hyperdimensional Computing" Cognitive Computation
    - Frady et al. (2018) "A Theory of Sequence Indexing and Working Memory in RNNs"
    """
    
    def __init__(self, config: Optional[SDMConfig] = None):
        self.config = config or SDMConfig()
        c = self.config
        
        # Hard location addresses: random binary vectors
        self._hard_locations = np.random.randint(
            0, 2, size=(c.num_hard_locations, c.address_bits), dtype=np.uint8
        )
        
        # Data counters at each hard location (accumulated, not overwritten)
        self._data_counters = np.zeros(
            (c.num_hard_locations, c.data_dim), dtype=np.float64
        )
        
        # Write count per location (for normalization)
        self._write_counts = np.zeros(c.num_hard_locations, dtype=np.int32)
        
        # Temporal weights (for recency)
        self._last_write_time = np.zeros(c.num_hard_locations, dtype=np.float64)
        
        # Current access radius
        self._radius = c.access_radius
        
        # Stats
        self._total_writes = 0
        self._total_reads = 0
        self._utilization = 0.0
    
    def write(self, address: np.ndarray, data: np.ndarray) -> int:
        """
        Write data to SDM at the given address.
        
        Args:
            address: Binary vector {0,1}^N (the "where" — like a visual fingerprint)
            data: Real-valued vector R^D (the "what" — embedding to store)
            
        Returns:
            Number of hard locations activated
        """
        self._total_writes += 1
        
        # Compute Hamming distances to all hard locations
        distances = self._hamming_distances(address)
        
        # Activate locations within radius
        activated = distances <= self._radius
        n_activated = activated.sum()
        
        # Adaptive radius: ensure minimum activation
        if self.config.adaptive_radius and n_activated < self.config.min_activated:
            sorted_distances = np.sort(distances)
            self._radius = int(sorted_distances[self.config.min_activated - 1]) + 1
            activated = distances <= self._radius
            n_activated = activated.sum()
        
        # Write: add data to all activated locations
        now = time.time()
        activated_indices = np.where(activated)[0]
        for idx in activated_indices:
            self._data_counters[idx] += data
            self._write_counts[idx] += 1
            self._last_write_time[idx] = now
        
        # Update utilization
        self._utilization = np.mean(self._write_counts > 0)
        
        return n_activated
    
    def read(
        self,
        address: np.ndarray,
        apply_temporal_weight: bool = True,
    ) -> Tuple[np.ndarray, float]:
        """
        Read data from SDM at the given address.
        
        The read is "soft" — it sums data from all activated locations,
        weighted by how close they are to the query address.
        
        Args:
            address: Binary query vector {0,1}^N
            apply_temporal_weight: Weight by recency
            
        Returns:
            (data_vector, confidence)
            - data_vector: Reconstructed data (real-valued)
            - confidence: 0-1, how reliable the recall is
        """
        self._total_reads += 1
        
        distances = self._hamming_distances(address)
        activated = distances <= self._radius
        n_activated = activated.sum()
        
        if n_activated == 0:
            return np.zeros(self.config.data_dim), 0.0
        
        activated_indices = np.where(activated)[0]
        
        # Compute weights: closer locations contribute more
        weights = np.zeros(n_activated)
        for i, idx in enumerate(activated_indices):
            dist_weight = 1.0 - (distances[idx] / self._radius)
            
            temporal_weight = 1.0
            if apply_temporal_weight and self._last_write_time[idx] > 0:
                age = time.time() - self._last_write_time[idx]
                temporal_weight = np.exp(-0.001 * age / 3600)  # slow decay
            
            write_weight = min(self._write_counts[idx] / 10.0, 1.0)
            
            weights[i] = dist_weight * temporal_weight * write_weight
        
        # Normalize weights
        total_weight = weights.sum()
        if total_weight == 0:
            return np.zeros(self.config.data_dim), 0.0
        weights /= total_weight
        
        # Weighted sum of data at activated locations
        result = np.zeros(self.config.data_dim, dtype=np.float64)
        for i, idx in enumerate(activated_indices):
            if self._write_counts[idx] > 0:
                normalized_data = self._data_counters[idx] / self._write_counts[idx]
                result += weights[i] * normalized_data
        
        # Compute confidence
        confidence = self._compute_confidence(n_activated, weights, activated_indices)
        
        return result.astype(np.float32), confidence
    
    def auto_associate(
        self,
        partial_pattern: np.ndarray,
        iterations: int = 5,
    ) -> Tuple[np.ndarray, float]:
        """
        Auto-associative recall: complete a pattern from a partial cue.
        
        Iteratively reads and re-addresses until convergence.
        This is how SDM acts as a "clean-up memory" — noisy input
        converges to the nearest stored clean pattern.
        
        Args:
            partial_pattern: Noisy/partial binary address
            iterations: Max cleanup iterations
            
        Returns:
            (cleaned_address, confidence)
        """
        current = partial_pattern.copy()
        prev_result = None
        
        for i in range(iterations):
            data, conf = self.read(current)
            
            # Threshold to binary for next iteration address
            if data.shape[0] >= self.config.address_bits:
                current = (data[:self.config.address_bits] > 0).astype(np.uint8)
            
            # Check convergence
            if prev_result is not None:
                change = np.mean(np.abs(data - prev_result))
                if change < 0.01:
                    break
            
            prev_result = data
        
        return data.astype(np.float32), conf
    
    def _hamming_distances(self, address: np.ndarray) -> np.ndarray:
        """Compute Hamming distance from address to all hard locations."""
        return np.sum(
            self._hard_locations != address.reshape(1, -1),
            axis=1
        )
    
    def _compute_confidence(
        self,
        n_activated: int,
        weights: np.ndarray,
        activated_indices: np.ndarray,
    ) -> float:
        """
        Compute confidence score for a read operation.
        
        High confidence when:
        - Many locations activated with data
        - Weights are concentrated (not diffuse)
        - Locations have been written to multiple times
        """
        # Factor 1: Activation coverage
        coverage = min(n_activated / self.config.min_activated, 1.0)
        
        # Factor 2: Weight concentration (entropy-based)
        entropy = -np.sum(weights * np.log2(weights + 1e-10))
        max_entropy = np.log2(max(n_activated, 1))
        concentration = 1.0 - (entropy / max(max_entropy, 1))
        
        # Factor 3: Data richness
        write_counts = self._write_counts[activated_indices]
        richness = min(write_counts.mean() / 5.0, 1.0)
        
        return float(np.clip(
            coverage * 0.3 + concentration * 0.4 + richness * 0.3,
            0, 1
        ))
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_writes": self._total_writes,
            "total_reads": self._total_reads,
            "utilization": float(self._utilization),
            "current_radius": self._radius,
            "locations_used": int(np.sum(self._write_counts > 0)),
            "total_locations": self.config.num_hard_locations,
            "avg_writes_per_location": float(self._write_counts[self._write_counts > 0].mean())
            if np.any(self._write_counts > 0) else 0.0,
        }
