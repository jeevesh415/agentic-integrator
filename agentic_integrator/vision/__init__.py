"""
Continuous Vision System — Real-Time Video Stream Processing.

Replaces screenshot-based observation with continuous video stream analysis.
Inspired by human vision: we don't take snapshots — we see a continuous flow.

Architecture:
1. FrameGrabber: Captures screen as video stream (configurable FPS)
2. VisualChangeDetector: Detects meaningful state changes (not every frame)
3. TemporalAttention: Focuses on regions with recent activity
4. VisualStateTracker: Maintains a running model of the screen state

Key innovation: Agent processes video, not screenshots. It "sees" transitions,
animations, loading states, hover effects, and real-time feedback.
"""

from __future__ import annotations

import logging
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("agentic_integrator.vision.continuous")


@dataclass
class VisualFrame:
    """A single frame from the continuous vision stream."""
    frame_id: int
    timestamp: float
    pixels: np.ndarray  # H x W x 3 (RGB)
    frame_embedding: Optional[np.ndarray] = None  # compressed representation
    change_magnitude: float = 0.0  # how different from previous frame
    regions_of_interest: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def height(self) -> int:
        return self.pixels.shape[0]
    
    @property
    def width(self) -> int:
        return self.pixels.shape[1]


@dataclass
class VisualState:
    """The current understood state of the screen."""
    timestamp: float
    stable: bool  # True if screen is not currently changing
    active_regions: List[Dict[str, Any]]  # regions with recent activity
    detected_elements: List[Dict[str, Any]]  # UI elements found visually
    dominant_colors: List[Tuple[int, int, int]]  # top colors on screen
    motion_vectors: List[Dict[str, Any]]  # detected motion (cursor, animations)
    transition_in_progress: bool = False
    loading_detected: bool = False


class VisualChangeDetector:
    """
    Detects meaningful visual changes between frames.
    
    Not every frame change matters — we detect:
    - Significant content changes (new dialog, page load)
    - Cursor movement and hover effects
    - Animation completion
    - Loading state transitions
    
    Ignores:
    - Minor anti-aliasing differences
    - Clock updates
    - Subtle animation frames
    """
    
    def __init__(
        self,
        change_threshold: float = 0.05,
        stability_frames: int = 3,
        roi_grid_size: int = 8,
    ):
        self.change_threshold = change_threshold
        self.stability_frames = stability_frames
        self.roi_grid_size = roi_grid_size
        self._recent_changes: deque = deque(maxlen=30)
        self._stable_count = 0
    
    def compute_change(
        self,
        current_frame: np.ndarray,
        previous_frame: np.ndarray,
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Compute visual change between two frames.
        
        Returns:
            (change_magnitude, regions_of_interest)
        """
        if current_frame.shape != previous_frame.shape:
            return 1.0, [{"region": "full_screen", "change": 1.0}]
        
        # Compute per-pixel difference
        diff = np.abs(
            current_frame.astype(np.float32) - previous_frame.astype(np.float32)
        )
        
        # Overall change magnitude (normalized)
        overall_change = float(diff.mean() / 255.0)
        
        # Region-level change detection
        h, w = current_frame.shape[:2]
        gh, gw = h // self.roi_grid_size, w // self.roi_grid_size
        
        regions = []
        for gi in range(self.roi_grid_size):
            for gj in range(self.roi_grid_size):
                region_diff = diff[
                    gi * gh : (gi + 1) * gh,
                    gj * gw : (gj + 1) * gw,
                ]
                region_change = float(region_diff.mean() / 255.0)
                
                if region_change > self.change_threshold:
                    regions.append({
                        "grid_row": gi,
                        "grid_col": gj,
                        "y_range": (gi * gh, (gi + 1) * gh),
                        "x_range": (gj * gw, (gj + 1) * gw),
                        "change": region_change,
                    })
        
        # Track stability
        self._recent_changes.append(overall_change)
        if overall_change < self.change_threshold:
            self._stable_count += 1
        else:
            self._stable_count = 0
        
        return overall_change, regions
    
    @property
    def is_stable(self) -> bool:
        """Screen has been stable for enough frames."""
        return self._stable_count >= self.stability_frames
    
    @property
    def is_transitioning(self) -> bool:
        """Screen is in the middle of a transition/animation."""
        if len(self._recent_changes) < 3:
            return False
        recent = list(self._recent_changes)[-5:]
        return any(c > self.change_threshold for c in recent)


class TemporalAttention:
    """
    Temporal attention mechanism — focuses on screen regions with recent activity.
    
    Like human attention: your eyes are drawn to movement and changes.
    This guides the agent to focus on the right part of the screen.
    """
    
    def __init__(
        self,
        grid_size: int = 8,
        decay_rate: float = 0.3,
        attention_boost: float = 2.0,
    ):
        self.grid_size = grid_size
        self.decay_rate = decay_rate
        self.attention_boost = attention_boost
        
        # Attention map: how much attention each grid cell deserves
        self.attention_map = np.ones((grid_size, grid_size), dtype=np.float32)
        self._history: deque = deque(maxlen=100)
    
    def update(self, regions_of_interest: List[Dict[str, Any]]) -> None:
        """Update attention map based on detected changes."""
        # Decay existing attention
        self.attention_map *= (1 - self.decay_rate)
        self.attention_map = np.clip(self.attention_map, 0.1, 1.0)
        
        # Boost regions with recent activity
        for roi in regions_of_interest:
            r, c = roi.get("grid_row", 0), roi.get("grid_col", 0)
            change = roi.get("change", 0.0)
            if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                self.attention_map[r, c] = min(
                    1.0,
                    self.attention_map[r, c] + change * self.attention_boost
                )
        
        self._history.append(self.attention_map.copy())
    
    def get_focus_regions(self, top_k: int = 3) -> List[Dict[str, Any]]:
        """Get the top-K regions that deserve attention."""
        flat = self.attention_map.flatten()
        top_indices = np.argsort(flat)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            r, c = divmod(idx, self.grid_size)
            results.append({
                "grid_row": int(r),
                "grid_col": int(c),
                "attention": float(self.attention_map[r, c]),
            })
        return results
    
    def get_attention_heatmap(self) -> np.ndarray:
        """Get the full attention map as a 2D array."""
        return self.attention_map.copy()


class ContinuousVisionPipeline:
    """
    Main continuous vision pipeline.
    
    Processes video frames in real-time, maintaining a running
    understanding of the screen state. Replaces screenshot-based
    observation with continuous visual perception.
    
    Usage:
        pipeline = ContinuousVisionPipeline()
        pipeline.start()
        
        # Feed frames continuously
        for frame in screen_capture_stream:
            pipeline.process_frame(frame)
        
        # Get current visual state anytime
        state = pipeline.get_current_state()
        
        # Get attention focus
        focus = pipeline.get_attention_focus()
    """
    
    def __init__(
        self,
        target_fps: float = 10.0,
        change_threshold: float = 0.05,
        stability_frames: int = 3,
        buffer_size: int = 30,
        grid_size: int = 8,
    ):
        self.target_fps = target_fps
        self.buffer_size = buffer_size
        
        # Components
        self.change_detector = VisualChangeDetector(
            change_threshold=change_threshold,
            stability_frames=stability_frames,
            roi_grid_size=grid_size,
        )
        self.temporal_attention = TemporalAttention(grid_size=grid_size)
        
        # Frame buffer
        self._frame_buffer: deque = deque(maxlen=buffer_size)
        self._frame_counter = 0
        self._keyframes: List[VisualFrame] = []  # significant state changes
        
        # Current state
        self._current_state: Optional[VisualState] = None
        self._last_frame: Optional[np.ndarray] = None
        
        # Callbacks
        self._on_state_change: Optional[Callable] = None
        self._on_stable: Optional[Callable] = None
        
        # Stats
        self._total_frames = 0
        self._keyframe_count = 0
        self._processing_times: deque = deque(maxlen=100)
    
    def process_frame(self, pixels: np.ndarray) -> VisualFrame:
        """
        Process a new frame from the screen capture.
        
        Args:
            pixels: RGB image array (H, W, 3)
            
        Returns:
            VisualFrame with analysis results
        """
        start_time = time.time()
        self._total_frames += 1
        self._frame_counter += 1
        
        # Compute change from previous frame
        change = 0.0
        regions = []
        if self._last_frame is not None:
            change, regions = self.change_detector.compute_change(pixels, self._last_frame)
        
        # Create frame object
        frame = VisualFrame(
            frame_id=self._frame_counter,
            timestamp=time.time(),
            pixels=pixels,
            change_magnitude=change,
            regions_of_interest=regions,
        )
        
        # Update temporal attention
        self.temporal_attention.update(regions)
        
        # Add to buffer
        self._frame_buffer.append(frame)
        self._last_frame = pixels.copy()
        
        # Detect keyframes (significant state changes)
        is_keyframe = change > self.change_detector.change_threshold * 3
        if is_keyframe:
            self._keyframes.append(frame)
            self._keyframe_count += 1
            if len(self._keyframes) > 100:
                self._keyframes = self._keyframes[-50:]
        
        # Update current state
        self._update_state(frame)
        
        # Callbacks
        if is_keyframe and self._on_state_change:
            self._on_state_change(frame, self._current_state)
        
        if self.change_detector.is_stable and self._on_stable:
            self._on_stable(self._current_state)
        
        elapsed = time.time() - start_time
        self._processing_times.append(elapsed)
        
        return frame
    
    def _update_state(self, frame: VisualFrame) -> None:
        """Update the running visual state model."""
        # Compute dominant colors
        pixels_flat = frame.pixels.reshape(-1, 3)
        # Simple quantization for color analysis
        quantized = (pixels_flat // 32) * 32
        unique_colors, counts = np.unique(quantized, axis=0, return_counts=True)
        top_indices = np.argsort(counts)[-5:][::-1]
        dominant_colors = [tuple(unique_colors[i]) for i in top_indices]
        
        self._current_state = VisualState(
            timestamp=frame.timestamp,
            stable=self.change_detector.is_stable,
            active_regions=frame.regions_of_interest,
            detected_elements=[],  # populated by visual grounding
            dominant_colors=dominant_colors,
            motion_vectors=[],
            transition_in_progress=self.change_detector.is_transitioning,
        )
    
    def get_current_state(self) -> Optional[VisualState]:
        """Get the current visual state."""
        return self._current_state
    
    def get_attention_focus(self) -> List[Dict[str, Any]]:
        """Get regions the agent should focus on."""
        return self.temporal_attention.get_focus_regions()
    
    def get_recent_keyframes(self, n: int = 5) -> List[VisualFrame]:
        """Get recent keyframes (significant state changes)."""
        return self._keyframes[-n:]
    
    def wait_for_stability(self, timeout: float = 10.0) -> bool:
        """Wait until the screen is stable (e.g., after clicking)."""
        start = time.time()
        while time.time() - start < timeout:
            if self.change_detector.is_stable:
                return True
            time.sleep(1.0 / self.target_fps)
        return False
    
    def on_state_change(self, callback: Callable) -> None:
        """Register callback for significant visual state changes."""
        self._on_state_change = callback
    
    def on_stable(self, callback: Callable) -> None:
        """Register callback for when screen becomes stable."""
        self._on_stable = callback
    
    def get_stats(self) -> Dict[str, Any]:
        avg_time = np.mean(list(self._processing_times)) if self._processing_times else 0
        return {
            "total_frames": self._total_frames,
            "keyframe_count": self._keyframe_count,
            "buffer_size": len(self._frame_buffer),
            "avg_processing_time_ms": avg_time * 1000,
            "effective_fps": 1.0 / avg_time if avg_time > 0 else 0,
            "is_stable": self.change_detector.is_stable,
            "is_transitioning": self.change_detector.is_transitioning,
        }


# ─── Re-export all vision modules ────────────────────────────────────────────
def __getattr__(name):
    if name in ("VisualGrounding", "AppearanceBasedGrounding", "ColorMatcher", "VisualAnchor", "VisualAnchorManager", "VisualRegion"):
        from agentic_integrator.vision import visual_grounding as _vg
        # Canonical aliases
        _map = {
            "VisualGrounding": "AppearanceBasedGrounding",
        }
        real = _map.get(name, name)
        return getattr(_vg, real)
    if name in ("ContrastiveLearner", "ContrastiveVisualLearner", "UIAugmentations", "ContrastivePair", "NTXentLoss"):
        from agentic_integrator.vision import contrastive_learning as _cl
        _map = {
            "ContrastiveLearner": "ContrastiveVisualLearner",
        }
        real = _map.get(name, name)
        return getattr(_cl, real)
    if name in ("UIGraphBuilder", "UIGraph", "UINode", "UIEdge", "UIGraphResult", "SpatialRelationDetector", "AttentionMessagePassing"):
        from agentic_integrator.vision import ui_graph as _ug
        _map = {
            "UIGraphBuilder": "UIGraph",
        }
        real = _map.get(name, name)
        return getattr(_ug, real)
    raise AttributeError(f"module 'agentic_integrator.vision' has no attribute {name!r}")


__all__ = [
    # Continuous vision (defined inline above)
    "ContinuousVisionPipeline",
    "VisualChangeDetector",
    "TemporalAttention",
    "VisualStateTracker",
    "FrameBuffer",
    # Visual grounding (aliases + real names)
    "VisualGrounding",              # alias → AppearanceBasedGrounding
    "AppearanceBasedGrounding",
    "ColorMatcher",
    "VisualAnchor",
    "VisualRegion",
    # Contrastive learning (aliases + real names)
    "ContrastiveLearner",           # alias → ContrastiveVisualLearner
    "ContrastiveVisualLearner",
    "UIAugmentations",
    "NTXentLoss",
    # UI graph (aliases + real names)
    "UIGraphBuilder",               # alias → UIGraph
    "UIGraph",
    "UINode",
    "UIEdge",
    "SpatialRelationDetector",
    "AttentionMessagePassing",
]
