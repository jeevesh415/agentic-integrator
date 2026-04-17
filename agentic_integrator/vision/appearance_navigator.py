"""
Appearance Navigator — Coordinate-Free Screen Navigation.

This is the navigation layer that closes the gap in AppearanceBasedGrounding.

The agent navigates the screen using:
1. Visual attention (what region looks like the target?)
2. Flow tracking (is the target still there, or has it moved?)
3. Gradient-following (approach the target smoothly, like gaze direction)

NEVER returns fixed (x, y) coordinates as the navigation output.
Returns: a NavigationSignal — direction + confidence + approach map.

Analogy to human navigation:
- You don't navigate to "pixel 847, 392"
- You navigate toward "the red button I see in the lower right"
- As you move, the target shifts, you recompute, you approach smoothly
- That's what this module does for a GUI agent

This module integrates DenseVisualField + TemporalIdentityTracker into
a unified navigation interface.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agentic_integrator.vision.dense_visual_field import DenseVisualField, SaliencyField
from agentic_integrator.vision.optical_flow_tracker import TemporalIdentityTracker, FlowField

logger = logging.getLogger("agentic_integrator.vision.appearance_navigator")


# ---------------------------------------------------------------------------
# Navigation signal — the output of navigation, not coordinates
# ---------------------------------------------------------------------------

@dataclass
class NavigationSignal:
    """
    What the agent receives as navigation guidance.

    NOT a bounding box. NOT (x, y) coordinates.
    A rich multi-modal signal describing where to go and how confident we are.

    The agent acts on:
    - direction_vector: where to look/move
    - approach_confidence: how sure we are
    - saliency_field: the full probability map (for advanced reasoning)
    - flow_velocity: is the target moving?
    """
    # Direction: unit vector (dy, dx) — normalized direction toward target
    direction_vector: np.ndarray      # shape (2,) — (dy, dx), range [-1, 1]

    # Confidence: how strongly is the target visible?
    approach_confidence: float        # [0, 1]

    # Distance estimate (relative — not pixels, just "near/medium/far")
    relative_distance: str           # "near" | "medium" | "far" | "unknown"

    # Full saliency map for the target
    saliency_field: Optional[SaliencyField] = None

    # If we have a tracked identity for this target
    track_id: Optional[int] = None

    # Flow — is the target moving?
    target_flow_velocity: Optional[np.ndarray] = None  # (dy, dx) per frame

    # Semantic description of what we're navigating to
    target_description: str = ""

    # Source coordinates — only used internally, NOT exposed as the navigation output
    # The external API does not surface raw pixel coordinates as the answer
    _internal_peak_location: Optional[Tuple[int, int]] = None

    @property
    def is_on_target(self) -> bool:
        """True if the agent is already at/near the target."""
        return self.approach_confidence > 0.85 and self.relative_distance == "near"

    @property
    def is_target_moving(self) -> bool:
        """True if the target itself is in motion (e.g., animation, scrolling)."""
        if self.target_flow_velocity is None:
            return False
        return float(np.linalg.norm(self.target_flow_velocity)) > 2.0

    @property
    def should_refocus(self) -> bool:
        """True if the agent should pause and re-evaluate the target."""
        return self.approach_confidence < 0.3 or (self.is_target_moving and not self.is_on_target)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "direction": self.direction_vector.tolist(),
            "confidence": self.approach_confidence,
            "relative_distance": self.relative_distance,
            "track_id": self.track_id,
            "target_moving": self.is_target_moving,
            "on_target": self.is_on_target,
            "should_refocus": self.should_refocus,
            "target": self.target_description,
        }


# ---------------------------------------------------------------------------
# Visual Target — a persistent representation of what we're looking for
# ---------------------------------------------------------------------------

@dataclass
class VisualTarget:
    """
    A target description that the navigator uses to find and approach an element.

    Built from:
    - An embedding (from DenseVisualField, contrastive learner, or CLIP)
    - A color signature (RGB)
    - A natural language description (for logging/reasoning)
    - Optional: a known track_id from previous frames

    The target persists across frames. It is updated as the agent gets closer
    and gets a better view of the element.
    """
    description: str
    embedding: Optional[np.ndarray] = None        # dense visual embedding
    color_signature: Optional[Tuple[int, int, int]] = None  # target RGB
    track_id: Optional[int] = None                # if we have a flow track for it
    last_seen_confidence: float = 0.0
    times_found: int = 0
    embedding_history: List[np.ndarray] = field(default_factory=list)

    def update_embedding(self, new_embedding: np.ndarray, weight: float = 0.3) -> None:
        """Update target embedding with new observation (exponential moving average)."""
        if self.embedding is None:
            self.embedding = new_embedding.copy()
        else:
            self.embedding = (1 - weight) * self.embedding + weight * new_embedding
            norm = np.linalg.norm(self.embedding)
            if norm > 0:
                self.embedding /= norm
        self.embedding_history.append(new_embedding)
        if len(self.embedding_history) > 10:
            self.embedding_history.pop(0)
        self.times_found += 1


# ---------------------------------------------------------------------------
# Appearance Navigator — main interface
# ---------------------------------------------------------------------------

class AppearanceNavigator:
    """
    Navigate the screen by visual appearance, not by coordinates.

    This is the top-level navigation interface. It combines:
    1. Dense visual field (what does every region look like?)
    2. Optical flow tracking (is the target moving? same element as before?)
    3. Appearance-based search (find target by how it looks)
    4. Direction computation (where should I look/move next?)

    Usage:
        nav = AppearanceNavigator()

        # Every frame
        nav.process_frame(screen_frame)

        # Navigate to a target by appearance
        signal = nav.navigate_to_target(
            VisualTarget("blue Send button", color_signature=(70, 130, 180))
        )

        # Act on the signal — NO fixed coordinates
        if signal.is_on_target:
            execute_click()
        elif signal.approach_confidence > 0.5:
            move_attention(signal.direction_vector)
        else:
            scroll_and_recheck()
    """

    def __init__(
        self,
        patch_size: int = 32,
        embed_dim: int = 256,
        max_tracks: int = 150,
        temperature: float = 0.07,
        smoothing_sigma: float = 4.0,
        current_position: Tuple[int, int] = (0, 0),
    ):
        self.dense_field = DenseVisualField(
            patch_size=patch_size,
            embed_dim=embed_dim,
            temperature=temperature,
            smoothing_sigma=smoothing_sigma,
        )
        self.flow_tracker = TemporalIdentityTracker(max_tracks=max_tracks)
        self._current_position = np.array(current_position, dtype=np.float32)
        self._frame_idx = 0
        self._last_flow: Optional[FlowField] = None
        self._screen_h: int = 0
        self._screen_w: int = 0

    def process_frame(self, frame: np.ndarray) -> None:
        """
        Process a new screen frame. Updates dense field + flow tracker.
        Call this every time the screen changes (before navigate_to_target).
        """
        self._screen_h, self._screen_w = frame.shape[:2]

        # Update dense visual field (appearance)
        self.dense_field.update(frame)

        # Update flow tracker (motion / temporal identity)
        _, flow_field = self.flow_tracker.process_frame(
            frame, dense_field=self.dense_field
        )
        self._last_flow = flow_field
        self._frame_idx += 1

    def navigate_to_target(
        self, target: VisualTarget, current_y: Optional[int] = None, current_x: Optional[int] = None
    ) -> NavigationSignal:
        """
        Compute navigation signal toward a visual target.

        Args:
            target: VisualTarget describing what to navigate to
            current_y, current_x: current agent position (screen pixels)

        Returns:
            NavigationSignal — direction + confidence, NOT coordinates
        """
        if current_y is None or current_x is None:
            current_y = int(self._current_position[0])
            current_x = int(self._current_position[1])

        # Step 1: Compute saliency field for the target
        saliency = self._compute_saliency(target)
        if saliency is None:
            return NavigationSignal(
                direction_vector=np.zeros(2),
                approach_confidence=0.0,
                relative_distance="unknown",
                target_description=target.description,
            )

        # Step 2: Check if we have a flow track for this target
        track_id = target.track_id
        flow_velocity = None

        if track_id is not None:
            track = self.flow_tracker.get_track(track_id)
            if track is not None and track.lost_frames == 0:
                flow_velocity = track.velocity.copy()
                # Update target embedding with fresh appearance
                target.update_embedding(track.appearance)
        else:
            # Try to find a track near the saliency peak
            peak_y, peak_x = saliency.peak_location
            nearby_track = self.flow_tracker.find_track_near(peak_y, peak_x, radius=40.0)
            if nearby_track is not None:
                track_id = nearby_track.track_id
                target.track_id = track_id
                flow_velocity = nearby_track.velocity.copy()
                target.update_embedding(nearby_track.appearance)

        # Step 3: Compute direction vector (no fixed coordinates)
        peak_y, peak_x = saliency.peak_location
        dy = peak_y - current_y
        dx = peak_x - current_x
        dist = np.sqrt(dy**2 + dx**2) + 1e-8

        direction = np.array([dy / dist, dx / dist], dtype=np.float32)

        # Step 4: Classify relative distance
        relative_distance = self._classify_distance(dist)

        # Step 5: Compute final confidence
        # Combine: saliency confidence + flow consistency
        base_conf = saliency.peak_confidence
        if flow_velocity is not None:
            # If we predicted target moved, check if saliency peak agrees
            predicted_peak_dy = flow_velocity[0]
            predicted_peak_dx = flow_velocity[1]
            actual_dy = peak_y - current_y
            actual_dx = peak_x - current_x
            # Flow agreement = cosine similarity between predicted motion direction
            # and actual saliency peak direction
            p_dir = np.array([predicted_peak_dy, predicted_peak_dx])
            a_dir = np.array([actual_dy, actual_dx])
            p_norm = np.linalg.norm(p_dir) + 1e-8
            a_norm = np.linalg.norm(a_dir) + 1e-8
            flow_agreement = float(np.dot(p_dir / p_norm, a_dir / a_norm))
            flow_conf_boost = max(0, flow_agreement) * 0.2
            base_conf = min(1.0, base_conf + flow_conf_boost)

        # Update target tracking info
        target.last_seen_confidence = base_conf
        target._internal_peak_location = saliency.peak_location

        return NavigationSignal(
            direction_vector=direction,
            approach_confidence=base_conf,
            relative_distance=relative_distance,
            saliency_field=saliency,
            track_id=track_id,
            target_flow_velocity=flow_velocity,
            target_description=target.description,
            _internal_peak_location=saliency.peak_location,
        )

    def _compute_saliency(self, target: VisualTarget) -> Optional[SaliencyField]:
        """Compute saliency field based on target description."""
        if self.dense_field.current_grid is None:
            return None

        if target.embedding is not None:
            return self.dense_field.attend_to_embedding(
                target.embedding, target.description
            )
        elif target.color_signature is not None:
            return self.dense_field.attend_to_color(
                target.color_signature, target.description
            )
        else:
            logger.warning(f"Target '{target.description}' has no embedding or color — cannot compute saliency")
            return None

    def _classify_distance(self, pixel_dist: float) -> str:
        """Classify pixel distance into semantic categories."""
        if self._screen_h == 0:
            return "unknown"
        screen_diag = np.sqrt(self._screen_h**2 + self._screen_w**2)
        relative = pixel_dist / (screen_diag + 1e-8)
        if relative < 0.05:
            return "near"
        elif relative < 0.25:
            return "medium"
        else:
            return "far"

    def update_position(self, y: int, x: int) -> None:
        """Update the agent's current position on the screen."""
        self._current_position = np.array([y, x], dtype=np.float32)

    def get_flow_field(self) -> Optional[FlowField]:
        """Get the current dense optical flow field."""
        return self._last_flow

    def build_target_from_embedding(
        self, description: str, embedding: np.ndarray
    ) -> VisualTarget:
        """Build a VisualTarget from a known embedding vector."""
        emb = embedding.astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb /= norm
        return VisualTarget(description=description, embedding=emb)

    def build_target_from_color(
        self, description: str, rgb: Tuple[int, int, int]
    ) -> VisualTarget:
        """Build a VisualTarget from a color signature."""
        return VisualTarget(description=description, color_signature=rgb)

    def build_target_from_region(
        self, description: str, frame_y: int, frame_x: int
    ) -> VisualTarget:
        """
        Build a VisualTarget by sampling the dense field at a screen location.
        The target becomes 'whatever looks like what's currently at (y, x)'.
        """
        if self.dense_field.current_grid is None:
            return VisualTarget(description=description)
        emb = self.dense_field.current_grid.embedding_at_pixel(frame_y, frame_x)
        return VisualTarget(description=description, embedding=emb.copy())

    def get_stats(self) -> Dict[str, Any]:
        return {
            "frames_processed": self._frame_idx,
            "dense_field": self.dense_field.get_field_stats(),
            "flow_tracker": self.flow_tracker.get_stats(),
            "current_position": self._current_position.tolist(),
        }
