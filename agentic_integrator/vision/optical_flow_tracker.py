"""
Optical Flow Tracker — Frame-to-Frame Element Tracking Without Re-Detection.

This module solves the temporal identity problem:
"Is this the same element as before?" answered WITHOUT bounding box re-detection.

Instead of detecting elements each frame and matching by IoU (intersection over
union of rectangles), we track feature points using optical flow. A point tracked
continuously across frames maintains its identity through motion alone.

This is how biological vision works. The eye doesn't re-identify every object
from scratch each frame. It tracks motion continuously and associates identity
through temporal continuity.

Theoretical grounding:
- Lucas-Kanade sparse optical flow (Lucas & Kanade, 1981)
- Farneback dense optical flow (Farneback, 2003)
- Temporal consistency via Kalman filter prediction (Kalman, 1960)
- SORT tracking concept — but NO bounding boxes required (Bewley et al., 2016)

Key innovation over SORT/DeepSORT:
- No detection step. No rectangles.
- Identity = continuous optical flow path.
- If flow breaks → identity uncertainty (not re-detection from scratch).
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("agentic_integrator.vision.optical_flow")


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class TrackedPoint:
    """
    A single tracked feature point with persistent identity.

    This is NOT a bounding box. It is a continuously tracked pixel location
    with a motion history and confidence score.
    """
    track_id: int
    current_position: np.ndarray   # (y, x) float
    velocity: np.ndarray           # (dy, dx) — current motion vector
    acceleration: np.ndarray       # (d²y, d²x) — change in velocity
    appearance: np.ndarray         # appearance embedding at this point (from DenseVisualField)
    confidence: float              # tracking confidence [0, 1]
    age_frames: int = 0            # frames since first detected
    lost_frames: int = 0           # consecutive frames where flow tracking failed
    position_history: deque = field(default_factory=lambda: deque(maxlen=30))

    def predict_next_position(self) -> np.ndarray:
        """
        Predict position in the next frame using kinematic model.
        Uses velocity + acceleration (second-order linear motion).
        """
        return self.current_position + self.velocity + 0.5 * self.acceleration

    def update_position(self, new_pos: np.ndarray, new_conf: float) -> None:
        """Update with new tracked position. Smooth velocity and acceleration."""
        self.position_history.append(self.current_position.copy())

        prev_velocity = self.velocity.copy()
        alpha = 0.7  # exponential moving average smoothing
        new_velocity = new_pos - self.current_position
        self.velocity = alpha * new_velocity + (1 - alpha) * self.velocity
        self.acceleration = self.velocity - prev_velocity

        self.current_position = new_pos
        self.confidence = 0.9 * self.confidence + 0.1 * new_conf
        self.age_frames += 1
        self.lost_frames = 0

    @property
    def is_stable(self) -> bool:
        """Point is stably tracked (low velocity, high confidence)."""
        speed = float(np.linalg.norm(self.velocity))
        return speed < 3.0 and self.confidence > 0.6

    @property
    def is_moving(self) -> bool:
        """Point is actively in motion."""
        return float(np.linalg.norm(self.velocity)) > 5.0


@dataclass
class FlowField:
    """
    Dense optical flow field — every pixel has a motion vector.
    This is a continuous vector field, not a discrete set of detections.

    Shape: (h, w, 2) — flow[y, x] = (dy, dx) displacement
    """
    flow: np.ndarray              # (h, w, 2) — float32 motion vectors
    magnitude: np.ndarray         # (h, w) — scalar motion magnitude
    frame_h: int
    frame_w: int
    timestamp: float = field(default_factory=time.time)
    frame_delta_ms: float = 0.0   # time since last frame in ms

    def motion_at(self, y: int, x: int) -> Tuple[float, float]:
        """Get motion vector at pixel (y, x)."""
        y = min(max(y, 0), self.frame_h - 1)
        x = min(max(x, 0), self.frame_w - 1)
        return float(self.flow[y, x, 0]), float(self.flow[y, x, 1])

    def moving_regions_mask(self, threshold: float = 2.0) -> np.ndarray:
        """Binary mask of significantly moving pixels — no rectangles, just a field."""
        return (self.magnitude > threshold).astype(np.uint8)

    def dominant_motion(self) -> Tuple[float, float]:
        """
        Overall dominant motion direction (e.g., page scroll).
        Returns (mean_dy, mean_dx) weighted by magnitude.
        """
        weights = self.magnitude
        total = weights.sum() + 1e-8
        mean_dy = (self.flow[:, :, 0] * weights).sum() / total
        mean_dx = (self.flow[:, :, 1] * weights).sum() / total
        return float(mean_dy), float(mean_dx)

    def motion_energy(self) -> float:
        """Total kinetic energy of the frame — how much is happening."""
        return float(self.magnitude.mean())


# ---------------------------------------------------------------------------
# Lucas-Kanade Sparse Flow Tracker (no OpenCV required — pure numpy)
# ---------------------------------------------------------------------------

class LKFlowTracker:
    """
    Sparse optical flow tracker using Lucas-Kanade pyramid method.

    Pure numpy implementation — no OpenCV dependency.
    Tracks sparse feature points across frames.

    When cv2 is available, uses the optimized C++ implementation for speed.
    Falls back to pure numpy for portability.

    Reference: Lucas-Kanade (1981); Tomasi-Kanade feature detector (1991)
    """

    def __init__(
        self,
        max_corners: int = 200,
        quality_level: float = 0.01,
        min_distance: float = 10.0,
        window_size: int = 21,
        max_pyramid_levels: int = 3,
    ):
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.window_size = window_size
        self.max_pyramid_levels = max_pyramid_levels
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_points: Optional[np.ndarray] = None
        self._use_cv2 = self._check_cv2()

    @staticmethod
    def _check_cv2() -> bool:
        try:
            import cv2
            return True
        except ImportError:
            return False

    def detect_features(self, gray: np.ndarray) -> np.ndarray:
        """
        Detect feature points worth tracking.
        Returns (N, 2) array of (y, x) float32 coordinates.
        """
        if self._use_cv2:
            import cv2
            corners = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=self.max_corners,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance,
            )
            if corners is not None:
                # cv2 returns (N, 1, 2) in (x, y) format
                pts = corners.reshape(-1, 2)
                return pts[:, ::-1]  # convert to (y, x)
            return np.zeros((0, 2), dtype=np.float32)
        else:
            return self._detect_features_numpy(gray)

    def _detect_features_numpy(self, gray: np.ndarray) -> np.ndarray:
        """
        Detect corner features using Shi-Tomasi eigenvalue criterion.
        Pure numpy implementation.
        """
        h, w = gray.shape
        gf = gray.astype(np.float32)

        # Compute gradients
        dy = np.gradient(gf, axis=0)
        dx = np.gradient(gf, axis=1)

        # Structure tensor components
        Ixx = dx * dx
        Ixy = dx * dy
        Iyy = dy * dy

        # Box blur (3x3) the structure tensor components
        k = np.ones((3, 3)) / 9.0
        Ixx_s = self._box_filter(Ixx, 3)
        Ixy_s = self._box_filter(Ixy, 3)
        Iyy_s = self._box_filter(Iyy, 3)

        # Min eigenvalue of structure tensor = Shi-Tomasi score
        trace = Ixx_s + Iyy_s
        det = Ixx_s * Iyy_s - Ixy_s * Ixy_s
        disc = np.sqrt(np.maximum((trace / 2) ** 2 - det, 0))
        min_eig = trace / 2 - disc

        # Threshold
        threshold = min_eig.max() * self.quality_level
        candidates = np.argwhere(min_eig > threshold)

        if len(candidates) == 0:
            return np.zeros((0, 2), dtype=np.float32)

        # Score-based selection
        scores = min_eig[candidates[:, 0], candidates[:, 1]]
        order = np.argsort(-scores)
        candidates = candidates[order]

        # Non-maximum suppression with min_distance
        selected = []
        used = np.zeros(len(candidates), dtype=bool)
        for i, pt in enumerate(candidates):
            if used[i]:
                continue
            selected.append(pt)
            if len(selected) >= self.max_corners:
                break
            # Suppress nearby points
            dists = np.sqrt(((candidates - pt) ** 2).sum(axis=1))
            used |= dists < self.min_distance

        if not selected:
            return np.zeros((0, 2), dtype=np.float32)
        return np.array(selected, dtype=np.float32)

    def track(
        self, prev_gray: np.ndarray, curr_gray: np.ndarray, prev_points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Track points from prev frame to curr frame.

        Returns:
            (new_points, status) where status[i] = 1 if point i was tracked
            new_points: (N, 2) in (y, x) format
        """
        if len(prev_points) == 0:
            return prev_points, np.array([], dtype=np.uint8)

        if self._use_cv2:
            import cv2
            # cv2 expects (N, 1, 2) in (x, y) format
            prev_xy = prev_points[:, ::-1].reshape(-1, 1, 2).astype(np.float32)
            lk_params = dict(
                winSize=(self.window_size, self.window_size),
                maxLevel=self.max_pyramid_levels,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
            )
            new_xy, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, prev_xy, None, **lk_params
            )
            if new_xy is None:
                return prev_points, np.zeros(len(prev_points), dtype=np.uint8)
            new_yx = new_xy.reshape(-1, 2)[:, ::-1]
            return new_yx, status.ravel()
        else:
            return self._track_numpy(prev_gray, curr_gray, prev_points)

    def _track_numpy(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        prev_points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Lucas-Kanade single-level tracking (numpy fallback)."""
        h, w = prev_gray.shape
        hw = self.window_size // 2
        new_points = prev_points.copy()
        status = np.zeros(len(prev_points), dtype=np.uint8)

        # Compute gradients on current (actually on prev for stability)
        Iy = np.gradient(prev_gray.astype(np.float32), axis=0)
        Ix = np.gradient(prev_gray.astype(np.float32), axis=1)

        for i, (py, px) in enumerate(prev_points):
            py, px = int(py), int(px)
            # Window bounds
            y0, y1 = max(py - hw, 0), min(py + hw + 1, h)
            x0, x1 = max(px - hw, 0), min(px + hw + 1, w)
            if y1 - y0 < 3 or x1 - x1 < 3:
                continue

            # Structure tensor in window
            iy = Iy[y0:y1, x0:x1].ravel()
            ix = Ix[y0:y1, x0:x1].ravel()

            A = np.array([
                [(ix * ix).sum(), (ix * iy).sum()],
                [(ix * iy).sum(), (iy * iy).sum()],
            ])
            det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
            if abs(det) < 1e-6:
                continue

            It = (curr_gray[y0:y1, x0:x1] - prev_gray[y0:y1, x0:x1]).astype(np.float32).ravel()
            b = np.array([-(ix * It).sum(), -(iy * It).sum()])

            try:
                v = np.linalg.solve(A, b)  # [dx, dy]
                new_py = py + v[1]
                new_px = px + v[0]
                if 0 <= new_py < h and 0 <= new_px < w:
                    new_points[i] = [new_py, new_px]
                    status[i] = 1
            except np.linalg.LinAlgError:
                pass

        return new_points, status

    @staticmethod
    def _box_filter(arr: np.ndarray, k: int) -> np.ndarray:
        """Fast box filter using cumulative sum."""
        h, w = arr.shape
        integral = np.cumsum(np.cumsum(arr, axis=0), axis=1)
        r = k // 2
        out = np.zeros_like(arr)
        for y in range(h):
            for x in range(w):
                y0, y1 = max(y - r, 0), min(y + r + 1, h) - 1
                x0, x1 = max(x - r, 0), min(x + r + 1, w) - 1
                total = integral[y1, x1]
                if y0 > 0:
                    total -= integral[y0 - 1, x1]
                if x0 > 0:
                    total -= integral[y1, x0 - 1]
                if y0 > 0 and x0 > 0:
                    total += integral[y0 - 1, x0 - 1]
                count = (y1 - y0 + 1) * (x1 - x0 + 1)
                out[y, x] = total / count
        return out


# ---------------------------------------------------------------------------
# Farneback Dense Flow (approximated in pure numpy)
# ---------------------------------------------------------------------------

class DenseFlowEstimator:
    """
    Dense optical flow — every pixel gets a motion vector.

    Unlike sparse tracking (which tracks specific points), dense flow creates
    a continuous vector field over the entire screen. This is the foundation
    for understanding what's happening everywhere, not just at tracked points.

    When cv2 is available: uses Farneback algorithm (fast, accurate).
    Fallback: gradient-based differential flow (numpy only).

    Reference: Farneback (2003), "Two-frame motion estimation based on
    polynomial expansion."
    """

    def __init__(self, pyr_scale: float = 0.5, levels: int = 3, winsize: int = 15):
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self._use_cv2 = self._check_cv2()

    @staticmethod
    def _check_cv2() -> bool:
        try:
            import cv2
            return True
        except ImportError:
            return False

    def compute(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> FlowField:
        """
        Compute dense optical flow between two frames.

        Returns:
            FlowField with (h, w, 2) flow vectors — direction + magnitude
        """
        t0 = time.time()

        if self._use_cv2:
            import cv2
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray,
                None,
                pyr_scale=self.pyr_scale,
                levels=self.levels,
                winsize=self.winsize,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            # flow: (h, w, 2) in (dx, dy) format — convert to (dy, dx)
            flow_yx = flow[:, :, ::-1]
        else:
            flow_yx = self._compute_numpy(prev_gray, curr_gray)

        magnitude = np.sqrt(flow_yx[:, :, 0]**2 + flow_yx[:, :, 1]**2)
        elapsed_ms = (time.time() - t0) * 1000

        h, w = prev_gray.shape
        return FlowField(
            flow=flow_yx.astype(np.float32),
            magnitude=magnitude.astype(np.float32),
            frame_h=h,
            frame_w=w,
            frame_delta_ms=elapsed_ms,
        )

    def _compute_numpy(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        """
        Gradient-based optical flow (Horn-Schunck approximation, numpy only).
        Reference: Horn & Schunck (1981).
        """
        pf = prev.astype(np.float32)
        cf = curr.astype(np.float32)

        # Temporal derivative
        It = cf - pf

        # Spatial derivatives
        Iy = np.gradient(pf, axis=0)
        Ix = np.gradient(pf, axis=1)

        # Estimate flow assuming local constancy (least squares in window)
        flow = np.zeros((*pf.shape, 2), dtype=np.float32)
        epsilon = 1e-6

        # Simplified: per-pixel flow from brightness constancy + smoothness prior
        # u = dx motion, v = dy motion
        denom = Ix**2 + Iy**2 + epsilon
        alpha = 0.01  # smoothness weight

        # Iterative solution (5 iterations)
        u = np.zeros_like(pf)
        v = np.zeros_like(pf)

        for _ in range(5):
            u_avg = self._laplacian_smooth(u)
            v_avg = self._laplacian_smooth(v)
            P = Ix * u_avg + Iy * v_avg + It
            D = alpha + Ix**2 + Iy**2
            u = u_avg - Ix * P / D
            v = v_avg - Iy * P / D

        flow[:, :, 0] = v  # dy
        flow[:, :, 1] = u  # dx
        return flow

    @staticmethod
    def _laplacian_smooth(arr: np.ndarray) -> np.ndarray:
        """Average of 4-connected neighbors (Laplacian smoothing)."""
        out = np.zeros_like(arr)
        out[1:-1, 1:-1] = (
            arr[:-2, 1:-1] + arr[2:, 1:-1] +
            arr[1:-1, :-2] + arr[1:-1, 2:]
        ) / 4.0
        # Handle borders
        out[0, :] = arr[1, :]
        out[-1, :] = arr[-2, :]
        out[:, 0] = arr[:, 1]
        out[:, -1] = arr[:, -2]
        return out


# ---------------------------------------------------------------------------
# Temporal Identity Manager — track-based identity across frames
# ---------------------------------------------------------------------------

class TemporalIdentityTracker:
    """
    Assigns and maintains persistent identity to screen locations across frames.

    This is the answer to: "Is this the same button I was looking at 5 frames ago?"
    Not by re-detecting and matching boxes — by continuous optical flow tracking.

    Identity persists as long as flow tracking is consistent. If a tracked point
    is lost (e.g., element scrolled off screen), its identity is marked uncertain
    rather than re-assigned.

    This implements appearance-based temporal consistency:
    - Track by flow (geometric)
    - Verify by appearance (semantic)
    - Recover by appearance when flow breaks
    """

    def __init__(
        self,
        max_tracks: int = 100,
        max_lost_frames: int = 10,
        min_confidence: float = 0.3,
    ):
        self.max_tracks = max_tracks
        self.max_lost_frames = max_lost_frames
        self.min_confidence = min_confidence
        self._tracks: Dict[int, TrackedPoint] = {}
        self._next_id = 0
        self._lk_tracker = LKFlowTracker()
        self._dense_flow = DenseFlowEstimator()
        self._prev_gray: Optional[np.ndarray] = None
        self._frame_idx = 0

    def process_frame(
        self,
        frame: np.ndarray,
        dense_field=None,  # Optional DenseVisualField for appearance verification
    ) -> Tuple[Dict[int, TrackedPoint], FlowField]:
        """
        Process a new frame. Update all tracked points via optical flow.

        Args:
            frame: (H, W, C) uint8 screen frame
            dense_field: Optional DenseVisualField for appearance-based recovery

        Returns:
            (active_tracks, flow_field) — tracks with updated positions
        """
        gray = self._to_gray(frame)

        if self._prev_gray is None:
            self._prev_gray = gray
            self._init_tracks(gray, frame, dense_field)
            dummy_flow = FlowField(
                flow=np.zeros((*gray.shape, 2), dtype=np.float32),
                magnitude=np.zeros(gray.shape, dtype=np.float32),
                frame_h=gray.shape[0],
                frame_w=gray.shape[1],
            )
            self._frame_idx += 1
            return dict(self._tracks), dummy_flow

        # Compute dense flow field
        flow_field = self._dense_flow.compute(self._prev_gray, gray)

        # Update sparse tracks via LK
        if self._tracks:
            self._update_tracks_lk(gray, flow_field, frame, dense_field)

        # Add new tracks in empty regions
        if len(self._tracks) < self.max_tracks // 2:
            self._add_new_tracks(gray, frame, dense_field)

        # Prune lost tracks
        self._prune_lost_tracks()

        self._prev_gray = gray
        self._frame_idx += 1

        return dict(self._tracks), flow_field

    def _init_tracks(
        self, gray: np.ndarray, frame: np.ndarray, dense_field
    ) -> None:
        """Initialize tracks on the first frame."""
        points = self._lk_tracker.detect_features(gray)
        for pt in points[:self.max_tracks]:
            self._create_track(pt, frame, dense_field)

    def _update_tracks_lk(
        self,
        curr_gray: np.ndarray,
        flow_field: FlowField,
        curr_frame: np.ndarray,
        dense_field,
    ) -> None:
        """Update existing tracks using LK optical flow."""
        track_ids = list(self._tracks.keys())
        prev_points = np.array(
            [self._tracks[tid].current_position for tid in track_ids],
            dtype=np.float32,
        )

        new_points, status = self._lk_tracker.track(
            self._prev_gray, curr_gray, prev_points
        )

        for i, tid in enumerate(track_ids):
            track = self._tracks[tid]
            if i < len(status) and status[i]:
                new_pos = new_points[i]
                # Verify with dense flow (consistency check)
                py, px = int(track.current_position[0]), int(track.current_position[1])
                flow_dy, flow_dx = flow_field.motion_at(py, px)
                flow_predicted = track.current_position + np.array([flow_dy, flow_dx])
                # If LK and dense flow disagree significantly, reduce confidence
                flow_conf = 1.0 - min(
                    np.linalg.norm(new_pos - flow_predicted) / 10.0, 0.5
                )
                # Update appearance if dense field available
                if dense_field is not None and dense_field.current_grid is not None:
                    appearance = dense_field.current_grid.embedding_at_pixel(
                        int(new_pos[0]), int(new_pos[1])
                    )
                    track.appearance = appearance
                track.update_position(new_pos, flow_conf)
            else:
                # Flow lost — try to recover with appearance
                track.lost_frames += 1
                if dense_field is not None and track.appearance is not None:
                    self._try_recover_track(track, dense_field)

    def _try_recover_track(self, track: TrackedPoint, dense_field) -> None:
        """Try to recover a lost track using appearance similarity."""
        if dense_field.current_grid is None:
            return
        # Search in predicted region (kinematic extrapolation)
        predicted = track.predict_next_position()
        py, px = int(predicted[0]), int(predicted[1])

        # Search a local window for the best appearance match
        grid = dense_field.current_grid
        search_radius = 50 + track.lost_frames * 10  # expand search over time
        r0 = max(0, py - search_radius)
        r1 = min(grid.screen_h - 1, py + search_radius)
        c0 = max(0, px - search_radius)
        c1 = min(grid.screen_w - 1, px + search_radius)

        best_sim = 0.3  # minimum threshold
        best_pos = None
        step = grid.patch_size

        for sy in range(r0, r1, step):
            for sx in range(c0, c1, step):
                emb = grid.embedding_at_pixel(sy, sx)
                sim = float(
                    np.dot(track.appearance, emb) /
                    (np.linalg.norm(track.appearance) * np.linalg.norm(emb) + 1e-8)
                )
                if sim > best_sim:
                    best_sim = sim
                    best_pos = np.array([sy, sx], dtype=np.float32)

        if best_pos is not None:
            track.update_position(best_pos, best_sim * 0.7)  # reduced confidence from recovery
            logger.debug(
                f"Track {track.track_id} recovered at {best_pos} "
                f"(sim={best_sim:.3f}, was lost {track.lost_frames} frames)"
            )

    def _add_new_tracks(
        self, gray: np.ndarray, frame: np.ndarray, dense_field
    ) -> None:
        """Detect and add new feature points in regions with no tracks."""
        existing_positions = np.array(
            [t.current_position for t in self._tracks.values()],
            dtype=np.float32,
        ) if self._tracks else np.zeros((0, 2), dtype=np.float32)

        new_pts = self._lk_tracker.detect_features(gray)

        for pt in new_pts:
            # Skip if too close to an existing track
            if len(existing_positions) > 0:
                dists = np.sqrt(((existing_positions - pt) ** 2).sum(axis=1))
                if dists.min() < 20.0:
                    continue
            self._create_track(pt, frame, dense_field)
            if len(self._tracks) >= self.max_tracks:
                break

    def _create_track(
        self, position: np.ndarray, frame: np.ndarray, dense_field
    ) -> TrackedPoint:
        tid = self._next_id
        self._next_id += 1

        appearance = None
        if dense_field is not None and dense_field.current_grid is not None:
            appearance = dense_field.current_grid.embedding_at_pixel(
                int(position[0]), int(position[1])
            )

        track = TrackedPoint(
            track_id=tid,
            current_position=position.astype(np.float32),
            velocity=np.zeros(2, dtype=np.float32),
            acceleration=np.zeros(2, dtype=np.float32),
            appearance=appearance if appearance is not None else np.zeros(1, dtype=np.float32),
            confidence=1.0,
        )
        self._tracks[tid] = track
        return track

    def _prune_lost_tracks(self) -> None:
        """Remove tracks that have been lost too long or have low confidence."""
        to_remove = [
            tid for tid, t in self._tracks.items()
            if t.lost_frames > self.max_lost_frames or t.confidence < self.min_confidence
        ]
        for tid in to_remove:
            del self._tracks[tid]

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame.astype(np.uint8)
        if frame.shape[2] == 1:
            return frame[:, :, 0].astype(np.uint8)
        # Luminance conversion
        r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
        return (0.2126 * r + 0.7152 * g + 0.0722 * b).astype(np.uint8)

    def get_track(self, track_id: int) -> Optional[TrackedPoint]:
        return self._tracks.get(track_id)

    def find_track_near(
        self, y: int, x: int, radius: float = 30.0
    ) -> Optional[TrackedPoint]:
        """Find the tracked point closest to (y, x) within radius."""
        best = None
        best_dist = radius
        for track in self._tracks.values():
            dist = float(np.linalg.norm(track.current_position - np.array([y, x])))
            if dist < best_dist:
                best_dist = dist
                best = track
        return best

    def get_stats(self) -> Dict[str, Any]:
        active = sum(1 for t in self._tracks.values() if t.lost_frames == 0)
        moving = sum(1 for t in self._tracks.values() if t.is_moving)
        return {
            "total_tracks": len(self._tracks),
            "active_tracks": active,
            "moving_tracks": moving,
            "frames_processed": self._frame_idx,
            "next_id": self._next_id,
        }
