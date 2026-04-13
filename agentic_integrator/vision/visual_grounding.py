"""
Visual Grounding — Find UI Elements by Appearance, Not Coordinates.

The key innovation: instead of "click at (523, 417)", the agent thinks
"click the blue Submit button" — just like a human would.

This module provides:
1. ColorMatcher: Find elements by color signature
2. VisualFeatureExtractor: Extract visual features (shape, color, texture, size)
3. AppearanceBasedGrounding: Locate UI elements by visual description
4. VisualAnchor: Remember elements by how they look, not where they are
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("agentic_integrator.vision.visual_grounding")


@dataclass
class VisualRegion:
    """A detected visual region on screen."""
    region_id: str
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    visual_features: Dict[str, Any]
    confidence: float
    label: str = ""
    
    @property
    def center(self) -> Tuple[int, int]:
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)
    
    @property
    def area(self) -> int:
        return self.bbox[2] * self.bbox[3]


@dataclass
class VisualAnchor:
    """
    A visual anchor — remembers a UI element by its appearance.
    
    Instead of remembering "the button at (523, 417)",
    remember "the blue rectangular element with white text 'Submit'
    in the lower-right area of the screen".
    """
    anchor_id: str
    color_signature: np.ndarray  # histogram of colors
    shape_descriptor: str  # "rectangle", "circle", "pill", etc.
    size_category: str  # "small", "medium", "large"
    relative_position: str  # "top-left", "center", "bottom-right", etc.
    text_content: str = ""
    surrounding_context: str = ""  # what's near this element
    
    # Tracking across frames
    last_seen_bbox: Optional[Tuple[int, int, int, int]] = None
    last_seen_timestamp: float = 0.0
    times_found: int = 0
    
    def matches(self, other_features: Dict[str, Any], threshold: float = 0.6) -> float:
        """Compute match score against candidate features."""
        score = 0.0
        checks = 0
        
        # Color similarity
        if "color_signature" in other_features and self.color_signature is not None:
            other_color = other_features["color_signature"]
            if isinstance(other_color, np.ndarray):
                color_sim = 1.0 - np.mean(np.abs(self.color_signature - other_color))
                score += color_sim * 0.35
                checks += 0.35
        
        # Shape match
        if "shape" in other_features:
            if other_features["shape"] == self.shape_descriptor:
                score += 0.2
            checks += 0.2
        
        # Size match
        if "size_category" in other_features:
            if other_features["size_category"] == self.size_category:
                score += 0.15
            checks += 0.15
        
        # Position match
        if "relative_position" in other_features:
            if other_features["relative_position"] == self.relative_position:
                score += 0.15
            checks += 0.15
        
        # Text match
        if "text" in other_features and self.text_content:
            if other_features["text"].lower() in self.text_content.lower():
                score += 0.15
            checks += 0.15
        
        return score / max(checks, 0.01)


class ColorMatcher:
    """
    Find UI elements by their color signature.
    
    "Click the red button" → find regions that are predominantly red.
    "Find the green notification" → locate green-colored areas.
    """
    
    # Named color ranges in HSV space
    COLOR_RANGES: Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = {
        "red": ((0, 100, 100), (10, 255, 255)),
        "red2": ((160, 100, 100), (180, 255, 255)),
        "orange": ((10, 100, 100), (25, 255, 255)),
        "yellow": ((25, 100, 100), (35, 255, 255)),
        "green": ((35, 100, 100), (85, 255, 255)),
        "blue": ((85, 100, 100), (130, 255, 255)),
        "purple": ((130, 100, 100), (160, 255, 255)),
        "white": ((0, 0, 200), (180, 30, 255)),
        "black": ((0, 0, 0), (180, 255, 30)),
        "gray": ((0, 0, 30), (180, 30, 200)),
    }
    
    def __init__(self, min_region_area: int = 100):
        self.min_region_area = min_region_area
    
    def find_color_regions(
        self,
        image: np.ndarray,
        target_color: str,
    ) -> List[Dict[str, Any]]:
        """
        Find regions in the image that match a target color name.
        
        Args:
            image: RGB image (H, W, 3)
            target_color: Color name (e.g., "red", "blue", "green")
            
        Returns:
            List of region dictionaries with bounding boxes
        """
        # Convert RGB to a simple HSV-like space
        # Using a simplified approach without OpenCV dependency
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        
        regions = []
        target_lower = target_color.lower()
        
        # Simple color matching based on RGB dominance
        if target_lower in ("red", "red2"):
            mask = (r > 150) & (g < 100) & (b < 100)
        elif target_lower == "green":
            mask = (g > 150) & (r < 100) & (b < 100)
        elif target_lower == "blue":
            mask = (b > 150) & (r < 100) & (g < 100)
        elif target_lower == "yellow":
            mask = (r > 150) & (g > 150) & (b < 100)
        elif target_lower == "orange":
            mask = (r > 180) & (g > 100) & (g < 180) & (b < 80)
        elif target_lower == "purple":
            mask = (r > 100) & (b > 100) & (g < 80)
        elif target_lower == "white":
            mask = (r > 200) & (g > 200) & (b > 200)
        elif target_lower == "black":
            mask = (r < 30) & (g < 30) & (b < 30)
        elif target_lower == "gray":
            diff_rg = np.abs(r.astype(int) - g.astype(int))
            diff_rb = np.abs(r.astype(int) - b.astype(int))
            mask = (diff_rg < 30) & (diff_rb < 30) & (r > 50) & (r < 200)
        else:
            return []
        
        # Find connected regions (simplified: grid-based)
        h, w = mask.shape
        grid_h, grid_w = max(1, h // 20), max(1, w // 20)
        
        for gi in range(0, h, grid_h):
            for gj in range(0, w, grid_w):
                cell = mask[gi:gi + grid_h, gj:gj + grid_w]
                if cell.sum() > self.min_region_area // 4:
                    # Extract average color
                    cell_pixels = image[gi:gi + grid_h, gj:gj + grid_w]
                    avg_color = cell_pixels[cell].mean(axis=0) if cell.any() else [0, 0, 0]
                    
                    regions.append({
                        "bbox": (gj, gi, grid_w, grid_h),
                        "center": (gj + grid_w // 2, gi + grid_h // 2),
                        "color": target_color,
                        "avg_rgb": tuple(int(c) for c in avg_color),
                        "pixel_count": int(cell.sum()),
                        "density": float(cell.mean()),
                    })
        
        # Merge nearby regions
        return self._merge_nearby_regions(regions)
    
    def compute_color_signature(self, region_pixels: np.ndarray, bins: int = 16) -> np.ndarray:
        """Compute a color histogram signature for a region."""
        if region_pixels.size == 0:
            return np.zeros(bins * 3, dtype=np.float32)
        
        signature = []
        for channel in range(3):
            hist, _ = np.histogram(region_pixels[:, channel], bins=bins, range=(0, 256))
            hist = hist.astype(np.float32)
            if hist.sum() > 0:
                hist /= hist.sum()
            signature.extend(hist)
        
        return np.array(signature, dtype=np.float32)
    
    @staticmethod
    def _merge_nearby_regions(regions: List[Dict], distance_threshold: int = 50) -> List[Dict]:
        """Merge regions that are close together."""
        if len(regions) <= 1:
            return regions
        
        merged = []
        used = set()
        
        for i, r1 in enumerate(regions):
            if i in used:
                continue
            group = [r1]
            for j, r2 in enumerate(regions[i + 1:], i + 1):
                if j in used:
                    continue
                dx = abs(r1["center"][0] - r2["center"][0])
                dy = abs(r1["center"][1] - r2["center"][1])
                if dx < distance_threshold and dy < distance_threshold:
                    group.append(r2)
                    used.add(j)
            
            # Merge group into single region
            all_x = [r["bbox"][0] for r in group]
            all_y = [r["bbox"][1] for r in group]
            all_x2 = [r["bbox"][0] + r["bbox"][2] for r in group]
            all_y2 = [r["bbox"][1] + r["bbox"][3] for r in group]
            
            merged_bbox = (min(all_x), min(all_y), max(all_x2) - min(all_x), max(all_y2) - min(all_y))
            merged.append({
                "bbox": merged_bbox,
                "center": (merged_bbox[0] + merged_bbox[2] // 2, merged_bbox[1] + merged_bbox[3] // 2),
                "color": group[0]["color"],
                "pixel_count": sum(r["pixel_count"] for r in group),
                "density": np.mean([r["density"] for r in group]),
            })
        
        return merged


class VisualFeatureExtractor:
    """
    Extract visual features from screen regions for appearance-based grounding.
    
    Features extracted:
    - Color signature (histogram)
    - Shape descriptor (aspect ratio → rectangle/square/wide/tall)
    - Size category (small/medium/large relative to screen)
    - Relative position (top-left, center, bottom-right, etc.)
    - Edge density (text-heavy vs solid color)
    """
    
    def __init__(self, screen_width: int = 1920, screen_height: int = 1080):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.color_matcher = ColorMatcher()
    
    def extract_features(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Dict[str, Any]:
        """
        Extract visual features from a region of the screen.
        
        Args:
            image: Full screen RGB image
            bbox: (x, y, width, height) of the region
            
        Returns:
            Dictionary of visual features
        """
        x, y, w, h = bbox
        region = image[y:y + h, x:x + w]
        
        if region.size == 0:
            return {}
        
        features = {}
        
        # Color signature
        pixels_flat = region.reshape(-1, 3)
        features["color_signature"] = self.color_matcher.compute_color_signature(pixels_flat)
        features["dominant_color"] = self._get_dominant_color(pixels_flat)
        
        # Shape descriptor
        aspect_ratio = w / max(h, 1)
        if 0.8 <= aspect_ratio <= 1.2:
            features["shape"] = "square"
        elif aspect_ratio > 2.5:
            features["shape"] = "wide_rectangle"
        elif aspect_ratio < 0.4:
            features["shape"] = "tall_rectangle"
        else:
            features["shape"] = "rectangle"
        
        features["aspect_ratio"] = aspect_ratio
        
        # Size category
        screen_area = self.screen_width * self.screen_height
        region_area = w * h
        area_ratio = region_area / max(screen_area, 1)
        
        if area_ratio < 0.005:
            features["size_category"] = "tiny"
        elif area_ratio < 0.02:
            features["size_category"] = "small"
        elif area_ratio < 0.1:
            features["size_category"] = "medium"
        elif area_ratio < 0.3:
            features["size_category"] = "large"
        else:
            features["size_category"] = "very_large"
        
        # Relative position (9-grid)
        cx, cy = x + w // 2, y + h // 2
        col = "left" if cx < self.screen_width * 0.33 else ("right" if cx > self.screen_width * 0.67 else "center")
        row = "top" if cy < self.screen_height * 0.33 else ("bottom" if cy > self.screen_height * 0.67 else "middle")
        features["relative_position"] = f"{row}-{col}" if row != "middle" or col != "center" else "center"
        
        # Edge density (proxy for text content)
        gray = np.mean(region, axis=2)
        edges_h = np.abs(np.diff(gray, axis=1)).mean()
        edges_v = np.abs(np.diff(gray, axis=0)).mean()
        features["edge_density"] = float((edges_h + edges_v) / 2)
        features["has_text_likely"] = features["edge_density"] > 15.0
        
        # Color uniformity
        features["color_uniformity"] = 1.0 - float(np.std(pixels_flat) / 128.0)
        
        return features
    
    def _get_dominant_color(self, pixels: np.ndarray) -> str:
        """Determine the dominant named color of a region."""
        if len(pixels) == 0:
            return "unknown"
        
        avg = pixels.mean(axis=0)
        r, g, b = avg[0], avg[1], avg[2]
        
        # Simple named color classification
        if r > 200 and g > 200 and b > 200:
            return "white"
        if r < 30 and g < 30 and b < 30:
            return "black"
        if r > 150 and g < 100 and b < 100:
            return "red"
        if g > 150 and r < 100 and b < 100:
            return "green"
        if b > 150 and r < 100 and g < 100:
            return "blue"
        if r > 150 and g > 150 and b < 100:
            return "yellow"
        if r > 180 and g > 100 and b < 80:
            return "orange"
        if r > 100 and b > 100 and g < 80:
            return "purple"
        
        # Check if grayish
        if abs(r - g) < 30 and abs(r - b) < 30:
            return "gray"
        
        return "mixed"


class AppearanceBasedGrounding:
    """
    Locate UI elements by visual appearance instead of coordinates.
    
    The human way: "I see a blue button labeled Submit in the bottom-right"
    NOT: "Click at pixel (1523, 847)"
    
    This module:
    1. Maintains a library of VisualAnchors (known element appearances)
    2. Can find elements on screen by matching visual features
    3. Handles UI changes (moved elements, theme changes) gracefully
    4. Learns new element appearances on-the-fly
    """
    
    def __init__(
        self,
        screen_width: int = 1920,
        screen_height: int = 1080,
        match_threshold: float = 0.5,
    ):
        self.feature_extractor = VisualFeatureExtractor(screen_width, screen_height)
        self.color_matcher = ColorMatcher()
        self.match_threshold = match_threshold
        
        # Known visual anchors
        self.anchors: Dict[str, VisualAnchor] = {}
        
        # Stats
        self._total_groundings = 0
        self._successful_groundings = 0
    
    def create_anchor(
        self,
        name: str,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        text_content: str = "",
    ) -> VisualAnchor:
        """
        Create a visual anchor for a UI element.
        
        Args:
            name: Human-readable name (e.g., "submit_button")
            image: Full screen image
            bbox: Bounding box of the element
            text_content: Known text in the element
            
        Returns:
            VisualAnchor
        """
        features = self.feature_extractor.extract_features(image, bbox)
        
        anchor = VisualAnchor(
            anchor_id=name,
            color_signature=features.get("color_signature", np.zeros(48)),
            shape_descriptor=features.get("shape", "unknown"),
            size_category=features.get("size_category", "medium"),
            relative_position=features.get("relative_position", "center"),
            text_content=text_content,
            last_seen_bbox=bbox,
            last_seen_timestamp=__import__("time").time(),
            times_found=1,
        )
        
        self.anchors[name] = anchor
        return anchor
    
    def find_element(
        self,
        description: Dict[str, str],
        image: np.ndarray,
        candidate_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> Optional[VisualRegion]:
        """
        Find a UI element on screen by visual description.
        
        Args:
            description: Visual description, e.g.:
                {"color": "blue", "shape": "rectangle", "position": "bottom-right"}
                {"text": "Submit", "color": "green"}
                {"anchor_name": "submit_button"}  # use a known anchor
            image: Current screen image
            candidate_bboxes: Optional pre-detected element bounding boxes
            
        Returns:
            VisualRegion if found, None otherwise
        """
        self._total_groundings += 1
        
        # If referencing a known anchor, use anchor matching
        if "anchor_name" in description and description["anchor_name"] in self.anchors:
            anchor = self.anchors[description["anchor_name"]]
            return self._find_by_anchor(anchor, image, candidate_bboxes)
        
        # Color-based search
        if "color" in description:
            color_regions = self.color_matcher.find_color_regions(image, description["color"])
            if color_regions:
                # Filter by other description features
                best_match = self._filter_candidates(color_regions, description, image)
                if best_match:
                    self._successful_groundings += 1
                    return best_match
        
        # Feature-based search over candidates
        if candidate_bboxes:
            best_score = 0.0
            best_region = None
            
            for bbox in candidate_bboxes:
                features = self.feature_extractor.extract_features(image, bbox)
                score = self._score_match(features, description)
                if score > best_score and score > self.match_threshold:
                    best_score = score
                    best_region = VisualRegion(
                        region_id=f"found_{self._total_groundings}",
                        bbox=bbox,
                        visual_features=features,
                        confidence=score,
                        label=description.get("text", ""),
                    )
            
            if best_region:
                self._successful_groundings += 1
            return best_region
        
        return None
    
    def _find_by_anchor(
        self,
        anchor: VisualAnchor,
        image: np.ndarray,
        candidate_bboxes: Optional[List[Tuple[int, int, int, int]]],
    ) -> Optional[VisualRegion]:
        """Find an element matching a known visual anchor."""
        # Try last known position first
        if anchor.last_seen_bbox:
            features = self.feature_extractor.extract_features(image, anchor.last_seen_bbox)
            score = anchor.matches(features)
            if score > self.match_threshold:
                anchor.times_found += 1
                self._successful_groundings += 1
                return VisualRegion(
                    region_id=anchor.anchor_id,
                    bbox=anchor.last_seen_bbox,
                    visual_features=features,
                    confidence=score,
                    label=anchor.text_content,
                )
        
        # Search candidates
        if candidate_bboxes:
            best_score = 0.0
            best_bbox = None
            best_features = None
            
            for bbox in candidate_bboxes:
                features = self.feature_extractor.extract_features(image, bbox)
                score = anchor.matches(features)
                if score > best_score:
                    best_score = score
                    best_bbox = bbox
                    best_features = features
            
            if best_score > self.match_threshold and best_bbox:
                anchor.last_seen_bbox = best_bbox
                anchor.times_found += 1
                self._successful_groundings += 1
                return VisualRegion(
                    region_id=anchor.anchor_id,
                    bbox=best_bbox,
                    visual_features=best_features or {},
                    confidence=best_score,
                    label=anchor.text_content,
                )
        
        return None
    
    def _filter_candidates(
        self,
        color_regions: List[Dict],
        description: Dict[str, str],
        image: np.ndarray,
    ) -> Optional[VisualRegion]:
        """Filter color-matched regions by additional description criteria."""
        for region in color_regions:
            bbox = region["bbox"]
            features = self.feature_extractor.extract_features(image, bbox)
            score = self._score_match(features, description)
            
            if score > self.match_threshold:
                return VisualRegion(
                    region_id=f"color_match_{self._total_groundings}",
                    bbox=bbox,
                    visual_features=features,
                    confidence=score,
                    label=description.get("text", ""),
                )
        
        return None
    
    @staticmethod
    def _score_match(features: Dict[str, Any], description: Dict[str, str]) -> float:
        """Score how well extracted features match a description."""
        score = 0.0
        total_weight = 0.0
        
        if "shape" in description and "shape" in features:
            total_weight += 0.2
            if features["shape"] == description["shape"]:
                score += 0.2
        
        if "position" in description and "relative_position" in features:
            total_weight += 0.2
            if description["position"] in features["relative_position"]:
                score += 0.2
        
        if "size" in description and "size_category" in features:
            total_weight += 0.15
            if features["size_category"] == description["size"]:
                score += 0.15
        
        if "color" in description and "dominant_color" in features:
            total_weight += 0.25
            if features["dominant_color"] == description["color"]:
                score += 0.25
        
        # Bonus for text-like regions when looking for text
        if "text" in description and features.get("has_text_likely"):
            total_weight += 0.2
            score += 0.2
        
        return score / max(total_weight, 0.01) if total_weight > 0 else 0.5
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_anchors": len(self.anchors),
            "total_groundings": self._total_groundings,
            "successful_groundings": self._successful_groundings,
            "success_rate": self._successful_groundings / max(self._total_groundings, 1),
        }
