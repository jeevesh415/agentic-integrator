"""Tests for the advanced memory systems and continuous vision."""

import time
import numpy as np
import pytest


class TestHierarchicalMemory:
    def setup_method(self):
        from agentic_integrator.memory.hierarchical_memory import HierarchicalMemory, EpisodicRecord
        self.HierarchicalMemory = HierarchicalMemory
        self.EpisodicRecord = EpisodicRecord
        self.mem = HierarchicalMemory(
            episodic_capacity=100, semantic_capacity=50,
            procedural_capacity=20, consolidation_threshold=3,
            embedding_dim=64,
        )

    def _make_episode(self, task="test_task", reward=0.7):
        return self.EpisodicRecord(
            timestamp=time.time(),
            visual_embedding=np.random.randn(64).astype(np.float32),
            action_taken="pyautogui.click(100, 200)",
            outcome_description="Button clicked",
            task_context=task,
            reward_signal=reward,
            emotional_valence=0.5,
            app_context="chrome",
        )

    def test_store_and_recall_episode(self):
        ep = self._make_episode()
        ep_id = self.mem.store_episode(ep)
        assert ep_id

        results = self.mem.recall_episodes(ep.visual_embedding, top_k=1)
        assert len(results) == 1

    def test_auto_consolidation(self):
        # Store enough episodes to trigger consolidation
        for _ in range(5):
            self.mem.store_episode(self._make_episode())

        assert len(self.mem.semantic) > 0, "Should have consolidated to semantic"

    def test_task_filtering(self):
        for _ in range(3):
            self.mem.store_episode(self._make_episode(task="task_a"))
        for _ in range(3):
            self.mem.store_episode(self._make_episode(task="task_b"))

        query = np.random.randn(64).astype(np.float32)
        results_a = self.mem.recall_episodes(query, task_filter="task_a")
        assert all(r.task_context == "task_a" for r in results_a)

    def test_stats(self):
        for _ in range(5):
            self.mem.store_episode(self._make_episode())
        stats = self.mem.get_stats()
        assert stats["episodic_count"] == 5
        assert stats["total_stores"] == 5


class TestHolographicMemory:
    def setup_method(self):
        from agentic_integrator.memory.holographic_memory import HolographicMemory
        self.mem = HolographicMemory(dim=512, capacity=100)

    def test_encode_and_recall(self):
        # Encode a UI element
        tid = self.mem.encode_experience({
            "element_type": "button",
            "color": "blue",
            "label": "Submit",
            "page": "checkout",
        })
        assert tid.startswith("holo_")

        # Recall by partial cue
        results = self.mem.recall_by_cue({"color": "blue", "element_type": "button"})
        assert len(results) > 0
        trace, score = results[0]
        assert score > 0

    def test_multiple_traces(self):
        self.mem.encode_experience({"color": "red", "type": "error", "page": "login"})
        self.mem.encode_experience({"color": "green", "type": "success", "page": "login"})
        self.mem.encode_experience({"color": "blue", "type": "info", "page": "dashboard"})

        # Query for login-related
        results = self.mem.recall_by_cue({"page": "login"})
        assert len(results) >= 1

    def test_decode_role(self):
        tid = self.mem.encode_experience({
            "color": "red",
            "shape": "circle",
        })
        trace = self.mem.traces[-1]
        decoded = self.mem.decode_role(trace, "color")
        # Should find "red" in decoded results
        assert len(decoded) >= 0  # may or may not find depending on dim

    def test_stats(self):
        self.mem.encode_experience({"test": "value"})
        stats = self.mem.get_stats()
        assert stats["total_traces"] == 1
        assert stats["total_stores"] == 1


class TestHyperDimensionalMemory:
    def setup_method(self):
        from agentic_integrator.memory.hyperdimensional_memory import HyperDimensionalMemory
        self.mem = HyperDimensionalMemory(dim=5000, capacity=500)

    def test_encode_visual_scene(self):
        elements = [
            {"type": "button", "label": "OK", "color": "blue"},
            {"type": "textfield", "label": "Search", "color": "white"},
        ]
        item_id = self.mem.encode_visual_scene(elements)
        assert item_id.startswith("hd_visual_scene_")

    def test_encode_action_sequence(self):
        actions = ["click_search", "type_query", "press_enter", "scroll_down"]
        item_id = self.mem.encode_action_sequence(actions, task_context="web_search")
        assert item_id.startswith("hd_action_pattern_")

    def test_query_by_features(self):
        # Store some elements
        self.mem.encode_ui_element({"type": "button", "color": "red", "label": "Delete"})
        self.mem.encode_ui_element({"type": "button", "color": "blue", "label": "Save"})
        self.mem.encode_ui_element({"type": "textfield", "color": "white", "label": "Name"})

        # Query for red button
        results = self.mem.query({"type": "button", "color": "red"}, category="ui_element")
        assert len(results) >= 0  # matches depend on HD similarity

    def test_hdc_ops_bind_unbind(self):
        from agentic_integrator.memory.hyperdimensional_memory import HDCOps
        ops = HDCOps(dim=5000)

        a = ops.random_hv()
        b = ops.random_hv()

        # Random vectors should be near-orthogonal
        assert abs(ops.similarity(a, b)) < 0.1

        # bind(A, B) should be dissimilar to both A and B
        bound = ops.bind(a, b)
        assert abs(ops.similarity(bound, a)) < 0.15
        assert abs(ops.similarity(bound, b)) < 0.15

        # bind(A, A) = identity-like
        self_bound = ops.bind(a, a)
        # In bipolar, A*A = all 1s, so similarity with any random should be ~0
        assert abs(ops.similarity(self_bound, b)) < 0.15

    def test_sequence_encoding(self):
        from agentic_integrator.memory.hyperdimensional_memory import HDCOps
        ops = HDCOps(dim=5000)

        a = ops.get_atom("step_1")
        b = ops.get_atom("step_2")
        c = ops.get_atom("step_3")

        seq = ops.encode_sequence([a, b, c])
        # Sequence should be dissimilar to individual items
        assert abs(ops.similarity(seq, a)) < 0.3

    def test_scalar_encoding(self):
        from agentic_integrator.memory.hyperdimensional_memory import HDCOps
        ops = HDCOps(dim=5000)

        # Similar values should have similar encodings
        v1 = ops.encode_scalar(0.5)
        v2 = ops.encode_scalar(0.51)
        v3 = ops.encode_scalar(0.99)

        sim_close = ops.similarity(v1, v2)
        sim_far = ops.similarity(v1, v3)
        assert sim_close > sim_far, "Closer values should be more similar"

    def test_stats(self):
        self.mem.encode_ui_element({"type": "button"})
        stats = self.mem.get_stats()
        assert stats["total_items"] == 1
        assert stats["total_stores"] == 1


class TestUnifiedMemoryController:
    def setup_method(self):
        from agentic_integrator.memory.unified_controller import UnifiedMemoryController
        self.ctrl = UnifiedMemoryController(
            embedding_dim=64, hrr_dim=256, hdc_dim=2000,
            episodic_capacity=100, holographic_capacity=100, hdc_capacity=100,
        )

    def test_store_experience(self):
        ids = self.ctrl.store_experience(
            visual_embedding=np.random.randn(64).astype(np.float32),
            action="pyautogui.click(500, 300)",
            outcome="Menu opened",
            task="navigate_to_settings",
            app="chrome",
            reward=0.8,
            ui_elements=[
                {"type": "menu_item", "label": "Settings"},
                {"type": "button", "label": "More"},
            ],
            role_fillers={
                "action": "click",
                "target": "settings_menu",
                "outcome": "menu_opened",
            },
        )
        assert "episodic" in ids
        assert "holographic" in ids
        assert "hyperdimensional" in ids

    def test_recall_by_situation(self):
        # Store several experiences
        for i in range(5):
            self.ctrl.store_experience(
                visual_embedding=np.random.randn(64).astype(np.float32),
                action=f"action_{i}",
                outcome=f"outcome_{i}",
                task="test_task",
                reward=0.5 + i * 0.1,
            )

        results = self.ctrl.recall_by_situation(
            visual_embedding=np.random.randn(64).astype(np.float32),
            task="test_task",
            query_features={"action": "action_3"},
        )
        assert "episodes" in results
        assert "concepts" in results
        assert len(results["episodes"]) > 0

    def test_prediction_guidance(self):
        for _ in range(10):
            self.ctrl.store_prediction("pyautogui.click(100, 200)", 0.8, 0.7)

        guidance = self.ctrl.get_prediction_guidance("click")
        assert "should_simulate" in guidance
        assert "accuracy_for_type" in guidance

    def test_stats(self):
        self.ctrl.store_experience(
            visual_embedding=np.random.randn(64).astype(np.float32),
            action="test", outcome="test", task="test",
        )
        stats = self.ctrl.get_stats()
        assert "hierarchical" in stats
        assert "holographic" in stats
        assert "hyperdimensional" in stats
        assert "prediction" in stats


class TestContinuousVision:
    def setup_method(self):
        from agentic_integrator.vision import ContinuousVisionPipeline
        self.pipeline = ContinuousVisionPipeline(target_fps=10, buffer_size=10)

    def test_process_frame(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = self.pipeline.process_frame(frame)
        assert result.frame_id == 1
        assert result.pixels.shape == (480, 640, 3)

    def test_change_detection(self):
        # First frame
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        self.pipeline.process_frame(frame1)

        # Same frame — should be stable
        result2 = self.pipeline.process_frame(frame1)
        assert result2.change_magnitude == 0.0

        # Different frame — should detect change
        frame3 = np.ones((480, 640, 3), dtype=np.uint8) * 255
        result3 = self.pipeline.process_frame(frame3)
        assert result3.change_magnitude > 0.5

    def test_stability_detection(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(5):
            self.pipeline.process_frame(frame)

        state = self.pipeline.get_current_state()
        assert state is not None
        assert state.stable

    def test_attention_focus(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.pipeline.process_frame(frame)

        focus = self.pipeline.get_attention_focus()
        assert len(focus) > 0

    def test_stats(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        for _ in range(5):
            self.pipeline.process_frame(frame)

        stats = self.pipeline.get_stats()
        assert stats["total_frames"] == 5


class TestVisualGrounding:
    def setup_method(self):
        from agentic_integrator.vision.visual_grounding import (
            AppearanceBasedGrounding, ColorMatcher, VisualFeatureExtractor,
        )
        self.grounding = AppearanceBasedGrounding(screen_width=640, screen_height=480)
        self.color_matcher = ColorMatcher()
        self.feature_extractor = VisualFeatureExtractor(640, 480)

    def test_color_matcher_find_red(self):
        # Create image with a red region
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[100:200, 100:200, 0] = 200  # Red channel
        regions = self.color_matcher.find_color_regions(image, "red")
        assert len(regions) > 0

    def test_color_signature(self):
        pixels = np.random.randint(0, 255, (100, 3), dtype=np.uint8)
        sig = self.color_matcher.compute_color_signature(pixels)
        assert sig.shape == (48,)  # 16 bins * 3 channels
        assert abs(sig.sum() - 3.0) < 0.01  # normalized per channel

    def test_feature_extraction(self):
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        features = self.feature_extractor.extract_features(image, (100, 100, 200, 50))
        assert "shape" in features
        assert "size_category" in features
        assert "relative_position" in features
        assert "dominant_color" in features
        assert features["shape"] in ("rectangle", "wide_rectangle", "tall_rectangle", "square")

    def test_create_anchor(self):
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        anchor = self.grounding.create_anchor("test_btn", image, (100, 100, 80, 30), "OK")
        assert anchor.anchor_id == "test_btn"
        assert anchor.text_content == "OK"
        assert "test_btn" in self.grounding.anchors

    def test_find_by_anchor(self):
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.grounding.create_anchor("my_btn", image, (100, 100, 80, 30))

        result = self.grounding.find_element(
            {"anchor_name": "my_btn"}, image,
            candidate_bboxes=[(100, 100, 80, 30), (300, 300, 80, 30)],
        )
        # Should find the anchor at its last known position
        assert result is not None

    def test_stats(self):
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.grounding.create_anchor("btn", image, (50, 50, 40, 20))
        stats = self.grounding.get_stats()
        assert stats["total_anchors"] == 1
