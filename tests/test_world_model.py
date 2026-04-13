"""Tests for the World Model components."""

import pytest
from PIL import Image

from agentic_integrator.world_model.base import SimulationResult, PlanningResult
from agentic_integrator.world_model.safety_gate import SafetyGate, SafetyCheckResult
from agentic_integrator.memory.prediction_memory import PredictionMemory


class TestSimulationResult:
    def test_composite_score(self):
        result = SimulationResult(
            action="pyautogui.click(100, 200)",
            predicted_state_description="Button clicked, menu opened",
            task_progress_score=0.8,
            confidence=0.9,
            safety_score=1.0,
        )
        # 0.8*0.5 + 1.0*0.3 + 0.9*0.2 = 0.4 + 0.3 + 0.18 = 0.88
        assert abs(result.composite_score - 0.88) < 0.01

    def test_composite_score_dangerous(self):
        result = SimulationResult(
            action="rm -rf /",
            predicted_state_description="All files deleted",
            task_progress_score=0.9,
            confidence=0.95,
            safety_score=0.0,
        )
        # High progress but zero safety should penalize
        assert result.composite_score < 0.7


class TestSafetyGate:
    def setup_method(self):
        self.gate = SafetyGate(engine_params={}, enabled=True)

    def test_safe_action(self):
        sim = SimulationResult(
            action="pyautogui.click(500, 300)",
            predicted_state_description="Clicked on search bar",
            task_progress_score=0.5,
            confidence=0.8,
            safety_score=1.0,
        )
        result = self.gate.check(sim)
        assert not result.blocked

    def test_dangerous_file_deletion(self):
        sim = SimulationResult(
            action="rm -rf /home/user/documents",
            predicted_state_description="Delete all documents",
            task_progress_score=0.5,
            confidence=0.8,
            safety_score=0.1,
        )
        result = self.gate.check(sim)
        assert result.blocked
        assert "high" in result.severity or "critical" in result.severity

    def test_dangerous_database_operation(self):
        sim = SimulationResult(
            action="DROP TABLE users",
            predicted_state_description="Database table deleted",
            task_progress_score=0.3,
            confidence=0.9,
            safety_score=0.0,
        )
        result = self.gate.check(sim)
        assert result.blocked

    def test_financial_action(self):
        sim = SimulationResult(
            action="pyautogui.click(300, 400)  # click purchase button",
            predicted_state_description="Purchase completed",
            task_progress_score=0.9,
            confidence=0.8,
            safety_score=0.5,
        )
        result = self.gate.check(sim)
        assert result.blocked

    def test_disabled_gate(self):
        gate = SafetyGate(engine_params={}, enabled=False)
        sim = SimulationResult(
            action="rm -rf /",
            predicted_state_description="Everything deleted",
            task_progress_score=0.5,
            confidence=0.5,
            safety_score=0.0,
        )
        result = gate.check(sim)
        assert not result.blocked


class TestPredictionMemory:
    def setup_method(self):
        self.memory = PredictionMemory(window_size=10)

    def test_record_and_accuracy(self):
        self.memory.record("pyautogui.click(100, 200)", 0.8, 0.75)
        assert self.memory.get_overall_accuracy() == 0.75

    def test_accuracy_by_type(self):
        # Record some click actions
        for _ in range(5):
            self.memory.record("pyautogui.click(100, 200)", 0.7, 0.8)
        # Record some type actions
        for _ in range(5):
            self.memory.record("pyautogui.typewrite('hello')", 0.6, 0.5)

        assert self.memory.get_accuracy_for_type("click") == 0.8
        assert self.memory.get_accuracy_for_type("type") == 0.5

    def test_should_simulate(self):
        # With no data, should default to True
        assert self.memory.should_simulate("click") is True

        # With good accuracy data, should still simulate
        for _ in range(10):
            self.memory.record("pyautogui.click(100, 200)", 0.8, 0.8)
        assert self.memory.should_simulate("click") is True

    def test_reset_episode(self):
        self.memory.record("action1", 0.5, 0.5)
        self.memory.record("action2", 0.6, 0.6)
        self.memory.reset_episode()
        # Episode records reset, but total records remain
        assert len(self.memory.records) == 2
        assert len(self.memory._episode_records) == 0

    def test_window_bounding(self):
        memory = PredictionMemory(window_size=5)
        for i in range(25):
            memory.record(f"action_{i}", 0.5, 0.5)
        # Should be bounded
        assert len(memory.records) <= 10  # window_size * 2


class TestStats:
    def test_safety_gate_stats(self):
        gate = SafetyGate(engine_params={}, enabled=True)
        safe_sim = SimulationResult(
            action="pyautogui.click(100, 200)",
            predicted_state_description="Safe click",
            task_progress_score=0.5,
            confidence=0.8,
            safety_score=1.0,
        )
        gate.check(safe_sim)
        stats = gate.get_stats()
        assert stats["total_checks"] == 1
        assert stats["blocks"] == 0
