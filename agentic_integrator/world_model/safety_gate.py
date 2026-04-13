"""Safety Gate — detects and blocks irreversible or dangerous actions."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List

from agentic_integrator.world_model.base import SimulationResult

logger = logging.getLogger("agentic_integrator.safety_gate")


# Patterns that indicate potentially dangerous actions
DANGEROUS_PATTERNS: List[Dict[str, Any]] = [
    # File operations
    {"pattern": r"(rm\s+-rf|rmdir|shutil\.rmtree|os\.remove|os\.unlink)", "category": "file_deletion", "severity": "high"},
    {"pattern": r"(format|mkfs|fdisk|diskpart)", "category": "disk_operation", "severity": "critical"},
    {"pattern": r"(overwrite|truncate)", "category": "data_overwrite", "severity": "medium"},
    # Web form actions
    {"pattern": r"(submit|\.submit\(\)|form.*submit)", "category": "form_submit", "severity": "medium"},
    {"pattern": r"(purchase|buy|checkout|pay|payment|place.?order)", "category": "financial", "severity": "critical"},
    {"pattern": r"(send|deliver|publish|broadcast)", "category": "send_action", "severity": "medium"},
    # System operations
    {"pattern": r"(shutdown|reboot|restart|poweroff)", "category": "system_power", "severity": "critical"},
    {"pattern": r"(sudo|chmod\s+777|chown)", "category": "privilege_escalation", "severity": "high"},
    {"pattern": r"(pip\s+install|apt\s+install|brew\s+install|npm\s+install)", "category": "package_install", "severity": "low"},
    # Database operations
    {"pattern": r"(DROP\s+TABLE|DELETE\s+FROM|TRUNCATE|DROP\s+DATABASE)", "category": "database_destructive", "severity": "critical"},
    {"pattern": r"(clear.?all|reset.?all|wipe|purge)", "category": "bulk_clear", "severity": "high"},
    # Account/auth actions
    {"pattern": r"(delete.?account|deactivate|unsubscribe|cancel.?subscription)", "category": "account_action", "severity": "high"},
    {"pattern": r"(logout|sign.?out|revoke)", "category": "auth_action", "severity": "medium"},
]


@dataclass
class SafetyCheckResult:
    """Result of a safety check on an action."""

    blocked: bool
    reason: str
    category: str
    severity: str  # critical, high, medium, low
    matched_patterns: List[str]


class SafetyGate:
    """
    Analyzes actions for safety risks before execution.

    Checks for:
    - Irreversible file operations (delete, format)
    - Financial transactions (purchase, pay)
    - System-level operations (shutdown, privilege escalation)
    - Database destructive operations (DROP, TRUNCATE)
    - Account-level changes (delete account, cancel subscription)
    """

    def __init__(self, engine_params: Dict[str, Any], enabled: bool = True):
        self.engine_params = engine_params
        self.enabled = enabled
        self._total_checks = 0
        self._blocks = 0

    def check(self, simulation: SimulationResult) -> SafetyCheckResult:
        """
        Check if an action is safe to execute.

        Args:
            simulation: The simulation result containing the action and predictions

        Returns:
            SafetyCheckResult indicating if the action should be blocked
        """
        self._total_checks += 1

        if not self.enabled:
            return SafetyCheckResult(
                blocked=False,
                reason="Safety gate disabled",
                category="none",
                severity="none",
                matched_patterns=[],
            )

        # Check the action string against dangerous patterns
        action_text = simulation.action + " " + simulation.predicted_state_description
        matched = []
        max_severity = "none"
        categories = []

        severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1, "none": 0}

        for pattern_def in DANGEROUS_PATTERNS:
            if re.search(pattern_def["pattern"], action_text, re.IGNORECASE):
                matched.append(pattern_def["pattern"])
                categories.append(pattern_def["category"])
                if severity_order.get(pattern_def["severity"], 0) > severity_order.get(max_severity, 0):
                    max_severity = pattern_def["severity"]

        # Also check if the simulation itself flagged irreversibility
        if simulation.is_irreversible:
            max_severity = max(max_severity, "high", key=lambda s: severity_order.get(s, 0))
            categories.append("simulation_flagged_irreversible")

        # Decision: block critical and high severity
        should_block = max_severity in ("critical", "high")

        if should_block:
            self._blocks += 1
            reason = (
                f"Dangerous action detected ({max_severity} severity): "
                f"categories={categories}. Action: {simulation.action[:100]}"
            )
            logger.warning(f"SAFETY GATE BLOCKED: {reason}")
        else:
            reason = ""

        return SafetyCheckResult(
            blocked=should_block,
            reason=reason,
            category=", ".join(categories) if categories else "none",
            severity=max_severity,
            matched_patterns=matched,
        )

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_checks": self._total_checks,
            "blocks": self._blocks,
            "block_rate": self._blocks / max(self._total_checks, 1),
        }
