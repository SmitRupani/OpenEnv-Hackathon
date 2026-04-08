from __future__ import annotations

import math
from typing import Any, Dict

from openenv.core.rubrics import Rubric


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _squash_to_open_interval(signal: float, scale: float) -> float:
    if not math.isfinite(signal):
        signal = 0.0
    if not math.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    normalized = 0.5 + math.atan(signal / scale) / math.pi
    
    # Strictly keep the final score between 0 and 1
    min_val = 0.0001
    max_val = 0.9999
    return min(max(round(normalized, 4), min_val), max_val)


class TaskGrader(Rubric):
    def __init__(
        self,
        name: str,
        *,
        target_surge: float,
        scale: float,
        balance_scale: float,
        balance_bonus: float,
        stability_bonus: float,
        completion_bonus: float,
        terminal_bonus: float,
    ):
        super().__init__()
        self.name = name
        self.target_surge = target_surge
        self.scale = scale
        self.balance_scale = balance_scale
        self.balance_bonus = balance_bonus
        self.stability_bonus = stability_bonus
        self.completion_bonus = completion_bonus
        self.terminal_bonus = terminal_bonus

    def forward(self, action: Any, observation: Any) -> float:
        metadata = getattr(observation, "metadata", {}) or {}
        raw_reward = _coerce_float(metadata.get("raw_reward"), 0.0)
        completed_rides = _coerce_float(metadata.get("completed_rides"), 0.0)
        demand = _coerce_float(getattr(observation, "demand", 0.0), 0.0)
        supply = _coerce_float(getattr(observation, "supply", 0.0), 0.0)
        surge_multiplier = _coerce_float(getattr(observation, "surge_multiplier", 1.0), 1.0)

        balance_gap = abs(demand - supply)
        balance_signal = max(0.0, 1.0 - (balance_gap / max(self.balance_scale, 1.0)))
        surge_alignment = max(
            0.0,
            1.0 - (abs(surge_multiplier - self.target_surge) / 1.5),
        )

        signal = (
            raw_reward
            + balance_signal * self.balance_bonus
            + surge_alignment * self.stability_bonus
            + completed_rides * self.completion_bonus
        )

        if getattr(observation, "done", False):
            signal += self.terminal_bonus

        return _squash_to_open_interval(signal, self.scale)


class DynapriceTaskGraderSuite(Rubric):
    TASK_NAMES = ("easy", "medium", "hard")

    def __init__(self):
        super().__init__()
        self.easy = TaskGrader(
            "easy",
            target_surge=1.15,
            scale=1600.0,
            balance_scale=50.0,
            balance_bonus=90.0,
            stability_bonus=100.0,
            completion_bonus=0.25,
            terminal_bonus=25.0,
        )
        self.medium = TaskGrader(
            "medium",
            target_surge=1.45,
            scale=2200.0,
            balance_scale=75.0,
            balance_bonus=120.0,
            stability_bonus=135.0,
            completion_bonus=0.2,
            terminal_bonus=35.0,
        )
        self.hard = TaskGrader(
            "hard",
            target_surge=1.75,
            scale=2800.0,
            balance_scale=95.0,
            balance_bonus=150.0,
            stability_bonus=170.0,
            completion_bonus=0.15,
            terminal_bonus=45.0,
        )

    def forward(self, action: Any, observation: Any) -> float:
        metadata = getattr(observation, "metadata", {}) or {}
        task_id = str(metadata.get("task_id", "easy"))
        if task_id not in self.TASK_NAMES:
            task_id = "easy"
        return getattr(self, task_id)(action, observation)

    def task_scores(self, action: Any, observation: Any) -> Dict[str, float]:
        return {
            task_name: getattr(self, task_name)(action, observation)
            for task_name in self.TASK_NAMES
        }
