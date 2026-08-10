"""Composite reward from the SAL-TD3 navigation paper."""

from __future__ import annotations

import math


def navigation_reward(
    heading_error: float,
    previous_distance: float,
    current_distance: float,
    minimum_obstacle_distance: float,
    *,
    goal_reached: bool = False,
    collision: bool = False,
) -> float:
    """Compute the paper's terminal or dense navigation reward."""
    if not -math.pi <= heading_error <= math.pi:
        raise ValueError("heading_error must be within [-pi, pi]")
    if previous_distance < 0 or current_distance < 0:
        raise ValueError("goal distances must be non-negative")
    if goal_reached and collision:
        raise ValueError("goal_reached and collision cannot both be true")
    if goal_reached:
        return 1000.0
    if collision:
        return -800.0
    heading = 4.0 * (1.0 - abs(heading_error))
    ratio = previous_distance / max(current_distance, 1e-8)
    progress = 2.0 * min(4.0, max(0.5, ratio))
    safety = -5.0 if minimum_obstacle_distance < 0.5 else 0.0
    return heading * progress + safety
