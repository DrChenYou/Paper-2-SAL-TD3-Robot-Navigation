import math

import pytest

from sal_td3.reward import navigation_reward


def test_terminal_rewards():
    assert navigation_reward(0, 2, 1, 2, goal_reached=True) == 1000
    assert navigation_reward(0, 2, 1, 2, collision=True) == -800


def test_dense_reward_matches_equation():
    assert navigation_reward(0.0, 2.0, 1.0, 1.0) == 16.0
    assert navigation_reward(0.0, 2.0, 1.0, 0.4) == 11.0
    with pytest.raises(ValueError):
        navigation_reward(math.pi + 0.01, 2.0, 1.0, 1.0)
