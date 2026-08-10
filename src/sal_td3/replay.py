"""Rank-based prioritized replay with n-step returns."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    steps: int


class RankBasedNStepReplay:
    """Bounded rank-based PER buffer with terminal-safe n-step aggregation."""

    def __init__(
        self,
        capacity: int = 100_000,
        n_step: int = 3,
        gamma: float = 0.98,
        alpha: float = 0.6,
        priority_epsilon: float = 1e-6,
        seed: int = 42,
    ):
        if capacity <= 0 or n_step <= 0:
            raise ValueError("capacity and n_step must be positive")
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.alpha = alpha
        self.priority_epsilon = priority_epsilon
        self._storage: list[Transition] = []
        self._priorities: list[float] = []
        self._next_index = 0
        self._pending: deque[tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = deque()
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._storage)

    def add(self, state, action, reward: float, next_state, done: bool) -> None:
        item = (
            np.asarray(state, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        )
        self._pending.append(item)
        if len(self._pending) >= self.n_step:
            self._store_pending_prefix()
        if done:
            while self._pending:
                self._store_pending_prefix()

    def _store_pending_prefix(self) -> None:
        state, action, _, _, _ = self._pending[0]
        total_reward = 0.0
        final_next_state = self._pending[0][3]
        final_done = False
        used_steps = 0
        for used_steps, (_, _, reward, next_state, done) in enumerate(self._pending, start=1):
            total_reward += (self.gamma ** (used_steps - 1)) * reward
            final_next_state = next_state
            final_done = done
            if done or used_steps == self.n_step:
                break
        transition = Transition(
            state.copy(), action.copy(), total_reward, final_next_state.copy(), final_done, used_steps
        )
        max_priority = max(self._priorities, default=1.0)
        if len(self._storage) < self.capacity:
            self._storage.append(transition)
            self._priorities.append(max_priority)
        else:
            self._storage[self._next_index] = transition
            self._priorities[self._next_index] = max_priority
            self._next_index = (self._next_index + 1) % self.capacity
        self._pending.popleft()

    def sample(self, batch_size: int, beta: float = 0.4) -> dict[str, np.ndarray]:
        if batch_size <= 0 or batch_size > len(self):
            raise ValueError("batch_size must be positive and no larger than the buffer")
        priorities = np.asarray(self._priorities, dtype=np.float64)
        order = np.argsort(-priorities, kind="stable")
        ranks = np.empty(len(self), dtype=np.float64)
        ranks[order] = np.arange(1, len(self) + 1, dtype=np.float64)
        probabilities = ranks ** (-self.alpha)
        probabilities /= probabilities.sum()
        indices = self._rng.choice(len(self), size=batch_size, replace=False, p=probabilities)
        weights = (len(self) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        samples = [self._storage[index] for index in indices]
        return {
            "states": np.stack([item.state for item in samples]),
            "actions": np.stack([item.action for item in samples]),
            "rewards": np.asarray([item.reward for item in samples], dtype=np.float32),
            "next_states": np.stack([item.next_state for item in samples]),
            "dones": np.asarray([item.done for item in samples], dtype=np.float32),
            "steps": np.asarray([item.steps for item in samples], dtype=np.int64),
            "weights": weights.astype(np.float32),
            "indices": indices,
        }

    def update_priorities(self, indices, td_errors) -> None:
        indices = np.asarray(indices)
        td_errors = np.asarray(td_errors)
        if indices.shape != td_errors.shape:
            raise ValueError("indices and td_errors must have matching shapes")
        for index, error in zip(indices, td_errors, strict=True):
            self._priorities[int(index)] = float(abs(error)) + self.priority_epsilon
