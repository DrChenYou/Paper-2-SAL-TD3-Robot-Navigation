"""Twin-delayed actor-critic update logic for SAL-TD3."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch

from .networks import TemporalActor, TemporalCritic


class SALTD3Agent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 2,
        *,
        hidden_dim: int = 256,
        lstm_hidden_dim: int = 128,
        attention_heads: int = 4,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 8e-4,
        gamma: float = 0.98,
        tau: float = 0.005,
        policy_delay: int = 3,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        device: str | torch.device = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip
        self.updates = 0
        self.actor = TemporalActor(
            state_dim, action_dim, hidden_dim, lstm_hidden_dim, attention_heads
        ).to(self.device)
        self.critic_1 = TemporalCritic(
            state_dim, action_dim, hidden_dim, lstm_hidden_dim
        ).to(self.device)
        self.critic_2 = TemporalCritic(
            state_dim, action_dim, hidden_dim, lstm_hidden_dim
        ).to(self.device)
        self.actor_target = deepcopy(self.actor).eval()
        self.critic_1_target = deepcopy(self.critic_1).eval()
        self.critic_2_target = deepcopy(self.critic_2).eval()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_1_optimizer = torch.optim.Adam(
            self.critic_1.parameters(), lr=critic_learning_rate
        )
        self.critic_2_optimizer = torch.optim.Adam(
            self.critic_2.parameters(), lr=critic_learning_rate
        )

    @torch.no_grad()
    def act(self, state: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if tensor.ndim in (1, 2):
            tensor = tensor.unsqueeze(0)
        action = self.actor(tensor)
        if noise_std:
            action = action + torch.randn_like(action) * noise_std
        return action.clamp(0.0, 1.0).squeeze(0).cpu().numpy()

    def train_step(self, batch: dict[str, np.ndarray]) -> dict[str, float | np.ndarray]:
        states = self._tensor(batch["states"])
        actions = self._tensor(batch["actions"])
        rewards = self._tensor(batch["rewards"]).unsqueeze(-1)
        next_states = self._tensor(batch["next_states"])
        dones = self._tensor(batch["dones"]).unsqueeze(-1)
        steps = self._tensor(batch["steps"]).unsqueeze(-1)
        weights = self._tensor(batch["weights"]).unsqueeze(-1)

        with torch.no_grad():
            noise = torch.randn_like(actions) * self.target_policy_noise
            noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(0.0, 1.0)
            target_q = torch.minimum(
                self.critic_1_target(next_states, next_actions),
                self.critic_2_target(next_states, next_actions),
            )
            discount = torch.pow(torch.full_like(steps, self.gamma), steps)
            target = rewards + discount * (1.0 - dones) * target_q

        current_1 = self.critic_1(states, actions)
        current_2 = self.critic_2(states, actions)
        td_1 = current_1 - target
        td_2 = current_2 - target
        critic_1_loss = (weights * td_1.square()).mean()
        critic_2_loss = (weights * td_2.square()).mean()
        self._optimize(self.critic_1_optimizer, critic_1_loss)
        self._optimize(self.critic_2_optimizer, critic_2_loss)

        self.updates += 1
        actor_loss_value = float("nan")
        if self.updates % self.policy_delay == 0:
            actor_loss = -self.critic_1(states, self.actor(states)).mean()
            self._optimize(self.actor_optimizer, actor_loss)
            actor_loss_value = actor_loss.item()
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic_1, self.critic_1_target)
            self._soft_update(self.critic_2, self.critic_2_target)

        mean_td = ((td_1.abs() + td_2.abs()) * 0.5).detach().squeeze(-1).cpu().numpy()
        return {
            "critic_1_loss": critic_1_loss.item(),
            "critic_2_loss": critic_2_loss.item(),
            "actor_loss": actor_loss_value,
            "td_errors": mean_td,
        }

    def _tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    @staticmethod
    def _optimize(optimizer: torch.optim.Optimizer, loss: torch.Tensor) -> None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    def _soft_update(self, source: torch.nn.Module, target: torch.nn.Module) -> None:
        with torch.no_grad():
            for source_parameter, target_parameter in zip(
                source.parameters(), target.parameters(), strict=True
            ):
                target_parameter.mul_(1.0 - self.tau).add_(source_parameter, alpha=self.tau)
