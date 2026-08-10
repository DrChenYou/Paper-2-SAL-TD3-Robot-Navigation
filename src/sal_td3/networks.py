"""Neural components described for SAL-TD3."""

from __future__ import annotations

import torch
from torch import nn


def _as_sequence(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x.unsqueeze(1)
    if x.ndim != 3:
        raise ValueError("expected [batch, features] or [batch, time, features]")
    return x


def _initialize(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.xavier_uniform_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
        elif isinstance(child, nn.LSTM):
            for name, parameter in child.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(parameter)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(parameter)
                elif "bias" in name:
                    nn.init.zeros_(parameter)


class MultimodalObservationEncoder(nn.Module):
    """Fuse 360-sample LiDAR scans with goal/pose features."""

    def __init__(self, lidar_points: int = 360, goal_pose_dim: int = 4, output_dim: int = 256):
        super().__init__()
        self.lidar_points = lidar_points
        self.goal_pose_dim = goal_pose_dim
        self.lidar_branch = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(16),
            nn.Flatten(),
            nn.Linear(64 * 16, 128),
            nn.ReLU(),
        )
        self.goal_branch = nn.Sequential(nn.Linear(goal_pose_dim, 64), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(192, output_dim), nn.LayerNorm(output_dim), nn.ReLU())
        _initialize(self)

    def forward(self, lidar: torch.Tensor, goal_pose: torch.Tensor) -> torch.Tensor:
        lidar = _as_sequence(lidar)
        goal_pose = _as_sequence(goal_pose)
        if lidar.shape[:2] != goal_pose.shape[:2]:
            raise ValueError("LiDAR and goal/pose tensors must share batch and time dimensions")
        if lidar.shape[-1] != self.lidar_points or goal_pose.shape[-1] != self.goal_pose_dim:
            raise ValueError("observation feature dimensions do not match the encoder configuration")
        batch, steps, _ = lidar.shape
        lidar_features = self.lidar_branch(lidar.reshape(batch * steps, 1, self.lidar_points))
        goal_features = self.goal_branch(goal_pose.reshape(batch * steps, self.goal_pose_dim))
        fused = self.fusion(torch.cat((lidar_features, goal_features), dim=-1))
        return fused.reshape(batch, steps, -1)


class TemporalActor(nn.Module):
    """LSTM actor with four-head self-attention and bounded actions."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 2,
        hidden_dim: int = 256,
        lstm_hidden_dim: int = 128,
        attention_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if lstm_hidden_dim % attention_heads:
            raise ValueError("lstm_hidden_dim must be divisible by attention_heads")
        self.input = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()
        )
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True)
        self.lstm_dropout = nn.Dropout(dropout)
        self.attention = nn.MultiheadAttention(
            lstm_hidden_dim, attention_heads, dropout=dropout, batch_first=True
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(lstm_hidden_dim)
        self.output = nn.Sequential(
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim),
            nn.ReLU(),
            nn.Linear(lstm_hidden_dim, action_dim),
            nn.Sigmoid(),
        )
        _initialize(self)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        state = _as_sequence(state)
        sequence, _ = self.lstm(self.input(state))
        sequence = self.lstm_dropout(sequence)
        attended, _ = self.attention(sequence, sequence, sequence, need_weights=False)
        sequence = self.attention_norm(sequence + self.attention_dropout(attended))
        return self.output(sequence[:, -1])


class TemporalCritic(nn.Module):
    """LSTM state-action value estimator used twice by TD3."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 2,
        hidden_dim: int = 256,
        lstm_hidden_dim: int = 128,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.features = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU())
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True)
        self.value = nn.Linear(lstm_hidden_dim, 1)
        _initialize(self)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state = _as_sequence(state)
        if action.ndim == 2:
            action = action.unsqueeze(1).expand(-1, state.shape[1], -1)
        if action.ndim != 3 or action.shape[:2] != state.shape[:2]:
            raise ValueError("action must align with the state sequence")
        features = self.features(torch.cat((state, action), dim=-1))
        sequence, _ = self.lstm(features)
        return self.value(sequence[:, -1])
