"""
Q-Network for LunarLander.

MLP that maps an 8-dim state to Q-values over 4 discrete actions.
"""
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    # Two-hidden-layer MLP

    def __init__(self, state_dim: int = 8, action_dim: int = 4, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
